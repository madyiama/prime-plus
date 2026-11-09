"""Closed-loop tip-tilt propagation with explicit physical signal locations.

References
----------
van Dam et al. (2006), PASP 118, 310-318, section 2.4 and error-budget
equations: Keck STRAP closed-loop PSD and measurement-noise propagation.
Juvénal, Kulcsár, Raynaud & Conan (2018), JOSA A 35, 1465-1476:
linear AO controller error budgets from disturbance PSD and controller TFs.
Ragland et al. (2016), Proc. SPIE 9909, 99091P, section 5.2:
calibrated Keck II tip-tilt mirror dynamic response.

Definitions (complex Fourier transfer functions, positive-feedback sign
absorbed into the controller's negative-feedback convention):
  d : physical incoming tip-tilt disturbance, RMS Zernike OPD [m]
  e : physical residual image-motion tilt [m]
  y : measured residual STRAP centroids [m]
  n : additive white sensor measurement noise [m]
  u : control command, in a unit mapped by plant P to physical tilt
  W : sensor integration/measurement response, dimensionless
  K : discrete controller response
  P : actuator/command-to-physical-correction response, including latency

  e = d - P*u ; y = W*e + n ; u = K*y
  L = W*P*K ; S = (1+L)^(-1)
  y = W*S*d + S*n
  e = S*d - P*K*S*n

These are LINEAR equations. No source fitting, Strehl optimization or
empirical coefficient is used. P3's existing loop data do not independently
validate W, K, the physical plant P or the Keck controller gain.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch, csd


@dataclass(frozen=True)
class TTClosedLoop:
    """Complex transfer functions on a one-sided frequency grid in Hz."""
    frequency_hz: np.ndarray
    sensor: np.ndarray
    controller: np.ndarray
    plant: np.ndarray
    loop: np.ndarray
    sensitivity: np.ndarray
    measured_disturbance: np.ndarray
    measured_sensor_noise: np.ndarray
    physical_disturbance: np.ndarray
    physical_sensor_noise: np.ndarray
    command_sensor_noise: np.ndarray


def keck_ii_mirror_ragland_2016(frequency_hz):
    """Published Keck II TT mirror model, Ragland et al. (2016) §5.2.

    H_TTM(s) = 2.55e10 / (2.55e10 + 4.5e7*s + 4e3*s^2 + s^3).
    This historical calibration is NOT verified for the 2020 dataset and
    must not become a silent instrument default.
    """
    freq = np.asarray(frequency_hz, dtype=float)
    if not np.all(np.isfinite(freq)) or np.any(freq < 0):
        raise ValueError("Frequency must be finite and non-negative")
    s = 2j*np.pi*freq
    return 2.55e10 / (2.55e10 + 4.5e7*s + 4e3*s**2 + s**3)


def closed_loop_transfers(frequency_hz, *, sensor, controller, plant):
    """Return exact linear negative-feedback transfers for given W, K, P."""
    f = np.asarray(frequency_hz, dtype=float)
    if f.ndim != 1 or len(f) < 2 or not np.all(np.isfinite(f)):
        raise ValueError("Frequency grid must be a finite one-dimensional array")
    if np.any(np.diff(f) <= 0) or f[0] < 0:
        raise ValueError("Frequency grid must be strictly increasing and non-negative")
    W = np.broadcast_to(np.asarray(sensor, dtype=complex), f.shape)
    K = np.broadcast_to(np.asarray(controller, dtype=complex), f.shape)
    P = np.broadcast_to(np.asarray(plant, dtype=complex), f.shape)
    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(K)) and
            np.all(np.isfinite(P))):
        raise ValueError("Loop transfers must be finite")
    L = W*P*K
    denominator = 1.0 + L
    if np.any(np.abs(denominator) < 1e-10):
        raise ValueError("Near-singular closed-loop denominator (1+WPK)")
    S = 1.0 / denominator
    return TTClosedLoop(
        frequency_hz=f, sensor=W, controller=K, plant=P, loop=L,
        sensitivity=S, measured_disturbance=W*S,
        measured_sensor_noise=S, physical_disturbance=S,
        physical_sensor_noise=-P*K*S, command_sensor_noise=K*S,
    )


def from_p3_tiptilt_loop(ttloop, mirror_model="unity"):
    """Build a model from P3's WFS, servo, lag terms without altering P3.

    mirror_model='unity' is the HISTORICAL P3 assumption P=lag.
    mirror_model='ragland_2016_keck_ii' applies published Keck II mirror
    dynamics to the existing latency response for sensitivity studies.
    Neither option independently calibrates an observation's actuator.
    """
    tf = ttloop.tf
    f = np.asarray(tf.freq, dtype=float)
    if mirror_model == "unity":
        mirror = np.ones_like(f)
    elif mirror_model == "ragland_2016_keck_ii":
        mirror = keck_ii_mirror_ragland_2016(f)
    else:
        raise ValueError("Unsupported mirror model: " + str(mirror_model))
    model = closed_loop_transfers(
        f, sensor=tf.wfs, controller=tf.servo,
        plant=np.asarray(tf.lag)*mirror,
    )
    if mirror_model == "unity" and not np.allclose(
        model.sensitivity, tf.rtf, rtol=1e-8, atol=1e-9,
    ):
        raise ValueError("P3 tip-tilt transfer definitions are inconsistent")
    return model


def one_sided_noise_spectral_density(sensor_noise_cov_m2, sample_rate_hz):
    """White two-sided sampled noise -> one-sided 2*Cn/fs [m²/Hz].

    The one-sided PSD integrates to Cn on [0, fs/2] in the continuum
    approximation; DC and Nyquist bins have zero measure here.
    """
    cov = np.asarray(sensor_noise_cov_m2, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        raise ValueError("Sensor covariance must be a finite 2x2 array")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive and finite")
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-25):
        raise ValueError("Sensor noise covariance must be symmetric")
    if np.linalg.eigvalsh(cov).min() < -1e-12*max(np.linalg.norm(cov), 1e-30):
        raise ValueError("Sensor noise covariance must be positive semidefinite")
    return 2.0 * cov / sample_rate_hz


def measured_tiptilt_cospectrum(samples_opd_m, sample_rate_hz, nperseg=512):
    """Estimate one-sided symmetric 2x2 cospectral density via Welch."""
    data = np.asarray(samples_opd_m, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 128:
        raise ValueError("STRAP series must have shape (n>=128,2)")
    if not np.all(np.isfinite(data)):
        raise ValueError("STRAP series must be finite")
    if nperseg < 128 or nperseg > len(data):
        raise ValueError("Welch nperseg must be >=128 and <= number of samples")
    kwargs = dict(fs=sample_rate_hz, window="hann", nperseg=nperseg,
                  noverlap=nperseg//2, detrend="constant", scaling="density")
    frequency, xx = welch(data[:, 0], **kwargs)
    _, yy = welch(data[:, 1], **kwargs)
    _, xy = csd(data[:, 0], data[:, 1], **kwargs)
    power = np.empty((2, 2, len(frequency)), dtype=float)
    power[0, 0], power[1, 1] = xx, yy
    power[0, 1] = power[1, 0] = np.real(xy)
    return frequency, power


def _integrate_cospectrum(power, freq):
    matrix = trapezoid(power, x=freq, axis=-1)
    return (matrix + matrix.T)/2


def propagate_spectral_covariances(
    model: TTClosedLoop, measured_cospectrum_m2_hz,
    sensor_noise_cov_m2, sample_rate_hz,
):
    """Separate measured residuals and physical residual in PSD space.

    The fitted white noise Cn is sensor-input covariance, not a physical
    jitter covariance. Assumes stationary, independent d and n.

    P_y(f) = |W*S|² P_d(f) + |S|² P_n(f)
    P_e(f) = |S|² P_d(f) + |P*K*S|² P_n(f)

    Eliminate the unknown incident disturbance:
      P_e,atmos = (P_y - |S|² P_n)/|W|²
      P_e,noise = |P*K*S|² P_n
    and integrate over [0, fs/2]. No clipping of negative residual spectra.
    """
    f = model.frequency_hz
    fs = float(sample_rate_hz)
    if f[0] < 0 or f[-1] > fs/2 + 1e-8:
        raise ValueError("Frequency grid lies outside [0, Nyquist]")
    full_spectrum = (
        f[0] <= 0.001*fs/2
        and abs(f[-1]-fs/2) <= 0.001*fs/2
    )
    if not full_spectrum:
        raise ValueError(
            "Physical covariance requires the full one-sided spectrum "
            "from DC through Nyquist; a partial band is not a covariance."
        )
    power = np.asarray(measured_cospectrum_m2_hz, dtype=float)
    if power.shape != (2, 2, len(f)) or not np.all(np.isfinite(power)):
        raise ValueError("Measured cospectrum must have shape (2,2,n_frequency)")
    noise_density = one_sided_noise_spectral_density(sensor_noise_cov_m2, fs)
    if np.any(np.abs(model.sensor)**2 < 1e-6):
        raise ValueError("Cannot invert a near-zero WFS response")
    noise_y = noise_density[:, :, None] * (
        np.abs(model.measured_sensor_noise)**2
    )[None, None, :]
    measured_atmos = power - noise_y
    # Physical residual atmospheric disturbance = measured atmospheric
    # residual / |W|². No inversion of S (which can be zero at DC).
    physical_atmos = measured_atmos / (
        np.abs(model.sensor)**2
    )[None, None, :]
    physical_noise = noise_density[:, :, None] * (
        np.abs(model.physical_sensor_noise)**2
    )[None, None, :]
    measured_noise_cov = _integrate_cospectrum(noise_y, f)
    measured_atmos_cov = _integrate_cospectrum(measured_atmos, f)
    physical_atmos_cov = _integrate_cospectrum(physical_atmos, f)
    physical_noise_cov = _integrate_cospectrum(physical_noise, f)
    physical_total_cov = physical_atmos_cov + physical_noise_cov
    noise_gain_measured = 2 / fs * trapezoid(
        np.abs(model.measured_sensor_noise)**2, x=f,
    )
    noise_gain_physical = 2 / fs * trapezoid(
        np.abs(model.physical_sensor_noise)**2, x=f,
    )
    command_noise_gain = 2 / fs * trapezoid(
        np.abs(model.command_sensor_noise)**2, x=f,
    )
    return {
        "frequency_min_hz": float(f[0]),
        "frequency_max_hz": float(f[-1]),
        "spectral_coverage_full_nyquist": True,
        "white_sensor_noise_cov_m2": np.asarray(sensor_noise_cov_m2).tolist(),
        "measured_residual_cov_m2": _integrate_cospectrum(power, f).tolist(),
        "measured_noise_cov_m2": measured_noise_cov.tolist(),
        "measured_atmos_cov_m2": measured_atmos_cov.tolist(),
        "physical_atmos_cov_m2": physical_atmos_cov.tolist(),
        "physical_noise_cov_m2": physical_noise_cov.tolist(),
        "physical_total_cov_m2": physical_total_cov.tolist(),
        "gain_sensor_to_measured_noise": float(noise_gain_measured),
        "gain_sensor_to_physical_noise": float(noise_gain_physical),
        "gain_sensor_to_command_noise": float(command_noise_gain),
        "physical_atmos_min_eig_m2": float(
            np.linalg.eigvalsh(physical_atmos_cov).min()
        ),
        "physical_total_min_eig_m2": float(
            np.linalg.eigvalsh(physical_total_cov).min()
        ),
        "is_validated_for_observation": False,
        "assumptions": [
            "linear time-invariant negative feedback",
            "independent stationary disturbance and white sensor noise",
            "W, K, plant P calibrated independently (not done for Keck n0013)",
            "cospectral estimates may have negative residual bins; not clipped",
            "finite Welch PSD resolution; covariance is an estimator",
        ],
    }


def integrator_delay_transfers(frequency_hz, sample_rate_hz, gain):
    """Exact transfer of the discrete unit-WFS, one-frame-delay integrator.

    Loop equations:
      e[k] = d[k] - u[k-1]
      y[k] = e[k] + n[k]
      u[k] = u[k-1] + gain*y[k]

    With q=exp(-i*2pi*f/fs) and den=1+(gain-1)*q:
      S=(1-q)/den, P*K*S=gain*q/den, and S+P*K*S=1.
    These stable expressions avoid the integrator pole singularity at DC.
    """
    freq = np.asarray(frequency_hz, dtype=float)
    if not np.isfinite(gain) or not 0 < gain < 2:
        raise ValueError("Stable integrator gain must satisfy 0 < gain < 2")
    if sample_rate_hz <= 0 or not np.isfinite(sample_rate_hz):
        raise ValueError("Sample rate must be positive")
    q = np.exp(-2j*np.pi*freq/sample_rate_hz)
    den = 1+(gain-1)*q
    S = (1-q)/den
    T = gain*q/den
    return S, T


def simulate_integrator_delay(disturbance_opd_m, noise_opd_m, gain):
    """Independent causal time-domain simulation; input arrays shape (N,2)."""
    disturbance = np.asarray(disturbance_opd_m, dtype=float)
    noise = np.asarray(noise_opd_m, dtype=float)
    if (disturbance.shape != noise.shape or disturbance.ndim != 2
            or disturbance.shape[1] != 2 or
            not np.all(np.isfinite(disturbance)) or
            not np.all(np.isfinite(noise))):
        raise ValueError("Disturbance and noise must be finite (N,2) arrays")
    if not np.isfinite(gain) or not 0 < gain < 2:
        raise ValueError("Stable integrator gain must satisfy 0 < gain < 2")
    n = len(disturbance)
    residual = np.empty_like(disturbance)
    measured = np.empty_like(disturbance)
    command = np.empty_like(disturbance)
    u_last = np.zeros(2)
    for t in range(n):
        residual[t] = disturbance[t] - u_last
        measured[t] = residual[t] + noise[t]
        u_last = u_last + gain*measured[t]
        command[t] = u_last
    return {"physical_residual_m": residual, "measured_centroids_m": measured,
            "control_command_m": command}
