"""Keck STRAP tip-tilt white measurement-noise estimators.

Inputs: residual centroids in metres of RMS Zernike tip-tilt OPD.
Outputs: white *sensor* noise covariance in m², not loop-propagated jitter.
Van Dam (2006) uses a loop-rejection-corrected high-frequency PSD.
Gendron & Lena (1995) uses quadratic extrapolation of temporal covariance.
"""
import warnings

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch, csd


TT_METHODS = ("van_dam", "gendron", "legacy_autocorrelation")


def _lag_covariance(data, lag):
    a = data[lag:].T @ data[:-lag] / (len(data) - lag)
    return (a + a.T)/2


def _time_domain(data, method):
    c0 = data.T @ data / len(data)
    if method == "gendron":
        c1, c2 = _lag_covariance(data, 1), _lag_covariance(data, 2)
        # Extrapolate C_atmos(tau)=a+b*tau² to lag zero using lag 1, 2.
        cov = c0 - (4*c1-c2)/3
    else:
        # Historical P3 one-frame cyclic autocorrelation estimator.
        cov = 0.5 * data.T @ (
            2*data-np.roll(data, -1, axis=0)-np.roll(data, 1, axis=0)
        ) / len(data)
    cov = (cov + cov.T)/2
    return cov, {
        "method": method, "observed_covariance_m2": c0.tolist(),
        "noise_covariance_m2": cov.tolist(),
        "noise_model_validated": False, "units": "m² RMS-Zernike tip-tilt OPD",
    }


def _trimmed_mean(values, fraction):
    vals = np.sort(np.asarray(values, dtype=float))
    trim = int(len(vals)*fraction)
    if 2*trim >= len(vals):
        raise ValueError("Too few high-frequency bins")
    return float(vals[trim:len(vals)-trim].mean()) if trim else float(vals.mean())


def _project_covariance_psd(cov):
    cov = (cov + cov.T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues.min() < -1e-10*max(np.abs(eigenvalues).max(), 1e-30):
        warnings.warn(
            "van_dam estimated non-PSD noise covariance; projecting and "
            "preserving unprojected eigenvalues in diagnostics.",
            RuntimeWarning, stacklevel=2,
        )
    epos = np.maximum(eigenvalues, 0)
    projected = (eigenvectors * epos) @ eigenvectors.T
    return (projected + projected.T)/2, eigenvalues


def estimate_tt_noise(
    samples_opd_m, *, method="van_dam", sample_rate_hz=None,
    rejection_frequency_hz=None, rejection_transfer=None,
    high_frequency_band=(0.65, 0.95), nperseg=512, trim_fraction=0.10,
):
    """Estimate sensor noise covariance from STRAP residuals.

    For van_dam, rejection_transfer is S(f)=1/(1+H(f)), not the
    noise-to-mirror transfer function. The high-frequency band is
    expressed as a fraction of Nyquist, where white noise must dominate.
    """
    if method not in TT_METHODS:
        raise ValueError("Unknown TT noise method: " + str(method))
    data = np.asarray(samples_opd_m, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 8:
        raise ValueError("STRAP samples must be an (n>=8, 2) array")
    if not np.isfinite(data).all():
        raise ValueError("STRAP samples must be finite")
    data = data - np.mean(data, axis=0)
    if method != "van_dam":
        return _time_domain(data, method)

    if (sample_rate_hz is None or rejection_frequency_hz is None
            or rejection_transfer is None):
        raise ValueError("van_dam requires the sampling rate and loop rejection")
    fs = float(sample_rate_hz)
    nperseg = int(nperseg)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sample rate must be positive")
    if nperseg < 128 or len(data) < nperseg:
        raise ValueError("van_dam requires at least nperseg >= 128 samples")
    lo, hi = [float(x) for x in high_frequency_band]
    if not (0 < lo < hi < 1) or not (0 <= trim_fraction < 0.5):
        raise ValueError("Invalid frequency band or trim fraction")

    f_loop = np.asarray(rejection_frequency_hz, dtype=float).reshape(-1)
    s_loop = np.asarray(rejection_transfer).reshape(-1)
    if (len(f_loop) < 2 or len(f_loop) != len(s_loop)
            or not np.all(np.isfinite(f_loop))
            or not np.all(np.isfinite(s_loop))
            or not np.all(np.diff(f_loop) > 0)):
        raise ValueError("Invalid frequency grid or loop rejection")
    kwargs = dict(fs=fs, nperseg=nperseg, noverlap=nperseg//2,
                  window="hann", detrend="constant", scaling="density")
    f, pxx = welch(data[:, 0], **kwargs)
    _, pyy = welch(data[:, 1], **kwargs)
    _, pxy = csd(data[:, 0], data[:, 1], **kwargs)
    rejection_power = np.interp(
        f, f_loop, np.abs(s_loop)**2, left=np.nan, right=np.nan
    )
    nyquist = fs/2
    fit = (f >= lo*nyquist) & (f <= hi*nyquist)
    if np.count_nonzero(fit) < 8:
        raise ValueError("Too few spectral bins in van_dam fit band")
    if (not np.isfinite(rejection_power[fit]).all()
            or np.any(rejection_power[fit] < 1e-4)):
        raise ValueError("Rejection model does not cover the fitting band")

    # In sensor-centroid PSD: P_measurement_noise(f)=|S(f)|² * N0.
    # N0 is a one-sided white noise floor in m²/Hz.
    floor = np.array([
        [_trimmed_mean(pxx[fit]/rejection_power[fit], trim_fraction),
         _trimmed_mean(np.real(pxy[fit])/rejection_power[fit], trim_fraction)],
        [_trimmed_mean(np.real(pxy[fit])/rejection_power[fit], trim_fraction),
         _trimmed_mean(pyy[fit]/rejection_power[fit], trim_fraction)],
    ])
    # White noise variance = integral of one-sided PSD over [0, fs/2].
    raw_cov = floor * nyquist
    cov, eigenvalues = _project_covariance_psd(raw_cov)
    observed = data.T @ data / len(data)

    diag = {
        "method": method,
        "sample_count": int(len(data)),
        "sampling_frequency_hz": fs,
        "welch_nperseg": nperseg,
        "welch_noverlap": nperseg//2,
        "welch_segment_count": int(1+(len(data)-nperseg)//(nperseg//2)),
        "high_frequency_band_fraction_nyquist": [lo, hi],
        "high_frequency_band_hz": [lo*nyquist, hi*nyquist],
        "fit_bin_count": int(fit.sum()),
        "trim_fraction": float(trim_fraction),
        "noise_floor_one_sided_m2_per_hz": floor.tolist(),
        "unprojected_noise_covariance_m2": raw_cov.tolist(),
        "unprojected_noise_eigenvalues_m2": eigenvalues.tolist(),
        "noise_covariance_m2": cov.tolist(),
        "observed_covariance_m2": observed.tolist(),
        "observed_variance_from_psd_m2": [
            float(trapezoid(pxx, f)), float(trapezoid(pyy, f)),
        ],
        "frequency_hz": f.tolist(),
        "high_frequency_fit_mask": fit.tolist(),
        "measured_psd_x_m2_per_hz": pxx.tolist(),
        "measured_psd_y_m2_per_hz": pyy.tolist(),
        "measured_cross_psd_real_m2_per_hz": np.real(pxy).tolist(),
        "rejection_power": rejection_power.tolist(),
        "model_noise_psd_x_m2_per_hz": (floor[0, 0]*rejection_power).tolist(),
        "model_noise_psd_y_m2_per_hz": (floor[1, 1]*rejection_power).tolist(),
        "noise_model_validated": False,
        "units": "m² RMS-Zernike tip-tilt OPD",
    }
    if np.any(np.diag(cov) >= np.diag(observed)):
        warnings.warn(
            "van_dam noise variance exceeds the observed TT variance on "
            "at least one axis; inspect gain, vibrations and fit band.",
            RuntimeWarning, stacklevel=2,
        )
    return cov, diag
