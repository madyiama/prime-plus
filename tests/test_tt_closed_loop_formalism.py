"""Test linear closed-loop AO physics independently of on-sky PSF fitting."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.telemetry.ttClosedLoop import (
    closed_loop_transfers,
    from_p3_tiptilt_loop,
    integrator_delay_transfers,
    keck_ii_mirror_ragland_2016,
    measured_tiptilt_cospectrum,
    one_sided_noise_spectral_density,
    propagate_spectral_covariances,
    simulate_integrator_delay,
)


def _cov(data):
    centered = data-data.mean(axis=0)
    return centered.T @ centered/len(centered)


def test_closed_loop_algebra_preserves_signal_location_and_signs():
    f = np.linspace(0, 62.5, 301)
    W = np.sinc(f/125)*np.exp(-1j*np.pi*f/125)
    P = np.exp(-2j*np.pi*f/125)
    K = 0.4*np.ones_like(f)
    model = closed_loop_transfers(
        f, sensor=W, controller=K, plant=P,
    )
    np.testing.assert_allclose(
        model.sensitivity + W*P*K*model.sensitivity,
        np.ones_like(f), atol=1e-14,
    )
    np.testing.assert_allclose(
        model.measured_disturbance, W*model.physical_disturbance
    )
    np.testing.assert_allclose(
        model.measured_sensor_noise, model.sensitivity
    )
    np.testing.assert_allclose(
        model.physical_sensor_noise,
        -model.plant*model.command_sensor_noise,
    )
    assert model.physical_sensor_noise[0] != model.measured_sensor_noise[0]


def test_closed_loop_plant_can_use_published_keck_ii_mirror():
    f = np.linspace(0, 62.5, 101)
    mirror = keck_ii_mirror_ragland_2016(f)
    assert mirror[0] == pytest.approx(1.0+0j)
    assert np.all(np.isfinite(mirror))
    assert np.any(abs(mirror[1:]-1) > 1e-5)

    p3 = NS(tf=NS(
        freq=f, wfs=np.ones_like(f), servo=np.ones_like(f)*0.35,
        lag=np.exp(-2j*np.pi*f/125),
    ))
    p3.tf.rtf = 1/(1+p3.tf.wfs*p3.tf.servo*p3.tf.lag)
    legacy = from_p3_tiptilt_loop(p3, mirror_model="unity")
    published = from_p3_tiptilt_loop(
        p3, mirror_model="ragland_2016_keck_ii",
    )
    np.testing.assert_allclose(legacy.sensitivity, p3.tf.rtf)
    assert not np.allclose(
        published.physical_sensor_noise,
        legacy.physical_sensor_noise, rtol=1e-9, atol=1e-12,
    )


def test_one_sided_white_noise_integrates_to_sensor_covariance():
    fs = 125.0
    cn = np.array([[4, 1], [1, 9]])*1e-14
    f = np.linspace(0, fs/2, 1001)
    pn = one_sided_noise_spectral_density(cn, fs)
    np.testing.assert_allclose(
        np.trapezoid(np.broadcast_to(pn[:,:,None], (2,2,len(f))), f, axis=-1),
        cn, rtol=1e-12,
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        one_sided_noise_spectral_density(np.diag([-1e-13, 2e-13]), fs)


def test_spectral_decomposition_matches_exact_gaussian_linear_model():
    fs = 125.0
    f = np.linspace(0, fs/2, 1001)
    W = np.sinc(f/fs)*np.exp(-1j*np.pi*f/fs)
    P = np.exp(-2j*np.pi*f/fs)
    K = np.full_like(f, 0.30)
    model = closed_loop_transfers(f, sensor=W, controller=K, plant=P)
    cn = np.array([[4, 0.8], [0.8, 9]])*1e-14
    cd = np.array([[2, 0.5], [0.5, 6]])*1e-13
    noise_density = one_sided_noise_spectral_density(cn, fs)
    disturbance_density = (2/fs)*cd[:,:,None]*(
        1+0.4*np.cos(np.pi*f/(fs/2))
    )[None,None,:]
    measured_psd = (
        np.abs(model.measured_disturbance)[None,None,:]**2*disturbance_density
        + np.abs(model.measured_sensor_noise)[None,None,:]**2
        * noise_density[:,:,None]
    )
    expected_atmos = np.trapezoid(
        np.abs(model.physical_disturbance)[None,None,:]**2
        * disturbance_density, f, axis=-1,
    )
    expected_noise = np.trapezoid(
        np.abs(model.physical_sensor_noise)[None,None,:]**2
        * noise_density[:,:,None], f, axis=-1,
    )
    result = propagate_spectral_covariances(
        model, measured_psd, cn, fs,
    )
    np.testing.assert_allclose(
        result["physical_atmos_cov_m2"], expected_atmos, rtol=1e-11,
    )
    np.testing.assert_allclose(
        result["physical_noise_cov_m2"], expected_noise, rtol=1e-11,
    )
    np.testing.assert_allclose(
        result["physical_total_cov_m2"],
        expected_atmos+expected_noise, rtol=1e-11,
    )
    assert result["physical_atmos_min_eig_m2"] > 0
    assert not result["is_validated_for_observation"]


def test_integrator_transfer_identity_matches_causal_recurrence():
    fs, gain = 125.0, 0.35
    f = np.linspace(0, fs/2, 4097)
    S, T = integrator_delay_transfers(f, fs, gain)
    np.testing.assert_allclose(S+T, np.ones_like(S), atol=2e-14)
    assert abs(S[0]) < 1e-14
    assert abs(T[0]-1) < 1e-14
    # Independent time-domain simulation of a white-noise-only loop.
    rng = np.random.default_rng(981)
    n = rng.normal(0, (2e-7, 3e-7), size=(65536,2))
    d = np.zeros_like(n)
    sim = simulate_integrator_delay(d, n, gain)
    observed = sim["measured_centroids_m"][2048:]
    physical = sim["physical_residual_m"][2048:]
    actual_meas = np.diag(_cov(observed))
    actual_physical = np.diag(_cov(physical))
    noise_var = np.array([(2e-7)**2, (3e-7)**2])
    predicted_meas = noise_var * np.trapezoid(np.abs(S)**2, f)/(fs/2)
    predicted_physical = noise_var * np.trapezoid(np.abs(T)**2, f)/(fs/2)
    np.testing.assert_allclose(actual_meas, predicted_meas, rtol=0.07)
    np.testing.assert_allclose(actual_physical, predicted_physical, rtol=0.07)


def test_measured_psd_recovers_physical_variance_from_known_noise():
    fs, gain = 125.0, 0.35
    rng = np.random.default_rng(28)
    n = rng.normal(0, (2e-7, 3e-7), size=(65536,2))
    sim = simulate_integrator_delay(np.zeros_like(n), n, gain)
    measured = sim["measured_centroids_m"][1024:]
    physical = sim["physical_residual_m"][1024:]
    freq, power = measured_tiptilt_cospectrum(measured, fs, nperseg=1024)
    S, T = integrator_delay_transfers(freq, fs, gain)
    # For this special integrator, S and -T are exact measured-noise and
    # physical-noise responses, so set W=1 in an equivalent TTClosedLoop.
    from p3.telemetry.ttClosedLoop import TTClosedLoop
    model = TTClosedLoop(
        frequency_hz=freq, sensor=np.ones_like(S),
        controller=np.full_like(S, np.nan), plant=np.full_like(S, np.nan),
        loop=np.full_like(S, np.nan), sensitivity=S,
        measured_disturbance=S, measured_sensor_noise=S,
        physical_disturbance=S, physical_sensor_noise=-T,
        command_sensor_noise=-T,
    )
    cn = np.diag([(2e-7)**2, (3e-7)**2])
    result = propagate_spectral_covariances(model, power, cn, fs)
    predicted = np.diag(np.asarray(result["physical_total_cov_m2"]))
    actual = np.diag(_cov(physical))
    np.testing.assert_allclose(predicted, actual, rtol=0.13)


def test_transfer_and_noise_validation_reject_nonphysical_inputs():
    f = np.array([0., 10., 20.])
    with pytest.raises(ValueError, match="strictly increasing"):
        closed_loop_transfers(f[::-1], sensor=1., controller=0.5, plant=1.)
    with pytest.raises(ValueError, match="Near-singular"):
        closed_loop_transfers(f, sensor=1., controller=-1., plant=1.)
    with pytest.raises(ValueError, match="Stable integrator gain"):
        integrator_delay_transfers(f, 125., 2.1)
    full_freq = np.array([0., 31.25, 62.5])
    with pytest.raises(ValueError, match="near-zero WFS"):
        model = closed_loop_transfers(
            full_freq, sensor=0., controller=0.5, plant=1.
        )
        propagate_spectral_covariances(
            model, np.zeros((2,2,3)), np.eye(2)*1e-13, 125.
        )
    with pytest.raises(ValueError, match="full one-sided spectrum"):
        incomplete = closed_loop_transfers(
            np.array([5., 10., 20.]), sensor=1., controller=0.5, plant=1.
        )
        propagate_spectral_covariances(
            incomplete, np.zeros((2,2,3)), np.eye(2)*1e-13, 125.
        )


def test_measured_psd_recovers_physical_jitter_with_correlated_turbulence():
    """Independent closed-loop time series with colored disturbance and noise."""
    fs, gain, n_samples = 125.0, 0.35, 65536
    rng = np.random.default_rng(222)
    noise = rng.normal(0, (2e-7, 3e-7), (n_samples, 2))
    innovation = rng.normal(0, (5e-8, 7e-8), (n_samples, 2))
    disturbance = np.zeros_like(noise)
    for i in range(1, n_samples):
        disturbance[i] = 0.98*disturbance[i-1] + innovation[i]
    sim = simulate_integrator_delay(disturbance, noise, gain)
    y = sim["measured_centroids_m"][2048:]
    e = sim["physical_residual_m"][2048:]
    frequency, power = measured_tiptilt_cospectrum(y, fs, nperseg=1024)
    S, T = integrator_delay_transfers(frequency, fs, gain)
    from p3.telemetry.ttClosedLoop import TTClosedLoop
    # Effective *closed-loop* responses exactly derived for this integrator;
    # the open-loop K has a pole at f=0, so test the finite S/T forms.
    model = TTClosedLoop(
        frequency_hz=frequency,
        sensor=np.ones_like(S),
        controller=np.full_like(S, np.nan),
        plant=np.full_like(S, np.nan),
        loop=np.full_like(S, np.nan),
        sensitivity=S, measured_disturbance=S,
        measured_sensor_noise=S, physical_disturbance=S,
        physical_sensor_noise=-T, command_sensor_noise=-T,
    )
    covariance = np.diag([(2e-7)**2, (3e-7)**2])
    result = propagate_spectral_covariances(model, power, covariance, fs)
    physical_from_spectra = np.diag(result["physical_total_cov_m2"])
    physical_from_time_series = np.diag(_cov(e))
    np.testing.assert_allclose(
        physical_from_spectra, physical_from_time_series, rtol=0.12
    )
    assert np.trace(result["physical_atmos_cov_m2"]) > 0
    assert np.trace(result["physical_noise_cov_m2"]) > 0


def test_p3_pn_equals_physical_noise_gain_only_for_unit_modulus_plant():
    """Historical P3 ntf=K*S has physical meaning iff |P|=1."""
    fs = 125.0
    f = np.linspace(0, fs/2, 4097)
    W = np.sinc(f/fs)*np.exp(-1j*np.pi*f/fs)
    K = np.full(f.shape, 0.35+0j)
    pure_delay = np.exp(-2j*np.pi*f/fs)
    model_delay = closed_loop_transfers(
        f, sensor=W, controller=K, plant=pure_delay,
    )
    historical_gain = 2/fs * np.trapezoid(
        np.abs(model_delay.command_sensor_noise)**2, f,
    )
    physical_gain = 2/fs * np.trapezoid(
        np.abs(model_delay.physical_sensor_noise)**2, f,
    )
    assert physical_gain == pytest.approx(historical_gain, rel=1e-12)
    mirror = keck_ii_mirror_ragland_2016(f)
    model_mirror = closed_loop_transfers(
        f, sensor=W, controller=K, plant=pure_delay*mirror,
    )
    actual_mirror_gain = 2/fs * np.trapezoid(
        np.abs(model_mirror.physical_sensor_noise)**2, f,
    )
    command_only_gain = 2/fs * np.trapezoid(
        np.abs(model_mirror.command_sensor_noise)**2, f,
    )
    assert abs(actual_mirror_gain - command_only_gain) > 0.01*command_only_gain
