"""Synthetic validation of van Dam, Gendron, and legacy STRAP noise estimates."""
import numpy as np
import pytest

from p3.telemetry.ttNoise import TT_METHODS, estimate_tt_noise


def _filtered_white_noise(n=32768, sample_rate=125.0, seed=18):
    rng = np.random.default_rng(seed)
    sigmas = np.array([2.0e-7, 3.0e-7])
    white = rng.standard_normal((n, 2)) * sigmas
    freq = np.fft.rfftfreq(n, d=1/sample_rate)
    # Synthetic rejection transfer, high-pass-ish and bounded away from zero.
    sf = 0.80 + 0.20 * (freq/(sample_rate/2))**2
    observed = np.fft.irfft(
        np.fft.rfft(white, axis=0) * sf[:, None],
        n=n, axis=0,
    )
    f_model = np.linspace(0, sample_rate/2, 4096)
    s_model = 0.80 + 0.20 * (f_model/(sample_rate/2))**2
    return observed, sigmas, f_model, s_model


def test_van_dam_recovers_known_input_noise_covariance():
    observed, sigmas, freq, rejection = _filtered_white_noise()
    covariance, diagnostics = estimate_tt_noise(
        observed, method="van_dam", sample_rate_hz=125.0,
        rejection_frequency_hz=freq, rejection_transfer=rejection,
        nperseg=1024,
    )
    np.testing.assert_allclose(np.diag(covariance), sigmas**2, rtol=0.17)
    assert np.linalg.eigvalsh(covariance).min() >= 0
    assert diagnostics["method"] == "van_dam"
    assert diagnostics["welch_segment_count"] > 10
    assert diagnostics["fit_bin_count"] >= 8
    assert not diagnostics["noise_model_validated"]


def test_van_dam_ignores_slow_turbulence_in_high_frequency_band():
    observed, sigmas, freq, rejection = _filtered_white_noise()
    t = np.arange(len(observed)) / 125.0
    atmospheric = np.column_stack((
        2.0e-6*np.sin(2*np.pi*1.0*t),
        1.5e-6*np.cos(2*np.pi*1.4*t),
    ))
    noisy = observed + atmospheric
    covariance, _ = estimate_tt_noise(
        noisy, method="van_dam", sample_rate_hz=125.0,
        rejection_frequency_hz=freq, rejection_transfer=rejection,
        nperseg=1024,
    )
    np.testing.assert_allclose(np.diag(covariance), sigmas**2, rtol=0.17)


def test_van_dam_psd_integral_agrees_with_observed_variance():
    observed, _, freq, rejection = _filtered_white_noise()
    _, diagnostics = estimate_tt_noise(
        observed, sample_rate_hz=125.0,
        rejection_frequency_hz=freq, rejection_transfer=rejection,
        nperseg=1024,
    )
    sampled = np.diag(np.asarray(diagnostics["observed_covariance_m2"]))
    psd_integrated = np.asarray(diagnostics["observed_variance_from_psd_m2"])
    np.testing.assert_allclose(psd_integrated, sampled, rtol=0.10)


def test_gendron_quadratic_extrapolation_white_noise_and_slow_signal():
    rng = np.random.default_rng(47)
    n = 65536
    time = np.arange(n)/125
    measurement_noise = rng.normal(0, [2e-7, 3e-7], (n, 2))
    slow = np.column_stack([
        1e-6 * np.sin(2*np.pi*0.7*time),
        1e-6 * np.cos(2*np.pi*0.5*time),
    ])
    cov, diag = estimate_tt_noise(slow+measurement_noise, method="gendron")
    np.testing.assert_allclose(np.diag(cov), [4e-14, 9e-14], rtol=0.15)
    assert diag["method"] == "gendron"


def test_legacy_estimator_matches_original_p3_circular_formula_exactly():
    data = np.random.default_rng(1).normal(size=(2500, 2))*1e-7
    demeaned = data - data.mean(axis=0)
    expected = 0.5 * demeaned.T @ (
        2*demeaned - np.roll(demeaned, -1, axis=0)
        - np.roll(demeaned, 1, axis=0)
    ) / len(data)
    result, diag = estimate_tt_noise(data, method="legacy_autocorrelation")
    np.testing.assert_allclose(result, expected, rtol=1e-13)
    assert diag["method"] == "legacy_autocorrelation"


def test_van_dam_rejects_invalid_configuration_instead_of_falling_back():
    data, _, freq, response = _filtered_white_noise(n=4096)
    with pytest.raises(ValueError, match="does not cover"):
        estimate_tt_noise(
            data, sample_rate_hz=125.0,
            rejection_frequency_hz=freq[freq<30],
            rejection_transfer=response[freq<30],
        )
    with pytest.raises(ValueError, match="Unknown TT noise method"):
        estimate_tt_noise(data, method="unexpected")
    with pytest.raises(ValueError, match="sampling rate and loop rejection"):
        estimate_tt_noise(data, method="van_dam")


def test_available_methods_are_explicit():
    assert TT_METHODS == ("van_dam", "gendron", "legacy_autocorrelation")
