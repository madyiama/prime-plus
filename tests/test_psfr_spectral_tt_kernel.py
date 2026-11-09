"""Opt-in spectral physical TT jitter OTF: theoretical and regression tests.

These tests do not fit or calibrate NIRC2 images. Synthetic input has known
disturbance and sensor-noise covariance, with independent signal paths.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

from p3.aoSystem.tipTiltGeometry import (
    angular_jitter_structure_function,
    opd_covariance_to_angular,
)
from p3.telemetry.ttClosedLoop import physical_tt_covariance_from_telemetry
from p3.psfr.psfR import psfR


def _mock_model(mode):
    model = psfR.__new__(psfR)
    model.tt_covariance_mode = mode
    model.tt_mirror_model = "unity"
    model.tt_welch_nperseg = 512
    samples = np.array([
        [1e-6, 0], [-1e-6, 0], [0, 1e-6], [0, -1e-6],
    ])
    fs = 125.
    frequency = np.linspace(0, fs/2, 1025)
    loop = NS(
        freq=fs,
        tf=NS(
            freq=frequency, wfs=np.ones_like(frequency),
            servo=np.full_like(frequency, 0.3),
            lag=np.ones_like(frequency),
            rtf=np.full_like(frequency, 1/1.3),
            pn=(0.3/1.3)**2,
        ),
    )
    model.trs = NS(
        aoMode="LGS",
        tipTilt=NS(
            slopes=samples, nExp=len(samples),
            Cn_tt=np.eye(2)*1e-13,
            command_units_verified=False,
        ),
        ttloop=loop, tel=NS(pupilAngle=0.),
    )
    u, v = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    model.freq = NS(
        wvlRef=2.1e-6, sampRef=2.0, U_=u, V_=v,
    )
    model.ao = NS(tel=NS(D=10.5))
    return model


def test_spectral_mode_injects_total_physical_covariance_without_double_count():
    model = _mock_model("spectral_closed_loop")
    c_atmos = np.diag([1.3e-13, 1.7e-13])
    c_noise = np.diag([0.4e-13, 0.6e-13])
    c_total = c_atmos + c_noise
    diagnostics = {
        "gain_sensor_to_measured_noise": 0.9,
        "gain_sensor_to_physical_noise": 0.05,
        "physical_atmos_min_eig_m2": 1.3e-13,
        "physical_total_min_eig_m2": 1.7e-13,
    }
    with patch(
        "p3.psfr.psfR.physical_tt_covariance_from_telemetry",
        return_value=(c_atmos, c_noise, c_total, diagnostics),
    ) as estimate:
        dphi = model.jitter_phase_structure_function()
    estimate.assert_called_once()
    np.testing.assert_allclose(model.tt_atmos_cov, c_atmos)
    np.testing.assert_allclose(model.tt_noise_cov, c_noise)
    np.testing.assert_allclose(model.tt_model_cov, c_total)
    theta = opd_covariance_to_angular(c_total, model.ao.tel.D)
    expected = angular_jitter_structure_function(
        theta, model.ao.tel.D, model.freq.wvlRef,
        model.freq.U_, model.freq.V_,
    )
    np.testing.assert_allclose(dphi, expected, rtol=1e-13)
    assert np.max(np.exp(-0.5*dphi)) <= 1.0 + 1e-13
    assert model.tiptilt_diagnostics["jitter_covariance_method"] == "spectral_closed_loop"
    assert model.tiptilt_diagnostics["covariance_projection_applied"] is False
    assert model.tiptilt_diagnostics["noise_model_validated"] is False


def test_legacy_mode_keeps_original_covariance_formula():
    model = _mock_model("legacy_psfr")
    with patch("p3.psfr.psfR.physical_tt_covariance_from_telemetry") as estimate:
        dphi = model.jitter_phase_structure_function()
    estimate.assert_not_called()
    assert dphi.shape == model.freq.U_.shape
    assert model.tiptilt_diagnostics["jitter_covariance_method"] == "legacy_psfr"
    assert model.tiptilt_diagnostics["tt_mirror_model"] == "unity"
    assert np.trace(model.tt_model_cov) >= 0


def _known_closed_loop_samples(n=32768, seed=31):
    fs = 125.
    rng = np.random.default_rng(seed)
    cn = np.diag([(2e-7)**2, (3e-7)**2])
    noise = rng.normal(0, [2e-7, 3e-7], (n, 2))
    time = np.arange(n) / fs
    disturbance = np.column_stack([
        1.1e-6*np.sin(2*np.pi*0.73*time),
        0.9e-6*np.cos(2*np.pi*0.51*time),
    ])
    gain = .3
    S = 1/(1+gain)
    centroids = S*(disturbance + noise)
    frequency = np.linspace(0, fs/2, 1025)
    loop = NS(freq=fs, tf=NS(
        freq=frequency,
        wfs=np.ones_like(frequency),
        servo=np.full_like(frequency, gain),
        lag=np.ones_like(frequency),
        rtf=np.full_like(frequency, S),
    ))
    # Physical residual e = S*d - gain*S*n for instantaneous W=1 plant=1.
    physical = S*disturbance - (gain*S)*noise
    return centroids, cn, loop, physical


def test_physical_jitter_spectrum_matches_independent_linear_simulation():
    data, cn, loop, expected_physical_series = _known_closed_loop_samples()
    atmos, noise, total, diagnostics = physical_tt_covariance_from_telemetry(
        data, cn, loop, mirror_model="unity", nperseg=1024,
    )
    expected = np.cov(expected_physical_series.T, bias=True)
    np.testing.assert_allclose(np.diag(total), np.diag(expected), rtol=0.13)
    np.testing.assert_allclose(total, atmos + noise, rtol=1e-12, atol=1e-26)
    assert np.trace(atmos) > 0
    assert np.trace(noise) > 0
    assert diagnostics["gain_sensor_to_physical_noise"] == pytest.approx(
        (0.3/1.3)**2, rel=1e-12,
    )
    assert diagnostics["is_validated_for_observation"] is False


def test_spectral_method_rejects_negative_atmospheric_covariance_no_clipping():
    _, cn, loop, _ = _known_closed_loop_samples(n=4096)
    samples = np.zeros((4096, 2))
    with pytest.raises(ValueError, match="non-PSD"):
        physical_tt_covariance_from_telemetry(
            samples, cn, loop, mirror_model="unity", nperseg=512,
        )


def test_spectral_mode_prohibits_ngs_without_validation():
    model = _mock_model("spectral_closed_loop")
    model.trs.aoMode = "NGS"
    with pytest.raises(ValueError, match="requires Keck LGS"):
        model.jitter_phase_structure_function()


def test_ragland_noise_transfer_mismatch_is_visible_not_silently_validated():
    model = _mock_model("spectral_closed_loop")
    model.tt_mirror_model = "ragland_2016_keck_ii"
    model.trs.tipTilt.noise_estimation_method = "van_dam"
    c_atmos = np.diag([1.3e-13, 1.7e-13])
    c_noise = np.diag([0.4e-13, 0.6e-13])
    c_total = c_atmos + c_noise
    diagnostics = {
        "gain_sensor_to_measured_noise": 1.01,
        "gain_sensor_to_physical_noise": 0.0512,
        "physical_atmos_min_eig_m2": 1.3e-13,
        "physical_total_min_eig_m2": 1.7e-13,
    }
    with patch(
        "p3.psfr.psfR.physical_tt_covariance_from_telemetry",
        return_value=(c_atmos, c_noise, c_total, diagnostics),
    ):
        with pytest.warns(RuntimeWarning, match="sensitivity study only"):
            model.jitter_phase_structure_function()
    diag = model.tiptilt_diagnostics
    assert diag["measurement_noise_estimator"] == "van_dam"
    assert diag["van_dam_rejection_model"] == "p3_pure_delay"
    assert diag["noise_transfer_mismatch"] is True
    assert diag["noise_model_validated"] is False
