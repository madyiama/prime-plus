"""Regression tests for Keck NGS finite-record HO noise and centered PSF views."""

from types import SimpleNamespace as NS

import numpy as np

from p3.telemetry.systemDiagnosis import (
    autocorrelation_noise_covariance, systemDiagnosis,
)
from scripts.keck_n0013_smoke import _central_crop, _ho_noise_endpoint_audit


def test_comparison_zoom_is_centered_on_array_not_brightest_pixel():
    image = np.zeros((128, 128), dtype=float)
    image[8, 12] = 1000.0
    image[64, 64] = 7.0
    crop = _central_crop(image, width=48)
    assert crop.shape == (48, 48)
    np.testing.assert_array_equal(crop, image[40:88, 40:88])
    assert crop[24, 24] == 7.0
    assert crop.max() == 7.0


def test_central_crop_odd_size_and_small_input():
    image = np.arange(35).reshape(5, 7)
    np.testing.assert_array_equal(_central_crop(image, width=3), image[1:4, 2:5])
    np.testing.assert_array_equal(_central_crop(image, width=48), image)


def test_periodic_autocorrelation_reproduces_historical_roll_equation():
    rng = np.random.default_rng(1283)
    signal = rng.normal(size=(61, 4))
    centered = signal - signal.mean(axis=0)
    for lag in (1, 3):
        old = 0.5 * centered.T @ (
            2 * centered
            - np.roll(centered, -lag, axis=0)
            - np.roll(centered, lag, axis=0)
        ) / len(centered)
        np.testing.assert_allclose(
            autocorrelation_noise_covariance(signal, lag, periodic=True),
            old, rtol=1e-13, atol=1e-14,
        )


def test_ngs_nonperiodic_noise_excludes_nonexistent_wrap():
    # A ramp has identical adjacent increments; linking final to first
    # fabricates a large noise increment, potentially a negative HO mode.
    ramp = np.arange(30, dtype=float)[:, None]
    finite = autocorrelation_noise_covariance(ramp, periodic=False)
    circular = autocorrelation_noise_covariance(ramp, periodic=True)
    np.testing.assert_allclose(finite, [[0.5]])
    assert circular[0, 0] > finite[0, 0] * 10
    assert np.linalg.eigvalsh(finite).min() >= 0


def test_ngs_system_diagnosis_selects_finite_noise_for_ho_and_tt():
    ramp = np.arange(30, dtype=float)[:, None]
    tt = np.column_stack((np.arange(30, dtype=float), np.sin(np.arange(30))))
    trs = NS(
        aoMode="NGS",
        dm=NS(com=ramp.copy()),
        tipTilt=NS(slopes=tt.copy()),
        holoop=NS(tf=NS(ctf=1.0, wfs=1.0)),
        ttloop=NS(tf=NS(ctf=1.0, wfs=1.0)),
    )
    diagnosis = systemDiagnosis.__new__(systemDiagnosis)
    diagnosis.trs = trs
    c_ho, c_tt = diagnosis.get_noise_variance(noiseMethod="autocorrelation")
    np.testing.assert_allclose(
        c_ho, autocorrelation_noise_covariance(ramp, periodic=False)
    )
    np.testing.assert_allclose(
        c_tt, autocorrelation_noise_covariance(tt, periodic=False)
    )
    assert trs.tipTilt.noise_estimation_method == "autocorrelation_nonperiodic"


def test_endpoint_audit_reports_source_of_negative_ho_covariance():
    commands = np.arange(30, dtype=float)[:, None]
    cn = autocorrelation_noise_covariance(commands, periodic=False)
    trs = NS(
        aoMode="NGS",
        dm=NS(com=commands),
        holoop=NS(gain=1.0),
        wfs=NS(Cn_ao=cn),
    )
    info = _ho_noise_endpoint_audit(trs)
    assert info["covariance_after_subtraction_used"]["min_eigenvalue_m2"] >= -1e-14
    assert info["covariance_after_subtraction_periodic"]["min_eigenvalue_m2"] < 0
    assert info["closing_jump_covariance_trace_m2"] > 0
    assert info["noise_trace_periodic_m2"] > info["noise_trace_nonperiodic_m2"]
