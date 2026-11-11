"""P4 STRAP frame and ablation contracts, independent of Keck archive files."""
import json
import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.ttFrame import (
    RAD_TO_MAS, angular_covariance_metrics, camera_angular_covariance,
    psf_aperture_metrics, sensor_frame_frequencies,
)
from p3.psfr.psfR import psfR


def test_frame_rotation_reproduces_legacy_axis_formula_exactly():
    u, v = np.meshgrid(np.linspace(-1, 1, 17), np.linspace(-1, 1, 17))
    angle = np.deg2rad(46.76723)
    us, vs = sensor_frame_frequencies(u, v, 46.76723)
    np.testing.assert_allclose(us, u*np.cos(angle)+v*np.sin(angle),
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(vs, -u*np.sin(angle)+v*np.cos(angle),
                               rtol=0, atol=1e-14)


def test_frame_rotation_and_reflection_preserve_covariance_trace_and_eigenvalues():
    c_opd = np.array([[3.0, 0.3], [0.3, 0.7]])*1e-12
    original = camera_angular_covariance(c_opd, 10.5, 0)
    for angle in (-71., 0., 23.1, 90., 173.):
        for parity in ("native", "flip_x", "flip_y"):
            c = camera_angular_covariance(c_opd, 10.5, angle, parity)
            assert np.trace(c) == pytest.approx(np.trace(original), rel=1e-12)
            np.testing.assert_allclose(np.linalg.eigvalsh(c),
                                       np.linalg.eigvalsh(original),
                                       rtol=1e-12, atol=1e-25)


def test_reflections_apply_to_sensor_axes_after_rotation():
    u, v = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    native = sensor_frame_frequencies(u, v, 47)
    fx = sensor_frame_frequencies(u, v, 47, parity="flip_x")
    fy = sensor_frame_frequencies(u, v, 47, parity="flip_y")
    np.testing.assert_allclose(fx[0], -native[0])
    np.testing.assert_allclose(fx[1], native[1])
    np.testing.assert_allclose(fy[0], native[0])
    np.testing.assert_allclose(fy[1], -native[1])


def _bare_jitter_model(pupil_angle=46.76723, axis_angle=None, parity="native"):
    model = psfR.__new__(psfR)
    model.trs = NS(tel=NS(pupilAngle=pupil_angle))
    model.ao = NS(tel=NS(D=10.5))
    y, x = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
    model.freq = NS(U_=x, V_=y, sampRef=2.0, wvlRef=2.06e-6)
    model.tt_axis_angle_deg = axis_angle
    model.tt_axis_parity = parity
    model.tt_kernel_dphi_multiplier_vs_noll = 1.
    return model


def test_explicit_camera_axes_are_independent_of_pupil_angle():
    covariance = np.diag([4e-13, 1e-13])
    m1 = _bare_jitter_model(pupil_angle=46, axis_angle=23)
    m2 = _bare_jitter_model(pupil_angle=-37, axis_angle=23)
    np.testing.assert_allclose(m1.jitter_phase_structure_function(Ctt=covariance),
                               m2.jitter_phase_structure_function(Ctt=covariance),
                               rtol=1e-12, atol=1e-12)
    legacy = _bare_jitter_model(pupil_angle=46)
    assert not np.allclose(legacy.jitter_phase_structure_function(Ctt=covariance),
                           m1.jitter_phase_structure_function(Ctt=covariance),
                           rtol=1e-5, atol=1e-12)


def test_legacy_no16_multiplies_dphi_by_exactly_one_sixteenth():
    covariance = np.diag([4e-13, 1e-13])
    model = _bare_jitter_model()
    reference = model.jitter_phase_structure_function(Ctt=covariance)
    model.tt_kernel_dphi_multiplier_vs_noll = 1.0/16.0
    np.testing.assert_allclose(model.jitter_phase_structure_function(Ctt=covariance),
                               reference/16.0, rtol=1e-12, atol=1e-12)


def test_camera_tilt_rms_has_correct_arcsecond_and_fwhm_conversion():
    D = 10.5
    rms_mas = [10.0, 25.0]
    sigma_rad = np.array(rms_mas)/RAD_TO_MAS
    opd = np.diag((D*sigma_rad/4)**2)
    metrics = angular_covariance_metrics(opd, D, 71.0, "flip_x")
    np.testing.assert_allclose(metrics["principal_rms_mas_minor_major"],
                               rms_mas, rtol=1e-12)
    np.testing.assert_allclose(metrics["principal_fwhm_mas_minor_major"],
                               np.asarray(rms_mas)*2*np.sqrt(2*np.log(2)),
                               rtol=1e-12)
    assert metrics["STRAP_optical_gain_calibrated"] is False


def test_fft_centred_delta_psf_has_unit_energy_in_all_circular_apertures():
    p = np.zeros((64, 64))
    p[32, 32] = 1
    metrics = psf_aperture_metrics(p)
    assert metrics["normalized_peak"] == 1
    assert metrics["centroid_x_y_px"] == [32, 32]
    assert metrics["second_moment_rms_px"] == pytest.approx(0)
    assert all(v == 1 for v in metrics["encircled_energy_at_center_px"].values())


def test_reject_invalid_frame_and_non_psd_covariances():
    with pytest.raises(ValueError, match="parity"):
        sensor_frame_frequencies(np.ones((2, 2)), np.ones((2, 2)), 0, "unknown")
    with pytest.raises(ValueError, match="positive-semidefinite"):
        angular_covariance_metrics(np.diag([-1e-12, 1e-12]), 10.5)
    with pytest.raises(ValueError, match="non-negative"):
        psf_aperture_metrics(np.array([[1.0, -0.1], [0., 0.]]))


def _report_function():
    script = Path(__file__).resolve().parents[1]/"scripts"/"keck_p4_tt_benchmark.py"
    return runpy.run_path(str(script))["summarize_existing_outputs"]


def _synthetic_four_folders(root):
    for name, (tt_model, kernel, disabled) in {
        "A0_no_tt": ("legacy_psfr", "noll_angular", True),
        "A1_legacy_physical": ("legacy_psfr", "noll_angular", False),
        "A2_legacy_no16_diagnostic": ("legacy_psfr", "legacy_unscaled_opd", False),
        "A3_spectral_physical": ("spectral_closed_loop", "noll_angular", False),
    }.items():
        path = root/name
        path.mkdir()
        metadata = {
            "input_fits": "/data/n0013.fits",
            "input_sav": "/data/n0013.sav",
            "code_revision": "test-sha",
            "ao_mode": "LGS",
            "forward_parameters_diagnostics": {
                "atmosphere_layers": 7, "r0_model_m": 0.43,
            },
            "psf_reference_wavelength_m": 2.05e-6,
            "p3_pupil_resampling": "physical_binary",
            "p3_pupil_parity": "native",
            "p3_array_rotation_applied_deg": 46.7,
            "p4_tt_axis_angle_deg": 46.7,
            "p4_tt_axis_parity": "native",
            "p4_tt_axis_uses_legacy_pupil": True,
            "tt_noise_method": "van_dam",
            "tt_noise_rejection_mirror_model": "unity",
            "tt_jitter_covariance_method": tt_model,
            "tt_kernel_convention": kernel,
            "tt_jitter_otf_enabled": not disabled,
        }
        metrics = {
            "psf": {"normalized_peak": 0.02},
            "camera_angular_covariance": (
                None if disabled else {"trace_rad2": 2e-14,
                                        "rms_per_axis_equivalent_mas": 20}
            ),
        }
        (path/"metadata.json").write_text(json.dumps(metadata))
        (path/"tiptilt_diagnostics.json").write_text("{}")
        (path/"p4_tt_metrics.json").write_text(json.dumps(metrics))
        (path/"tt_noise_diagnostics.json").write_text(json.dumps({
            "noise_covariance_m2": [[1e-12, 0], [0, 1e-12]]
        }))
    return root


def test_four_ablation_output_invariants_are_checked_before_comparison(tmp_path):
    analyse = _report_function()
    root = _synthetic_four_folders(tmp_path)
    report = analyse(root)
    assert report["status"] == "model_only_ablation_complete"
    assert not report["best_fit_model_selected"]
    assert report["STRAP_noise_identical_all_four"]
    metadata_path = root/"A3_spectral_physical"/"metadata.json"
    j = json.loads(metadata_path.read_text())
    j["forward_parameters_diagnostics"]["r0_model_m"] = 0.54
    metadata_path.write_text(json.dumps(j))
    with pytest.raises(ValueError, match="r0_model_m"):
        analyse(root)


def test_ablation_refuses_sensor_noise_difference(tmp_path):
    analyse = _report_function()
    root = _synthetic_four_folders(tmp_path)
    path = root/"A2_legacy_no16_diagnostic"/"tt_noise_diagnostics.json"
    path.write_text(json.dumps({
        "noise_covariance_m2": [[2e-12, 0], [0, 1e-12]]
    }))
    with pytest.raises(ValueError, match="noise floor"):
        analyse(root)
