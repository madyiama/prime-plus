"""P2: masked nonstationary cone OTF from physical covariance and modal projection.

All tests are synthetic, CPU-only and independent of Keck FITS/SAV.
No production P3 source code is exercised or changed by the prototype.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.keck_p2_masked_cone_otf import (
    load_case_geometry, _write_outputs, pupil_clipping_diagnostics,
    _sample_pupil,
    _cone_covariance,
    _project_modes,
    _nonstationary_pupil_otf,
    _otf_psf_metrics,
    evaluate_configuration,
    phase_structure_function_von_karman,
)


def _geometry(n_layer=1):
    if n_layer == 1:
        heights = np.array([8208.713037464015])
        weights = np.array([1.0])
        hardy = 312.41730593493025
    else:
        heights = np.array([
            0.0, 679.1794205843375, 1358.358841168675,
            2716.71768233735, 5433.4353646747,
            10866.8707293494, 21733.7414586988,
        ])
        weights = np.array([
            0.4837829375341058, 2.8954981850976237e-11,
            7.28398762188621e-11, 4.0265521636513835e-06,
            0.044337315959307365, 0.4071794322793533,
            0.06469628757327503,
        ])
        hardy = 295.00749428261554
    return {
        "diameter_m": 10.5,
        "r0_m": 0.43911888960217066,
        "wavelength_m": 1.530e-6,
        "lgs_height_m": 122252.29570518076,
        "heights": heights,
        "weights": weights,
        "hardy_wfe_nm": hardy,
        "pupil_source": "circle",
    }


def test_infinite_outer_scale_is_exact_kolmogorov_and_finite_is_smaller():
    rho = np.linspace(0.0, 10.5, 17)
    r0 = 0.4391
    inf = phase_structure_function_von_karman(rho, r0, np.inf)
    np.testing.assert_allclose(
        inf, 6.88*(rho/r0)**(5/3), rtol=1e-14, atol=1e-13
    )
    finite = phase_structure_function_von_karman(rho, r0, 31)
    assert finite[0] == pytest.approx(0.0)
    assert np.all(finite >= 0)
    assert np.all(finite[1:] < inf[1:])
    with pytest.raises(ValueError, match="Invalid"):
        phase_structure_function_von_karman(rho, 0.0, 31)
    with pytest.raises(ValueError, match="Invalid"):
        phase_structure_function_von_karman(rho, r0, -1)


def test_piston_projection_does_not_change_phase_pair_differences():
    g = _geometry()
    p = _sample_pupil(g["diameter_m"], 11, "circle")
    C = _cone_covariance(g, p, np.inf)
    Cp, _ = _project_modes(C, p, remove_tilt=False)
    a = np.diag(C)
    b = np.diag(Cp)
    dphi_before = a[:, None]+a[None, :]-2*C
    dphi_after = b[:, None]+b[None, :]-2*Cp
    np.testing.assert_allclose(
        dphi_before, dphi_after, atol=2e-10, rtol=5e-11
    )


def test_exact_masked_otf_zero_baseline_hermitian_jensen_and_psf():
    g = _geometry()
    p = _sample_pupil(g["diameter_m"], 11, "circle")
    C = _cone_covariance(g, p, np.inf)
    Cp, _ = _project_modes(C, p, remove_tilt=True)
    dl, exact, stationary, _ = _nonstationary_pupil_otf(Cp, p)
    mid = p["n"]-1
    assert dl[mid, mid] == pytest.approx(1.0, abs=1e-12)
    assert exact[mid, mid] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(exact, exact[::-1, ::-1], atol=1e-13)
    assert np.all(exact >= -1e-12)
    assert np.all(exact <= dl + 1e-12)
    assert np.all(exact >= stationary - 1e-12)
    m, psf = _otf_psf_metrics(dl, exact)
    assert m["strehl_from_otf_integral"] == pytest.approx(
        m["strehl_from_psf_peak_ratio"], rel=1e-9
    )
    assert m["negative_psf_flux_fraction"] < 1e-8
    assert psf.sum() == pytest.approx(1.0, rel=1e-13)


@pytest.mark.parametrize("layers", [1, 7])
def test_circular_pupil_projected_kolmogorov_cone_matches_hardy(layers):
    g = _geometry(layers)
    p = _sample_pupil(g["diameter_m"], 21, "circle")
    result, _ = evaluate_configuration(g, p, np.inf)
    no_tilt = result["piston_removed"]["pupil_projected_wfe_nm"]
    tilt = result["piston_and_tilt_removed"]["pupil_projected_wfe_nm"]
    assert tilt == pytest.approx(g["hardy_wfe_nm"], rel=0.02)
    assert no_tilt > tilt + 45.0
    assert result["piston_and_tilt_removed"][
        "otf_dc"
    ] == pytest.approx(1, abs=1e-12)


def test_sampled_keck_mask_changes_modal_and_diffraction_otfs():
    g = _geometry()
    D = g["diameter_m"]
    # Stand-in Keck-like map with segmented gaps and a central obscuration.
    N = 101
    yy, xx = np.indices((N, N))
    r = np.hypot(xx-50, yy-50)
    aperture = (
        (r <= 48) & (r > 10)
        & (np.abs(xx-50) > 1)
        & (np.abs(yy-50) > 1)
    ).astype(float)
    keck = _sample_pupil(D, 21, "keck", full_pupil=aperture)
    circle = _sample_pupil(D, 21, "circle")
    assert keck["illuminated_points"] < circle["illuminated_points"]
    result_keck, out_keck = evaluate_configuration(g, keck, np.inf)
    result_circle, out_circle = evaluate_configuration(g, circle, np.inf)
    assert result_keck["piston_and_tilt_removed"][
        "pupil_projected_wfe_nm"
    ] != pytest.approx(
        result_circle["piston_and_tilt_removed"]["pupil_projected_wfe_nm"],
        rel=0.001
    )
    assert not np.allclose(
        out_keck["piston_and_tilt_removed"]["otf_dl"],
        out_circle["piston_and_tilt_removed"]["otf_dl"]
    )


def test_ground_layer_has_no_focal_cone_error_and_unit_otf():
    g = _geometry()
    g["heights"] = np.array([0.0])
    p = _sample_pupil(g["diameter_m"], 11, "circle")
    result, _ = evaluate_configuration(g, p, np.inf)
    assert result["piston_removed"]["pupil_projected_wfe_nm"] == 0
    assert result["piston_and_tilt_removed"]["pupil_projected_wfe_nm"] == 0
    assert result["piston_and_tilt_removed"][
        "strehl_from_otf_integral"
    ] == pytest.approx(1, abs=1e-12)


def test_modal_tilt_basis_preserves_weighted_orthogonality():
    g = _geometry()
    p = _sample_pupil(g["diameter_m"], 15, "circle")
    C = _cone_covariance(g, p, np.inf)
    Cp, _ = _project_modes(C, p, remove_tilt=True)
    w = p["amplitude"]**2
    w /= w.sum()
    modes = np.column_stack([
        np.ones(len(w)), p["x"]/g["diameter_m"],
        p["y"]/g["diameter_m"],
    ])
    v = Cp@(w[:, None]*modes)
    assert np.max(np.abs(v)) < 1e-9


def test_completed_p2_case_loader_transfers_hardy_reference_to_prototype(tmp_path):
    """Regression for the real n0120 CLI KeyError: 'hardy_wfe_nm'."""
    case = tmp_path / "n0120_P2_1layers"
    case.mkdir()
    # This fixture uses the actual output FILE SCHEMA, not the synthetic
    # hand-built geometry dict used by the other unit tests.
    aniso = {
        "reference_wavelength_m": 1.530e-6,
        "atmosphere_r0_m": 0.43911888960217066,
        "atmosphere_layers": 1,
        "layer_weights": [1.0],
        "layer_heights_los_m": [8208.713037464015],
        "lgs_height_los_m": 122252.29570518076,
        "hardy_cone_wfe_nm": 312.41730593493025,
        "focal_model_conditions": {
            "von_karman_outer_scale_m": 31.045692184830187,
        },
    }
    grid = {
        "telescope_diameter_m": 10.5,
        "otf_sampling_sampRef": 3.0231069111135773,
    }
    (case/"anisoplanatism_diagnostics.json").write_text(
        json.dumps(aniso), encoding="utf-8"
    )
    (case/"frequency_grid_diagnostics.json").write_text(
        json.dumps(grid), encoding="utf-8"
    )
    (case/"Keck_NIRC2.ini").write_text(
        "[DM]\nNumberActuators = [1]\n", encoding="utf-8"
    )
    np.save(case/"diffraction_otf.npy", np.ones((8, 8)))

    g = load_case_geometry(case)
    assert g["hardy_wfe_nm"] == pytest.approx(312.41730593493025)
    assert g["weights"].tolist() == [1.0]
    assert g["n_actu"] == 1

    # Exercise the exact first computation reached by the user CLI,
    # including Hardy metric formatting, not just a key presence check.
    sampled = _sample_pupil(g["diameter_m"], 15, "circle")
    report, _ = evaluate_configuration(g, sampled, np.inf)
    assert report["piston_and_tilt_removed"]["hardy_reference_wfe_nm"] == pytest.approx(
        312.41730593493025
    )
    assert report["piston_and_tilt_removed"]["pupil_projected_wfe_nm"] > 0

    del aniso["hardy_cone_wfe_nm"]
    (case/"anisoplanatism_diagnostics.json").write_text(
        json.dumps(aniso), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Missing hardy_cone_wfe_nm"):
        load_case_geometry(case)


def test_masked_cone_report_writes_flattened_numpy_products(tmp_path):
    """Regression for real CLI AttributeError: ndarray has no .items()."""
    g = _geometry()
    g["pupil_source"] = "circle"
    sampled = _sample_pupil(g["diameter_m"], 9, "circle")
    metrics, products = evaluate_configuration(g, sampled, np.inf)
    flattened = {
        f"n9_L0inf_{mode}_{key}": array
        for mode, arrays in products.items()
        for key, array in arrays.items()
    }
    assert flattened and all(
        isinstance(array, np.ndarray) for array in flattened.values()
    )
    output = tmp_path / "masked_cone_results"
    summary = {"evaluations": [metrics]}
    _write_outputs(summary, flattened, output)
    assert (output / "masked_cone_otf_audit.json").is_file()
    assert (output / "masked_cone_otf_audit.csv").is_file()
    assert (output / "masked_cone_otf_audit.png").is_file()
    for name, array in flattened.items():
        target = output / f"{name}.npy"
        assert target.is_file()
        np.testing.assert_array_equal(
            np.load(target, allow_pickle=False), array
        )


def test_rotated_pupil_clipping_report_separates_tiny_from_material_overshoots():
    """Thousands of tiny spline negatives need not be physically important."""
    raw = np.ones((100, 100), dtype=float)
    raw[:43, :] = -1e-12    # 4300 out-of-range pixels, negligible flux
    raw[90, 90] = 1.0 + 1e-12
    report = pupil_clipping_diagnostics(raw)
    assert report["out_of_range_pixel_count"] == 4301
    assert report["out_of_range_pixel_count_above_1e-8"] == 0
    assert report["material_pupil_clipping_warning"] is False
    assert abs(report["relative_intensity_change_from_clipping"]) < 1e-9

    raw[90, 90] = 10.0
    report_material = pupil_clipping_diagnostics(raw)
    assert report_material["out_of_range_pixel_count_above_1e-2"] == 1
    assert report_material["material_pupil_clipping_warning"] is True
    assert report_material["maximum_amplitude_clipping_change"] == 9.0
    assert report_material["relative_intensity_change_from_clipping"] < 0


def test_rotated_pupil_clipping_requires_finite_2d_data():
    with pytest.raises(ValueError, match="finite 2-D"):
        pupil_clipping_diagnostics(np.array([1.0]))
    with pytest.raises(ValueError, match="finite 2-D"):
        pupil_clipping_diagnostics(np.array([[np.nan]]))
