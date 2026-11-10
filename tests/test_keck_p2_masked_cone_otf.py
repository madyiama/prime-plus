"""P2: masked nonstationary cone OTF from physical covariance and modal projection.

All tests are synthetic, CPU-only and independent of Keck FITS/SAV.
No production P3 source code is exercised or changed by the prototype.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.keck_p2_masked_cone_otf import (
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
