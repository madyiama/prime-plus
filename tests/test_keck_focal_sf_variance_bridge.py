"""P2: bridge the physical pupil SF second moment and OTF-derived WFE.

For a real nonnegative pupil autocorrelation and a pupil-averaged phase
structure function, <phi(x)-mean_pupil(phi)>^2 is exactly one-half the
OTF_DL-weighted Dphi, whereas -log(SR) has an additional nonlinear
Jensen gap. The Hardy infinite-L0 analytical result should only be
compared to the *linear second moment* after matching physical inputs.
"""

import numpy as np
import pytest

from scripts.keck_n0013_smoke import _structure_function_wfe_bridge


def _example(n=48, amplitude=0.002):
    y, x = np.indices((n, n))
    radial2 = (x-n//2)**2 + (y-n//2)**2
    otf = np.exp(-0.004*radial2)
    return amplitude*radial2, otf


def test_pupil_weighted_phase_variance_is_linear_and_exact():
    dphi, otf = _example()
    lam = 1.53e-6
    diagnostics = _structure_function_wfe_bridge(dphi, otf, lam)
    p = otf / otf.sum()
    expected_phase_var = 0.5 * np.sum(p*dphi)
    expected_sr = np.sum(p * np.exp(-0.5*dphi))
    assert diagnostics["pupil_piston_removed_phase_variance_rad2"] == pytest.approx(
        expected_phase_var, rel=1e-13
    )
    assert diagnostics["anisoplanatism_only_strehl_factor"] == pytest.approx(
        expected_sr, rel=1e-13
    )
    assert diagnostics["strehl_equivalent_phase_variance_rad2"] == pytest.approx(
        -np.log(expected_sr), rel=1e-13
    )
    assert diagnostics["jensen_gap_phase_variance_rad2"] > 0
    assert diagnostics["strehl_equivalent_wfe_nm"] < diagnostics[
        "pupil_piston_removed_wfe_nm"
    ]
    assert diagnostics["sf_zero_lag_rad2"] == 0.0


def test_linear_bridge_respects_turbulence_strength_scaling():
    dphi, otf = _example()
    a = _structure_function_wfe_bridge(dphi, otf, 1.53e-6)
    b = _structure_function_wfe_bridge(2*dphi, otf, 1.53e-6)
    assert b["pupil_piston_removed_phase_variance_rad2"] == pytest.approx(
        2*a["pupil_piston_removed_phase_variance_rad2"], rel=1e-13
    )
    assert b["pupil_piston_removed_wfe_nm"] == pytest.approx(
        2**0.5*a["pupil_piston_removed_wfe_nm"], rel=1e-13
    )
    # The SR-equivalent variance is nonlinear and does not double.
    assert b["strehl_equivalent_phase_variance_rad2"] < (
        2*a["strehl_equivalent_phase_variance_rad2"]
    )


def test_strehl_and_pupil_variances_converge_at_weak_phase():
    dphi, otf = _example(amplitude=1e-7)
    report = _structure_function_wfe_bridge(dphi, otf, 1.53e-6)
    assert report["strehl_equivalent_phase_variance_rad2"] == pytest.approx(
        report["pupil_piston_removed_phase_variance_rad2"],
        rel=1e-4
    )


def test_opd_bridge_is_invariant_under_consistent_wavelength_change():
    dphi, otf = _example()
    lam1, lam2 = 1.53e-6, 2.2e-6
    a = _structure_function_wfe_bridge(dphi, otf, lam1)
    b = _structure_function_wfe_bridge(
        dphi*(lam1/lam2)**2, otf, lam2
    )
    assert b["pupil_piston_removed_wfe_nm"] == pytest.approx(
        a["pupil_piston_removed_wfe_nm"], rel=1e-13
    )


def test_structure_function_bridge_rejects_nonzero_dc_and_bad_weights():
    dphi, otf = _example()
    with pytest.raises(ValueError, match="Dphi\\(0\\)=0"):
        _structure_function_wfe_bridge(dphi+0.1, otf, 1.53e-6)
    otf[0, 0] = -0.1
    with pytest.raises(ValueError, match="Negative diffraction"):
        _structure_function_wfe_bridge(dphi, otf, 1.53e-6)


def test_zero_structure_function_gives_unit_strehl_zero_wfe():
    dphi = np.zeros((16, 16), dtype=float)
    otf = np.ones((16, 16), dtype=float)
    report = _structure_function_wfe_bridge(dphi, otf, 1.5e-6)
    assert report["anisoplanatism_only_strehl_factor"] == 1
    assert report["pupil_piston_removed_wfe_nm"] == 0
    assert report["strehl_equivalent_wfe_nm"] == 0
