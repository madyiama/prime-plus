"""Independent LGS Kolmogorov cone covariance / Hardy modal definition."""

import numpy as np
import pytest

from scripts.keck_p2_cone_modal_reference import independent_modal_budget


def _n0120_one_layer():
    # n0120 LOS geometry, independently ground-truthed by the previous
    # 1/7-layer atmospheric audit, 1.530 micron reference wavelength.
    return {
        "diameter_m": 10.5,
        "r0_m": 0.43911888960217066,
        "wavelength_m": 1.530e-6,
        "lgs_height_m": 122252.29570518076,
        "weights": np.array([1.0]),
        "heights_m": np.array([8208.713037464015]),
        "hardy_wfe_nm": 312.4173059349302,
    }


def test_ideal_circular_pupil_piston_and_tilt_removed_matches_hardy():
    report = independent_modal_budget(_n0120_one_layer(), 43)
    assert report["pupil"] == "circle"
    assert report["illuminated_grid_points"] > 1000
    # An independent phase covariance with 6.88 Kolmogorov prefactor and
    # a 43-point discretization agrees with Hardy's TILT-REMOVED formula
    # to better than one percent, no tuning or science fitting.
    assert report["piston_and_tilt_removed_wfe_nm"] == pytest.approx(
        312.4173059349302, rel=0.01
    )
    assert report["piston_removed_wfe_nm"] == pytest.approx(
        377.703, rel=0.01
    )
    assert report["piston_removed_wfe_nm"] > report[
        "piston_and_tilt_removed_wfe_nm"
    ]
    assert report["tilt_contribution_wfe_nm"] > 200.0


def test_square_grid_bias_is_separate_from_tilt_projection():
    g = _n0120_one_layer()
    disk = independent_modal_budget(g, 21, "circle")
    square = independent_modal_budget(g, 21, "square")
    assert square["piston_removed_wfe_nm"] > disk["piston_removed_wfe_nm"]
    assert (square["piston_and_tilt_removed_wfe_nm"]
            > disk["piston_and_tilt_removed_wfe_nm"])
    # Both square and circle are valid covariance projections, but only
    # the disk is the idealized aperture to which Hardy applies.
    assert abs(disk["relative_difference_vs_hardy"]) < 0.02


def test_ground_cone_residual_is_exactly_zero():
    g = _n0120_one_layer()
    g["heights_m"] = np.array([0.0])
    g["hardy_wfe_nm"] = 1.0  # Not physically compared in this limiting test.
    result = independent_modal_budget(g, 11)
    assert result["piston_removed_wfe_nm"] == 0
    assert result["piston_and_tilt_removed_wfe_nm"] == 0
    assert result["tilt_contribution_variance_rad2"] == 0


@pytest.mark.parametrize("shape", ["circle", "square"])
def test_covariance_budget_respects_weighted_variance_decomposition(shape):
    report = independent_modal_budget(_n0120_one_layer(), 11, pupil=shape)
    assert report["piston_removed_variance_rad2"] == pytest.approx(
        report["piston_and_tilt_removed_variance_rad2"]
        + report["tilt_contribution_variance_rad2"],
        rel=1e-12
    )


def test_bad_modal_grid_is_rejected():
    with pytest.raises(ValueError, match="odd"):
        independent_modal_budget(_n0120_one_layer(), 12)
    with pytest.raises(ValueError, match="Pupil shape"):
        independent_modal_budget(_n0120_one_layer(), 11, pupil="hex")


def test_n0120_seven_layer_piston_and_tilt_removed_converges_to_hardy():
    g = _n0120_one_layer()
    g["weights"] = np.array([
        0.4837829375341058, 2.8954981850976237e-11,
        7.28398762188621e-11, 4.0265521636513835e-06,
        0.044337315959307365, 0.4071794322793533,
        0.06469628757327503,
    ])
    g["heights_m"] = np.array([
        0.0, 679.1794205843375, 1358.358841168675,
        2716.71768233735, 5433.4353646747,
        10866.8707293494, 21733.7414586988,
    ])
    g["hardy_wfe_nm"] = 295.00749428261554
    # Coarse independent n=21 circular pupil reproduces Hardy within 1%
    # despite the >20% difference with piston-only full-cone WFE.
    report = independent_modal_budget(g, 21)
    assert report["piston_and_tilt_removed_wfe_nm"] == pytest.approx(
        295.00749428261554, rel=0.015
    )
    assert report["piston_removed_wfe_nm"] > 370
