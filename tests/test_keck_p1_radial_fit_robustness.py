"""P1-C fixed-L0 stability diagnostics; no Keck telemetry is required."""

import numpy as np
import pytest

from scripts.keck_p1_radial_fit_robustness import (
    analyse_radial_robustness, write_results,
)


def _diagnostics(excess=False):
    orders = np.arange(2, 23)
    model = 40.0*(2/orders)**(11/3)
    ratio = (1 + .03*(orders-2)) if excess else np.ones_like(model)
    return {
        "fit": {
            "r0_wfs_los_m": .115,
            "L0_m": 31.,
            "n_modes": 261,
            "n_valid_actuators": 349,
            "noise_proxy_fraction_of_total_modal_variance": .0008,
        },
        "radial": [{
            "radial_order": int(n),
            "median_measured_rad2": float(y*1.001),
            "median_corrected_rad2": float(y),
            "median_theory_fit_rad2": float(m),
        } for n,m,y in zip(orders,model,model*ratio)],
    }


def test_perfect_model_is_invariant_to_radial_range_and_weighting():
    d = analyse_radial_robustness(_diagnostics(excess=False))
    assert d["quality_flags"] == [
        "HIGH_RADIAL_ORDER_DM_REPRESENTABILITY_REVIEW",
        "SUBTRACTED_DM_LAG_NOISE_PROXY_SMALL_VS_MODAL_VARIANCE",
    ]
    for b in d["subsets"]:
        assert b["r0_absolute_loss_m"] == pytest.approx(.115, rel=1e-12)
        assert b["r0_log_loss_m"] == pytest.approx(.115, rel=1e-12)
        assert b["shape_rms_log_residual"] == pytest.approx(0, abs=1e-12)


def test_high_order_excess_makes_fixed_L0_r0_depend_on_order_cut():
    d = analyse_radial_robustness(_diagnostics(excess=True))
    assert "HIGH_ORDER_EXCESS_NOT_CAPTURED_BY_FITTED_MODEL" in d["quality_flags"]
    lo = next(b for b in d["subsets"] if (b["order_min"],b["order_max"]) == (2,4))
    hi = next(b for b in d["subsets"] if (b["order_min"],b["order_max"]) == (17,22))
    assert lo["r0_absolute_loss_m"] > hi["r0_absolute_loss_m"]
    assert lo["r0_log_loss_m"] > hi["r0_log_loss_m"]
    assert d["reference"]["high_order_median_data_over_model"] > 1.2
    assert len(d["per_order"]) == 21


def test_report_output_json_csv_and_plot(tmp_path):
    d = analyse_radial_robustness(_diagnostics(excess=True))
    write_results(d, tmp_path)
    for n in (
        "p1_radial_fit_robustness.json",
        "p1_radial_fit_robustness_subsets.csv",
        "p1_radial_fit_robustness_orders.csv",
        "p1_radial_fit_robustness.png",
    ):
        assert (tmp_path/n).is_file()


def test_invalid_or_nonpositive_radial_variances_are_rejected():
    d = _diagnostics()
    d["radial"][-1]["median_corrected_rad2"] = -1
    with pytest.raises(ValueError, match="nonpositive"):
        analyse_radial_robustness(d)


def test_input_modal_diagnostics_are_not_modified():
    d = _diagnostics(excess=True)
    before = [row.copy() for row in d["radial"]]
    analyse_radial_robustness(d)
    assert d["radial"] == before
