"""P1-C read-only radial-order sensitivity from p1_modal_noise_audit.json.

Compare per-order measured/theory excess, then infer an *amplitude-only*
r0 at the already-fitted, FIXED L0 for several radial-order subsets.

THIS IS NOT a new Keck turbulence model or a joint (r0,L0) fit.
It explicitly exposes the absolute-variance weight bias (orders 2-4
dominating) versus log/relative-variance weighting and a possible
unmodeled modal response / WFS aliasing at high orders.

Usage:
 uv run python scripts/keck_p1_radial_fit_robustness.py \
   --input results/n0120_P1_modal_noise/p1_modal_noise_audit.json \
   --output results/n0120_P1_radial_robustness

No SAV, FITS, internet or re-execution of the estimator required.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_RANGES = (
    (2, 4), (2, 8), (2, 12), (2, 16), (2, 22),
    (5, 8), (9, 12), (13, 16), (17, 22),
    (5, 22), (9, 22), (12, 22),
)


def _valid_radial_arrays(d):
    rows = d["radial"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing populated radial array")
    order = np.asarray([a["radial_order"] for a in rows], dtype=int)
    obs = np.asarray([a["median_corrected_rad2"] for a in rows], dtype=float)
    raw = np.asarray([a["median_measured_rad2"] for a in rows], dtype=float)
    model = np.asarray([a["median_theory_fit_rad2"] for a in rows], dtype=float)
    if (len(np.unique(order)) != len(order) or np.any(order < 2)
            or not np.all(np.isfinite(obs))
            or not np.all(np.isfinite(raw))
            or not np.all(np.isfinite(model))
            or np.any(obs <= 0) or np.any(model <= 0)
            or np.any(raw < obs)):
        raise ValueError("Invalid or nonpositive radial variances")
    r0 = float(d["fit"]["r0_wfs_los_m"])
    L0 = float(d["fit"]["L0_m"])
    if not (np.isfinite(r0) and r0 > 0 and np.isfinite(L0) and L0 > 0):
        raise ValueError("Invalid original fitted r0 or L0")
    return order, raw, obs, model, r0, L0


def _amplitude_fit(y, m, r0_reference):
    """At fixed L0, Zernike variance ∝ r0^(-5/3).

    Two inequivalent loss functions emphasize different orders:
      - absolute: argmin_a sum_i(y_i-a*m_i)^2
      - logarithmic: argmin_a sum_i(log(y_i)-log(a*m_i))^2

    The logarithmic loss is a diagnostic and assumes equal relative
    credibility per radial bin (not an established statistical weight).
    """
    a = float(np.dot(m, y)/np.dot(m, m))
    log_a = float(np.mean(np.log(y/m)))
    a_log = float(np.exp(log_a))
    if a <= 0 or a_log <= 0:
        raise ValueError("Nonpositive variance amplitude")
    return {
        "amplitude_absolute_loss": a,
        "r0_absolute_loss_m": float(r0_reference*a**(-3/5)),
        "amplitude_log_loss": a_log,
        "r0_log_loss_m": float(r0_reference*a_log**(-3/5)),
        "shape_rms_log_residual": float(np.sqrt(np.mean(
            (np.log(y/m)-log_a)**2
        ))),
        "shape_rms_fractional_residual": float(np.sqrt(np.mean(
            (y/(a*m)-1)**2
        ))),
    }


def analyse_radial_robustness(
    diagnostics: dict, ranges=DEFAULT_RANGES,
) -> dict:
    order, raw, obs, model, r0, L0 = _valid_radial_arrays(diagnostics)
    ratios = obs/model
    noise_pct = 100*(raw-obs)/raw
    per_order = []
    for n, y, m, ratio, noise_removed_pct in zip(
        order, obs, model, ratios, noise_pct
    ):
        per_order.append({
            "radial_order": int(n),
            "observed_corrected_rad2": float(y),
            "theory_fit_rad2": float(m),
            "observed_over_theory": float(ratio),
            "noise_subtracted_fraction_percent": float(noise_removed_pct),
            "r0_fixed_L0_single_order_m": float(r0*ratio**(-3/5)),
        })
    subsets = []
    for lo, hi in ranges:
        subset = (order >= lo)&(order <= hi)
        if np.count_nonzero(subset) < 2:
            continue
        fit = _amplitude_fit(obs[subset], model[subset], r0)
        subsets.append({
            "order_min": int(lo), "order_max": int(hi),
            "n_radial_orders": int(subset.sum()),
            **fit,
        })
    maxorder = int(order.max())
    low = (order>=2)&(order<=4)
    high = (order>=max(10, maxorder-6))
    low_med = float(np.median(ratios[low])) if np.any(low) else None
    high_med = float(np.median(ratios[high])) if np.any(high) else None
    flags = []
    if (low_med is not None and high_med is not None
            and high_med > 1.20*low_med):
        flags.append("HIGH_ORDER_EXCESS_NOT_CAPTURED_BY_FITTED_MODEL")
    if maxorder >= 20:
        flags.append("HIGH_RADIAL_ORDER_DM_REPRESENTABILITY_REVIEW")
    if diagnostics["fit"].get("noise_proxy_fraction_of_total_modal_variance", 0) < .01:
        flags.append("SUBTRACTED_DM_LAG_NOISE_PROXY_SMALL_VS_MODAL_VARIANCE")
    return {
        "scope": (
            "posthoc radial-order variance audit; L0 fixed at production "
            "value, no independent seeing model or parameter error bars"
        ),
        "reference": {
            "r0_fit_los_m": r0, "L0_fit_m": L0,
            "n_modes": diagnostics["fit"].get("n_modes"),
            "n_valid_actuators": diagnostics["fit"].get("n_valid_actuators"),
            "radial_order_min": int(order.min()),
            "radial_order_max": maxorder,
            "low_order_median_data_over_model": low_med,
            "high_order_median_data_over_model": high_med,
            "noise_fraction_of_modal_trace": diagnostics["fit"].get(
                "noise_proxy_fraction_of_total_modal_variance"
            ),
            "inferred_model_bias_not_unique_cause": (
                "DM/modal pupil mismatch, WFS spatial aliasing, servo "
                "or noise/optical gain can each affect high orders; "
                "the diagnostic alone does not identify which"
            ),
        },
        "subsets": subsets,
        "per_order": per_order,
        "quality_flags": flags,
    }


def write_results(report: dict, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"p1_radial_fit_robustness.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for filename, data in (
        ("p1_radial_fit_robustness_subsets.csv", report["subsets"]),
        ("p1_radial_fit_robustness_orders.csv", report["per_order"]),
    ):
        if data:
            with (output/filename).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = report["per_order"]
    subset = report["subsets"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 8))
    axs[0].plot(
        [v["radial_order"] for v in p],
        [v["observed_over_theory"] for v in p],
        "o-", label="Corrected / theoretical (current fit)"
    )
    axs[0].axhline(1, color="black", linestyle="--")
    axs[0].set_ylabel("Variance ratio")
    axs[0].legend()
    labels = [f"{v['order_min']}–{v['order_max']}" for v in subset]
    x = np.arange(len(subset))
    axs[1].plot(
        x, [v["r0_absolute_loss_m"] for v in subset],
        "o-", label="Absolute variance least squares"
    )
    axs[1].plot(
        x, [v["r0_log_loss_m"] for v in subset],
        "s-", label="Log variance least squares"
    )
    axs[1].axhline(
        report["reference"]["r0_fit_los_m"], color="black",
        linestyle="--", label="Current full-range r0"
    )
    axs[1].set_xticks(x, labels, rotation=30, ha="right")
    axs[1].set_xlabel("Zernike radial-order range")
    axs[1].set_ylabel("Inferred r0 [m], L0 fixed")
    axs[1].legend()
    for ax in axs:
        ax.grid(alpha=.2)
    fig.suptitle("P1: r0 sensitivity to radial-order selection and loss")
    fig.tight_layout()
    fig.savefig(output/"p1_radial_fit_robustness.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyse_radial_robustness(d)
    write_results(result, args.output)
    print(
        f"[P1 radial] r0={result['reference']['r0_fit_los_m']:.6f} m, "
        f"L0={result['reference']['L0_fit_m']:.3f} m; "
        f"order {result['reference']['radial_order_min']}"
        f"–{result['reference']['radial_order_max']}; "
        f"flags={result['quality_flags']}"
    )
    for subset in result["subsets"]:
        print(
            f"  orders {subset['order_min']:2d}–"
            f"{subset['order_max']:2d}: "
            f"r0(abs)={subset['r0_absolute_loss_m']:.6f}m, "
            f"r0(log)={subset['r0_log_loss_m']:.6f}m"
        )
    print(f"[P1 radial] Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
