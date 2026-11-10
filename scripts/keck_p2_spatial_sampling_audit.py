"""P2 Keck/NIRC2: discriminate Flicker lag-grid bias from finite outer scale.

Does NOT change the production PSF-R or fit a science image. Reuses the
on-axis turbulence, sodium altitude and true NIRC2 diffraction OTF from
an existing Keck smoke result. For each L0 it evaluates three variants:

  legacy            Native P3 focal map + index-only spline zoom (baseline)
  remap_legacy      SAME native legacy map but sampled at physical OTF lags
                    (may lack coverage; quantified and NOT a correction)
  physical_pupil    Recompute Flicker pair covariance on a pupil coordinate
                    grid spanning [-D/2, D/2], then resample at the actual
                    OTF physical lags, retaining the same square-grid
                    matrix_to_map mean (NOT the exact nonstationary OTF)

A separate stationary Dphi ~ |rho|^(5/3) synthetic calibrates the
coordinate mapping without relying on Hardy, r0 or an astronomical PSF.

Example:
    uv run python scripts/keck_p2_spatial_sampling_audit.py \
      --case results/n0120_P2_1layers \
      --output results/n0120_P2_spatial_audit

Use one layer first: each additional layer is expensive on CPU.
The exact L0=infinity branch is used, never L0=1e9 as a surrogate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3.aoSystem.anisoplanatismModel import (
    focal_angular_function, focal_anisoplanatism_phase_variance,
)
from p3.aoSystem.FourierUtils import interpolateSupport
from p3.psfr.psfR import _anisoplanatism_zero_lag
from scripts.keck_n0013_smoke import _structure_function_wfe_bridge
from scripts.keck_l0_cone_sweep import load_geometry


def otf_lag_axis(n_otf: int, diameter_m: float, samp_ref: float) -> np.ndarray:
    """Science OTF separation in metres, from NIRC2 pixel angular scale.

    delta_rho = lambda / (N * pixel_scale_rad) = D*sampRef/N in the
    standard kRef=1 Keck smoke run. The OTF includes padded *zero*
    support beyond the physical pupil difference D; that padding is NOT
    itself a physically sampled turbulent pupil.
    """
    if n_otf < 2 or diameter_m <= 0 or samp_ref <= 0:
        raise ValueError("Invalid OTF sampling geometry")
    return (np.arange(n_otf) - n_otf//2) * (
        diameter_m * samp_ref / n_otf
    )


def native_lag_axis(
    n_actu: int, diameter_m: float, samp_ref: float, pupil_grid_mode: str,
) -> np.ndarray:
    """Actual native Flicker map lag axis, in metres."""
    n_ph = 2*int(n_actu) + 1
    if n_ph < 3 or diameter_m <= 0 or samp_ref <= 0:
        raise ValueError("Invalid focal covariance grid")
    if pupil_grid_mode == "legacy":
        # Current code: x_i = umax*(i - n_ph//2)/n_ph,
        # umax = (D/4) * max(1,sampRef).
        step = (diameter_m / 4) * max(1.0, samp_ref) / n_ph
    elif pupil_grid_mode == "physical_diameter":
        # Exact on-axis physical pupil support [-D/2, +D/2].
        step = diameter_m/(n_ph-1)
    else:
        raise ValueError(f"Unknown pupil grid {pupil_grid_mode}")
    return (np.arange(2*n_ph-1) - (n_ph-1)) * step


def resample_native_to_otf(
    sf_native: np.ndarray, native_axis_m: np.ndarray,
    target_axis_m: np.ndarray, order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate an explicitly metrically calibrated 2-D lag map.

    Values outside the native domain are set to ZERO ONLY because the
    physical OTF should also have no illuminated pairs beyond ±D.
    The caller MUST inspect the weighted missing-support fraction before
    interpreting the result. Do not use a clipped extrapolated SF.
    """
    sf_native = np.asarray(sf_native, dtype=float)
    native_axis_m = np.asarray(native_axis_m, dtype=float)
    target_axis_m = np.asarray(target_axis_m, dtype=float)
    if (sf_native.ndim != 2 or sf_native.shape !=
            (native_axis_m.size, native_axis_m.size)
            or np.any(~np.isfinite(sf_native))
            or np.any(~np.isfinite(native_axis_m))):
        raise ValueError("Invalid native SF/metric lag axis")
    if not np.allclose(np.diff(native_axis_m), np.diff(native_axis_m)[0]):
        raise ValueError("Native focal SF axis must be regularly spaced")
    step = float(native_axis_m[1] - native_axis_m[0])
    if step <= 0:
        raise ValueError("Native SF coordinate step must be positive")
    coords_1d = (target_axis_m - native_axis_m[0])/step
    yy, xx = np.meshgrid(coords_1d, coords_1d, indexing="ij")
    outside = (
        (yy < 0) | (xx < 0)
        | (yy > sf_native.shape[0]-1)
        | (xx > sf_native.shape[1]-1)
    )
    mapped = map_coordinates(
        sf_native, [yy, xx], order=order,
        mode="constant", cval=0.0, prefilter=order > 1,
    )
    return mapped, outside


def relative_error_on_otf_support(
    actual, expected, otf, relative_floor=1e-12,
):
    positive = np.maximum(np.asarray(otf, dtype=float), 0.0)
    weight = positive / positive.sum()
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    denom = float(np.sum(weight * expected**2))
    return float(np.sqrt(np.sum(weight*(actual-expected)**2)
                         / max(denom, relative_floor)))


def synthetic_spatial_calibration(geometry):
    """Stationary power law on *known physical metre axes*.

    This is deliberately separate from the Flicker phase covariance:
    if this fails the error is in coordinate mapping / interpolation, not
    in the nonstationary covariance or the atmospheric normalization.
    """
    g = geometry
    target = otf_lag_axis(
        g["otf_size"], g["diameter_m"], g["samp_ref"]
    )
    yy, xx = np.meshgrid(target, target, indexing="ij")
    expected = np.hypot(xx, yy)**(5.0/3.0)
    otf = g["otf"]
    modes = {}
    for mode in ("legacy", "physical_diameter"):
        native = native_lag_axis(
            g["n_actu"], g["diameter_m"], g["samp_ref"], mode
        )
        ny, nx = np.meshgrid(native, native, indexing="ij")
        src = np.hypot(nx, ny)**(5.0/3.0)
        corrected, outside = resample_native_to_otf(src, native, target)
        weight = np.maximum(otf, 0.0)
        missing_fraction = (
            float(weight[outside].sum()/weight.sum())
        )
        record = {
            "native_lag_step_m": float(native[1]-native[0]),
            "native_lag_half_span_m": float(native.max()),
            "otf_lag_step_m": float(target[1]-target[0]),
            "otf_lag_half_span_m": float(target.max()),
            "otf_weight_outside_native_support": missing_fraction,
            "physical_interpolation_relative_rms_error": (
                relative_error_on_otf_support(corrected, expected, otf)
            ),
        }
        if mode == "legacy":
            zoom = interpolateSupport(src, g["otf_size"])
            record["legacy_index_zoom_relative_rms_error"] = (
                relative_error_on_otf_support(zoom, expected, otf)
            )
        modes[mode] = record
    return modes


def _summary(sf, geometry, hardy_nm, outside_fraction=0.0):
    sf = _anisoplanatism_zero_lag(sf)
    report = _structure_function_wfe_bridge(
        sf, geometry["otf"], geometry["wavelength_m"]
    )
    return {
        "pupil_wfe_nm": report["pupil_piston_removed_wfe_nm"],
        "strehl_equivalent_wfe_nm": report[
            "strehl_equivalent_wfe_nm"
        ],
        "pupil_variance_rad2": report[
            "pupil_piston_removed_phase_variance_rad2"
        ],
        "hardy_over_pupil_wfe": (
            float(hardy_nm/report["pupil_piston_removed_wfe_nm"])
            if report["pupil_piston_removed_wfe_nm"] > 0 else None
        ),
        "otf_weight_outside_native_support": float(outside_fraction),
        "sf_zero_lag_rad2": report["sf_zero_lag_rad2"],
        "negative_otf_weight_fraction": report[
            "diffraction_otf_negative_weight_fraction"
        ],
    }


def _make_instrument_objects(geometry, outer_scale):
    g = geometry
    tel = NS(D=g["diameter_m"])
    atm = NS(
        L0=float(outer_scale),
        nL=len(g["weights"]),
        heights=np.asarray(g["heights"], dtype=float),
        weights=np.asarray(g["weights"], dtype=float),
        r0=g["r0_m"], wvl=g["wavelength_m"],
    )
    src = NS(nSrc=1, direction=[np.zeros(1), np.zeros(1)])
    lgs = NS(
        height=np.array([g["lgs_height_m"]], dtype=float),
        direction=[np.zeros(1), np.zeros(1)],
    )
    return tel, atm, src, lgs


def compare_one_outer_scale(geometry, outer_scale):
    """Three variants, one physical L0, one set of atmospheric parameters."""
    g = geometry
    tel, atm, src, lgs = _make_instrument_objects(g, outer_scale)
    phase_var = float(focal_anisoplanatism_phase_variance(tel, atm, lgs))
    hardy_nm = np.sqrt(phase_var)*g["wavelength_m"]*1e9/(2*np.pi)
    target_axis = otf_lag_axis(
        g["otf_size"], g["diameter_m"], g["samp_ref"]
    )
    strengths = g["weights"]*g["r0_m"]**(-5.0/3.0)
    n_src = g["n_actu"]

    print(f"[P2 spatial] L0={outer_scale:g}, {len(strengths)} layer(s), "
          f"actuators={n_src}, nOtf={g['otf_size']}", flush=True)
    legacy = focal_angular_function(
        tel, atm, src, lgs, np.array([n_src]),
        g["otf_size"], g["samp_ref"],
        output_grid="native", pupil_grid_mode="legacy",
    )
    # Preserve exact original index-based zoom for a strict baseline.
    legacy_sf = np.einsum("slxy,l->xy", legacy, strengths)
    zoom_sf = interpolateSupport(legacy_sf, g["otf_size"])
    legacy_report = _summary(zoom_sf, g, hardy_nm)

    native_axis_old = native_lag_axis(
        n_src, g["diameter_m"], g["samp_ref"], "legacy"
    )
    remap, old_outside = resample_native_to_otf(
        legacy_sf, native_axis_old, target_axis
    )
    otf_weights = np.maximum(g["otf"], 0.0)
    missing_old = float(
        otf_weights[old_outside].sum()/otf_weights.sum()
    )
    old_report = _summary(remap, g, hardy_nm, missing_old)

    # Evaluate exactly the same Flicker formula using x_i in [-D/2,D/2]
    # before the matrix_to_map() average (still square/unmasked).
    physical_native = focal_angular_function(
        tel, atm, src, lgs, np.array([n_src]),
        g["otf_size"], g["samp_ref"],
        output_grid="native", pupil_grid_mode="physical_diameter",
    )
    physical_sf = np.einsum("slxy,l->xy", physical_native, strengths)
    physical_axis = native_lag_axis(
        n_src, g["diameter_m"], g["samp_ref"], "physical_diameter"
    )
    corrected, physical_outside = resample_native_to_otf(
        physical_sf, physical_axis, target_axis
    )
    missing_physical = float(
        otf_weights[physical_outside].sum()/otf_weights.sum()
    )
    physical_report = _summary(
        corrected, g, hardy_nm, missing_physical
    )

    return {
        "outer_scale_m": (
            "infinity" if np.isposinf(outer_scale) else float(outer_scale)
        ),
        "hardy_wfe_nm": float(hardy_nm),
        "legacy": legacy_report,
        "physical_resampling_of_legacy_grid": old_report,
        "physical_pupil_grid": physical_report,
    }


def run_audit(geometry, outer_scales):
    synthetic = synthetic_spatial_calibration(geometry)
    results = [compare_one_outer_scale(geometry, l0) for l0 in outer_scales]
    orig = geometry["aniso"].get("pupil_sf_bridge_focal_only", {})
    original_wfe = orig.get("pupil_piston_removed_wfe_nm")
    baseline = geometry["aniso"].get(
        "focal_model_conditions", {}
    ).get("von_karman_outer_scale_m")
    closest = next(
        (r for r in results if isinstance(r["outer_scale_m"], float)
         and baseline is not None
         and np.isclose(r["outer_scale_m"], baseline, rtol=1e-9)),
        None,
    )
    baseline_relative_error = None
    if original_wfe and closest:
        baseline_relative_error = (
            closest["legacy"]["pupil_wfe_nm"]/original_wfe - 1
        )
    quality_flags = []
    if baseline_relative_error is None:
        quality_flags.append("BASELINE_NOT_INCLUDED")
    elif abs(baseline_relative_error) > 1e-4:
        quality_flags.append("LEGACY_BASELINE_MISMATCH")
    if synthetic["physical_diameter"][
        "physical_interpolation_relative_rms_error"
    ] > 0.05:
        quality_flags.append("PHYSICAL_STATIONARY_GRID_TEST_FAIL")
    if any(
        row["physical_resampling_of_legacy_grid"][
            "otf_weight_outside_native_support"
        ] > 0.01 for row in results
    ):
        quality_flags.append("LEGACY_REMAP_INCOMPLETE_SUPPORT")
    if any(
        row["physical_pupil_grid"][
            "otf_weight_outside_native_support"
        ] > 0.01 for row in results
    ):
        quality_flags.append("PHYSICAL_GRID_OUTSIDE_OTF_SUPPORT")
    return {
        "source_code_scope": (
            "audit only; production focal_angular_function default "
            "and matrix_to_map are untouched"
        ),
        "geometry": {
            "diameter_m": geometry["diameter_m"],
            "n_actu_1d": geometry["n_actu"],
            "n_layers": len(geometry["weights"]),
            "n_otf": geometry["otf_size"],
            "samp_ref": geometry["samp_ref"],
            "r0_m": geometry["r0_m"],
            "wavelength_m": geometry["wavelength_m"],
            "lgs_height_m": geometry["lgs_height_m"],
            "otf_delta_lag_m": (
                geometry["diameter_m"] * geometry["samp_ref"]
                / geometry["otf_size"]
            ),
        },
        "synthetic_stationary_spatial_control": synthetic,
        "baseline_relative_legacy_wfe_error": baseline_relative_error,
        "quality_flags": quality_flags,
        "sweep": results,
        "limitations": [
            "physical_pupil_grid still averages all square-grid pairs "
            "through matrix_to_map, not a true masked nonstationary OTF",
            "physical resampling sets values outside computed native "
            "lags to zero; missing positive diffraction-OTF weight "
            "is reported and must be negligible before interpretation",
            "Hardy analytical formula assumes a different idealized "
            "pupil and infinite Kolmogorov outer scale",
            "no observational PSF/photometry fitting or TT adjustment",
        ],
    }


def _write_artifacts(result, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "p2_spatial_sampling_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    rows = []
    for item in result["sweep"]:
        for mode in (
            "legacy", "physical_resampling_of_legacy_grid",
            "physical_pupil_grid",
        ):
            row = {
                "L0_m": item["outer_scale_m"],
                "method": mode, "hardy_wfe_nm": item["hardy_wfe_nm"],
            }
            row.update(item[mode])
            rows.append(row)
    with (destination / "p2_spatial_sampling_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [str(x["outer_scale_m"]) for x in result["sweep"]]
    xx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    for mode, label in [
        ("legacy", "P3 legacy (index zoom)"),
        ("physical_resampling_of_legacy_grid",
         "Physical lag resampling (incomplete support)"),
        ("physical_pupil_grid", "Physical pupil grid + lag resampling"),
    ]:
        ax.plot(
            xx, [row[mode]["pupil_wfe_nm"] for row in result["sweep"]],
            "o-", label=label,
        )
    ax.plot(
        xx, [row["hardy_wfe_nm"] for row in result["sweep"]],
        "k--", label="Hardy (Kolmogorov)",
    )
    ax.set_xticks(xx, labels)
    ax.set_xlabel("L0 [m] (infinity uses analytic Kolmogorov limit)")
    ax.set_ylabel("Piston-removed, pupil-SF-equivalent WFE [nm OPD]")
    ax.set_title("Keck P2: grid sampling vs outer-scale effects")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "p2_spatial_sampling_audit.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--outer-scales", default=None,
        help="Comma-separated L0 [m], e.g. 31.045692,100,300,inf. "
             "Default: measured L0,100,300,inf.",
    )
    parser.add_argument("--n-actu", type=int, default=None)
    args = parser.parse_args()
    geom = load_geometry(args.case, args.n_actu)
    measured_l0 = geom["aniso"]["focal_model_conditions"][
        "von_karman_outer_scale_m"
    ]
    raw = (
        args.outer_scales if args.outer_scales is not None
        else f"{measured_l0},100,300,inf"
    )
    outer_scales = [float(v.strip()) for v in raw.split(",")]
    if not outer_scales or not all(
        v > 0 and (np.isfinite(v) or np.isposinf(v))
        for v in outer_scales
    ):
        parser.error("Outer scales must be positive metres or inf")
    result = run_audit(geom, outer_scales)
    _write_artifacts(result, args.output)
    print(f"[P2 audit] files written to {args.output}", flush=True)
    print(json.dumps({
        "quality_flags": result["quality_flags"],
        "baseline_relative_legacy_wfe_error": result[
            "baseline_relative_legacy_wfe_error"
        ],
        "synthetic_control": result[
            "synthetic_stationary_spatial_control"
        ],
        "sweep": [
            {
                "L0_m": r["outer_scale_m"],
                "hardy_nm": r["hardy_wfe_nm"],
                "legacy_nm": r["legacy"]["pupil_wfe_nm"],
                "physically_sampled_nm": r[
                    "physical_pupil_grid"
                ]["pupil_wfe_nm"],
                "coverage_lost": r[
                    "physical_pupil_grid"
                ]["otf_weight_outside_native_support"],
            }
            for r in result["sweep"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
