"""Isolated Keck focal-cone outer-scale sweep without a science-PSF fit.

Reuses a completed Keck smoke run's atmospheric geometry and *real* NIRC2
diffraction OTF. Unlike a full reconstruction, this does not run telemetry,
the HO model, or a source fit. It uses Flicker's exact analytical L0=inf
asymptotic (not an enormous finite number prone to cancellation).

Example (run from repo root):
  uv run python scripts/keck_l0_cone_sweep.py \
    --case results/n0120_P2_1layers \
    --output results/n0120_P2_1layers/l0_sweep

Use the 1-layer case first (fastest). Then optionally repeat for 7 layers.
"""

from __future__ import annotations

import argparse
import ast
from configparser import ConfigParser
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3.aoSystem.anisoplanatismModel import (
    focal_angular_function, focal_anisoplanatism_phase_variance,
)
from p3.psfr.psfR import _anisoplanatism_zero_lag
from scripts.keck_n0013_smoke import _structure_function_wfe_bridge


def _json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required Keck audit missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _actuator_count(folder: Path):
    ini = list(folder.glob("*.ini"))
    if len(ini) != 1:
        raise ValueError(
            "Cannot identify a unique Keck configuration INI in "
            f"{folder}; use --n-actu to supply the DM nActu1D"
        )
    cfg = ConfigParser()
    cfg.optionxform = str
    cfg.read(ini[0])
    raw = ast.literal_eval(cfg.get("DM", "NumberActuators"))
    values = np.asarray(raw, dtype=float).reshape(-1)
    if (len(values) != 1 or values[0] < 1
            or values[0] != int(values[0])):
        raise ValueError("Expected exactly one positive integer DM nActu1D")
    return int(values[0])


def load_geometry(case: Path, override_nactu=None):
    aniso = _json(case / "anisoplanatism_diagnostics.json")
    grid = _json(case / "frequency_grid_diagnostics.json")
    otf = np.load(case / "diffraction_otf.npy", allow_pickle=False)
    otf = np.asarray(otf, dtype=float)
    if otf.ndim != 2 or otf.shape[0] != otf.shape[1]:
        raise ValueError("Expected an [nOtf,nOtf] NIRC2 diffraction OTF")
    weights = np.asarray(aniso["layer_weights"], dtype=float)
    heights = np.asarray(aniso["layer_heights_los_m"], dtype=float)
    if not (weights.size == heights.size == aniso["atmosphere_layers"]
            and np.isclose(weights.sum(), 1, atol=1e-6)
            and np.all(weights >= 0)):
        raise ValueError("Invalid MASS-DIMM layer geometry or weights")
    nactu = (_actuator_count(case)
             if override_nactu is None else int(override_nactu))
    if nactu <= 0:
        raise ValueError("nActu1D must be positive")
    wl = float(aniso["reference_wavelength_m"])
    r0 = float(aniso["atmosphere_r0_m"])
    diam = float(grid["telescope_diameter_m"])
    samp = float(grid["otf_sampling_sampRef"])
    h_lgs = float(aniso["lgs_height_los_m"])
    if not (wl > 0 and r0 > 0 and diam > 0 and samp > 0 and h_lgs > 0):
        raise ValueError("All physical scales must be positive")
    return {
        "aniso": aniso, "otf": otf, "weights": weights,
        "heights": heights, "n_actu": nactu, "wavelength_m": wl,
        "r0_m": r0, "diameter_m": diam, "samp_ref": samp,
        "lgs_height_m": h_lgs, "otf_size": otf.shape[0],
    }


def sweep_outer_scale(geometry, outer_scales):
    """Return numerical focal SF RMS vs Hardy at fixed geometry and r0."""
    g = geometry
    tel = NS(D=g["diameter_m"])
    src = NS(nSrc=1, direction=[np.zeros(1), np.zeros(1)])
    lgs = NS(
        height=np.array([g["lgs_height_m"]], dtype=float),
        direction=[np.zeros(1), np.zeros(1)],
    )
    atm = NS(
        nL=len(g["weights"]), heights=g["heights"],
        weights=g["weights"], r0=g["r0_m"],
        wvl=g["wavelength_m"], L0=1.0,
    )
    hardy_var = float(focal_anisoplanatism_phase_variance(tel, atm, lgs))
    hardy_nm = np.sqrt(hardy_var) * g["wavelength_m"] * 1e9/(2*np.pi)
    rows = []
    for outer_scale in outer_scales:
        atm.L0 = float(outer_scale)
        print(
            f"[Cone outer scale] L0={atm.L0:g}m, nL={atm.nL}, "
            f"nActu1D={g['n_actu']}, nOtf={g['otf_size']}",
            flush=True,
        )
        # Actual P3 numerical focal-cone kernel; no HO, TT or science fit.
        per_layer = focal_angular_function(
            tel, atm, src, lgs,
            np.array([g["n_actu"]]), g["otf_size"], g["samp_ref"],
        )
        strength = g["weights"] * g["r0_m"]**(-5.0/3.0)
        sf = _anisoplanatism_zero_lag(
            np.einsum("slxy,l->sxy", per_layer, strength)[0]
        )
        report = _structure_function_wfe_bridge(
            sf, g["otf"], g["wavelength_m"]
        )
        rows.append({
            "L0_m": (float(atm.L0) if np.isfinite(atm.L0) else "infinity"),
            "hardy_piston_removed_variance_rad2": hardy_var,
            "hardy_wfe_nm": float(hardy_nm),
            "flicker_pupil_variance_rad2": report[
                "pupil_piston_removed_phase_variance_rad2"
            ],
            "flicker_pupil_wfe_nm": report[
                "pupil_piston_removed_wfe_nm"
            ],
            "flicker_strehl_equivalent_wfe_nm": report[
                "strehl_equivalent_wfe_nm"
            ],
            "hardy_over_flicker_pupil_wfe": (
                float(hardy_nm / report["pupil_piston_removed_wfe_nm"])
                if report["pupil_piston_removed_wfe_nm"] > 0 else None
            ),
            "fractional_pupil_variance_vs_hardy": (
                report["pupil_piston_removed_phase_variance_rad2"]
                / hardy_var
            ) if hardy_var > 0 else None,
            "jensen_gap_rad2": report["jensen_gap_phase_variance_rad2"],
            "sf_zero_lag_rad2": report["sf_zero_lag_rad2"],
            "negative_diffraction_otf_weight_fraction": report[
                "diffraction_otf_negative_weight_fraction"
            ],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--outer-scales", default="31.045692184830187,100,300,1000,inf",
        help="Comma-separated L0 in metres; inf uses exact Kolmogorov limit",
    )
    parser.add_argument(
        "--n-actu", type=int, default=None,
        help="Optional override for DM nActu1D read from the run's INI",
    )
    args = parser.parse_args()
    values = []
    for item in args.outer_scales.split(","):
        item = item.strip().lower()
        value = float(item)
        if not (value > 0 and (np.isfinite(value) or np.isposinf(value))):
            parser.error(f"Invalid outer scale {item!r}")
        values.append(value)
    geom = load_geometry(args.case, args.n_actu)
    rows = sweep_outer_scale(geom, values)
    original = geom["aniso"].get("pupil_sf_bridge_focal_only") or {}
    baseline = next((
        row for row in rows
        if isinstance(row["L0_m"], float)
        and np.isclose(row["L0_m"], geom["aniso"]["focal_model_conditions"][
            "von_karman_outer_scale_m"
        ], rtol=1e-9)
    ), None)
    baseline_rel_error = None
    if baseline and original.get("pupil_piston_removed_wfe_nm"):
        baseline_rel_error = (
            baseline["flicker_pupil_wfe_nm"]
            / original["pupil_piston_removed_wfe_nm"] - 1
        )
    result = {
        "source_case": str(args.case.resolve()),
        "physics": (
            "P3 focal_angular_function with fixed on-axis Keck geometry; "
            "infinite outer scale uses analytic McDonald asymptotic, "
            "no science fitting"
        ),
        "n_actu_1d": geom["n_actu"],
        "n_layers": len(geom["weights"]),
        "reference_wavelength_m": geom["wavelength_m"],
        "r0_m": geom["r0_m"],
        "sampling": geom["samp_ref"],
        "n_otf": geom["otf_size"],
        "baseline_relative_wfe_error_vs_original_run": baseline_rel_error,
        "baseline_warning": (
            "A mismatch at the original L0 means the standalone setup "
            "does not reproduce the original source/GS or sampling geometry."
            if baseline_rel_error is not None
            and abs(baseline_rel_error) > 1e-3 else None
        ),
        "sweep": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "l0_cone_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    import csv
    with (args.output / "l0_cone_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [str(row["L0_m"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, [row["hardy_wfe_nm"] for row in rows],
            "k--", label="Hardy (Kolmogorov)")
    ax.plot(x, [row["flicker_pupil_wfe_nm"] for row in rows],
            "o-", label="Flicker: pupil-weighted SF RMS")
    ax.plot(x, [row["flicker_strehl_equivalent_wfe_nm"] for row in rows],
            "s-", label="Flicker: Strehl-equivalent RMS")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Outer scale L0 [m] (inf = exact asymptotic limit)")
    ax.set_ylabel("Focal anisoplanatism OPD RMS [nm]")
    ax.set_title(
        f"Keck focal cone: {len(geom['weights'])} layers, "
        "fixed WFS r0 and NIRC2 pupil OTF"
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "l0_cone_sweep.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
