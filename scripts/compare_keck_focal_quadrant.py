"""Strict real Keck A/B: symmetry vs direct quadrant focal covariance.

Both cases MUST be computed from identical FITS/SAV, pupil settings,
HO/TT configuration and WITHOUT geometric cache or cProfile. The
whole [y,x,wavelength] science PSF cube is the acceptance gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_case(folder: Path, method: str):
    meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    profile = json.loads(
        (folder/"psfr_profile_breakdown.json").read_text(encoding="utf-8"))
    if meta.get("focal_anisoplanatism_algorithm") != method:
        raise ValueError(f"Expected focal method {method!r} in {folder}")
    psf = np.load(folder/"telemetry_only_psf.npy", allow_pickle=False)
    if psf.ndim != 3 or min(psf.shape) == 0 or not np.isfinite(psf).all():
        raise ValueError(f"{folder}: expected finite 3-D science PSF cube")
    if meta.get("focal_cache_configured"):
        raise ValueError("Disable geometric cache for speed/parity A/B")
    if profile.get("cprofile_enabled"):
        raise ValueError("Disable cProfile for absolute timings")
    kernel = profile.get("focal_geometric_kernel", {})
    if kernel.get("cache_status") != "disabled":
        raise ValueError("Expected disabled focal geometric cache")
    return meta, profile, psf


def _moments(psf):
    y, x = np.indices(psf.shape[:2], dtype=np.float64)
    total = np.sum(psf, axis=(0, 1))
    if not np.isfinite(total).all() or np.any(total <= 0):
        raise ValueError("Integrated Keck science flux must be positive")
    return {
        "integrated_flux": total.tolist(),
        "peak_flux": np.max(psf, axis=(0, 1)).tolist(),
        "centroid_x_pixels": (np.sum(psf*x[:, :, None],
                                    axis=(0, 1))/total).tolist(),
        "centroid_y_pixels": (np.sum(psf*y[:, :, None],
                                    axis=(0, 1))/total).tolist(),
    }


def compare(symmetry_folder, quadrant_folder, *, rtol=1e-12, atol=1e-12):
    m0, p0, psf0 = _load_case(Path(symmetry_folder), "symmetry")
    m1, p1, psf1 = _load_case(Path(quadrant_folder), "quadrant")
    required = (
        "input_fits", "input_sav", "p3_pupil_rotation_hypothesis",
        "p3_pupil_resampling", "p3_pupil_parity", "ho_vii_method",
    )
    missing = [key for key in required if key not in m0 or key not in m1]
    if missing:
        raise ValueError("Missing Keck provenance fields: "+str(missing))
    changed = [key for key in required if m0[key] != m1[key]]
    if changed:
        raise ValueError("Different science input/pupil/HO settings: "
                         +str(changed))
    if psf0.shape != psf1.shape:
        raise ValueError("Keck science PSF cubes have different shapes")
    if p0.get("focal_geometric_kernel", {}).get("n_computed_layers") != \
            p1.get("focal_geometric_kernel", {}).get("n_computed_layers"):
        raise ValueError("Compared a different number of focal layers")
    error = psf1-psf0
    same = bool(np.allclose(psf0, psf1, rtol=rtol, atol=atol))
    mom0, mom1 = _moments(psf0), _moments(psf1)
    delta = {key: (np.asarray(mom1[key])-np.asarray(mom0[key])).tolist()
             for key in mom0}
    k0 = p0["focal_geometric_kernel"]
    k1 = p1["focal_geometric_kernel"]
    f0 = (float(k0["covariance_ms"])+float(k0["grouping_ms"]))*1e-3
    f1 = (float(k1["covariance_ms"])+float(k1["grouping_ms"]))*1e-3
    init0 = float(p0["psfr_stage_timings_ms"]["psfr_init_total"])*1e-3
    init1 = float(p1["psfr_stage_timings_ms"]["psfr_init_total"])*1e-3
    return {
        "kind": "real_Keck_symmetry_vs_direct_quadrant",
        "symmetry_case": str(symmetry_folder),
        "quadrant_case": str(quadrant_folder),
        "science_psf_accepted": same,
        "rtol": rtol, "atol": atol,
        "science_psf_shape": list(psf0.shape),
        "science_psf_max_absolute_error": float(np.max(np.abs(error))),
        "science_psf_relative_l2_error": float(
            np.linalg.norm(error)/max(np.linalg.norm(psf0), 1e-30)),
        "science_psf_moments_symmetry": mom0,
        "science_psf_moments_quadrant": mom1,
        "science_psf_moment_differences": delta,
        "focal_layer_kernel_seconds": {"symmetry": f0, "quadrant": f1},
        "focal_layer_kernel_speedup": f0/f1 if f1 > 0 else None,
        "constructor_seconds": {"symmetry": init0, "quadrant": init1},
        "constructor_speedup": init0/init1 if init1 > 0 else None,
        "quadrant_diagnostics": k1,
        "caution": "Full NIRC2 science parity is required for n0120 and "
                   "another acquisition with a different pupil rotation. "
                   "No claim of new instrument or scientific calibration.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symmetry", type=Path, required=True)
    p.add_argument("--quadrant", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("results/compare_keck_focal_quadrant.json"))
    args = p.parse_args()
    report = compare(args.symmetry, args.quadrant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["science_psf_accepted"]:
        raise SystemExit("FAIL: full Keck science PSF parity at 1e-12")


if __name__ == "__main__":
    main()
