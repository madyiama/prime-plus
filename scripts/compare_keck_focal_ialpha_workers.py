"""Compare real Keck science PSFs using 1 vs 2/4 exact Ialpha CPU workers.

Inputs, Keck NIRC2 pupil rotation/parity/resampling, HO recipe,
focal method, static config, cache status and cProfile must match.
Only requested worker count may differ. NO cache, NO cProfile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_keck_focal_quadrant import _moments


def _load(folder: Path):
    meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    prof = json.loads((folder/"psfr_profile_breakdown.json").read_text(encoding="utf-8"))
    psf = np.load(folder/"telemetry_only_psf.npy",allow_pickle=False)
    if psf.ndim != 3 or 0 in psf.shape or not np.isfinite(psf).all():
        raise ValueError(f"{folder}: invalid spectral science PSF")
    if meta.get("focal_anisoplanatism_algorithm") != "quadrant":
        raise ValueError("Both Keck cases must use focal-algorithm quadrant")
    if meta.get("focal_cache_configured") or prof.get("cprofile_enabled"):
        raise ValueError("Disable geometric cache and cProfile for A/B")
    diag = prof.get("focal_geometric_kernel",{})
    if diag.get("cache_status") != "disabled" or not diag.get("quadrant_direct"):
        raise ValueError("Expected cold direct quadrant computation, no cache")
    return meta,prof,psf


def compare(serial_dir: Path, parallel_dir: Path,
            *, rtol=1e-12, atol=1e-12):
    m0,p0,a = _load(Path(serial_dir))
    m1,p1,b = _load(Path(parallel_dir))
    for key in (
        "input_fits","input_sav","p3_pupil_rotation_hypothesis",
        "p3_pupil_resampling","p3_pupil_parity","ho_vii_method",
        "focal_anisoplanatism_algorithm","focal_cache_configured",
    ):
        if key not in m0 or key not in m1 or m0[key] != m1[key]:
            raise ValueError("Missing or differing science provenance: "+key)
    if m0.get("focal_workers") != 1:
        raise ValueError("Serial reference must use focal_workers=1")
    worker = m1.get("focal_workers")
    if worker not in (2,4):
        raise ValueError("Parallel candidate must use 2 or 4 workers")
    if a.shape != b.shape:
        raise ValueError("Different Keck science wavelength-cube shapes")
    da = p0["focal_geometric_kernel"]
    db = p1["focal_geometric_kernel"]
    if da.get("n_computed_layers") != db.get("n_computed_layers"):
        raise ValueError("Different number of actual focal layers")
    if db.get("ialpha_workers_requested") != worker:
        raise ValueError("Parallel worker provenance mismatch")
    if db.get("ialpha_parallel_calls",0) < 1:
        raise ValueError("Parallel worker path did not actually run Ialpha")
    passed = bool(np.allclose(a,b,rtol=rtol,atol=atol))
    delta = b-a
    obs_a, obs_b = _moments(a),_moments(b)
    focal_a = (float(da["covariance_ms"])+float(da["grouping_ms"]))*1e-3
    focal_b = (float(db["covariance_ms"])+float(db["grouping_ms"]))*1e-3
    total_a = float(p0["psfr_stage_timings_ms"]["psfr_init_total"])*1e-3
    total_b = float(p1["psfr_stage_timings_ms"]["psfr_init_total"])*1e-3
    return {
        "kind":"real_Keck_Ialpha_workers_vs_serial",
        "science_psf_accepted":passed,
        "serial_case":str(serial_dir),
        "parallel_case":str(parallel_dir),
        "workers":worker,"rtol":rtol,"atol":atol,
        "shape":list(a.shape),
        "max_absolute_difference":float(np.max(np.abs(delta))),
        "relative_l2_difference":float(
            np.linalg.norm(delta)/max(np.linalg.norm(a),1e-30)),
        "observable_changes":{
            key:(np.asarray(obs_b[key])-np.asarray(obs_a[key])).tolist()
            for key in obs_a
        },
        "focal_kernel_seconds":{"serial":focal_a,"parallel":focal_b},
        "focal_kernel_speedup":focal_a/focal_b if focal_b else None,
        "psfr_init_seconds":{"serial":total_a,"parallel":total_b},
        "psfr_init_speedup":total_a/total_b if total_b else None,
        "parallel_diagnostics":db,
        "note":"One Keck acquisition is not a physical calibration. Repeat and "
               "test a second pupil rotation; retain 1 worker as default.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial",type=Path,required=True)
    parser.add_argument("--parallel",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path(
        "results/compare_keck_focal_ialpha_workers.json"))
    args=parser.parse_args()
    result=compare(args.serial,args.parallel)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result["science_psf_accepted"]:
        raise SystemExit("FAIL: full science PSF differs at 1e-12")


if __name__=="__main__":
    main()
