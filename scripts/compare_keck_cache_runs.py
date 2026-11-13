"""Compare a cold and warm Keck geometric focal SF cache on SAME science input.

The pupil/OPD/OTFs are independently reconstructed in both runs.
Never compare unequal rotations or different observations as if their
science PSFs should match. Output is audit JSON and a failing exit code
for any mismatch.

Example:
 uv run python scripts/compare_keck_cache_runs.py \
     --cold results/focal_cache_cold/n0120 \
     --warm results/focal_cache_warm/n0120
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def compare(cold, warm, *, rtol=1e-12, atol=1e-12):
    cold=Path(cold)
    warm=Path(warm)
    def load(folder):
        m=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        p=json.loads(
            (folder/"psfr_profile_breakdown.json").read_text(encoding="utf-8")
        )
        img=np.load(folder/"telemetry_only_psf.npy",allow_pickle=False)
        return m,p,np.asarray(img,dtype=np.float64)
    meta_c,profile_c,a=load(cold)
    meta_w,profile_w,b=load(warm)
    comparable=(
        "input_fits","input_sav","focal_anisoplanatism_algorithm",
        "p3_pupil_rotation_hypothesis","p3_pupil_parity",
        "p3_pupil_resampling",
    )
    if any(meta_c.get(k)!=meta_w.get(k) for k in comparable):
        raise ValueError("Not the same FITS/SAV, focal algorithm, or pupil hypothesis")
    if (a.shape!=b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise ValueError("Keck science PSF shape/finite validation failed")
    c=profile_c["focal_geometric_kernel"]
    w=profile_w["focal_geometric_kernel"]
    if (c.get("cache_status")!="miss" or
            w.get("cache_status")!="hit" or
            c.get("cache_key")!=w.get("cache_key")):
        raise ValueError(
            "Expected same-key cold=miss / warm=hit, found "+
            str((c.get("cache_status"),w.get("cache_status")))
        )
    diff=b-a
    total_c=float(meta_c["psfr_total_init_elapsed_s"])
    total_w=float(meta_w["psfr_total_init_elapsed_s"])
    frequency_c=float(meta_c["frequency_anisoplanatism_elapsed_s"])
    frequency_w=float(meta_w["frequency_anisoplanatism_elapsed_s"])
    parity=bool(np.allclose(a,b,rtol=rtol,atol=atol))
    return {
        "kind":"real_Keck_cold_vs_warm_geometry_only_cache",
        "parity_passed":parity,
        "shape":list(a.shape),
        "max_absolute_error":float(np.max(np.abs(diff))),
        "relative_l2_error":float(np.linalg.norm(diff)/np.linalg.norm(a)),
        "rtol":rtol,"atol":atol,
        "cache_key":c["cache_key"],
        "cold_cache_status":c["cache_status"],
        "warm_cache_status":w["cache_status"],
        "initialization_seconds":{"cold":total_c,"warm":total_w},
        "initialization_speedup":total_c/total_w,
        "nested_anisoplanatism_seconds":{
            "cold":frequency_c,"warm":frequency_w,
        },
        "nested_anisoplanatism_speedup":frequency_c/frequency_w,
        "cprofile_enabled_on_either_run":(
            bool(profile_c.get("cprofile_enabled")) or
            bool(profile_w.get("cprofile_enabled"))
        ),
        "warning":"Do not compare profiled/unprofiled absolute runtimes; "
                  "pupil and derived optical OTF always recomputed.",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold",required=True,type=Path)
    ap.add_argument("--warm",required=True,type=Path)
    ap.add_argument("--output",type=Path,
                    default=Path("results/compare_keck_focal_cache.json"))
    a=ap.parse_args()
    report=compare(a.cold,a.warm)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps(report,indent=2,allow_nan=False))
    if not report["parity_passed"]:
        raise SystemExit("FAIL: cached Keck PSF differs from uncached Keck PSF")


if __name__=="__main__":
    main()
