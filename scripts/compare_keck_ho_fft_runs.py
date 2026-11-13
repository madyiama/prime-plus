"""Numerical and timing acceptance for same Keck data, old vs batched HO FFT.

Requires TWO independently reconstructed FITS/SAV runs with same pupil
rotation/hypothesis and same focal-cache hit state, differing ONLY
in HO FFT mode. Compares actual HO SF and full spectral PSF arrays.
An unchanged final PSF alone is not sufficient scientific evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_case(folder,mode):
    folder=Path(folder)
    meta=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    profile=json.loads(
        (folder/"psfr_profile_breakdown.json").read_text(encoding="utf-8")
    )
    if meta.get("ho_fft_mode")!=mode:
        raise ValueError(f"Expected {mode!r} in {folder}/metadata.json")
    psf=np.load(folder/"telemetry_only_psf.npy",allow_pickle=False)
    dphi=np.load(folder/"ho_dphi_ao.npy",allow_pickle=False)
    return meta,profile,psf,dphi


def _compare_array(a,b,*,name,rtol,atol):
    if (a.shape!=b.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise ValueError(name+" must have equal dimensions and finite values")
    diff=b-a
    return {
        "shape":list(a.shape),
        "parity_passed":bool(np.allclose(a,b,rtol=rtol,atol=atol)),
        "max_abs":float(np.max(np.abs(diff))),
        "relative_l2":float(
            np.linalg.norm(diff)/max(np.linalg.norm(a),1e-30)
        ),
    }


def compare(old,new,*,rtol=1e-12,atol=1e-12):
    m0,p0,psf0,dphi0=_load_case(old,"legacy")
    m1,p1,psf1,dphi1=_load_case(new,"batched")
    keys=(
        "input_fits","input_sav","focal_anisoplanatism_algorithm",
        "p3_pupil_rotation_hypothesis","p3_pupil_resampling",
        "p3_pupil_parity","focal_cache_configured",
    )
    changed=[k for k in keys if m0.get(k)!=m1.get(k)]
    if changed:
        raise ValueError(
            "Cannot compare different Keck science inputs or pupil settings: "
            +", ".join(changed)
        )
    # Never attribute cold vs warm cache speedup to HO batching.
    cache0=p0["focal_geometric_kernel"]
    cache1=p1["focal_geometric_kernel"]
    if cache0.get("cache_status")!=cache1.get("cache_status"):
        raise ValueError("Different focal-cache status biases the HO benchmark")
    if cache0.get("cache_key")!=cache1.get("cache_key"):
        raise ValueError("Different focal-cache key invalidates comparison")
    if p0.get("cprofile_enabled") or p1.get("cprofile_enabled"):
        raise ValueError("Disable cProfile for comparable wall times")
    science=_compare_array(psf0,psf1,name="Keck PSF",
                           rtol=rtol,atol=atol)
    ho=_compare_array(dphi0,dphi1,name="HO SF",rtol=rtol,atol=atol)
    stages0=p0["psfr_stage_timings_ms"]
    stages1=p1["psfr_stage_timings_ms"]
    old_ho=float(stages0["ho_sf"])/1000
    new_ho=float(stages1["ho_sf"])/1000
    old_total=float(stages0["psfr_init_total"])/1000
    new_total=float(stages1["psfr_init_total"])/1000
    return {
        "kind":"real_Keck_HO_Vii_fft_batching_vs_legacy",
        "parity_passed":science["parity_passed"] and ho["parity_passed"],
        "rtol":rtol,"atol":atol,
        "science_psf":science,"ho_structure_function":ho,
        "ho_fft_options":p1["ho_fft"],
        "focal_cache_status":cache0.get("cache_status"),
        "ho_sf_seconds":{"legacy":old_ho,"batched":new_ho},
        "ho_sf_speedup":old_ho/new_ho if new_ho>0 else None,
        "psfr_init_seconds":{"legacy":old_total,"batched":new_total},
        "psfr_init_speedup":old_total/new_total if new_total>0 else None,
        "warning":"One A/B pair measures one run; repeat before "
                  "claiming stable performance. HO/pupil Fourier transforms "
                  "are always recomputed per Keck observation.",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy",required=True,type=Path)
    ap.add_argument("--batched",required=True,type=Path)
    ap.add_argument("--output",type=Path,
                    default=Path("results/compare_keck_ho_fft.json"))
    a=ap.parse_args()
    result=compare(a.legacy,a.batched)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not result["parity_passed"]:
        raise SystemExit("FAIL: HO SF or Keck PSF numerical parity violated")


if __name__=="__main__":
    main()
