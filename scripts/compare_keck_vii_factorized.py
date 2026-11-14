"""Compare actual Keck legacy vs algebraically factorized Vii HO kernels.

No tolerance relaxation. Report strict HO phase SF parity separately
from full spectral science PSF parity at atol=rtol=1e-12.
The accepted gate is the *final PSF* (user-approved), while a HO-SF
mismatch is retained as an explicit numerical-diagnostic warning.
Time results are meaningless for different FITS/SAV, pupil or cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(folder:Path,method:str):
    metadata=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    if metadata.get("ho_vii_method")!=method:
        raise ValueError(f"Expected {method!r} in {folder}/metadata.json")
    profile=json.loads((folder/"psfr_profile_breakdown.json").read_text(encoding="utf-8"))
    psf=np.load(folder/"telemetry_only_psf.npy",allow_pickle=False)
    dphi=np.load(folder/"ho_dphi_ao.npy",allow_pickle=False)
    return metadata,profile,psf,dphi


def _compare(a,b,*,rtol,atol):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite or differently shaped science products")
    err=b-a
    return {
        "shape":list(a.shape),
        "parity_passed":bool(np.allclose(a,b,rtol=rtol,atol=atol)),
        "max_absolute_difference":float(np.max(np.abs(err))),
        "relative_l2_difference":float(
            np.linalg.norm(err)/max(np.linalg.norm(a),1e-30)
        ),
    }


def compare(legacy:Path,factorized:Path,*,rtol=1e-12,atol=1e-12):
    m0,p0,psf0,dphi0=_read(Path(legacy),"legacy")
    m1,p1,psf1,dphi1=_read(Path(factorized),"factorized")
    fixed=(
        "input_fits","input_sav",
        "p3_pupil_rotation_hypothesis","p3_pupil_resampling",
        "p3_pupil_parity","focal_anisoplanatism_algorithm",
        "focal_cache_configured",
    )
    changed=[k for k in fixed if m0.get(k)!=m1.get(k)]
    if changed:
        raise ValueError("Different Keck inputs/pupil/focal settings: "+str(changed))
    if p0.get("cprofile_enabled") or p1.get("cprofile_enabled"):
        raise ValueError("Do not use cProfile for wall-time comparisons")
    c0=p0.get("focal_geometric_kernel",{})
    c1=p1.get("focal_geometric_kernel",{})
    if (c0.get("cache_status")!=c1.get("cache_status") or
            c0.get("cache_key")!=c1.get("cache_key")):
        raise ValueError("Cold-vs-warm geometric cache would bias Vii speedup")
    ho=_compare(dphi0,dphi1,rtol=rtol,atol=atol)
    psf=_compare(psf0,psf1,rtol=rtol,atol=atol)
    t0=float(p0["psfr_stage_timings_ms"]["ho_sf"])*1e-3
    t1=float(p1["psfr_stage_timings_ms"]["ho_sf"])*1e-3
    init0=float(p0["psfr_stage_timings_ms"]["psfr_init_total"])*1e-3
    init1=float(p1["psfr_stage_timings_ms"]["psfr_init_total"])*1e-3
    return {
        "kind":"real_Keck_Vii_factorization_vs_historical",
        "parity_passed":bool(ho["parity_passed"] and psf["parity_passed"]),
        "science_psf_accepted":bool(psf["parity_passed"]),
        "ho_sf_strict_parity_passed":bool(ho["parity_passed"]),
        "numerical_diagnostic_warning":(
            None if ho["parity_passed"] else
            "Intermediate HO SF exceeds strict 1e-12 tolerance even "
            "though science PSF is accepted; retain this warning and "
            "verify optical/stellar metrics on further observations."
        ),
        "rtol":rtol,"atol":atol,
        "ho_structure_function":ho,"science_psf":psf,
        "ho_vii_diagnostics":p1.get("ho_vii_kernel",{}),
        "focal_cache_status":c0.get("cache_status"),
        "ho_sf_seconds":{"legacy":t0,"factorized":t1},
        "ho_sf_speedup":t0/t1,
        "psfr_init_seconds":{"legacy":init0,"factorized":init1},
        "psfr_init_speedup":init0/init1,
        "caution":"One Keck observation: repeat and test another pupil rotation "
                  "before promotion; mathematical equality is not the same "
                  "as numerical equality.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy",type=Path,required=True)
    p.add_argument("--factorized",type=Path,required=True)
    p.add_argument("--output",type=Path,
                   default=Path("results/compare_keck_vii_factorized.json"))
    args=p.parse_args()
    result=compare(args.legacy,args.factorized)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result["science_psf_accepted"]:
        raise SystemExit("FAILED: science PSF parity (rtol=atol=1e-12)")
    if not result["ho_sf_strict_parity_passed"]:
        print("WARNING: intermediate HO SF 1e-12 parity failed; "
              "science PSF passed. This difference remains auditable.")


if __name__=="__main__":
    main()
