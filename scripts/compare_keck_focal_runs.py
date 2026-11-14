"""Compare two real Keck PSF-R outputs: legacy vs opt-in focal symmetry.

  uv run python scripts/compare_keck_focal_runs.py \
    --legacy results/focal_legacy/n0120 \
    --symmetry results/focal_symmetry/n0120

Outputs PSF relative L2/max-absolute error and measured init/aniso
times from metadata. This validates exact numerical non-regression; it
does NOT validate any new scientific TT/NCPA calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(legacy: Path, symmetry: Path, *, rtol=1e-12, atol=1e-12):
    def load(path,expected_algo):
        metadata=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        actual=metadata.get("focal_anisoplanatism_algorithm")
        if actual!=expected_algo:
            raise ValueError(
                f"{path}: algorithm must be {expected_algo!r}, found {actual!r}"
            )
        return np.load(path/"telemetry_only_psf.npy",allow_pickle=False),metadata
    a,meta_a=load(legacy,"legacy")
    b,meta_b=load(symmetry,"symmetry")
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Inconsistent or nonfinite Keck PSF images")
    dif=b-a
    parity=bool(np.allclose(a,b,rtol=rtol,atol=atol))
    baseline=float(meta_a["psfr_total_init_elapsed_s"])
    optimized=float(meta_b["psfr_total_init_elapsed_s"])
    focal_a=float(meta_a["frequency_anisoplanatism_elapsed_s"])
    focal_b=float(meta_b["frequency_anisoplanatism_elapsed_s"])
    return {
        "kind":"real_Keck_E2E_focal_covariance_non_regression",
        "legacy_case":str(legacy),
        "symmetry_case":str(symmetry),
        "shape":list(a.shape),
        "parity_passed":parity,
        "max_absolute_difference":float(np.max(np.abs(dif))),
        "relative_l2_difference":float(np.linalg.norm(dif)/np.linalg.norm(a)),
        "rtol":rtol,"atol":atol,
        "psfr_initialization_legacy_s":baseline,
        "psfr_initialization_symmetry_s":optimized,
        "psfr_initialization_speedup":baseline/optimized,
        "frequency_anisoplanatism_legacy_s":focal_a,
        "frequency_anisoplanatism_symmetry_s":focal_b,
        "frequency_anisoplanatism_speedup":focal_a/focal_b,
        "caution":"The nested anisoplanatism timer is within PSF-R initialization",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy",type=Path,required=True)
    parser.add_argument("--symmetry",type=Path,required=True)
    parser.add_argument("--output",type=Path,
                        default=Path("results/compare_keck_focal.json"))
    args=parser.parse_args()
    report=compare(args.legacy,args.symmetry)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["parity_passed"]:
        raise SystemExit("FAILED: candidate does not reproduce legacy Keck PSF")


if __name__=="__main__":
    main()
