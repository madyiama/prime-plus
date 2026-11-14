#!/usr/bin/env python3
"""Keck/NIRC2 PSF-R V1 reproducible end-to-end validation on one/many acquisitions.

The script executes the *existing* telemetry-only PSF-R implementation with
its locked provisional V1 recipe, then validates essential artifact invariants.
It does not fit the science PSF or identify optical gain, STRAP APD excess
noise, or physical TT mirror calibration. The anisokinetic term is left to
P3's EXISTING dani_tt: no new TT term is added.

Example:
  uv run python scripts/keck_psfr_v1_e2e.py \
    --fits /path/n0120.fits /path/n0013.fits \
    --sav /path/n0120_LGS_trs.sav /path/n0013_LGS_trs.sav \
    --calib p3/aoSystem/data/KECK_CALIBRATION \
    --output results/psfr_v1_acceptance
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=ROOT/"config"/"psfr_keck_v1.toml"
SMOKE=ROOT/"scripts"/"keck_n0013_smoke.py"

NOMINAL = {
    "science_mode":"Keck_LGS_NIRC2",
    "n_layer":7,
    "pupil_resampling":"physical_binary",
    "pupil_rotation":"header",
    "pupil_parity":"native",
    "tt_profile":"reduced_jitter",
    "tt_kernel_convention":"noll_angular",
    "tt_axis_parity":"native",
    "ho_covariance_policy":"strict",
    "r0_method":"zernike",
    "ncpa_enabled":False,
    "science_psf_fitting":False,
    "tt_anisokinetism":"p3_dani_tt_only",
    "legacy_tt_kernel_divide_by_16":False,
    "default_science_jitter_calibrated":False,
}


def load_nominal(path):
    """Fail closed on unreviewed TOML changes or silently different presets."""
    from p3.telemetry.ttSelection import resolve_tt_selection,PROFILES

    with Path(path).open("rb") as f:
        config=tomllib.load(f)
    actual=config.get("psfr_v1",{})
    if actual!=NOMINAL:
        diff={k:{"expected":v,"actual":actual.get(k)}
              for k,v in NOMINAL.items() if actual.get(k)!=v}
        missing=set(actual)-set(NOMINAL)
        raise ValueError(
            "PSF-R V1 nominal recipe has changed without updating validation: "
            +json.dumps({"mismatch":diff,"unexpected_fields":sorted(missing)})
        )
    selection=resolve_tt_selection(config,profile=actual["tt_profile"])
    config_selection=resolve_tt_selection(config)
    required=PROFILES["reduced_jitter"]
    for key,expected in required.items():
        if getattr(selection,key)!=expected or getattr(config_selection,key)!=expected:
            raise ValueError("TT V1 profile/TOML mismatch at "+key)
    if config_selection.report()["instrument_calibrated"]:
        raise ValueError("The Keck V1 instrument gain is NOT calibrated")
    return config


def validate_run(directory):
    """Hard numerical checks; no PSF fit or empirical Strehl acceptance."""
    d=Path(directory)
    needed=[
        "telemetry_only_psf.npy","p4_otf_tt.npy",
        "metadata.json","tt_selection.json",
        "p4_tt_angular_chain.json","tiptilt_diagnostics.json",
        "anisoplanatism_diagnostics.json",
        "ho_covariance_diagnostics.json",
        "comparison_nirc2_psfr.png",
    ]
    missing=[name for name in needed if not (d/name).is_file()]
    if missing:
        raise ValueError("Missing V1 PSF-R artifacts: "+", ".join(missing))
    def read(name):
        with (d/name).open("r",encoding="utf-8") as f:return json.load(f)
    meta=read("metadata.json")
    sel=read("tt_selection.json")
    angular=read("p4_tt_angular_chain.json")
    tt=read("tiptilt_diagnostics.json")
    if meta.get("ao_mode")!="LGS":
        raise ValueError("V1 gate is for Keck LGS STRAP; refuse NGS")
    expected={
        "p3_pupil_resampling":"physical_binary",
        "p3_pupil_parity":"native",
        "p3_pupil_rotation_hypothesis":"header",
        "tt_kernel_convention":"noll_angular",
        "ho_covariance_policy":"strict",
    }
    for k,v in expected.items():
        if meta.get(k)!=v:raise ValueError(f"Unexpected {k}: {meta.get(k)!r}")
    if meta.get("ho_covariance_projection_applied") is not False:
        raise ValueError("HO covariance was repaired instead of validated")
    if not meta.get("tt_kernel_is_physical_noll"):
        raise ValueError("Rejected historical /16 TT OTF kernel")
    if sel.get("profile")!="reduced_jitter":
        raise ValueError("The nominal STRAP profile was not used")
    if (sel.get("noise_method")!="van_dam"
        or sel.get("covariance_method")!="legacy_psfr"
        or sel.get("mirror_model")!="unity"):
        raise ValueError("STRAP estimator or model differs from V1 nominal")
    if angular.get("tt_kernel_convention")!="noll_angular":
        raise ValueError("Incorrect angular OTf convention")
    if angular.get("status")!="numerical_chain_verified_instrument_uncalibrated":
        raise ValueError("TT angular chain is not numerically verified")
    if not angular.get("tt_source_angle_replay",{}).get(
            "tt_centroid_to_otf_units_verified"):
        raise ValueError("DTTCENTROIDS -> angular OTF invariant failed")
    if tt.get("jitter_covariance_method")!="legacy_psfr":
        raise ValueError("TT covariance model mismatch")
    if angular.get("sensor_gain_independently_calibrated") is not False:
        raise ValueError("Misleading STRAP optical gain calibration status")
    psf=np.load(d/"telemetry_only_psf.npy",allow_pickle=False)
    otf=np.load(d/"p4_otf_tt.npy",allow_pickle=False)
    if (psf.ndim not in (2,3) or psf.shape[0]<16 or
        psf.shape[1]!=psf.shape[0] or
        not np.isfinite(psf).all() or psf.min()<-1e-12 or psf.sum()<=0):
        raise ValueError("Nonphysical or empty PSF-R forward prediction")
    if (otf.ndim!=2 or otf.shape[0]!=otf.shape[1]
        or not np.isfinite(otf).all() or otf.min()<-1e-12
        or otf.max()>1+1e-12 or
        abs(otf[otf.shape[0]//2,otf.shape[1]//2]-1)>1e-12):
        raise ValueError("Nonphysical TT OTF kernel")
    return {
        "status":"passed_numerical_artifact_gates",
        "output":str(d.resolve()),
        "psf_shape":list(psf.shape),
        "psf_flux":float(psf.sum()),
        "psf_peak":float(psf.max()),
        "tt_otf_min":float(otf.min()),
        "tt_otf_centre":float(otf[otf.shape[0]//2,otf.shape[1]//2]),
        "tt_angular_unit_chain_verified":True,
        "nominal_method":"van_dam+legacy_psfr+unity+noll_angular",
        "pupil":"physical_binary/header/native",
        "anisokinetism":"existing_P3_dani_tt_only",
        "independent_instrument_calibration":False,
        "science_image_fit_performed":False,
        "telemetry_metadata_git":meta.get("code_revision"),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits",type=Path,nargs="+",required=True)
    ap.add_argument("--sav",type=Path,nargs="+",required=True)
    ap.add_argument("--calib",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    ap.add_argument("--output",type=Path,
                    default=Path("results/psfr_v1_acceptance"))
    ap.add_argument("--focal-workers",type=int,choices=(1,2,4),default=1,
                    help="CPU Ialpha workers, opt-in with quadrant only")
    ap.add_argument("--focal-algorithm",choices=("legacy", "symmetry", "quadrant"),
                    default="legacy",
                    help="Experimental focal covariance accelerator; default unchanged")
    ap.add_argument("--focal-cache-dir",type=Path,default=None,
                    help="Disk cache ONLY for unfiltered symmetric focal SF; "
                         "rotation-dependent pupil and OTF are never cached")
    ap.add_argument("--ho-vii-method",choices=("legacy","factorized","factorized_real"),
                    default="legacy",
                    help="Opt-in Vii full-FFT or one-sided real-FFT factorization")
    ap.add_argument("--profile-psfr",action="store_true",
                    help="Optional PSF-R cProfile; timing overhead is expected")
    ap.add_argument("--export-ao-basis",action="store_true",
                    help="Export frozen AO phase SF + optical kernels and validate nominal parity")
    ap.add_argument("--check-only",action="store_true",
                    help="Check existing case outputs; do not run P3")
    args=ap.parse_args()
    if args.focal_workers != 1 and args.focal_algorithm != "quadrant":
        ap.error("--focal-workers > 1 requires --focal-algorithm quadrant")
    if args.focal_cache_dir is not None and args.focal_algorithm not in ("symmetry", "quadrant"):
        ap.error("--focal-cache-dir requires --focal-algorithm symmetry or quadrant")
    if len(args.fits)!=len(args.sav):
        ap.error("Each FITS requires a matching SAV at the same argument index")
    if len({p.stem for p in args.fits})!=len(args.fits):
        ap.error("Distinct FITS basenames required to avoid overwriting outputs")
    load_nominal(args.config)
    if not args.check_only:
        for path in (*args.fits,*args.sav,args.calib):
            if not path.exists():ap.error("Missing input: "+str(path))
    args.output.mkdir(parents=True,exist_ok=True)
    report={
        "config":str(args.config.resolve()),
        "source_branch_requirement":"merged_P4_A_P4_C_core",
        "science_validation":"not_a_calibrated_Keck_2020_model",
        "method":"van_dam+legacy_psfr+unity+noll_angular",
        "focal_anisoplanatism_algorithm":args.focal_algorithm,
        "focal_workers":args.focal_workers,
        "focal_cache_configured":bool(args.focal_cache_dir is not None),
        "cprofile_enabled":bool(args.profile_psfr),
        "ho_vii_method":args.ho_vii_method,
        "cases":[],
    }
    failures=0
    for fits,sav in zip(args.fits,args.sav):
        destination=args.output/fits.stem
        entry={"fits":str(fits),"sav":str(sav)}
        if not args.check_only:
            destination.mkdir(parents=True,exist_ok=True)
            cmd=[
                sys.executable,str(SMOKE),
                "--fits",str(fits.resolve()),
                "--sav",str(sav.resolve()),
                "--calib",str(args.calib.resolve()),
                "--config",str(args.config.resolve()),
                "--output",str(destination.resolve()),
                "--n-layer","7",
                "--p3-pupil-resampling","physical_binary",
                "--p3-pupil-rotation","header",
                "--p3-pupil-parity","native",
                "--tt-profile","reduced_jitter",
                "--tt-kernel-convention","noll_angular",
                "--tt-axis-parity","native",
                "--ho-covariance-policy","strict",
            ]
            if args.focal_cache_dir is not None:
                cmd.extend(("--focal-cache-dir",
                            str(args.focal_cache_dir.resolve())))
            if args.profile_psfr:
                cmd.append("--profile-psfr")
            if args.ho_vii_method!="legacy":
                cmd.extend(("--ho-vii-method",args.ho_vii_method))
            if args.export_ao_basis:
                cmd.append("--export-ao-basis")
            if args.focal_algorithm != "legacy":
                cmd.extend(("--focal-algorithm",args.focal_algorithm))
            if args.focal_workers != 1:
                cmd.extend(("--focal-workers",str(args.focal_workers)))
            entry["command"]=cmd
            print("[PSFR V1] Reconstructing",fits.name,flush=True)
            with (destination/"psfr_v1_run.log").open("w",encoding="utf-8") as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                      check=False)
            entry["process_returncode"]=result.returncode
            if result.returncode:
                failures+=1
                entry["status"]="reconstruction_failed"
                entry["log"]=str(destination/"psfr_v1_run.log")
                report["cases"].append(entry)
                print("[PSFR V1] FAIL:",entry["log"],
                      "returncode=",result.returncode,flush=True)
                # Surface the traceback: without this, the parent merely
                # prints FAIL and conceals the underlying Keck exception.
                # Preserve the entire diagnostic in psfr_v1_run.log.
                try:
                    tail=(destination/"psfr_v1_run.log").read_text(
                        encoding="utf-8",errors="replace"
                    ).splitlines()[-65:]
                    print("[PSFR V1] ---- last 65 log lines ----",flush=True)
                    print("\n".join(tail),flush=True)
                    print("[PSFR V1] ---- end log ----",flush=True)
                    entry["log_tail"]=tail
                except OSError as err:
                    print("[PSFR V1] log unreadable:",err,flush=True)
                continue
        try:
            entry.update(validate_run(destination))
            print("[PSFR V1] PASS:",fits.stem,flush=True)
        except Exception as exc:
            failures+=1
            entry["status"]="artifact_gate_failed"
            entry["error"]=str(exc)
            print("[PSFR V1] FAIL:",fits.stem,exc,flush=True)
        report["cases"].append(entry)
    report["summary"]={
        "passed":sum(x.get("status")=="passed_numerical_artifact_gates"
                     for x in report["cases"]),
        "failed":failures,
        "total":len(report["cases"]),
        "framework_nominal_only":True,
        "scientific_absolute_calibration_proven":False,
    }
    out=args.output/"psfr_v1_acceptance.json"
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print("[PSFR V1] Report:",out,flush=True)
    return 1 if failures else 0


if __name__=="__main__":
    raise SystemExit(main())
