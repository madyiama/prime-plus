"""P4-F real-data controlled Keck PSF-R A/B: centroids vs DTT increments.

Runs two complete smoke reconstructions from the SAME FITS/SAV/Keck
calibration and the SAME noise/covariance recipe:
  A: B.DTTCENTROIDS[1:] (aligned baseline)
  B: diff(B.DTTCOMMANDS)/DT_SERVO[0] (increment replay)

Outputs two full diagnostics folders, a JSON invariance report and
a comparison PNG. This is a computational identity/integration test,
NOT an independent TT optical gain measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_tt_psfr_outputs(aligned_dir,replayed_dir, *,output,max_rel_psf=1e-4,
                             make_figure=True):
    """Compare end-to-end PSF, TT OTF, noise and archived samples.

    Reference and replay MUST have dropped the same first input frame
    and used the same physical Keck pipeline.
    """
    a=Path(aligned_dir);b=Path(replayed_dir);out=Path(output)
    ma=_load_json(a/"metadata.json")
    mb=_load_json(b/"metadata.json")
    ra=_load_json(a/"tt_command_replay.json")
    rb=_load_json(b/"tt_command_replay.json")
    if (ma.get("tt_slope_source")!="centroids_aligned"
        or mb.get("tt_slope_source")!="command_increment"):
        raise ValueError("A/B must compare centroids_aligned to command_increment")
    for key in ("input_fits","input_sav","tt_noise_method",
                "tt_jitter_covariance_method","p3_pupil_resampling",
                "tt_kernel_convention","psf_reference_wavelength_m"):
        if ma.get(key)!=mb.get(key):
            raise ValueError(f"A/B mismatch in metadata {key}")
    if not ra["first_frame_removed"] or not rb["first_frame_removed"]:
        raise ValueError("A/B sample index windows differ")
    pa=np.asarray(np.load(a/"telemetry_only_psf.npy"),dtype=float)
    pb=np.asarray(np.load(b/"telemetry_only_psf.npy"),dtype=float)
    oa=np.asarray(np.load(a/"p4_otf_tt.npy"),dtype=float)
    ob=np.asarray(np.load(b/"p4_otf_tt.npy"),dtype=float)
    if pa.shape!=pb.shape or oa.shape!=ob.shape:
        raise ValueError("A/B model array shapes differ")
    na=_load_json(a/"tt_noise_diagnostics.json")
    nb=_load_json(b/"tt_noise_diagnostics.json")
    ca=np.asarray(na["noise_covariance_m2"],dtype=float)
    cb=np.asarray(nb["noise_covariance_m2"],dtype=float)
    tt_a=_load_json(a/"tiptilt_diagnostics.json")
    tt_b=_load_json(b/"tiptilt_diagnostics.json")
    pa2=pa.sum(axis=-1) if pa.ndim==3 else pa
    pb2=pb.sum(axis=-1) if pb.ndim==3 else pb
    pa2=pa2/pa2.sum()
    pb2=pb2/pb2.sum()
    psfrel=float(np.linalg.norm(pa2-pb2)/max(np.linalg.norm(pa2),1e-30))
    otfrel=float(np.linalg.norm(oa-ob)/max(np.linalg.norm(oa),1e-30))
    noise_rel=float(np.linalg.norm(ca-cb)/max(np.linalg.norm(ca),1e-30))
    report={
        "label":"P4-F Keck real PSF-R centroid/command-increment A/B",
        "centroids_aligned_output":str(a.resolve()),
        "command_increment_output":str(b.resolve()),
        "same_fits_and_sav":True,
        "source_sample_alignment":"N-1; same archived Keck B frames",
        "source_nominal_arcsec_rms_mismatch":rb["relative_rms_error_vs_centroids"],
        "source_covariance_relative_frobenius_error":
            rb["covariance_relative_frobenius_error"],
        "tt_noise_covariance_relative_frobenius_error":noise_rel,
        "tt_otf_relative_l2_difference":otfrel,
        "psf_relative_l2_difference":psfrel,
        "psf_max_abs_difference_after_unit_flux":
            float(np.max(np.abs(pb2-pa2))),
        "psf_peak_centroids_aligned":float(pa2.max()),
        "psf_peak_command_increment":float(pb2.max()),
        "psf_peak_relative_change":float((pb2.max()-pa2.max())/pa2.max()),
        "tt_diagnostics_centroids":tt_a,
        "tt_diagnostics_command_increment":tt_b,
        "psf_arrays_close_at_requested_tolerance":
            bool(psfrel<=max_rel_psf),
        "expected_outcome_is_equivalent_psfs":True,
        "no_independent_optical_gain_calibration":True,
        "note":(
            "If A/B matches, replacing centroids with command increments "
            "has NOT reduced physical jitter; both are the same controller "
            "observable. If it differs, investigate a numerical pipeline "
            "inconsistency before declaring a physical improvement."
        ),
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"tt_source_ab_comparison.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8",
    )
    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
        vmax=max(pa2.max(),pb2.max())
        eps=max(vmax*1e-7,1e-300)
        for ax,im,title in [
            (axs[0],pa2,"A — STRAP centroids [1:]"),
            (axs[1],pb2,"B — ΔDTT command / 0.1"),
        ]:
            ax.imshow(np.log10(np.maximum(im,eps)),origin="lower",
                      vmin=np.log10(eps),vmax=np.log10(vmax),cmap="magma")
            ax.set_title(title)
            ax.set_xlabel("x pixel")
            ax.set_ylabel("y pixel")
        diff=pb2-pa2
        dmax=max(float(np.max(np.abs(diff))),1e-15)
        axes=axs[2]
        heat=axes.imshow(diff,origin="lower",vmin=-dmax,
                         vmax=dmax,cmap="RdBu_r")
        axes.set_title("B − A (equal unit flux)")
        axes.set_xlabel("x pixel")
        fig.colorbar(heat,ax=axes,label="PSF unit-flux difference",shrink=.7)
        fig.suptitle(
            "P4-F Keck TT replay — numerical consistency, not sky gain\n"
            f"PSF rel L2={psfrel:.3e}; TT OTF rel L2={otfrel:.3e}; "
            f"noise covariance rel={noise_rel:.3e}"
        )
        fig.savefig(out/"tt_source_ab_comparison.png",dpi=155)
        plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits",required=True,type=Path)
    parser.add_argument("--sav",required=True,type=Path)
    parser.add_argument("--calib",required=True,type=Path)
    parser.add_argument("--output",type=Path,
                        default=Path("results/n0120_tt_source_ab"))
    parser.add_argument("--n-layer",type=int,choices=(1,3,7),default=7)
    parser.add_argument("--tt-profile",
                        choices=("reduced_jitter","gendron_comparison",
                                 "spectral_comparison"),
                        default="reduced_jitter")
    parser.add_argument("--p3-pupil-resampling",
                        choices=("legacy","physical_binary"),
                        default="physical_binary")
    parser.add_argument("--compare-only",action="store_true",
                        help="Compare existing output folders without rerunning")
    parser.add_argument("--max-relative-psf-diff",type=float,default=1e-4)
    args,extra=parser.parse_known_args()
    if args.max_relative_psf_diff<=0 or not np.isfinite(args.max_relative_psf_diff):
        parser.error("max-relative-psf-diff must be finite and positive")
    for prefix in ("--tt-slope-source","--output","--fits","--sav","--calib"):
        if any(x==prefix or x.startswith(prefix+"=") for x in extra):
            parser.error(prefix+" is managed by the A/B harness")
    a=args.output/"centroids_aligned"
    b=args.output/"command_increment"
    script=Path(__file__).with_name("keck_n0013_smoke.py")
    if not args.compare_only:
        for source,dest in (("centroids_aligned",a),
                            ("command_increment",b)):
            cmd=[
                sys.executable,str(script),
                "--fits",str(args.fits),
                "--sav",str(args.sav),
                "--calib",str(args.calib),
                "--output",str(dest),
                "--n-layer",str(args.n_layer),
                "--tt-profile",args.tt_profile,
                "--p3-pupil-resampling",args.p3_pupil_resampling,
                "--tt-slope-source",source,*extra,
            ]
            print(f"[P4-F] Full PSF-R run: {source}",flush=True)
            subprocess.run(cmd,check=True)
    result=compare_tt_psfr_outputs(
        a,b,output=args.output,
        max_rel_psf=args.max_relative_psf_diff,
    )
    print("[P4-F] TT input RMS mismatch:",
          f"{result['source_nominal_arcsec_rms_mismatch']:.3e}")
    print("[P4-F] Noise covariance relative difference:",
          f"{result['tt_noise_covariance_relative_frobenius_error']:.3e}")
    print("[P4-F] TT OTF relative L2 difference:",
          f"{result['tt_otf_relative_l2_difference']:.3e}")
    print("[P4-F] PSF relative L2 difference:",
          f"{result['psf_relative_l2_difference']:.3e}")
    print("[P4-F] PSF equivalent within tolerance:",
          result["psf_arrays_close_at_requested_tolerance"])
    print("[P4-F] Report:",args.output/"tt_source_ab_comparison.json")
    print("[P4-F] Figure:",args.output/"tt_source_ab_comparison.png")
    if not result["psf_arrays_close_at_requested_tolerance"]:
        raise SystemExit(
            "TT A/B differs unexpectedly: investigate numerical pipeline "
            "and inputs; never infer an improved physical jitter from this"
        )


if __name__=="__main__":
    main()
