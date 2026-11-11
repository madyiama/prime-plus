"""P4-D: independently audit Keck DTT mechanical command/strain-gauge response.

Run:
  uv run python scripts/keck_p4d_dtt_mechanical_audit.py \\
    --sav "/path/to/n0120_LGS_trs.sav" --output results/n0120_dtt_mechanical

Outputs JSON and a diagnostic PNG. Does not reconstruct the PSF or
infer an absolute STRAP optical gain from correlated closed-loop data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import readsav

from p3.telemetry.ttMechanicalAudit import audit_dtt_mechanics


def _field(record, name):
    """IDL struct field names are case-insensitive across readsav versions."""
    names=record.dtype.names
    if names is None:
        raise ValueError("B telemetry is not an IDL record")
    lookup={str(k).upper():k for k in names}
    if name not in lookup:
        raise ValueError(f"B telemetry missing {name}; fields: {names}")
    return np.asarray(record[lookup[name]])


def _make_figure(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for ax,key,title in (
        (axes[0,0],"level_delay_scan","Levels: command → differential gauges"),
        (axes[0,1],"difference_delay_scan","Differences: fast command/gauge changes"),
    ):
        vals=[x for x in report[key] if x["valid"]]
        ax.plot([x["lag_frames"] for x in vals],
                [x["r2_train"] for x in vals],marker="o",label="train")
        ax.plot([x["lag_frames"] for x in vals],
                [x["r2_test"] for x in vals],marker="s",label="held out")
        ax.axvline(0,color="0.7",ls="--")
        ax.set_title(title)
        ax.set_xlabel("Lag frames (positive = gauge follows command)")
        ax.set_ylabel("R² (gauge units)")
        ax.legend()
        ax.grid(alpha=.3)
    spectral=report["spectral_response"]
    if spectral["status"]=="ok":
        for info in spectral["axes"]:
            f=np.asarray(info["frequency_hz"])
            axes[1,0].plot(f,info["coherence"],label=f"axis {info['axis']}")
            axes[1,1].semilogy(
                f,np.maximum(info["transfer_gain"],1e-10),
                label=f"axis {info['axis']}"
            )
        axes[1,0].set_ylim(0,1.05)
        axes[1,0].set_ylabel("Coherence (estimated)")
        axes[1,1].set_ylabel("Gauge command-equivalent / command")
        for ax in (axes[1,0],axes[1,1]):
            ax.set_xlim(0,150)
            ax.set_xlabel("Hz")
            ax.legend()
            ax.grid(alpha=.3)
    else:
        for ax in (axes[1,0],axes[1,1]):
            ax.text(.5,.5,
                    "No reliable segmented Welch spectrum:\\n"+
                    spectral["status"],ha="center",va="center",
                    transform=ax.transAxes)
    axes[1,0].set_title("Frequency coherence")
    axes[1,1].set_title("Frequency response magnitude")
    fig.suptitle(
        "P4-D Keck DTT: relative mechanical audit (NOT sky jitter calibration)\\n"
        f"Timestamp-gap count: {report['timing']['nonuniform_interval_count']}"
    )
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sav",type=Path,required=True)
    parser.add_argument(
        "--output",type=Path,default=Path("results/n0120_dtt_mechanical")
    )
    parser.add_argument("--max-lag",type=int,default=10)
    parser.add_argument("--nperseg",type=int,default=1024)
    parser.add_argument("--no-plot",action="store_true")
    args=parser.parse_args()
    if args.max_lag<1 or args.max_lag>100:
        parser.error("max-lag must be between 1 and 100 frames")
    if args.nperseg<128 or args.nperseg>8192:
        parser.error("nperseg must be between 128 and 8192")
    if not args.sav.is_file():
        parser.error("SAV file not found: "+str(args.sav))
    data=readsav(str(args.sav))
    if "b" not in data:
        parser.error("SAV missing STRAP B record")
    record=data["b"][0]
    report=audit_dtt_mechanics(
        _field(record,"DTTCOMMANDS"),
        _field(record,"DTTSTRAINGAUGE"),
        _field(record,"TIMESTAMP"),
        _field(record,"CONF_ID"),
        max_lag=args.max_lag,
        nperseg=args.nperseg,
    )
    clipped=None
    if "DTTCOMMANDSCLIPPED" in {str(x).upper() for x in record.dtype.names}:
        flags=_field(record,"DTTCOMMANDSCLIPPED").reshape(-1)
        clipped={
            "frame_count":int(len(flags)),
            "nonzero_count":int(np.count_nonzero(flags)),
            "no_clipping_reported":not bool(np.any(flags)),
        }
    report.update({
        "sav_filename":args.sav.name,
        "commands_clipped":clipped,
        "no_science_image_fitting":True,
        "raw_dt_centroids_rescaled":False,
        "no_optical_gain_or_mirror_model_fit":True,
    })
    args.output.mkdir(parents=True,exist_ok=True)
    dest=args.output/"dtt_mechanical_audit.json"
    dest.write_text(json.dumps(report,indent=2,allow_nan=False),
                    encoding="utf-8")
    t=report["timing"]
    lv=report["best_level_delay_train"]
    diff=report["best_difference_delay_train"]
    print(f"[Keck DTT] frames={report['frame_count']}, "
          f"duration={report['duration_seconds']:.2f} s")
    print(f"[RTC cadence] median={t['sampling_hz_median']:.4f} Hz, "
          f"mean={t['sampling_hz_mean']:.4f} Hz, "
          f"nonuniform intervals={t['nonuniform_interval_count']}")
    print(f"[Levels] lag={lv['lag_frames']} frames, "
          f"R² train/test={lv['r2_train']:.5f}/{lv['r2_test']:.5f}")
    print(f"[Differences] lag={diff['lag_frames']} frames, "
          f"R² train/test={diff['r2_train']:.5f}/{diff['r2_test']:.5f}")
    print("[Configurations] fit",
          report["configuration_stability"]["fit_segment_count"],
          "of",len(report["configuration_stability"]["segments"]))
    spec=report["spectral_response"]
    if spec["status"]=="ok":
        print(f"[Spectral] Welch={spec['nperseg_effective']}, "
              f"windows={spec['total_independent_welch_windows']}, "
              f"scope={spec['segment_role']}")
        for axis in spec["axes"]:
            print(f"  axis {axis['axis']}: phase delay "
                  f"{axis['phase_delay_fit_ms']} ms, fit "
                  f"{axis['phase_fit_bins']} bins")
    else:
        print("[Spectral]",spec["status"])
    for item in report["warnings"]:
        print("[Caution]",item)
    if not args.no_plot:
        _make_figure(report,args.output/"dtt_mechanical_audit.png")
    print("Saved:",dest)


if __name__=="__main__":
    main()
