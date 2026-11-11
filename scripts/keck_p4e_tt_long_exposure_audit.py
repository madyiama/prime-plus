"""Keck P4-E: statistical TT command/centroid and noise-gain sensitivity audit.

Run from the repository root:
  uv run python scripts/keck_p4e_tt_long_exposure_audit.py \\
    --sav "/path/to/n0120_LGS_trs.sav" \\
    --output results/n0120_tt_statistics

The script never fits a scalar from the NIRC2 science PSF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import readsav

from p3.telemetry.ttLongExposureAudit import (
    audit_command_increments,van_dam_gain_sensitivity,
)


def _field(rec,name):
    names=rec.dtype.names
    if not names:
        raise ValueError("Expected structured Keck B record")
    index={str(n).upper():n for n in names}
    if name not in index:
        raise ValueError(f"B field {name} absent; available: {names}")
    return np.asarray(rec[index[name]])


def _plot(audit,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    controller=audit["command_increment_audit"]
    noise=audit["gain_sensitivity"]
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    lag=controller["lag_scan"]
    axs[0,0].plot([s["lag_frames"] for s in lag],
                [s["r2_free_test"] for s in lag],"o-",label="fitted 2×2")
    axs[0,0].plot([s["lag_frames"] for s in lag],
                [s["r2_expected_controller_test"] for s in lag],
                "s-",label="nominal 0.1×I")
    axs[0,0].set(xlabel="Command vs centroid lag [frames]",ylabel="Held-out R²",
               title="Command increments vs recorded STRAP")
    axs[0,0].legend()
    axs[0,0].grid(alpha=.25)
    trials=noise["trials"]
    bands=sorted(set(tuple(v["band_fraction_of_nyquist"]) for v in trials))
    for band in bands:
        sub=sorted([v for v in trials
                    if tuple(v["band_fraction_of_nyquist"])==band],
                   key=lambda v:v["assumed_effective_optical_gain"])
        gs=[v["assumed_effective_optical_gain"] for v in sub]
        for axis in range(2):
            noise_rms=[
                np.sqrt(max(0,v["sensor_noise_covariance_recorded_arcsec2"][axis][axis]))*1000
                for v in sub
            ]
            axs[1,0].plot(gs,noise_rms,"o-",
                          label=f"{band[0]:.2f}–{band[1]:.2f} Nyq, ax {axis}")
            variance_fraction=[
                v["predicted_filtered_noise_fraction_per_axis"][axis]
                for v in sub
            ]
            axs[1,1].plot(gs,variance_fraction,"o-",
                          label=f"{band[0]:.2f}–{band[1]:.2f} Nyq, ax {axis}")
            conditional_rms=[
                (v["conditional_jitter_band_sensitivity"]
                 ["residual_band_rms_axes_conditional_mas"] or [float("nan")]*2)[axis]
                for v in sub
            ]
            axs[0,1].plot(gs,conditional_rms,"o-",
                          label=f"{band[0]:.2f}–{band[1]:.2f} Nyq, ax {axis}")
    axs[0,1].set(xlabel="Assumed g_opt (NOT estimated)",
                 ylabel="Band-limited residual RMS [conditional mas]",
                 title="Hypothetical jitter (P=one-frame unity only)")
    axs[1,0].set(xlabel="Assumed g_opt (NOT estimated)",
               ylabel="Sensor noise RMS [recorded mas]",
               title="van Dam noise is gain-dependent")
    axs[1,1].set(xlabel="Assumed g_opt (NOT estimated)",
               ylabel="Filtered noise fraction of measured variance",
               title="Conditional loop model (not sky jitter)")
    for ax in (axs[0,1],axs[1,0],axs[1,1]):
        ax.grid(alpha=.25)
        ax.legend(fontsize="xx-small")
    fig.suptitle(
        "P4-E TT sensitivity — NOT a calibrated sky jitter / optical gain fit"
    )
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sav",type=Path,required=True)
    parser.add_argument("--output",type=Path,
                        default=Path("results/n0120_tt_statistics"))
    parser.add_argument("--max-lag",type=int,default=10)
    parser.add_argument("--nperseg",type=int,default=512)
    parser.add_argument("--no-plot",action="store_true")
    args=parser.parse_args()
    if not args.sav.exists():
        parser.error("SAV file not found: "+str(args.sav))
    sav=readsav(str(args.sav))
    if "b" not in sav:
        parser.error("No Keck STRAP B telemetry available")
    b=sav["b"][0]
    y=_field(b,"DTTCENTROIDS")
    u=_field(b,"DTTCOMMANDS")
    times=_field(b,"TIMESTAMP")
    servo=np.asarray(sav["dt_servo"]).ravel()
    cmd=audit_command_increments(
        y,u,times,servo,max_lag=args.max_lag)
    noise=van_dam_gain_sensitivity(
        y,times,servo,nperseg=args.nperseg)
    audit={
        "file":args.sav.name,
        "mode":"LGS_STRAP_command_redundancy_and_conditional_noise",
        "command_increment_audit":cmd,
        "gain_sensitivity":noise,
        "conclusion":{
            "optical_gain_estimated":False,
            "long_exposure_physical_tt_covariance_identified":False,
            "recommended_gain":None,
            "changes_to_psfr":False,
            "next_step":"Independent physical DTT command calibration or "
                        "HO-derived TT turbulence PSD prior before g fit"
        },
    }
    args.output.mkdir(parents=True,exist_ok=True)
    output=args.output/"tt_statistical_audit.json"
    output.write_text(json.dumps(audit,indent=2,allow_nan=False),
                      encoding="utf-8")
    best=cmd["best_alignment"]
    print(f"[Integrator] K(z) = {cmd['integrator_gain']:.6g}/(1-z^-1)")
    print(f"[Timing] fs={cmd['timebase']['sampling_hz_median']:.5f} Hz; "
          f"nonuniform steps={cmd['timebase']['nonuniform_interval_count']}")
    print(f"[Increment alignment] lag={best['lag_frames']} frames "
          f"R²_free_test={best['r2_free_test']:.6f} "
          f"R²_nominal_test={best['r2_expected_controller_test']:.6f}")
    print(f"[Increment relation] nominal controller deviation "
          f"{best['relative_matrix_deviation_from_expected']:.4g}; "
          f"normalized nominal RMS test error "
          f"{best['relative_expected_controller_rms_error_test']:.4g}")
    print("[Gain sensitivity] No optical gain was fitted or selected")
    for trial in noise["trials"]:
        if trial["band_fraction_of_nyquist"]==[0.65,0.95]:
            std=np.sqrt(np.maximum(0,np.diag(
                trial["sensor_noise_covariance_recorded_arcsec2"]
            )))*1000
            cond=trial["conditional_jitter_band_sensitivity"]
            conditional=cond["residual_band_rms_axes_conditional_mas"]
            print(f"  assumed g={trial['assumed_effective_optical_gain']:.2f}, "
                  f"van Dam sensor RMS in RECORDED mas={std.round(3)}, "
                  f"conditional band jitter RMS mas={conditional}, "
                  f"band={cond['frequency_interval_hz']}, "
                  f"warnings={len(trial['warnings'])}")
    if not args.no_plot:
        _plot(audit,args.output/"tt_statistical_audit.png")
    print("Saved:",output)


if __name__=="__main__":
    main()
