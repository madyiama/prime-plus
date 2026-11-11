#!/usr/bin/env python3
"""P4-B Keck STRAP audit: three independent checks, one LGS FITS/SAV.

1. Prespecified van Dam high-frequency bands and Welch segment lengths.
2. Fixed contiguous APD-flux windows, with a conditional Poisson baseline.
3. 2020-uncalibrated W/K/P/S plant alternatives with frozen vs re-fit noise.

No science-image fitting, no noise-scale optimization, no covariance clipping.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import readsav

from p3.telemetry.ttP4BAudit import (
    BANDS, MIRRORS, WELCH_SIZES, sensitivity_audit, flux_segment_audit,
    initialize_keck_loop_transfers,
)
from p3.telemetry.ttApdNoise import reproduce_keck_centroids


def _revision():
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "p3", "scripts", "tests"],
            cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"git_revision": sha, "tracked_or_untracked_source_edits": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "tracked_or_untracked_source_edits": None}


def _figure(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["van_dam_and_loop_sensitivity"]["rows"]
    segments = result["apd_flux_segments"]["blocks"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for mirror in MIRRORS:
        x = [r for r in rows if r["mirror"] == mirror]
        for band in BANDS:
            y = [r for r in x if r["band_fraction_nyquist"] == list(band)]
            y.sort(key=lambda r: r["welch_nperseg"])
            ax[0].plot([r["welch_nperseg"] for r in y],
                       [r["sensor_noise"]["trace_arcsec2"]*1e6 for r in y],
                       "o-", label=f"{mirror}, band {band}")
    ax[0].set(title="van Dam robustness (no science fit)",
              xlabel="Welch nperseg", ylabel="noise trace (mas²)")
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.2)
    ax[1].plot(
        [x["mean_net_counts"] for x in segments],
        [x["van_dam_noise"]["trace_arcsec2"]*1e6 for x in segments],
        "o", label="van Dam per fixed time window"
    )
    ax[1].plot(
        [x["mean_net_counts"] for x in segments],
        [x["predicted_ideal_apd_noise"]["trace_arcsec2"]*1e6 for x in segments],
        "s", label="conditional ideal APD Poisson"
    )
    ax[1].set(xlabel="mean background-subtracted APD count/frame",
              ylabel="sensor noise trace (mas²)",
              title="Contiguous windows (gain uncalibrated)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.2)
    for mirror in MIRRORS:
        ref = next(
            r for r in rows if r["mirror"] == mirror
            and r["band_fraction_nyquist"] == [0.65,0.95]
            and r["welch_nperseg"] == 512
        )
        phys = ref["spectral_physical"]
        if phys["status"] == "computed_unvalidated":
            ax[2].bar(
                mirror.replace("ragland_2016_keck_ii", "Ragland 2016"),
                phys["total"]["trace_arcsec2"]*1e6,
                label="Physical TT total (hypothesis)")
    ax[2].set(title="Plant hypothesis and physical TT",
              ylabel="physical covariance trace (mas²)")
    ax[2].tick_params(axis="x", labelrotation=15)
    ax[2].grid(alpha=0.2,axis="y")
    fig.suptitle(
        "P4-B — Keck STRAP detector and closed-loop sensitivity; "
        "NO model-selection from NIRC2 image"
    )
    fig.savefig(path, dpi=155)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits", required=True, type=Path)
    p.add_argument("--sav", required=True, type=Path)
    p.add_argument("--calib", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--segment-length", type=int, default=512,
                   help="Contiguous, nonoverlapping APD windows, preset before seeing flux")
    p.add_argument("--segment-welch", type=int, default=256,
                   help="Must admit two Welch windows per APD segment")
    args = p.parse_args()
    for key in ("fits", "sav", "calib"):
        if not getattr(args, key).exists():
            p.error(f"Missing {key}: {getattr(args,key)}")
    args.output.mkdir(parents=True,exist_ok=True)

    archive = readsav(str(args.sav))
    if "b" not in archive:
        p.error("P4-B requires Keck LGS STRAP B telemetry with four APDs")
    b = archive["b"][0]
    raw = np.asarray(b["APDCOUNTS"],dtype=np.float64)
    centroids = np.asarray(b["DTTCENTROIDS"],dtype=np.float64)
    sky = np.asarray(archive["apd_sky_back"],dtype=np.float64)
    imat = np.asarray(archive["apd_imat"],dtype=np.float64)
    independent = reproduce_keck_centroids(raw,sky,imat,centroids)
    print("[P4-B] Keck APD centroid transform verified", flush=True)

    # This loads FITS/header metadata and historical Keck RTC calibration,
    # never inspects science image intensities as a fit target.
    from p3.telemetry.telemetryKeck import telemetryKeck
    trs = telemetryKeck(
        str(args.sav.resolve()),str(args.fits.resolve()),
        str(args.calib.resolve()),
        path_save=str(args.output.resolve()),
        cam_fov=128,nLayer=1,addNCPA=False,
    )
    if trs.aoMode.upper() != "LGS":
        raise ValueError("Expected LGS STRAP telemetry")
    if (trs.tipTilt.slopes_arcsec.shape != centroids.shape or
            not np.array_equal(trs.tipTilt.slopes_arcsec,centroids)):
        raise ValueError("Keck telemetry loader does not preserve aligned raw centroids")
    # telemetryKeck loads RTC controller coefficients, sample frequencies
    # and delays, but tf.freq/wfs/servo/lag/rtf are set ONLY by P3's
    # systemDiagnosis.temporal_transfer_function(). Initialize precisely
    # that stage, without running atmospheric r0/L0 or HO diagnosis.
    print("[P4-B] initializing P3 RTC STRAP W/K/P/S transfers", flush=True)
    loop_initialization = initialize_keck_loop_transfers(trs)
    loop = trs.ttloop
    print("[P4-B] van Dam preset bands/Welch and transfer hypotheses",flush=True)
    noise = sensitivity_audit(centroids,loop)
    print("[P4-B] APD contiguous-flux windows and photon-noise baseline",flush=True)
    flux = flux_segment_audit(
        raw,sky,imat,centroids,loop,
        segment_length=args.segment_length,
        nperseg=args.segment_welch,
    )
    report = {
        "project": "Keck NIRC2 P4-B",
        "science_fits": str(args.fits.resolve()),
        "telemetry_sav": str(args.sav.resolve()),
        "source_revision": _revision(),
        "sample_count": int(len(centroids)),
        "loop_frequency_hz": float(loop.freq),
        "loop_transfer_initialization": loop_initialization,
        "data_contract": independent,
        "van_dam_and_loop_sensitivity": noise,
        "apd_flux_segments": flux,
        "unvalidated_instrument_properties": [
            "APD electrons per archived count",
            "APD avalanche excess/read/sky noise",
            "centroid optical gain",
            "TT mirror and RTC 2020 transfer W/K/P/S",
            "Keck STRAP to NIRC2 axis registration",
        ],
        "r0_or_psf_fitting_performed": False,
        "historical_van_dam_psd_projection_is_flagged_and_disallowed_for_physical_inference": True,
        "no_hypothesis_ranked_by_image_agreement": True,
    }
    path = args.output/"p4b_strap_noise_audit.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    _figure(report,args.output/"p4b_strap_noise_audit.png")
    print("[P4-B] Audit saved:",path,flush=True)
    print("[P4-B] All quantities conditional on uncalibrated Keck 2020 detector/loop",flush=True)


if __name__ == "__main__":
    main()
