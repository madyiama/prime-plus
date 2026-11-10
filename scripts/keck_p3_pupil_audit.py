#!/usr/bin/env python3
"""P3 Keck/NIRC2 pupil audit: diffraction hypotheses, not science PSF fits.

Example:
uv run python scripts/keck_p3_pupil_audit.py \\
  --fits /path/to/n0120.fits \\
  --calib p3/aoSystem/data/KECK_CALIBRATION \\
  --output results/n0120_P3_pupil
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--oversample", type=int, default=2)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from p3.telemetry.keckPupilAudit import (
        pupil_orientation_report, read_mask, binary_pupil_on_grid,
        diffraction_intensity,
    )

    hdr = fits.getheader(args.fits)
    report = pupil_orientation_report(hdr)
    mask_path, mask = read_mask(args.calib, report)
    args.output.mkdir(parents=True, exist_ok=True)

    report["science_fits_name"] = args.fits.name
    report["archived_mask_path"] = str(mask_path)
    report["archived_mask_shape"] = list(mask.shape)
    report["analysis_resolution"] = args.resolution
    report["analysis_oversample"] = args.oversample
    report["comparison_limitation"] = (
        "Qualitative diffraction morphology only; uncalibrated FITS "
        "mask zero, mechanical angle and image-axis convention. No "
        "observational best-fit rotation, turbulence, focus or NCPA."
    )

    hypotheses = report["candidate_scipy_array_rotation_deg"]
    fig, axes = plt.subplots(2, len(hypotheses),
                             figsize=(3.4*len(hypotheses), 6.4))
    for k, (label, angle) in enumerate(hypotheses.items()):
        mask_k = binary_pupil_on_grid(mask, args.resolution, angle)
        psf_k = diffraction_intensity(mask_k, args.oversample)
        np.save(args.output / f"pupil_{label}.npy", mask_k)
        np.save(args.output / f"diffraction_{label}.npy", psf_k)
        axes[0, k].imshow(mask_k, origin="lower", interpolation="nearest",
                          vmin=0, vmax=1, cmap="gray")
        axes[0, k].set_title(f"{label}\\n{angle:+.3f} deg", fontsize=9)
        axes[1, k].imshow(
            np.log10(np.maximum(psf_k/psf_k.max(), 1e-7)),
            origin="lower", cmap="magma", vmin=-6, vmax=0,
        )
        axes[1, k].set_title("Diffraction only (log10 peak)", fontsize=9)
        for ax in (axes[0, k], axes[1, k]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"NIRC2 {args.fits.name}: uncalibrated pupil rotation hypotheses",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.output / "p3_pupil_diffraction_hypotheses.png", dpi=150)
    plt.close(fig)
    (args.output / "p3_pupil_orientation_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("[P3] Pupil report:", args.output / "p3_pupil_orientation_audit.json")
    print("[P3] Diffraction hypotheses:",
          args.output / "p3_pupil_diffraction_hypotheses.png")
    print("[P3] No rotation was calibrated or selected.")


if __name__ == "__main__":
    main()
