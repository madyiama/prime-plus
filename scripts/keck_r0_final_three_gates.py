#!/usr/bin/env python3
"""Read-only final r0/L0 three-gate preparation; does NOT change the r0 fit.

Example:
uv run python scripts/keck_r0_final_three_gates.py \\
  --fits /path/to/n0120.fits --sav /path/to/n0120_LGS_trs.sav \\
  --calib p3/aoSystem/data/KECK_CALIBRATION \\
  --output results/n0120_r0_final_three_gates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--sav", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-layer", type=int, default=7)
    p.add_argument("--max-frames", type=int, default=4096)
    args = p.parse_args()
    for candidate in (args.fits, args.sav, args.calib):
        if not candidate.exists():
            p.error(f"Missing input: {candidate}")
    args.output.mkdir(parents=True, exist_ok=True)

    from p3.telemetry.telemetryKeck import telemetryKeck
    from p3.telemetry.systemDiagnosis import systemDiagnosis
    from p3.telemetry.r0FinalAudit import analyse_final_three_gates

    trs = telemetryKeck(
        str(args.sav.resolve()), str(args.fits.resolve()),
        str(args.calib.resolve()), path_save=str(args.output.resolve()),
        nLayer=args.n_layer, cam_fov=128, addNCPA=False,
    )
    # Identical legacy constructor/default estimator: no functional changes.
    sd = systemDiagnosis(trs)
    report = analyse_final_three_gates(sd, frame_limit=args.max_frames)
    report["inputs"] = {
        "fits": args.fits.name, "sav": args.sav.name,
        "calibration_directory": str(args.calib.resolve()),
        "n_layer": args.n_layer,
        "max_frames": args.max_frames,
    }
    outfile = args.output / "r0_final_three_gates.json"
    outfile.write_text(json.dumps(report, indent=2, allow_nan=False),
                       encoding="utf-8")
    print(f"[P1-FINAL] Three gates: {outfile}")
    print("[P1-FINAL] Production r0/L0 unchanged")
    print("[P1-FINAL] RTC sign and gain are unverified: "
          "never select the best-fit hypothesis automatically")


if __name__ == "__main__":
    main()
