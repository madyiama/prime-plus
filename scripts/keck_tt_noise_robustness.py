"""Benchmark Keck STRAP van Dam sensor-noise robustness; NO PSF fitting.

Example:
  uv run python scripts/keck_tt_noise_robustness.py \
    --fits /path/to/n0013.fits --sav /path/to/n0013_LGS_trs.sav \
    --calib p3/aoSystem/data/KECK_CALIBRATION \
    --output results/n0013_vandam_robustness

The FITS and SAV remain on the user's machine and are never committed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import tomllib

from p3.telemetry.ttNoiseRobustness import (
    DEFAULT_BANDS, DEFAULT_WELCH, DEFAULT_PARTITIONS,
    run_robustness_benchmark, save_robustness_outputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--sav", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/n0013_vandam_robustness"))
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "config" / "keck_tt_noise.toml")
    parser.add_argument("--band", action="append", nargs=2, type=float,
                        metavar=("LOW", "HIGH"),
                        help="Repeat to override spectral bands (fraction of Nyquist)")
    parser.add_argument("--welch", nargs="+", type=int,
                        default=list(DEFAULT_WELCH),
                        help="Welch segment lengths in frames")
    parser.add_argument("--partitions", nargs="+", type=int,
                        default=list(DEFAULT_PARTITIONS),
                        help="Non-overlapping time splits (e.g. 2 3 4)")
    parser.add_argument("--min-segments", type=int, default=3,
                        help="Minimum overlapping Welch segments per estimate")
    args = parser.parse_args()
    for path in (args.fits, args.sav, args.calib, args.config):
        if not path.exists():
            parser.error(f"File or directory not found: {path}")

    with args.config.open("rb") as handle:
        config = tomllib.load(handle).get("tip_tilt_noise", {})
    band = tuple(config.get("high_frequency_band", [0.65, 0.95]))
    welch = int(config.get("welch_nperseg", 512))
    bands = tuple(tuple(x) for x in args.band) if args.band else DEFAULT_BANDS

    from p3.telemetry.telemetryKeck import telemetryKeck
    from p3.telemetry.systemDiagnosis import systemDiagnosis

    args.output.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    print("[telemetryKeck] START", flush=True)
    trs = telemetryKeck(
        str(args.sav.resolve()), str(args.fits.resolve()),
        str(args.calib.resolve()), path_save=str(args.output.resolve()),
        cam_fov=128, nLayer=1, addNCPA=False,
    )
    print(f"[telemetryKeck] OK ({time.perf_counter()-t:.1f}s)", flush=True)
    t = time.perf_counter()
    print("[systemDiagnosis] START", flush=True)
    # The nominal estimate is computed first to construct the same
    # transfer functions as the PSF-R smoke-test. Its result is not reused
    # for the grid (all combinations are independently recomputed).
    sd = systemDiagnosis(
        trs, tt_noise_method="van_dam",
        tt_noise_high_band=band, tt_noise_nperseg=welch,
    )
    print(f"[systemDiagnosis] OK ({time.perf_counter()-t:.1f}s)", flush=True)

    t = time.perf_counter()
    print("[robustness] START", flush=True)
    result = run_robustness_benchmark(
        sd.trs.tipTilt.slopes,
        sample_rate_hz=sd.trs.ttloop.freq,
        rejection_frequency_hz=sd.trs.ttloop.tf.freq,
        rejection_transfer=sd.trs.ttloop.tf.rtf,
        noise_transfer_factor=sd.trs.ttloop.tf.pn,
        bands=bands,
        welch_segments=args.welch,
        partitions=args.partitions,
        nominal_band=band, nominal_nperseg=welch,
        min_segments=args.min_segments,
    )
    result["input_fits"] = str(args.fits.resolve())
    result["input_sav"] = str(args.sav.resolve())
    result["nominal_config"] = str(args.config.resolve())
    save_robustness_outputs(result, args.output)
    successful = sum(
        row["status"] == "ok" for row in result["cases"]
    )
    skipped = sum(
        row["status"] == "skipped" for row in result["cases"]
    )
    errors = sum(
        row["status"] == "error" for row in result["cases"]
    )
    print(f"[robustness] OK ({time.perf_counter()-t:.1f}s): "
          f"{successful} fits; {skipped} skipped; {errors} errors",
          flush=True)
    nominal = result["nominal_full_record"]
    if nominal is not None:
        print("Nominal sensor-noise trace [m²]:",
              nominal["trace_sensor_noise_m2"], flush=True)
        print("Nominal measured residual atmosphere trace [m²]:",
              nominal["trace_atmos_measured_m2"], flush=True)
        print("Nominal P3-formula corrected trace [m²]:",
              nominal["trace_atmos_p3_m2"], flush=True)
    for name in ("van_dam_robustness.md", "van_dam_robustness.png",
                 "van_dam_robustness.csv", "van_dam_robustness.json"):
        print("Result:", args.output / name, flush=True)


if __name__ == "__main__":
    main()
