"""Synthetic reproducible CPU benchmark: Vii modal covariance -> OTF.

Run: uv run python scripts/bench_psfr_modes2otf.py --trials 5

This measures one *kernel*; it is not a Keck end-to-end benchmark.
Run with thread limits (OPENBLAS_NUM_THREADS=1, etc.) for fair comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

os.environ.setdefault("P3_DISABLE_GPU", "TRUE")

import numpy as np

from p3.psfr.psfrUtils import modes2Otf


def benchmark(*, n_pupil: int = 64, n_modes: int = 64, trials: int = 3) -> dict:
    if n_pupil < 16 or n_modes < 1 or trials < 1:
        raise ValueError("n_pupil >= 16, n_modes >= 1, trials >= 1 required")
    generator = np.random.default_rng(20261112)
    y, x = np.indices((n_pupil, n_pupil))
    radius2 = ((x + .5 - n_pupil / 2) / (n_pupil / 2))**2
    radius2 += ((y + .5 - n_pupil / 2) / (n_pupil / 2))**2
    pupil = ((radius2 < .94**2) & (radius2 > .23**2)).astype(float)
    modes = generator.normal(size=(n_pupil**2, n_modes))
    modes *= pupil.ravel()[:, None]
    factor = generator.normal(size=(n_modes, n_modes))
    covariance = factor @ factor.T / n_modes

    def evaluate():
        otf, dphi = modes2Otf(
            covariance, modes, pupil, 2 * n_pupil, samp=2,
            basis="Vii", covariance_psd_policy="strict",
        )
        if not np.isfinite(otf).all() or not np.isfinite(dphi).all():
            raise ValueError("Nonfinite kernel output")
        return otf, dphi

    evaluate()  # Warm-up (not timed)
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        evaluate()
        samples.append(time.perf_counter() - start)
    return {
        "kind": "synthetic_Vii_kernel_not_Keck_E2E",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_pupil": n_pupil,
        "n_modes": n_modes,
        "n_otf": 2*n_pupil,
        "trials": trials,
        "durations_s": samples,
        "median_s": statistics.median(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pupil", type=int, default=64)
    parser.add_argument("--n-modes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = benchmark(n_pupil=args.n_pupil, n_modes=args.n_modes,
                       trials=args.trials)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
