"""Isolated strict Flicker SF benchmark: full symmetry vs direct quadrant.

This is NOT a Keck FITS/SAV science PSF benchmark, and tracemalloc
does not report total native RSS. Single-process CPU NumPy/SciPy only.
"""
from __future__ import annotations

import argparse
import json
import tracemalloc
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from p3.aoSystem.anisoplanatismModel import focal_angular_function


def bench(*, nact, layers, notf, l0, trials, grid):
    tel = NS(D=10.5)
    atm = NS(nL=layers, heights=np.linspace(200., 14000., layers),
             L0=l0)
    src = NS(nSrc=1, direction=np.array([[2.1e-5], [-1.1e-5]]))
    gs = NS(height=np.array([90000.]),
            direction=np.array([[0.], [0.]]))
    profiles = {}
    products = {}
    for method in ("symmetry", "quadrant"):
        samples = []
        for _ in range(trials):
            # Absolute timings MUST exclude instrumentation overhead.
            start = perf_counter()
            val = focal_angular_function(
                tel, atm, src, gs, nact, notf, 2.,
                covariance_algorithm=method,
                pupil_grid_mode=grid, output_grid="otf",
            )
            samples.append(perf_counter()-start)
        tracemalloc.start()
        try:
            memory_product = focal_angular_function(
                tel, atm, src, gs, nact, notf, 2.,
                covariance_algorithm=method,
                pupil_grid_mode=grid, output_grid="otf",
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if not np.allclose(memory_product, val, rtol=1e-12, atol=2e-11):
            raise ValueError("Instrumented and uninstrumented SF differ")
        profiles[method] = {
            "elapsed_s": samples,
            "median_elapsed_s": median(samples),
            "tracemalloc_peak_bytes_separate_run": peak,
        }
        products[method] = val
    ref, candidate = products["symmetry"], products["quadrant"]
    err = candidate-ref
    parity = bool(np.allclose(ref, candidate, rtol=1e-12, atol=2e-11))
    t0 = profiles["symmetry"]["median_elapsed_s"]
    t1 = profiles["quadrant"]["median_elapsed_s"]
    return {
        "kind": "synthetic_focal_quadrant_not_Keck_science_validation",
        "geometry": {
            "n_actu": nact, "n_ph": 2*nact+1,
            "n_layers": layers, "n_otf": notf, "L0_m": l0,
            "pupil_grid_mode": grid,
        },
        "n_trials": trials,
        "parity_passed": parity,
        "relative_l2_error": float(
            np.linalg.norm(err)/max(np.linalg.norm(ref), 1e-30)),
        "max_absolute_error": float(np.max(np.abs(err))),
        "tolerances": {"rtol": 1e-12, "atol": 2e-11},
        "routines": profiles,
        "speedup_focal_only": t0/t1,
        "memory_note": "tracemalloc excludes untracked C/NumPy RSS",
        "qualification": "Benchmark on actual Keck n0120 and a second pupil rotation before promotion",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-actu", type=int, default=7)
    p.add_argument("--layers", type=int, default=7)
    p.add_argument("--n-otf", type=int, default=128)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--outer-scale", type=float, default=25.)
    p.add_argument("--pupil-grid-mode", choices=("legacy", "physical_diameter"),
                   default="legacy")
    p.add_argument("--output", type=Path,
                   default=Path("results/bench_psfr_focal_quadrant.json"))
    args = p.parse_args()
    if min(args.n_actu, args.layers, args.n_otf, args.trials) < 1:
        p.error("All positive integer arguments must be >= 1")
    if not (args.outer_scale > 0 or np.isposinf(args.outer_scale)):
        p.error("Outer scale must be positive or infinity")
    report = bench(nact=args.n_actu, layers=args.layers,
                   notf=args.n_otf, l0=args.outer_scale,
                   trials=args.trials, grid=args.pupil_grid_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["parity_passed"]:
        raise SystemExit("FAIL: direct quadrant changes exact focal SF")


if __name__ == "__main__":
    main()
