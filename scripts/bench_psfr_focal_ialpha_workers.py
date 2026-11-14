"""PR47 benchmark: identical exact Flicker quadrant SF with 1, 2, 4 CPU workers.

No cache, no cProfile, no GPU and no approximate Bessel kernel. Timings are
isolated focal-angular synthetic computations, NOT Keck FITS/SAV PSF speedups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from p3.aoSystem.anisoplanatismModel import focal_angular_function


def benchmark(*, nact=10, layers=7, notf=128, l0=25.,
              trials=5, grid="legacy", workers=(1, 2, 4)):
    if len(set(workers)) != len(workers) or 1 not in workers:
        raise ValueError("Worker list must include baseline 1 and be distinct")
    if any(type(x) is not int or x not in (1,2,4) for x in workers):
        raise ValueError("Workers must be 1, 2 or 4")
    tel = NS(D=10.5)
    atm = NS(nL=layers, heights=np.linspace(500., 14000., layers), L0=l0)
    src = NS(nSrc=1, direction=np.array([[1.75e-5],[-0.93e-5]]))
    gs = NS(height=np.array([90000.]), direction=np.array([[0.],[0.]]))
    durations, products = {}, {}
    diagnostics = {}
    for worker_count in workers:
        # Warm-up before timed trials: imports and one-off setup are excluded.
        focal_angular_function(
            tel, atm, src, gs, nact, notf, 2.,
            covariance_algorithm="quadrant", focal_workers=worker_count,
            pupil_grid_mode=grid)
        times = []
        for _ in range(trials):
            d = {}
            start = perf_counter()
            psf_kernel = focal_angular_function(
                tel, atm, src, gs, nact, notf, 2.,
                covariance_algorithm="quadrant", focal_workers=worker_count,
                pupil_grid_mode=grid, focal_diagnostics=d)
            times.append(perf_counter()-start)
        products[worker_count] = psf_kernel
        diagnostics[worker_count] = d
        durations[worker_count] = {
            "times_s": times,
            "median_s": median(times),
            "min_s": min(times),
            "max_s": max(times),
            "parallel_ialpha_calls_last": d.get("ialpha_parallel_calls",0),
            "serial_ialpha_calls_last": d.get("ialpha_serial_calls",0),
            "effective_workers": d.get("ialpha_workers_effective",1),
        }
    reference = products[1]
    parity = {}
    for w, val in products.items():
        diff = val-reference
        parity[w] = {
            "max_abs_error": float(np.max(np.abs(diff))),
            "relative_l2_error": float(
                np.linalg.norm(diff)/max(np.linalg.norm(reference),1e-30)),
            "rtol_1e12_atol_1e12_passed": bool(
                np.allclose(val,reference,rtol=1e-12,atol=1e-12)),
            "bitwise_equal": bool(np.array_equal(val,reference)),
        }
    result = {
        "kind": "synthetic_exact_Ialpha_CPU_workers_not_Keck_science",
        "n_actu": nact, "n_ph": 2*nact+1, "n_layers": layers,
        "n_otf": notf, "outer_scale_m": l0,
        "pupil_grid_mode": grid, "trials": trials,
        "workers": list(workers),
        "durations": {str(k): v for k,v in durations.items()},
        "parity": {str(k): v for k,v in parity.items()},
        "speedup_vs_serial": {
            str(k): durations[1]["median_s"]/v["median_s"]
            for k,v in durations.items()
        },
        "full_science_psf_validated": False,
        "note": "Use Keck n0120 full 128x128x3 comparator and another pupil "
                "orientation before considering a default. Thread scaling "
                "is architecture and problem-size dependent; slower is valid.",
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-actu",type=int,default=10)
    p.add_argument("--layers",type=int,default=7)
    p.add_argument("--n-otf",type=int,default=128)
    p.add_argument("--trials",type=int,default=5)
    p.add_argument("--outer-scale",type=float,default=25.)
    p.add_argument("--pupil-grid-mode",choices=("legacy","physical_diameter"),
                   default="legacy")
    p.add_argument("--output",type=Path,
                   default=Path("results/bench_psfr_focal_ialpha_workers.json"))
    args = p.parse_args()
    if min(args.n_actu,args.layers,args.n_otf,args.trials) < 1:
        p.error("All size and trial arguments must be positive")
    if not (args.outer_scale > 0 or np.isposinf(args.outer_scale)):
        p.error("Outer scale must be positive or infinity")
    report = benchmark(nact=args.n_actu,layers=args.layers,
                       notf=args.n_otf,trials=args.trials,
                       l0=args.outer_scale,grid=args.pupil_grid_mode)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(report,indent=2,allow_nan=False))
    if not all(row["rtol_1e12_atol_1e12_passed"]
               for row in report["parity"].values()):
        raise SystemExit("FAIL: exact Ialpha worker parity")


if __name__ == "__main__":
    main()
