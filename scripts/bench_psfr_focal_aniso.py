"""Benchmark opt-in focal anisoplanatism accelerator vs exact legacy math.

This benchmarks ONLY the Flicker focal-angular SF step, not the complete
Keck PSF-R. Use the same parameters for both algorithms; fail if outputs
do not match at conservative float64 tolerances.

  uv run python scripts/bench_psfr_focal_aniso.py --n-actu 7 --layers 7
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-actu",type=int,default=7)
    parser.add_argument("--layers",type=int,default=7)
    parser.add_argument("--n-otf",type=int,default=128)
    parser.add_argument("--trials",type=int,default=3)
    parser.add_argument("--outer-scale",type=float,default=25.)
    parser.add_argument("--output",type=Path,default=Path("results/bench_psfr_focal_aniso.json"))
    args=parser.parse_args()
    if min(args.n_actu,args.layers,args.n_otf,args.trials)<1:
        parser.error("All size and trial arguments must be positive")
    tel=NS(D=10.5)
    atm=NS(
        nL=args.layers,
        heights=np.linspace(0,14000,args.layers),
        L0=args.outer_scale,
    )
    source=NS(nSrc=1,direction=np.array([[0.0],[0.0]]))
    beacon=NS(height=90000.,direction=np.array([[0.0],[0.0]]))
    routines={}
    outputs={}
    for algorithm in ("legacy","symmetry"):
        times=[]
        for i in range(args.trials):
            t0=perf_counter()
            image=focal_angular_function(
                tel,atm,source,beacon,args.n_actu,args.n_otf,2.,
                covariance_algorithm=algorithm
            )
            times.append(perf_counter()-t0)
        routines[algorithm]={"durations_s":times,"median_s":median(times)}
        outputs[algorithm]=image
    difference=outputs["symmetry"]-outputs["legacy"]
    absolute=float(np.max(np.abs(difference)))
    relative=float(np.linalg.norm(difference)/max(
        np.linalg.norm(outputs["legacy"]),1e-30,
    ))
    passed=bool(np.allclose(
        outputs["symmetry"],outputs["legacy"],rtol=1e-12,atol=2e-11
    ))
    result={
        "kind":"synthetic_focal_angular_anisoplanatism_not_Keck_E2E",
        "n_actu":args.n_actu,
        "n_ph":2*args.n_actu+1,
        "n_layers":args.layers,
        "n_otf":args.n_otf,
        "outer_scale_m":args.outer_scale,
        "trials":args.trials,
        "routines":routines,
        "max_absolute_difference":absolute,
        "relative_l2_difference":relative,
        "parity_passed":passed,
        "symmetry_speedup_for_focal_step":(
            routines["legacy"]["median_s"]/routines["symmetry"]["median_s"]
        ),
        "warning":"This benchmark does NOT imply equivalent acceleration of full PSF-R",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not passed:
        raise SystemExit("FAIL: focal symmetry changes the six-term result")


if __name__=="__main__":
    main()
