"""Benchmark exact-physics Vii factorization against the legacy kernel.

Compares full HO phase structure function and HO OTF, NOT only the
intermediate Vii numerator; no parallelism or accelerator. The
algorithm has different floating operation order and can legitimately
FAIL the required strict 1e-12 numerical gate on larger grids.

Example:
  uv run python scripts/bench_psfr_vii_factorized.py \
    --n-pupil 48 --n-modes 120 --trials 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from p3.psfr.psfrUtils import modes2Otf


def benchmark(*,n_pupil=24,n_modes=32,trials=3,seed=1608):
    if n_pupil<8 or n_modes<1 or trials<1:
        raise ValueError("Invalid pupil size, number of HO modes or trials")
    rng=np.random.default_rng(seed)
    yy,xx=np.indices((n_pupil,n_pupil))
    pupil=((((xx-.5*n_pupil)/(.46*n_pupil))**2
            +((yy-.5*n_pupil)/(.46*n_pupil))**2<1)
           & ((xx-.5*n_pupil)**2+(yy-.5*n_pupil)**2>
              (.08*n_pupil)**2)).astype(float)
    pupil[yy>n_pupil*.72]=0.
    modes=rng.normal(scale=.04,size=(n_pupil*n_pupil,n_modes))
    # Diagonal covariance is PSD and cheap to diagonalize, isolating Vii.
    covariance=np.diag(np.linspace(.01,1.0,n_modes))
    reports={}
    output={}
    for method in ("legacy","factorized"):
        values=[]
        diagnostic={}
        for trial in range(trials):
            t0=perf_counter()
            image=modes2Otf(
                covariance,modes,pupil,2*n_pupil,samp=2,
                basis="Vii",covariance_psd_policy="strict",
                vii_method=method,vii_diagnostics=diagnostic,
            )
            values.append(perf_counter()-t0)
        reports[method]={
            "wall_times_s":values,"median_s":median(values),
            "vii_diagnostics":diagnostic,
        }
        output[method]=image
    l=output["legacy"]
    c=output["factorized"]
    errors=[]
    for title,a,b in zip(("otf","ho_dphi"),l,c):
        dif=b-a
        errors.append({
            "product":title,
            "max_absolute_difference":float(np.max(np.abs(dif))),
            "relative_l2_difference":float(
                np.linalg.norm(dif)/max(np.linalg.norm(a),1e-30)
            ),
            "parity_passed":bool(np.allclose(a,b,rtol=1e-12,atol=1e-12)),
        })
    result={
        "kind":"synthetic_Vii_factorization_not_Keck_E2E",
        "n_pupil":n_pupil,"n_modes":n_modes,
        "n_otf":2*n_pupil,"trials":trials,
        "single_cpu":True,"numpy_float64":True,
        "rtol":1e-12,"atol":1e-12,
        "results":reports,"parity":errors,
        "parity_passed":all(e["parity_passed"] for e in errors),
        "speedup":reports["legacy"]["median_s"]/
                   reports["factorized"]["median_s"],
    }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pupil",type=int,default=24)
    p.add_argument("--n-modes",type=int,default=32)
    p.add_argument("--trials",type=int,default=3)
    p.add_argument("--output",type=Path,
                   default=Path("results/bench_psfr_vii_factorized.json"))
    a=p.parse_args()
    result=benchmark(n_pupil=a.n_pupil,n_modes=a.n_modes,trials=a.trials)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result["parity_passed"]:
        raise SystemExit(
            "Vii factorization faster/slower metrics recorded, but strict "
            "HO SF/OTF parity failed. Do not enable in science."
        )


if __name__=="__main__":
    main()
