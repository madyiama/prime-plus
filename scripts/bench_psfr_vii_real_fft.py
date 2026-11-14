"""Single-CPU benchmark: full-complex vs one-sided real-FFT factorized Vii.

The historical full-complex factorized Vii is the mathematical reference
(no new physics, no GPU or worker pool). Reports a robust median over
repeated runs, HO SF and OTF parity, and Python-tracked allocation peak.
Native FFT RSS can exceed the tracemalloc peak: do not claim it as total
physical resident memory.

    uv run python scripts/bench_psfr_vii_real_fft.py \
      --n-pupil 48 --n-modes 120 --trials 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import tracemalloc

import numpy as np

from p3.psfr.psfrUtils import modes2Otf


def benchmark(*, n_pupil=48,n_modes=120,trials=5,seed=4512):
    if n_pupil<8 or n_modes<1 or trials<1:
        raise ValueError("Invalid pupil size, HO mode count or trial count")
    rng=np.random.default_rng(seed)
    yy,xx=np.indices((n_pupil,n_pupil))
    rho=((xx-.52*n_pupil)/(.44*n_pupil))**2+(
        (yy-.48*n_pupil)/(.41*n_pupil)
    )**2
    pupil=(rho<1).astype(float)
    pupil[(xx<n_pupil*.3)&(yy>n_pupil*.6)]=0.
    modes=rng.normal(scale=.035,size=(n_pupil*n_pupil,n_modes))
    covariance=np.diag(np.linspace(.01,1.0,n_modes))
    results={}
    outputs={}
    for method in ("factorized","factorized_real"):
        diagnostics={}
        args=(covariance,modes,pupil,2*n_pupil)
        # Prewarm FFT plans/library paths; no science result is cached.
        modes2Otf(*args,vii_method=method,vii_diagnostics=diagnostics)
        elapsed=[]
        peak=[]
        last=None
        for _ in range(trials):
            tracemalloc.start()
            start=perf_counter()
            last=modes2Otf(
                *args,vii_method=method,vii_diagnostics=diagnostics,
            )
            elapsed.append(perf_counter()-start)
            _,p=tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak.append(int(p))
        results[method]={
            "wall_seconds":elapsed,
            "median_seconds":median(elapsed),
            "python_tracemalloc_peak_bytes":max(peak),
            "vii_diagnostics":diagnostics,
        }
        outputs[method]=last
    comparisons={}
    ref=outputs["factorized"]
    candidate=outputs["factorized_real"]
    for label,a,b in zip(("ho_otf","ho_structure_function"),ref,candidate):
        diff=b-a
        comparisons[label]={
            "shape":list(a.shape),
            "parity_1e_12":bool(
                np.allclose(a,b,rtol=1e-12,atol=1e-12)
            ),
            "max_absolute_difference":float(np.max(np.abs(diff))),
            "relative_l2_difference":float(
                np.linalg.norm(diff)/max(np.linalg.norm(a),1e-30)
            ),
        }
    return {
        "kind":"synthetic_single_CPU_real_fft_vs_complex_Vii_not_Keck_E2E",
        "pupil_shape":[n_pupil,n_pupil],
        "n_modes":n_modes,"trials":trials,
        "numpy_float64":True,"gpu":False,"fft_workers":1,
        "rtol":1e-12,"atol":1e-12,
        "results":results,
        "comparisons":comparisons,
        "all_internal_parity_passed":all(
            r["parity_1e_12"] for r in comparisons.values()
        ),
        "speedup_real_vs_complex":(
            results["factorized"]["median_seconds"] /
            results["factorized_real"]["median_seconds"]
        ),
        "warning":"Synthetic HO OTF/SF parity is not real Keck science-PSF parity. "
                  "tracemalloc excludes some native allocator/RSS overhead.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pupil",type=int,default=48)
    parser.add_argument("--n-modes",type=int,default=120)
    parser.add_argument("--trials",type=int,default=5)
    parser.add_argument("--output",type=Path,
                        default=Path("results/bench_psfr_vii_real_fft.json"))
    args=parser.parse_args()
    report=benchmark(
        n_pupil=args.n_pupil,n_modes=args.n_modes,trials=args.trials,
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps(report,indent=2,allow_nan=False))
    if not report["all_internal_parity_passed"]:
        raise SystemExit(
            "Strict synthetic HO/OTF parity FAILED. "
            "Inspect report and do not enable on Keck without full PSF check."
        )


if __name__=="__main__":
    main()
