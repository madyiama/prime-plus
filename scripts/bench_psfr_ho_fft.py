"""Benchmark Vii HO FFT batches against unchanged per-mode NumPy baseline.

Synthetic spatial modes and PSD covariance only, not a claim about
real Keck throughput. No rotated pupil or HO FFT cache is used.

    uv run python scripts/bench_psfr_ho_fft.py --pupil-size 64 --modes 200 \
      --batch-sizes 8 16 32 --workers 1 --trials 3

The command exits nonzero if HO SF or OTF differ above rtol=atol=1e-12.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from p3.psfr.psfrUtils import modes2Otf


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pupil-size",type=int,default=64)
    p.add_argument("--modes",type=int,default=200)
    p.add_argument("--batch-sizes",type=int,nargs="+",default=[8,16,32])
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--trials",type=int,default=3)
    p.add_argument("--seed",type=int,default=812)
    p.add_argument("--output",type=Path,
                   default=Path("results/bench_psfr_ho_fft.json"))
    args=p.parse_args()
    if (args.pupil_size<8 or args.modes<1 or args.trials<1
            or any(b<1 for b in args.batch_sizes)
            or args.workers<1):
        p.error("Invalid pupil dimensions, FFT batch or iteration count")
    rng=np.random.default_rng(args.seed)
    n=args.pupil_size
    yy,xx=np.indices((n,n))
    pupil=(((xx-.5*n)/(n*.44))**2
           +((yy-.5*n)/(n*.44))**2<1).astype(float)
    # Asymmetric segment clipping exercises more than a circular pupil.
    pupil[(xx<n*.25)&(yy>n*.5)]=0
    modes=rng.normal(scale=.02,size=(n*n,args.modes))
    # Diagonal PSD covariance isolates FFT accuracy and avoids a
    # large eigensystem benchmark unrelated to this performance sprint.
    covariance=np.diag(np.linspace(.05,1.2,args.modes))
    n_otf=2*n
    reports={}
    baseline=None
    for label,batch,workers in [
        ("legacy",16,1),
        *((f"batched_{b}",b,args.workers) for b in args.batch_sizes),
    ]:
        durations=[]
        diagnostics={}
        for _ in range(args.trials):
            start=perf_counter()
            result=modes2Otf(
                covariance,modes,pupil,n_otf,samp=2,
                fft_mode=("legacy" if label=="legacy" else "batched"),
                fft_batch_size=batch,fft_workers=workers,
                fft_diagnostics=diagnostics,
            )
            durations.append(perf_counter()-start)
        if baseline is None:
            baseline=result
            parity={"passed":True,"otf_abs_error":0.,"sf_abs_error":0.}
        else:
            err_otf=float(np.max(np.abs(result[0]-baseline[0])))
            err_sf=float(np.max(np.abs(result[1]-baseline[1])))
            parity={
                "passed":bool(
                    np.allclose(result[0],baseline[0],rtol=1e-12,atol=1e-12)
                    and np.allclose(result[1],baseline[1],rtol=1e-12,atol=1e-12)
                ),
                "otf_abs_error":err_otf,"sf_abs_error":err_sf,
            }
        reports[label]={
            "wall_times_s":durations,
            "median_s":median(durations),
            "parity":parity,
            "fft_diagnostics":diagnostics,
        }
    for label,value in reports.items():
        value["speedup_vs_legacy"]=reports["legacy"]["median_s"]/value["median_s"]
    payload={
        "kind":"synthetic_Vii_modal_FFT_batching_not_Keck_E2E",
        "pupil_size":n,"mode_count":args.modes,
        "workers":args.workers,"trials":args.trials,
        "float64_fft":True,
        "rtol":1e-12,"atol":1e-12,
        "results":reports,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps(payload,indent=2,allow_nan=False))
    if any(not v["parity"]["passed"] for v in reports.values()):
        raise SystemExit("FAIL: batched Vii FFT violates strict parity")


if __name__=="__main__":
    main()
