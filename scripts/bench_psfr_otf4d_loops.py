"""Audit wavelength-loop removal in the Fourier OTF->PSF core (single CPU).

The historic sf_3D_to_psf_4D pipeline computes a different static
OTF for each wavelength, handles wavelength-specific pixel binning
and crops each plane. Those science-dependent operations CANNOT
simply be removed. This microbenchmark examines ONLY the shared
OTF multiplication + spatial inverse-FFT step, AFTER all wavelength
static OTFs and phasors have been correctly computed.

A positive microbenchmark does not imply an E2E Keck speedup.
The production science renderer is NOT modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import tracemalloc

import numpy as np


def _validate(sf, stat, phasor, waves):
    sf=np.asarray(sf,dtype=np.float64)
    stat=np.asarray(stat)
    phasor=np.asarray(phasor)
    waves=np.asarray(waves,dtype=np.float64)
    if (sf.ndim!=3 or stat.ndim!=3 or phasor.ndim!=4
            or sf.shape[:2]!=stat.shape[:2]
            or phasor.shape!=sf.shape[:2]+(sf.shape[2],stat.shape[2])
            or waves.shape!=(stat.shape[-1],)
            or not np.isfinite(sf).all()
            or not np.isfinite(stat).all()
            or not np.isfinite(phasor).all()
            or not np.isfinite(waves).all()
            or not np.all(waves>0)):
        raise ValueError("Invalid [y,x,source,wavelength] Fourier inputs")
    return sf,stat,phasor,waves


def otf4d_reference(sf,stat,phasor,waves):
    """Historic wavelength loop: only Fourier part of sf_3D_to_psf_4D."""
    sf,stat,phasor,waves=_validate(sf,stat,phasor,waves)
    n0,n1,ns=sf.shape
    nw=len(waves)
    out=np.empty((n0,n1,ns,nw),dtype=float)
    scale=(2*np.pi*1e-9/waves)**2
    for j in range(nw):
        turb=np.exp(-.5*sf*scale[j])
        tot=np.fft.fftshift(
            turb*stat[:,:,j,None]*phasor[:,:,:,j],axes=(0,1)
        )
        out[:,:,:,j]=np.real(
            np.fft.fftshift(np.fft.ifftn(tot,axes=(0,1)),axes=(0,1))
        )
    return out


def otf4d_vectorized(sf,stat,phasor,waves):
    """Potential all-wavelength vectorized Fourier core, not full renderer."""
    sf,stat,phasor,waves=_validate(sf,stat,phasor,waves)
    scale=(2*np.pi*1e-9/waves)**2
    turb=np.exp(-.5*sf[:,:,:,None]*scale[None,None,None,:])
    tot=np.fft.fftshift(
        turb*stat[:,:,None,:]*phasor,axes=(0,1),
    )
    return np.real(
        np.fft.fftshift(np.fft.ifftn(tot,axes=(0,1)),axes=(0,1))
    )


def benchmark(*,n=128,n_src=1,n_wvl=3,trials=7,seed=52):
    if min(n,n_src,n_wvl,trials)<1:
        raise ValueError("Array dimensions and trials must be positive")
    rng=np.random.default_rng(seed)
    # Fourier kernels are only numerical, not a calibrated Keck pupil.
    y,x=np.indices((n,n))
    radius=(x-n/2)**2+(y-n/2)**2
    sf=np.broadcast_to(
        (1e5*(radius/(n*n)))[:,:,None],(n,n,n_src)
    ).copy()
    stat=np.exp(-radius[:,:,None]/(n*n*.12))*np.ones(n_wvl)
    phasor=np.exp(1j*rng.normal(scale=.03,size=(n,n,n_src,n_wvl)))
    waves=np.linspace(1.9e-6,2.3e-6,n_wvl)
    results={}
    arrays={}
    for name,fn in (("wavelength_loop",otf4d_reference),
                    ("all_wavelengths",otf4d_vectorized)):
        fn(sf,stat,phasor,waves)
        times=[]
        peaks=[]
        for _ in range(trials):
            tracemalloc.start()
            start=perf_counter()
            answer=fn(sf,stat,phasor,waves)
            times.append(perf_counter()-start)
            _,peak=tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak)
        results[name]={
            "wall_s":times,
            "median_s":median(times),
            "tracemalloc_peak_bytes":max(peaks),
        }
        arrays[name]=answer
    delta=arrays["all_wavelengths"]-arrays["wavelength_loop"]
    parity=bool(np.allclose(
        arrays["all_wavelengths"],arrays["wavelength_loop"],
        rtol=1e-12,atol=1e-12
    ))
    return {
        "kind":"synthetic_OTF_to_4D_PSF_Fourier_core_only",
        "not_included":"getStaticOTF, wavelength-specific binning/crop, flux normalization",
        "physics_changed":False,
        "shape":[n,n,n_src,n_wvl],
        "trials":trials,"workers":1,"gpu":False,
        "rtol":1e-12,"atol":1e-12,"parity_passed":parity,
        "max_absolute_difference":float(np.max(np.abs(delta))),
        "relative_l2_difference":float(
            np.linalg.norm(delta)/
            max(np.linalg.norm(arrays["wavelength_loop"]),1e-30)
        ),
        "input_wavelength_cube_bytes":(
            stat.nbytes+phasor.nbytes+sf.nbytes
        ),
        "results":results,
        "loop_over_vectorized_speed_ratio":(
            results["wavelength_loop"]["median_s"]/
            results["all_wavelengths"]["median_s"]
        ),
        "warning":"Precomputed static-OTF FFT microbenchmark only. "
                  "Do not infer speedup of Keck E2E PSF-R.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pupil",type=int,default=128)
    p.add_argument("--sources",type=int,default=1)
    p.add_argument("--wavelengths",type=int,default=3)
    p.add_argument("--trials",type=int,default=7)
    p.add_argument("--output",type=Path,
                   default=Path("results/bench_psfr_otf4d_loops.json"))
    args=p.parse_args()
    r=benchmark(n=args.n_pupil,n_src=args.sources,
                n_wvl=args.wavelengths,trials=args.trials)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(r,indent=2,allow_nan=False))
    if not r["parity_passed"]:
        raise SystemExit("4D Fourier vectorization fails strict numerical parity")


if __name__=="__main__":
    main()
