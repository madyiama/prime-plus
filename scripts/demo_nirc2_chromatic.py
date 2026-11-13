"""Illustrate true wavelength-dependent diffraction with a measured KAI filter.

WARNING: pupil is an ideal circular aperture, NOT the Keck segmented pupil.
This demonstrates the optical and photon-weighting contracts only; it does
not reconstruct telemetry or validate absolute Keck PSF-R calibration.

Prerequisite:
  uv run python scripts/fetch_nirc2_filters.py --filters Kp
Example:
  uv run python scripts/demo_nirc2_chromatic.py --filter Kp --n-bins 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p3.psfr.chromatic import (
    prepare_pupil_lag_grid, detector_monochromatic_psf,
)
from p3.psfr.spectral import (
    read_kai_filter, photon_quadrature, integrate_monochromatic_psfs,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter",default="Kp")
    parser.add_argument("--filter-dir",type=Path,default=Path("data/nirc2_filters"))
    parser.add_argument("--n-bins",type=int,default=12)
    parser.add_argument("--pupil-size",type=int,default=96)
    parser.add_argument("--fov-pixels",type=int,default=128)
    parser.add_argument("--pixel-scale-mas",type=float,default=9.942)
    parser.add_argument("--diameter-m",type=float,default=10.5)
    parser.add_argument("--output",type=Path,
                        default=Path("results/nirc2_chromatic_demo"))
    args=parser.parse_args()

    raw=read_kai_filter(args.filter_dir/f"{args.filter}.dat")
    quadrature=photon_quadrature(raw,n_bins=args.n_bins)
    yy,xx=np.indices((args.pupil_size,args.pupil_size))
    centre=(args.pupil_size-1)/2
    aperture=(
        (xx-centre)**2+(yy-centre)**2 < (.47*args.pupil_size)**2
    ).astype(float)
    geometry=prepare_pupil_lag_grid(
        aperture,diameter_m=args.diameter_m
    )
    pixel_rad=args.pixel_scale_mas/1000/206265
    psf=[]
    for wavelength in quadrature.wavelength_m:
        result=detector_monochromatic_psf(
            geometry,wavelength_m=wavelength,
            pixel_scale_rad=pixel_rad,n_pixels=args.fov_pixels,
        )
        psf.append(result.psf)
    cube=np.stack(psf,axis=-1)
    polychromatic=integrate_monochromatic_psfs(
        cube,quadrature.weights
    )
    monochromatic=detector_monochromatic_psf(
        geometry,wavelength_m=quadrature.effective_wavelength_m,
        pixel_scale_rad=pixel_rad,n_pixels=args.fov_pixels,
    ).psf
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/"monochromatic_psfs.npy",cube)
    np.save(args.output/"polychromatic_psf.npy",polychromatic)
    np.save(args.output/"effective_wavelength_mono_psf.npy",monochromatic)

    report={
        "filter":args.filter,
        "source":raw.source,
        "passband_cleaning":raw.audit,
        "pupil_geometry":"IDEAL circular demonstration, NOT Keck segmented pupil",
        "spectral_approximation":"Flat photon SED; filter only; atmosphere not supplied",
        "n_bins":int(args.n_bins),
        "wavelength_um":(quadrature.wavelength_m*1e6).tolist(),
        "weights":quadrature.weights.tolist(),
        "effective_wavelength_um":float(quadrature.effective_wavelength_m*1e6),
        "poly_flux":float(polychromatic.sum()),
        "poly_peak":float(polychromatic.max()),
        "mono_peak":float(monochromatic.max()),
    }
    (args.output/"diagnostics.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    axes[0].plot(raw.wavelength_um,raw.transmission)
    axes[0].set(xlabel="Wavelength (µm)",ylabel="Measured filter transmission",
                title=f"KAI NIRC2 {args.filter}")
    for i,(image,title) in enumerate([
        (monochromatic,"Mono at effective λ"),
        (polychromatic,"Photon-weighted polychromatic"),
    ],start=1):
        extent=(np.array([-args.fov_pixels/2,args.fov_pixels/2,
                          -args.fov_pixels/2,args.fov_pixels/2])
                *args.pixel_scale_mas)
        axes[i].imshow(np.log10(np.maximum(image,1e-9)),
                       origin="lower",extent=extent,
                       vmin=-7,vmax=-1,cmap="viridis")
        axes[i].set(xlabel="Camera angle (mas)",ylabel="Camera angle (mas)",
                    title=title)
    fig.savefig(args.output/"chromatic_comparison.png",dpi=160)
    plt.close(fig)
    print(json.dumps(report,indent=2))
    print("Figure:",args.output/"chromatic_comparison.png")


if __name__=="__main__":
    main()
