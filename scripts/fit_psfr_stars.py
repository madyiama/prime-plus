"""Fit stellar photometry and astrometry using an existing Keck V1 PSF.

Workflow (the science image is the EXACT telemetryKeck cutout):
  uv run python scripts/export_psfr_template.py --case results/psfr_v1_fit/n0120
  uv run python scripts/fit_psfr_stars.py --case results/psfr_v1_fit/n0120

For two stars with known approximate detector coordinates:
  uv run python scripts/fit_psfr_stars.py --case ... --xy 61 64 --xy 83 75

Fitting never loads SAV telemetry, never re-estimates HO/TT, and never
changes the reference PSF. The supplied x/y positions are detector
column/row pixels, not sky position angles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p3.psfr.stellar import fit_stellar_scene
from p3.psfr.template import load_psfr_template


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True,
                        help="Existing Keck V1 case with saved exact cutout and template")
    parser.add_argument("--template",type=Path,default=None)
    parser.add_argument("--observed",type=Path,default=None,
                        help="Override the exact science cutout with co-registered .npy")
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--xy",type=float,nargs=2,action="append",
                        metavar=("X_COLUMN","Y_ROW"),
                        help="Initial pixel centre; repeat for multiple stars")
    parser.add_argument("--flux0",type=float,nargs="+",default=None,
                        help="Initial fluxes, one per --xy")
    parser.add_argument("--background0",type=float,default=None)
    parser.add_argument("--radius",type=float,default=None,
                        help="Use only pixels within radius from initial positions")
    parser.add_argument("--max-offset",type=float,default=5.0,
                        help="Maximum position shift from initial coordinates [px]")
    parser.add_argument("--sigma",type=Path,default=None,
                        help="Optional positive per-pixel noise sigma .npy")
    parser.add_argument("--mask",type=Path,default=None,
                        help="Optional boolean .npy mask where True=valid")
    parser.add_argument("--saturation",type=float,default=None,
                        help="Exclude observed pixels >= threshold (in image units)")
    parser.add_argument("--loss",choices=("linear","soft_l1","huber"),
                        default="linear")
    parser.add_argument("--f-scale",type=float,default=1.)
    parser.add_argument("--max-nfev",type=int,default=300)
    args=parser.parse_args()
    case=args.case
    tpath=args.template or (case/"psfr_fit_template.npz")
    observed_path=args.observed or (case/"observed_nirc2_cutout.npy")
    if not tpath.is_file():
        parser.error(
            "Missing PSF template: first run scripts/export_psfr_template.py "
            f"--case {case}"
        )
    if not observed_path.is_file():
        parser.error(
            "Missing co-registered science cutout: rebuild the Keck V1 "
            "case on this branch (old runs did not save this file)"
        )
    template=load_psfr_template(tpath)
    observed=np.load(observed_path,allow_pickle=False)
    sigma=np.load(args.sigma,allow_pickle=False) if args.sigma else None
    mask=np.load(args.mask,allow_pickle=False) if args.mask else None
    if args.saturation is not None:
        if not np.isfinite(args.saturation):
            parser.error("--saturation must be finite")
        saturation_mask=np.isfinite(observed)&(observed<args.saturation)
        mask=saturation_mask if mask is None else (
            np.asarray(mask,dtype=bool)&saturation_mask
        )
    if args.flux0 is not None and args.xy is None:
        parser.error("--flux0 requires at least one explicit --xy position")
    if args.flux0 is not None and len(args.flux0)!=len(args.xy):
        parser.error("Provide one --flux0 value per --xy star")

    fit=fit_stellar_scene(
        observed,template,
        initial_positions_xy_px=args.xy,
        initial_fluxes=args.flux0,
        initial_background=args.background0,
        sigma=sigma,valid_mask=mask,
        radius_px=args.radius,
        max_position_offset_px=args.max_offset,
        robust_loss=args.loss,f_scale=args.f_scale,
        max_nfev=args.max_nfev,
    )
    dest=args.output or (case/"stellar_fit")
    dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/"model.npy",fit.model)
    np.save(dest/"residual.npy",fit.residual)
    np.save(dest/"valid_pixels.npy",fit.valid_pixels)
    peak_y,peak_x=np.unravel_index(
        np.argmax(template.image),template.image.shape
    )
    summary={
        "converged":fit.converged,
        "optimizer_status":fit.status,
        "optimizer_message":fit.message,
        "nfev":fit.nfev,
        "parameters_fitted":"flux, x column, y row per source, shared constant background",
        "psfr_ao_parameters_fixed":True,
        "observation":str(observed_path.resolve()),
        "template":str(tpath.resolve()),
        "template_source":template.provenance,
        "n_stars":int(len(fit.fluxes)),
        "stars":[{
            "index":int(i),
            "flux_detector_counts_on_unshifted_template_support":float(flux),
            "x_column_px":float(x),
            "y_row_px":float(y),
            "dx_from_template_peak_px":float(x-peak_x),
            "dy_from_template_peak_px":float(y-peak_y),
            "dx_from_template_peak_mas":float((x-peak_x)*template.pixel_scale_mas),
            "dy_from_template_peak_mas":float((y-peak_y)*template.pixel_scale_mas),
        } for i,((x,y),flux) in enumerate(zip(
            fit.positions_xy_px,fit.fluxes
        ))],
        "background_detector_counts_per_pixel":fit.background,
        "pixel_scale_mas":template.pixel_scale_mas,
        "n_valid_pixels":int(fit.valid_pixels.sum()),
        "n_excluded_pixels":int(fit.valid_pixels.size-fit.valid_pixels.sum()),
        "rms_on_valid_pixels":fit.rms,
        "weighted_chi2":fit.weighted_chi2,
        "reduced_chi2":fit.reduced_chi2,
        "sigma_source":str(args.sigma) if args.sigma else "none_unweighted",
        "robust_loss":args.loss,
        "f_scale":args.f_scale,
        "interpolation_negative_to_peak":fit.interpolation_negative_to_peak,
        "warning":(
            "Fixed telemetry-only Keck V1 PSF; no independent TT calibration, "
            "no fitted AO parameters; flux refers to template support and "
            "x/y use detector orientation, not calibrated on-sky orientation."
        ),
    }
    (dest/"fit_report.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    finite=np.isfinite(observed)
    data_clean=np.where(finite,observed,np.nan)
    res_clean=np.where(fit.valid_pixels,fit.residual,np.nan)
    image_scale=np.nanpercentile(np.abs(data_clean),99) or 1.
    residual_scale=np.nanpercentile(np.abs(res_clean),99) or 1.
    ax[0].imshow(np.arcsinh(data_clean/image_scale),
                 origin="lower",cmap="viridis")
    ax[1].imshow(np.arcsinh(fit.model/image_scale),
                 origin="lower",cmap="viridis")
    ax[2].imshow(np.arcsinh(res_clean/residual_scale),
                 origin="lower",cmap="coolwarm",vmin=-1,vmax=1)
    for subplot,title in zip(
        ax,("NIRC2 measured cutout","Stellar fit (fixed PSF-R)","Data - model")
    ):
        subplot.set(title=title,xlabel="x column [px]",ylabel="y row [px]")
    fig.suptitle(
        f"Keck V1 stellar fit: {len(fit.fluxes)} star(s), "
        f"RMS={fit.rms:.4g}, success={fit.converged}"
    )
    fig.savefig(dest/"fit_comparison.png",dpi=160)
    plt.close(fig)
    print(json.dumps({
        "converged":fit.converged,
        "n_stars":len(fit.fluxes),
        "positions_xy_px":fit.positions_xy_px.tolist(),
        "fluxes":fit.fluxes.tolist(),
        "background":fit.background,
        "rms":fit.rms,
        "valid_pixels":int(fit.valid_pixels.sum()),
    },indent=2))
    print("Fit outputs:",dest)
    return 0 if fit.converged else 1


if __name__=="__main__":
    raise SystemExit(main())
