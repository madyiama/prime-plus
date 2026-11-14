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
from dataclasses import asdict
from pathlib import Path

import numpy as np

from p3.psfr.detection import detect_stellar_sources
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
    parser.add_argument("--detect-only",action="store_true",
                        help="Create detection catalog/figure without final fitting")
    parser.add_argument("--max-sources",type=int,default=8,
                        help="Hard upper limit on PSF-prior detections")
    parser.add_argument("--detection-snr",type=float,default=6.,
                        help="Minimum matched-PSF detection SNR")
    parser.add_argument("--halo-snr",type=float,default=4.,
                        help="Required off-centre PSF wing SNR (hot-pixel veto)")
    parser.add_argument("--min-separation",type=float,default=4.,
                        help="Minimum distance between accepted sources [pixels]")
    parser.add_argument("--core-radius",type=int,default=9,
                        help="PSF morphology radius in pixels")
    parser.add_argument("--max-detection-candidates",type=int,default=128)
    parser.add_argument("--fwhm-min-ratio",type=float,default=.70,
                        help="Minimum candidate/PSF half-max width ratio")
    parser.add_argument("--fwhm-max-ratio",type=float,default=1.55,
                        help="Maximum candidate/PSF half-max width ratio")
    parser.add_argument("--min-shape-correlation",type=float,default=.78,
                        help="Minimum local correlation with the PSF core")
    parser.add_argument("--exclusion-fwhm",type=float,default=2.0,
                        help="Exclude additional detections within this "
                             "many PSF FWHM from an accepted star")
    parser.add_argument("--min-relative-flux",type=float,default=.03,
                        help="Minimum candidate flux / brightest accepted star")
    parser.add_argument("--min-bic-improvement",type=float,default=12.,
                        help="Minimum penalized model improvement, heuristic "
                             "unless --sigma is calibrated")
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

    dest=args.output or (case/"stellar_fit")
    dest.mkdir(parents=True,exist_ok=True)

    # Manual --xy deliberately overrides automatic source detection.
    # Without --xy the detector counts candidates using the actual PSF-R,
    # rejecting isolated single-pixel shapes before the stellar fit.
    detected=None
    starts=args.xy
    flux0=args.flux0
    background0=args.background0
    fit_mask=mask
    if args.xy is None:
        detected=detect_stellar_sources(
            observed,template,sigma=sigma,valid_mask=mask,
            max_sources=args.max_sources,min_snr=args.detection_snr,
            min_halo_snr=args.halo_snr,
            min_separation_px=args.min_separation,
            core_radius_px=args.core_radius,
            max_examined=args.max_detection_candidates,
            fwhm_min_ratio=args.fwhm_min_ratio,
            fwhm_max_ratio=args.fwhm_max_ratio,
            min_shape_correlation=args.min_shape_correlation,
            exclusion_fwhm=args.exclusion_fwhm,
            min_relative_flux=args.min_relative_flux,
            min_bic_improvement=args.min_bic_improvement,
        )
        starts=detected.positions_xy_px
        flux0=detected.initial_fluxes
        background0=detected.background
        fit_mask=detected.fit_valid_pixels
        catalog={
            "method":"iterative PSF matched-filter + FWHM/shape, "
                     "wing SNR, spike and penalized model alternatives",
            "psf_fwhm_px":detected.psf_fwhm_px,
            "psf_fwhm_range_px":detected.fwhm_acceptance_range_px,
            "psf_fwhm_exclusion_radius_px":detected.exclusion_radius_px,
            "min_shape_correlation":detected.min_shape_correlation,
            "min_relative_flux":detected.min_relative_flux,
            "min_bic_improvement":detected.min_bic_improvement,
            "n_masked_compact_pixel_candidates":detected.n_masked_hot_pixel_candidates,
            "noise_source":detected.noise_source,
            "empirical_noise_sigma":detected.noise_sigma,
            "thresholds":{
                "min_psf_snr":detected.min_snr,
                "min_halo_snr":detected.min_halo_snr,
                "max_sources":detected.max_sources,
                "min_separation_px":args.min_separation,
                "core_radius_px_minimum":args.core_radius,
                "candidate_fwhm_ratio_min":args.fwhm_min_ratio,
                "candidate_fwhm_ratio_max":args.fwhm_max_ratio,
                "exclusion_fwhm":args.exclusion_fwhm,
                "min_shape_correlation":args.min_shape_correlation,
                "min_relative_flux":args.min_relative_flux,
                "min_bic_improvement":args.min_bic_improvement,
            },
            "n_detected":len(detected.accepted),
            "n_rejected":len(detected.rejected),
            "accepted":[asdict(c) for c in detected.accepted],
            "rejected":[asdict(c) for c in detected.rejected],
            "final_refined_xy_px":starts.tolist(),
            "initial_fluxes_for_fit":flux0.tolist(),
            "n_initially_valid_pixels":detected.input_valid_pixels,
            "n_valid_after_candidate_rejection":detected.final_detection_valid_pixels,
            "examined_candidates":detected.examined_candidates,
            "stopped_reason":detected.stopped_reason,
            "labels_are_morphological_not_verified_detector_truth":True,
            "AO_parameters_fitted":False,
        }
        (dest/"source_detection.json").write_text(
            json.dumps(catalog,indent=2,allow_nan=False),encoding="utf-8"
        )
        np.save(dest/"source_detection_valid_pixels.npy",fit_mask)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
        good=observed[np.isfinite(observed)]
        scale=float(np.percentile(np.abs(good),99)) if good.size else 1.
        ax.imshow(np.arcsinh(np.where(np.isfinite(observed),observed,0)/
                            max(scale,1e-12)),origin="lower",cmap="viridis")
        if len(starts):
            ax.scatter(starts[:,0],starts[:,1],s=90,facecolors="none",
                       edgecolors="lime",linewidths=1.8,
                       label=f"PSF-like ({len(starts)})")
            # Show the measured core FWHM around accepted detections.
            from matplotlib.patches import Circle
            for x,y in starts:
                ax.add_patch(Circle(
                    (x,y),.5*detected.psf_fwhm_px,fill=False,
                    edgecolor="lime",linewidth=1.3,linestyle="--",
                ))
        if detected.rejected:
            bad=np.asarray([c.seed_xy_px for c in detected.rejected])
            ax.scatter(bad[:,0],bad[:,1],marker="x",c="red",
                       s=45,label=f"Rejected ({len(bad)})")
        if len(starts) or detected.rejected:
            ax.legend(loc="upper right")
        ax.set(title=(
            "PSF-R candidates (FWHM="
            f"{detected.psf_fwhm_px:.2f} px, "
            f"exclusion={detected.exclusion_radius_px:.1f} px)"
        ),
               xlabel="x column [px]",ylabel="y row [px]")
        fig.savefig(dest/"source_detection.png",dpi=160)
        plt.close(fig)
        print("[PSF-R detection]",len(starts),"source(s),",
              len(detected.rejected),"rejected candidates,",
              "stop =",detected.stopped_reason,flush=True)
        if args.detect_only:
            print("Detection catalog:",dest/"source_detection.json")
            return 0
        if len(starts)==0:
            print("No PSF-like stellar source found; inspect source_detection.png "
                  "and adjust --detection-snr/--halo-snr or supply --xy.")
            return 2
    elif args.detect_only:
        parser.error("--detect-only requires automatic detection (omit --xy)")

    fit=fit_stellar_scene(
        observed,template,
        initial_positions_xy_px=starts,
        initial_fluxes=flux0,
        initial_background=background0,
        sigma=sigma,valid_mask=fit_mask,
        radius_px=args.radius,
        max_position_offset_px=args.max_offset,
        robust_loss=args.loss,f_scale=args.f_scale,
        max_nfev=args.max_nfev,
    )
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
        "source_detection_method":(
            "psf_prior_morphological" if detected is not None else
            "manual_initial_coordinates"
        ),
        "source_detection_rejected_pixels":(
            len(detected.rejected) if detected is not None else 0
        ),
        "psfr_prior_fwhm_px":(
            detected.psf_fwhm_px if detected is not None else None
        ),
        "source_detection_catalog":(
            str(dest/"source_detection.json") if detected is not None else None
        ),
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
