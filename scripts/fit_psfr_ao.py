"""AO parameter fit after Keck V1 PSF-R and stellar fit; NO AO reinitialization.

One-time Keck pass with --export-ao-basis is mandatory to capture the
expensive prepared HO/TT/fit/alias/anisoplanatism kernels. Example:

 uv run python scripts/fit_psfr_ao.py --case results/psfr_v1_fit/n0120
 uv run python scripts/fit_psfr_ao.py --case results/psfr_v1_fit/n0120 \
   --n-stars 2 --ao-params r0 gho

gtt must NEVER be activated without --allow-exploratory-gtt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p3.psfr.ao_basis import load_ao_basis
from p3.psfr.ao_fit import fit_hybrid_ao
from p3.psfr.template import load_psfr_template


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",type=Path,required=True)
    p.add_argument("--template",type=Path,default=None)
    p.add_argument("--basis",type=Path,default=None)
    p.add_argument("--observed",type=Path,default=None)
    p.add_argument("--stellar-report",type=Path,default=None)
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--xy",type=float,nargs=2,action="append",
                   metavar=("X_COLUMN","Y_ROW"))
    p.add_argument("--n-stars",type=int,default=None,
                   help="Select N brightest from the existing stellar fit; "
                        "for a catalogue use explicit --xy coordinates later")
    p.add_argument("--ao-params",choices=("r0","gho","gtt"),
                   nargs="+",default=["r0"],
                   help="Parameters to fit; default r0 only, gtt experimental")
    p.add_argument("--allow-exploratory-gtt",action="store_true",
                   help="Permit gtt under a tight prior, NOT STRAP calibration")
    p.add_argument("--sigma",type=Path,default=None,
                   help="Detector noise standard deviation map .npy; optional")
    p.add_argument("--mask",type=Path,default=None,
                   help="Boolean True=valid pixels; default previous fit mask")
    p.add_argument("--prior-r0",type=float,default=.15,
                   help="Gaussian sigma for log(r0/r0_telemetry)")
    p.add_argument("--prior-gho",type=float,default=.15,
                   help="Gaussian sigma of gho-1")
    p.add_argument("--prior-gtt",type=float,default=.07,
                   help="Gaussian sigma of exploratory gtt-1")
    p.add_argument("--alternations",type=int,default=2)
    p.add_argument("--ao-max-nfev",type=int,default=35)
    p.add_argument("--stellar-max-nfev",type=int,default=150)
    p.add_argument("--max-position-offset",type=float,default=3.)
    args=p.parse_args()

    case=args.case
    filename=args.basis or case/"ao_fit_basis.npz"
    if not filename.is_file():
        p.error("Missing AO basis: rerun scripts/keck_psfr_v1_e2e.py "
                "with --export-ao-basis once; old PSF-only results "
                "cannot recreate the AO phase structure kernels.")
    tpath=args.template or case/"psfr_fit_template.npz"
    if not tpath.is_file():
        p.error("Missing fitting PSF: run scripts/export_psfr_template.py "
                f"--case {case}")
    ipath=args.observed or case/"observed_nirc2_cutout.npy"
    if not ipath.is_file():
        p.error("Missing co-registered observed Keck cutout")
    basis=load_ao_basis(filename)
    template=load_psfr_template(tpath)
    image=np.load(ipath,allow_pickle=False)
    if args.xy is not None and args.n_stars is not None:
        p.error("Choose either --xy or --n-stars, not both")
    initial_fluxes=None
    initial_background=None
    if args.xy is not None:
        positions=np.asarray(args.xy,dtype=float)
    else:
        report=args.stellar_report or case/"stellar_fit"/"fit_report.json"
        if not report.is_file():
            p.error("Need previous stellar_fit/fit_report.json or --xy "
                    "coordinates. Review the source-detection catalog first.")
        history=json.loads(report.read_text(encoding="utf-8"))
        stars=history.get("stars",[])
        if not stars:
            p.error("Previous stellar fit contains no stars")
        if args.n_stars is not None:
            if args.n_stars<1 or args.n_stars>len(stars):
                p.error("--n-stars must be between 1 and existing fitted count")
            stars=sorted(
                stars,key=lambda x:x["flux_detector_counts_on_unshifted_template_support"],
                reverse=True
            )[:args.n_stars]
        positions=np.asarray(
            [[s["x_column_px"],s["y_row_px"]] for s in stars],dtype=float
        )
        initial_fluxes=np.asarray([
            s["flux_detector_counts_on_unshifted_template_support"] for s in stars
        ])
        initial_background=float(
            history["background_detector_counts_per_pixel"]
        )
    sigma=np.load(args.sigma,allow_pickle=False) if args.sigma else None
    maskfile=args.mask or case/"stellar_fit"/"valid_pixels.npy"
    mask=np.load(maskfile,allow_pickle=False) if maskfile.is_file() else None
    result=fit_hybrid_ao(
        image,basis,template,initial_positions_xy_px=positions,
        initial_fluxes=initial_fluxes,initial_background=initial_background,
        valid_mask=mask,sigma=sigma,
        active_parameters=tuple(args.ao_params),
        allow_exploratory_gtt=args.allow_exploratory_gtt,
        prior_sigmas={
            "r0":args.prior_r0,"gho":args.prior_gho,"gtt":args.prior_gtt
        },
        n_alternations=args.alternations,
        max_nfev=args.ao_max_nfev,
        stellar_max_nfev=args.stellar_max_nfev,
        max_position_offset_px=args.max_position_offset,
    )
    output=args.output or case/"hybrid_ao_fit"
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/"baseline_model.npy",result.baseline_model)
    np.save(output/"hybrid_model.npy",result.model)
    np.save(output/"hybrid_residual.npy",result.residual)

    details={
        "status":"exploratory_hybrid_AO_fit_not_independent_instrument_calibration",
        "active_AO_parameters":list(result.active_parameters),
        "baseline_AO":{"r0":basis.r0_m,"gho":1.,"gtt":1.},
        "best_AO":result.ao_parameters,
        "r0_wavelength_m":basis.atmosphere_wavelength_m,
        "prior_sigmas":result.prior_sigmas,
        "stages":list(result.optimizer_status),
        "numeric_nominal_PSF_parity":result.nominal_parity,
        "n_sources":int(result.stellar.fluxes.size),
        "stellar_parameters":[
            {"x_col":float(x),"y_row":float(y),"flux":float(f)}
            for (x,y),f in zip(
                result.stellar.positions_xy_px,result.stellar.fluxes
            )
        ],
        "stellar_background":float(result.stellar.background),
        "baseline_RMS":result.initial_rms,
        "hybrid_RMS":result.final_rms,
        "RMS_ratio":(float(result.final_rms/result.initial_rms)
                     if result.initial_rms>0 else None),
        "AO_shape_only_jacobian_condition":result.jacobian_condition,
        "notes":list(result.warnings),
        "noise_source":result.noise_source,
        "empirical_noise_sigma_not_calibrated":result.empirical_noise_scale,
        "TT_sensor_optical_gain_independently_calibrated":False,
        "forward_physics":"existing_Keck_V1_reference_wavelength_not_chromatic",
        "input_basis":str(filename),
        "input_template":str(tpath),
        "fit_pixels_mask":str(maskfile) if mask is not None else None,
        "sigma_map":str(args.sigma) if args.sigma else None,
    }
    (output/"ao_fit_report.json").write_text(
        json.dumps(details,indent=2,allow_nan=False),encoding="utf-8"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,4,figsize=(17,4.3),constrained_layout=True)
    vmax=float(np.nanpercentile(image[np.isfinite(image)],99)) or 1.
    for ax,array,title in zip(
        axs[:3],[image,result.baseline_model,result.model],
        ["Observed NIRC2","Telemetry AO + stellar","Hybrid AO + stellar"]
    ):
        ax.imshow(np.arcsinh(np.where(np.isfinite(array),array,0)/max(vmax,1e-12)),
                  origin="lower",cmap="viridis")
        ax.set(title=title,xlabel="x [px]",ylabel="y [px]")
    rclip=np.where(np.isfinite(result.residual),result.residual,0.)
    scale=float(np.percentile(np.abs(rclip),99)) or 1.
    axs[3].imshow(np.arcsinh(rclip/scale),origin="lower",
                  cmap="coolwarm",vmin=-1,vmax=1)
    axs[3].set(title="Data - hybrid",xlabel="x [px]",ylabel="y [px]")
    fig.savefig(output/"ao_fit_comparison.png",dpi=160)
    plt.close(fig)
    print(json.dumps({
        "AO":result.ao_parameters,
        "RMS_initial":result.initial_rms,
        "RMS_final":result.final_rms,
        "warnings":list(result.warnings),
        "nominal_parity":result.nominal_parity,
        "report":str(output/"ao_fit_report.json"),
    },indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
