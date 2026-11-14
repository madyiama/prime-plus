"""Export a validated Keck V1 PSF-R prediction to a portable fitting template.

Reuses an ALREADY computed E2E case: no FITS/SAV reconstruction, no AO
initialization, and no modification to existing science PSF arrays.

Example:
  uv run python scripts/export_psfr_template.py \
    --case results/psfr_v1_acceptance/n0120 \
    --pixel-scale-mas 9.942
The pixel scale is an example for narrow-camera observations only;
always use the value actually applicable to the FITS acquisition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p3.psfr.template import (
    template_from_psfr,save_psfr_template,
    KECK_V1_SPECTRAL_STATUS,
)
from scripts.keck_psfr_v1_e2e import validate_run


def export_keck_case(case, *, pixel_scale_mas, output=None):
    """Refuse nonnominal, incomplete or modified V1 cases before export."""
    directory=Path(case)
    acceptance=validate_run(directory)
    metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    recorded_scale=metadata.get("camera_pixel_scale_mas")
    if recorded_scale is not None and not np.isclose(
        float(recorded_scale),float(pixel_scale_mas),
        rtol=1e-8,atol=1e-8
    ):
        raise ValueError(
            "Supplied pixel scale differs from reconstruction metadata"
        )
    if metadata.get("psf_spectral_model") is None:
        raise ValueError("Missing PSF-R spectral provenance")
    # This V1 path MUST stay labelled as reference-wavelength, not a
    # validated wavelength-dependent Keck broadband propagation.
    spectrum=metadata["psf_spectral_model"]
    if "Single reference-wavelength PSF" not in spectrum:
        raise ValueError("Unexpected spectral physics in nominal Keck V1")
    predicted=np.load(directory/"telemetry_only_psf.npy",allow_pickle=False)
    provenance={
        "schema":"p3.psfr.template.v1",
        "model":"Keck_LGS_NIRC2_V1",
        "spectral_propagation":KECK_V1_SPECTRAL_STATUS,
        "reference_wavelength_m":metadata.get("psf_reference_wavelength_m"),
        "origin_case":str(directory.resolve()),
        "input_fits":metadata.get("input_fits"),
        "input_sav":metadata.get("input_sav"),
        "code_revision":metadata.get("code_revision"),
        "pupil_resampling":metadata.get("p3_pupil_resampling"),
        "pupil_rotation_hypothesis":metadata.get("p3_pupil_rotation_hypothesis"),
        "pupil_parity":metadata.get("p3_pupil_parity"),
        "tt_profile":metadata.get("tt_selection_profile"),
        "tt_kernel":metadata.get("tt_kernel_convention"),
        "tt_jitter_model":metadata.get("tt_jitter_covariance_method"),
        "ho_covariance_policy":metadata.get("ho_covariance_policy"),
        "absolute_strap_calibration":False,
        "science_image_psf_fit_performed":False,
        "acceptance_status":acceptance["status"],
    }
    template=template_from_psfr(
        predicted,pixel_scale_mas=pixel_scale_mas,provenance=provenance
    )
    destination=(
        Path(output) if output is not None else directory/"psfr_fit_template.npz"
    )
    saved=save_psfr_template(template,destination)
    manifest={
        "template":str(saved.resolve()),
        "shape":list(template.image.shape),
        "n_channels":int(template.cube.shape[-1]),
        "pixel_scale_mas":float(template.pixel_scale_mas),
        "flux":float(template.image.sum()),
        "spectral_fractions":template.spectral_fractions.tolist(),
        "provenance":template.provenance,
        "warning":"Fitting must not be interpreted as independent Keck STRAP calibration",
    }
    saved.with_suffix(".json").write_text(
        json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8"
    )
    return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True)
    parser.add_argument("--pixel-scale-mas",type=float,required=True,
                        help="Positive detector mas/pixel from the actual FITS camera")
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args()
    path=export_keck_case(
        args.case,pixel_scale_mas=args.pixel_scale_mas,output=args.output
    )
    print("Validated PSF-R fitting template:",path)
    print("Manifest:",path.with_suffix(".json"))


if __name__=="__main__":
    main()
