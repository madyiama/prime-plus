"""Validated, fitting-ready PSF-R template without recomputing AO telemetry.

This is a data boundary, NOT a new Keck physics model or a PSF fitter.
The Keck V1 output already contains per-channel weights; never apply an
NIRC2 transmission curve a second time to this template.

Only source-independent, zero-background PSF predictions are valid here.
The exported template is normalized on its finite detector support.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .products import normalize_psf


SCHEMA = "p3.psfr.template.v1"
KECK_V1_SPECTRAL_STATUS = "reference_wavelength_psf_reweighted_not_chromatic"


@dataclass(frozen=True)
class PSFTemplate:
    """A detector-grid, unit-flux template with explicit provenance."""
    image: np.ndarray
    cube: np.ndarray
    spectral_fractions: np.ndarray
    pixel_scale_mas: float
    provenance: dict


def _validate_template(image, cube, fractions, pixel_scale):
    image = np.asarray(image, dtype=np.float64)
    cube = np.asarray(cube, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    if (image.ndim != 2 or cube.ndim != 3 or
            image.shape != cube.shape[:2] or
            min(image.shape) < 2 or fractions.shape != (cube.shape[2],)):
        raise ValueError("PSF template must have [y,x] and [y,x,wavelength] shapes")
    if not (np.isfinite(image).all() and np.isfinite(cube).all()
            and np.isfinite(fractions).all()):
        raise ValueError("PSF template contains non-finite values")
    if (np.any(image < 0) or np.any(cube < 0) or
            np.any(fractions < 0)):
        raise ValueError("PSF template contains negative intensities")
    if not np.isclose(image.sum(), 1, rtol=1e-12, atol=1e-12):
        raise ValueError("PSF template image must have unit flux")
    if not np.allclose(image, cube.sum(axis=2), rtol=1e-13, atol=1e-14):
        raise ValueError("PSF template image and cube disagree")
    if not np.allclose(fractions, cube.sum(axis=(0,1)), rtol=1e-13, atol=1e-14):
        raise ValueError("PSF template spectral fractions disagree")
    pix = float(pixel_scale)
    if not np.isfinite(pix) or pix <= 0:
        raise ValueError("PSF template pixel scale must be finite positive mas/pixel")


def template_from_psfr(
    prediction, *, pixel_scale_mas, source_index=None,
    provenance=None,
) -> PSFTemplate:
    """Make an independent, normalized copy of the current PSF-R output.

    Source index is mandatory for multiple sources. No recentering or
    chromatic reweighting occurs. Default V1 provenance explicitly flags
    that Keck's legacy 3-bin model shares a single reference-wavelength PSF.
    """
    product = normalize_psf(prediction, source_index=source_index)
    pix = float(pixel_scale_mas)
    original = {} if provenance is None else dict(provenance)
    original.setdefault("schema", SCHEMA)
    original.setdefault("model", "telemetry_only_psfr")
    original.setdefault("spectral_propagation", KECK_V1_SPECTRAL_STATUS)
    original.setdefault("absolute_strap_calibration", False)
    original.setdefault("science_image_psf_fit_performed", False)
    original.setdefault("array_axes", "row_y_column_x")
    original.setdefault(
        "astrometric_orientation_calibrated", False
    )
    original["source_index"] = (
        None if source_index is None else int(source_index)
    )
    image=np.array(product.image,copy=True)
    cube=np.array(product.cube,copy=True)
    fractions=np.array(product.spectral_fractions,copy=True)
    _validate_template(image,cube,fractions,pix)
    # Reject accidental non-JSON metadata early.
    json.dumps(original,allow_nan=False)
    return PSFTemplate(image,cube,fractions,pix,original)


def save_psfr_template(template: PSFTemplate, path: str | Path) -> Path:
    """Save a portable .npz artifact with no pickled Python objects."""
    _validate_template(
        template.image,template.cube,
        template.spectral_fractions,template.pixel_scale_mas
    )
    if template.provenance.get("schema") != SCHEMA:
        raise ValueError("Unsupported PSF template provenance schema")
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        target,
        image=template.image,
        cube=template.cube,
        spectral_fractions=template.spectral_fractions,
        pixel_scale_mas=np.asarray(template.pixel_scale_mas),
        provenance_json=np.asarray(
            json.dumps(template.provenance,sort_keys=True,allow_nan=False)
        ),
    )
    # numpy appends '.npz' if caller did not supply it; don't silently
    # return a path that was not created.
    return target if target.suffix == ".npz" else Path(str(target)+".npz")


def load_psfr_template(path: str | Path) -> PSFTemplate:
    """Reload and validate the unit-flux artifact; never silently repair."""
    with np.load(path,allow_pickle=False) as data:
        expected={
            "image","cube","spectral_fractions",
            "pixel_scale_mas","provenance_json"
        }
        if set(data.files) != expected:
            raise ValueError("PSF template archive does not match v1 schema")
        image=np.array(data["image"],copy=True)
        cube=np.array(data["cube"],copy=True)
        fractions=np.array(data["spectral_fractions"],copy=True)
        pix=float(data["pixel_scale_mas"])
        provenance=json.loads(str(data["provenance_json"].item()))
    if not isinstance(provenance,dict) or provenance.get("schema") != SCHEMA:
        raise ValueError("Unsupported PSF template provenance schema")
    _validate_template(image,cube,fractions,pix)
    return PSFTemplate(image,cube,fractions,pix,provenance)
