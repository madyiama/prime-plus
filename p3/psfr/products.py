"""Pure functions to validate and normalize PSF-R forward predictions.

No atmospheric or AO model changes are made here. Spectral slices supplied
by legacy P3 are ALREADY weighted, and must not be weighted a second time.
This module does not assert that legacy P3's broadband PSF is chromatically
accurate; that is a separate scientific validation gate.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PSFProduct(NamedTuple):
    """Data-only output: unit-flux cube/image and existing spectral fractions.

    cube: (y, x, wavelength), image: (y, x), fractions: (wavelength,).
    Normalization refers to the *supplied finite image support*.
    """
    cube: np.ndarray
    image: np.ndarray
    spectral_fractions: np.ndarray


def normalize_psf(
    array: np.ndarray, *, source_index: int | None = None
) -> PSFProduct:
    """Return a unit-flux, nonnegative PSF without changing the input.

    Accepted layouts are (y,x), (y,x,wavelength), or
    (y,x,source,wavelength). The 4-D multi-source case requires an
    explicit source_index. The input must represent a PSF kernel computed
    with zero background, *not* an observed image or flux-added star field.

    Negative or nonfinite PSFs fail explicitly instead of being
    silently repaired. It is the caller's responsibility to diagnose
    diffraction/OTF interpolation errors.
    """
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 4:
        n_source = values.shape[2]
        if source_index is None:
            if n_source != 1:
                raise ValueError("Multiple sources: source_index is required")
            source_index = 0
        if (not isinstance(source_index, (int, np.integer))
                or not 0 <= source_index < n_source):
            raise ValueError("Invalid source_index")
        values = values[:, :, source_index, :]
    elif source_index is not None:
        raise ValueError("source_index is only valid for 4-D input")

    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    elif values.ndim != 3:
        raise ValueError("PSF must be 2-D, 3-D, or 4-D")
    if any(n <= 0 for n in values.shape):
        raise ValueError("PSF dimensions must be nonempty")
    if not np.isfinite(values).all():
        raise ValueError("PSF contains nonfinite values")
    if np.any(values < 0):
        raise ValueError("PSF contains negative intensities; audit OTF/FFT")
    total = float(np.sum(values, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF total flux must be positive and finite")

    cube = values / total
    image = np.sum(cube, axis=2)
    fractions = np.sum(cube, axis=(0, 1))
    return PSFProduct(cube=cube, image=image, spectral_fractions=fractions)
