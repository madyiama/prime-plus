"""Flux-bounded, subpixel-area sampling of a binary astronomical pupil.

The pixel value is the area-averaged *complex-field amplitude* of the
binary pupil over the output cell, not the square-root of throughput.
This preserves 0 <= A <= 1 and avoids cubic-spline overshoot. It does not
calibrate the physical FITS mask angle, parity or meters-per-pixel.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def resample_binary_pupil(mask, resolution, rotation_deg=0.0, supersample=4):
    """Rotate (scipy-positive angle) and area-downsample an archived binary mask.

    Source and destination grids span the same square physical field. Samples
    are taken at output subpixel *centres*. A strictly binary input is
    required; apodizers, NCPA maps, or pre-interpolated pupils need a
    different interpolator and cannot silently take this path.
    """
    arr = np.asarray(mask, dtype=float)
    n = int(resolution)
    s = int(supersample)
    angle = float(rotation_deg)
    if (arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 8
            or not np.all(np.isfinite(arr))):
        raise ValueError("Binary pupil must be finite square 2-D with side >=8")
    if (n < 16 or n != resolution or s < 1 or s != supersample
            or not np.isfinite(angle)):
        raise ValueError("Invalid pupil resolution, supersampling or rotation")
    if not np.any(arr > 0):
        raise ValueError("Pupil contains no open samples")
    if not np.all((np.abs(arr) < 1e-8) | (np.abs(arr-1.0) < 1e-8)):
        raise ValueError("Physical binary sampling requires a 0/1 FITS pupil")
    native_side = arr.shape[0]
    output_coords = (np.arange(n*s, dtype=float)+0.5)/s - n/2
    # Coordinates in pixels of the original, same-square-field image.
    output_coords *= native_side/n
    dy, dx = np.meshgrid(output_coords, output_coords, indexing="ij")
    phi = np.deg2rad(angle)
    centre = (native_side-1)/2
    # scipy.ndimage.rotate: source=[cos,sin;-sin,cos]*destination.
    sy = centre + np.cos(phi)*dy + np.sin(phi)*dx
    sx = centre - np.sin(phi)*dy + np.cos(phi)*dx
    samples = map_coordinates(arr, (sy, sx), order=0,
                              mode="constant", cval=0.0,
                              prefilter=False)
    pupil = samples.reshape(n, s, n, s).mean(axis=(1, 3))
    # The computed mean is bounded by construction, not by clipping.
    if not np.all((pupil >= 0) & (pupil <= 1)):
        raise AssertionError("Resampled pupil transmission became nonphysical")
    return pupil


def pupil_sampling_diagnostics(native, sampled, angle_deg, supersample):
    """Record optical-amplitude and effective intensity-throughput statistics."""
    native = np.asarray(native, dtype=float)
    sampled = np.asarray(sampled, dtype=float)
    return {
        "method": "physical_binary_subpixel_area_average",
        "rotation_deg_scipy": float(angle_deg),
        "supersampling_per_axis": int(supersample),
        "raw_mask_shape": list(native.shape),
        "output_mask_shape": list(sampled.shape),
        "output_amplitude_min": float(sampled.min()),
        "output_amplitude_max": float(sampled.max()),
        "fractional_amplitude_pixel_count": int(np.count_nonzero((sampled > 0) & (sampled < 1))),
        "input_binary_area_fraction": float(np.mean(native)),
        "output_mean_amplitude": float(np.mean(sampled)),
        "output_mean_intensity_transmission": float(np.mean(sampled**2)),
        "zero_angle_parity_calibrated": False,
        "warning": "Pixel amplitude is a subpixel area average; angle zero and camera handedness must be calibrated independently",
    }
