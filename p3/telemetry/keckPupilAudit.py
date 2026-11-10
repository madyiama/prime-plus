"""Keck/NIRC2 pupil orientation audit; no atmospheric or NCPA calibration.

KAON 1160 gives NIRC2-pupil = EL - ROTPPOSN and WFS-pupil
= ROTPPOSN - EL, in different reference frames.  The absolute zero of
the archived pupil FITS and scipy image-axis convention are not verified
by those two equations. Do not auto-select a rotation from science data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from p3.telemetry.keckUtils import getPupilAngles


# Explicit inventory of archive masks, not an assumption that every
# NIRC2 mask has a corresponding calibrated local FITS file.
_MASK_FILES = {
    "OPEN": "keck_pupil_open_240px.fits",
    "LARGEHEX": "keck_pupil_largeHex_272px.fits",
    "INCIRCLE": None,
}


def pupil_orientation_report(header):
    """Describe telescope, camera and mechanical pupil angles separately.

    Candidate array rotations are *hypotheses*, not validated transforms:
    KAON 1160's 90-degree WFS/camera offset cannot determine the
    zero-angle convention of a specific stored FITS mask.
    """
    angles = getPupilAngles(header)
    mask_name = str(header.get("PMSNAME", "")).strip().upper()
    if mask_name not in _MASK_FILES:
        raise ValueError(f"Unsupported or missing NIRC2 PMSNAME: {mask_name!r}")
    theta = float(angles["nirc2_pupil_angle_deg"])
    hypotheses = {
        "as_archived_no_rotation": 0.0,
        "kaon_nirc2_nominal": theta,
        "kaon_nirc2_plus_90": theta + 90.0,
        "kaon_nirc2_minus_90": theta - 90.0,
        "reverse_sign_diagnostic": -theta,
    }

    def mechanical(keyword):
        if keyword not in header:
            return None
        value = float(header[keyword])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite NIRC2 mechanical pupil angle: {keyword}")
        return value

    return {
        "source": "KAON_1160_Ragland_2017",
        "sky_pa_deg": float(angles["sky_position_angle_deg"]),
        "nirc2_pupil_angle_deg": theta,
        "wfs_pupil_angle_deg": float(angles["wfs_pupil_angle_deg"]),
        "relative_reference_offset_deg": 90.0,
        "pupil_mask_name": mask_name,
        "pupil_mask_file": _MASK_FILES[mask_name],
        "pmsangl_header_deg": mechanical("PMSANGL"),
        "pmrangl_header_deg": mechanical("PMRANGL"),
        "pupil_mask_mechanical_angle_deg": None,
        "mechanical_to_optical_angle_calibrated": False,
        "mask_fits_zero_angle_calibrated": False,
        "camera_array_rotation_sign_calibrated": False,
        "candidate_scipy_array_rotation_deg": hypotheses,
        "wfs_camera_reference_warning": (
            "WFS and NIRC2 have opposite angular signs and a 90-degree "
            "reference offset; do not apply both transforms twice"
        ),
        "status": "diagnostic_hypotheses_only",
    }


def read_mask(calibration_directory, report):
    """Load the actual archived mask; never silently substitute OPEN for INCIRCLE."""
    from astropy.io import fits

    filename = report.get("pupil_mask_file")
    if filename is None:
        raise FileNotFoundError(
            "No calibrated local FITS for this NIRC2 pupil mask; "
            "an alternate pupil must not be silently substituted"
        )
    path = Path(calibration_directory) / "NIRC2_MASK" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Archived Keck pupil FITS not found: {path}")
    data = np.asarray(fits.getdata(path), dtype=float)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError("Keck pupil FITS must contain a square 2D mask")
    if not np.isfinite(data).all() or not np.any(data > 0):
        raise ValueError("Keck pupil FITS must contain finite positive transmission")
    return path, data


def binary_pupil_on_grid(data, resolution, rotation_deg):
    """Use exactly the bounded, area-averaged sampling of the P3 telescope.

    Name kept for compatibility with the first P3 diagnostic script.
    Fractional boundary amplitudes represent subpixel pupil occupancy,
    not negative/overshooting cubic-spline transmission.
    """
    from p3.aoSystem.pupilTransforms import resample_binary_pupil
    return resample_binary_pupil(
        data, resolution, rotation_deg=rotation_deg, supersample=4,
    )


def diffraction_intensity(pupil_mask, oversample=2):
    """Unit-flux monochromatic diffraction illustration (no science fit)."""
    mask = np.asarray(pupil_mask)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("Square pupil required")
    factor = int(oversample)
    if factor < 1 or float(oversample) != factor or mask.sum() <= 0:
        raise ValueError("Positive integer oversample and nonempty pupil required")
    ft = np.fft.fftshift(np.fft.fft2(
        mask.astype(float), s=(factor * mask.shape[0],) * 2,
    ))
    psf = np.abs(ft) ** 2
    return psf / psf.sum()
