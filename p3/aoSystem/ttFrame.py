"""P4 STRAP/NIRC2 coordinate hypotheses and covariance diagnostics.

These routines separate camera-frame orientation from the pupil-mask angle.
They do NOT calibrate Keck's optical STRAP gain, handedness or zero angle.
"""
from __future__ import annotations

import numpy as np

RAD_TO_MAS = 180.0 * 3600.0 * 1000.0 / np.pi
_GAUSSIAN_FWHM = 2 * np.sqrt(2 * np.log(2))


def sensor_frame_frequencies(u, v, angle_deg, parity="native"):
    """Map camera OTF frequencies into STRAP sensor-axis coordinates.

    The convention reproduces the historical P3 mapping for parity='native':
        u_s = cos(a)*u_cam + sin(a)*v_cam
        v_s = -sin(a)*u_cam + cos(a)*v_cam
    With parity='flip_x'/'flip_y', the indicated *sensor* coordinate is
    reflected after the rotation. Rotation and reflection are diagnostics:
    their absolute Keck calibration is not supplied by the pupil angle.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Camera frequency grids must be finite and have equal shapes")
    if not np.isfinite(angle_deg) or parity not in ("native", "flip_x", "flip_y"):
        raise ValueError("Invalid STRAP-to-NIRC2 angle or parity hypothesis")
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    us = c*u + s*v
    vs = -s*u + c*v
    if parity == "flip_x":
        us = -us
    if parity == "flip_y":
        vs = -vs
    return us, vs


def camera_angular_covariance(covariance_opd_m2, diameter_m, angle_deg,
                              parity="native"):
    """Transform a STRAP RMS-Noll OPD covariance into detector-angle rad^2.

    Uses u_sensor = Q*u_camera; hence C_camera=Q.T*C_sensor*Q.
    Under this convention total variance and eigenvalues cannot change.
    """
    c = np.asarray(covariance_opd_m2, dtype=float)
    if c.shape != (2, 2) or not np.isfinite(c).all() or diameter_m <= 0:
        raise ValueError("Expected finite 2x2 OPD covariance and positive diameter")
    if not np.allclose(c, c.T, rtol=1e-10, atol=1e-24):
        raise ValueError("STRAP covariance must be symmetric")
    a = np.deg2rad(float(angle_deg))
    if not np.isfinite(a) or parity not in ("native", "flip_x", "flip_y"):
        raise ValueError("Invalid STRAP frame hypothesis")
    q = np.array([[np.cos(a), np.sin(a)],
                  [-np.sin(a), np.cos(a)]])
    if parity == "flip_x":
        q = np.diag([-1.0, 1.0]) @ q
    elif parity == "flip_y":
        q = np.diag([1.0, -1.0]) @ q
    c_theta = (4.0/float(diameter_m))**2 * c
    return q.T @ c_theta @ q


def angular_covariance_metrics(covariance_opd_m2, diameter_m, angle_deg=0.0,
                               parity="native"):
    """RMS, principal-axis FWHM and trace, with no PSD repair."""
    c = camera_angular_covariance(covariance_opd_m2, diameter_m, angle_deg, parity)
    e, directions = np.linalg.eigh((c+c.T)/2)
    tol = 1e-12 * max(float(np.max(np.abs(e))), 1e-30)
    if np.min(e) < -tol:
        raise ValueError("Physical TT covariance is not positive-semidefinite")
    # Remove only eigensolver-roundoff negativity within the declared tolerance.
    e_nonnegative = np.maximum(e, 0)
    rms = np.sqrt(e_nonnegative)*RAD_TO_MAS
    principal = directions[:, 1]
    return {
        "trace_rad2": float(np.trace(c)),
        "trace_mas2": float(np.trace(c)*RAD_TO_MAS**2),
        "rms_per_axis_equivalent_mas": float(np.sqrt(max(np.trace(c)/2, 0))*RAD_TO_MAS),
        "principal_rms_mas_minor_major": rms.tolist(),
        "principal_fwhm_mas_minor_major": (_GAUSSIAN_FWHM*rms).tolist(),
        "major_axis_angle_camera_deg_mod180": float(
            np.rad2deg(np.arctan2(principal[1], principal[0])) % 180.0
        ) if e_nonnegative[1] > 0 else None,
        "angle_deg_hypothesis": float(angle_deg),
        "parity_hypothesis": parity,
        "angle_calibrated": False,
        "STRAP_optical_gain_calibrated": False,
    }


def psf_aperture_metrics(psf, radii_px=(3, 5, 10, 20)):
    """Model-only finite-field, normalized PSF widths/energy, NOT observed FWHM.

    The second moment is over the entire supplied finite field and may be
    sensitive to PSF halo and field-of-view truncation. Never fit the binary.
    """
    p = np.asarray(psf, dtype=float)
    if (p.ndim != 2 or p.shape[0] != p.shape[1] or p.size == 0
            or not np.all(np.isfinite(p)) or np.min(p) < -1e-12):
        raise ValueError("PSF must be a finite square, non-negative image")
    p = np.maximum(p, 0)
    if np.sum(p) <= 0:
        raise ValueError("PSF contains no flux")
    p /= np.sum(p)
    y, x = np.indices(p.shape)
    # FFT-centred PSFs place the zero-shift pixel at N//2 for even N.
    c = p.shape[0]//2
    r = np.hypot(x-c, y-c)
    mx = float(np.sum(x*p))
    my = float(np.sum(y*p))
    xx, yy = x-mx, y-my
    cov = np.array([[np.sum(p*xx*xx), np.sum(p*xx*yy)],
                    [np.sum(p*xx*yy), np.sum(p*yy*yy)]])
    return {
        "psf_size_px": int(p.shape[0]),
        "normalized_peak": float(np.max(p)),
        "centroid_x_y_px": [mx, my],
        "second_moment_trace_px2": float(np.trace(cov)),
        "second_moment_rms_px": float(np.sqrt(np.trace(cov)/2)),
        "encircled_energy_at_center_px": {
            str(radius): float(np.sum(p[r <= radius])) for radius in radii_px
        },
        "no_science_source_fitting": True,
        "second_moment_fov_dependent": True,
    }
