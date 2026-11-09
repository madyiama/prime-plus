"""Unit-safe tip-tilt geometry and covariance handling for PSF reconstruction.

OPD coefficients are RMS-normalized Zernike tilt amplitudes (metres).
Angles are radians. Covariance projection is a numerical safeguard, not a
substitute for a physically calibrated detector-noise estimate.
"""
from __future__ import annotations

import warnings
import numpy as np


ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def zernike_tilt_opd_per_arcsec(telescope_diameter_m: float) -> float:
    """Metres RMS OPD per arcsec image motion for Z2/Z3 = 2*rho*cos/sin(phi)."""
    if not np.isfinite(telescope_diameter_m) or telescope_diameter_m <= 0:
        raise ValueError("Telescope diameter must be positive and finite")
    return float(telescope_diameter_m) * ARCSEC_TO_RAD / 4.0


def opd_covariance_to_angular(covariance_opd_m2, telescope_diameter_m):
    """Convert RMS Zernike tilt covariance [m^2] to image-angle covariance [rad^2]."""
    if telescope_diameter_m <= 0:
        raise ValueError("Telescope diameter must be positive")
    return np.asarray(covariance_opd_m2, dtype=float) * (4.0 / telescope_diameter_m) ** 2


def project_covariance_to_psd(covariance, label="tip-tilt covariance"):
    """Return a symmetric positive-semidefinite covariance and the *raw* eigenvalues.

    Negative eigenvalues are warned about and remain inspectable in diagnostics.
    The projection makes the Gaussian blur mathematically valid; it cannot
    recover the physical jitter omitted by a biased noise estimator.
    """
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 2x2 covariance matrix")
    sym = 0.5 * (matrix + matrix.T)
    eigenvalues, vectors = np.linalg.eigh(sym)
    scale = max(np.max(np.abs(eigenvalues)), 1e-30)
    if np.min(eigenvalues) < -1e-12 * scale:
        warnings.warn(
            f"{label} is not positive-semidefinite (eigenvalues: "
            f"{eigenvalues.tolist()}); projecting to PSD. Noise model "
            "must be validated independently.",
            RuntimeWarning,
            stacklevel=2,
        )
    clipped = np.maximum(eigenvalues, 0.0)
    projected = (vectors * clipped) @ vectors.T
    return 0.5 * (projected + projected.T), eigenvalues


def angular_jitter_structure_function(
    covariance_rad2, telescope_diameter_m, wavelength_m, u, v
):
    """D_phi = (2*pi*D/lambda)^2 [u,v] C_theta [u,v]^T.

    u and v are dimensionless pupil-baseline coordinates, including optical
    sampling and any instrument-to-pupil rotation. The OTF blur is exp(-D_phi/2).
    """
    cov = np.asarray(covariance_rad2, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError("Expected 2x2 angular covariance")
    if wavelength_m <= 0 or telescope_diameter_m <= 0:
        raise ValueError("Wavelength and telescope diameter must be positive")
    factor = (2.0 * np.pi * telescope_diameter_m / wavelength_m) ** 2
    return factor * (cov[0, 0] * u**2 + 2.0 * cov[0, 1] * u * v + cov[1, 1] * v**2)
