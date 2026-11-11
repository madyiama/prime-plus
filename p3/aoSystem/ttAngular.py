"""P4: unit-explicit Keck STRAP centroids -> angular covariance -> TT OTF.

Numerical identities are independent of van Dam/APD noise calibration.
* raw Keck DTTCENTROIDS: arcsec (2 axes)
* Gaussian jitter C_theta: rad^2 (2x2)
* detector OTF frequency: cycles/radian
* Dphi_tt = 4*pi^2 nu.T C_theta nu
* OTF_tt = exp(-Dphi_tt/2)

A direct angular replay of the historical and spectral covariance
estimators is provided to detect any misplaced D/4 or factor 16.
"""
from __future__ import annotations

import numpy as np

from p3.aoSystem.tipTiltGeometry import (
    ARCSEC_TO_RAD,
    project_covariance_to_psd,
    zernike_tilt_opd_per_arcsec,
)

def raw_centroid_covariance_rad2(centroids_arcsec):
    """Biased N-frame covariance, identical convention to P3's s.T@s/N."""
    a = np.asarray(centroids_arcsec, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite Keck DTTCENTROIDS array (frames,2) in arcsec")
    centered = (a - a.mean(axis=0))*ARCSEC_TO_RAD
    return centered.T@centered/len(centered)


def noll_opd_noise_to_angular(noise_opd_m2, diameter_m):
    """OPD covariance -> rad² only at boundary with legacy noise estimator."""
    a = np.asarray(noise_opd_m2, dtype=float)
    if a.shape != (2, 2) or not np.isfinite(a).all():
        raise ValueError("Finite 2x2 STRAP noise covariance required")
    gain = zernike_tilt_opd_per_arcsec(diameter_m)
    return a * (ARCSEC_TO_RAD/gain)**2


def direct_angular_tt_covariance(
    centroids_arcsec, noise_covariance_opd_m2, diameter_m, *,
    method, pn=None, ttloop=None, mirror_model="unity", nperseg=512,
):
    """Re-evaluate model's TT covariance in rad² from source angles.

    The existing sensor-noise identification remains in OPD units for
    compatibility with the pipeline. Its scale is cancelled at the boundary;
    NO new sensor-noise inference or calibration is performed.

    Legacy path exactly reproduces historical PSD projection and pn.
    Spectral path re-runs the same LTI filter algebra with angular signals.
    """
    values = np.asarray(centroids_arcsec, dtype=float)
    measured = raw_centroid_covariance_rad2(values)
    noise_rad2 = noll_opd_noise_to_angular(noise_covariance_opd_m2, diameter_m)
    if method == "legacy_psfr":
        gain = float(pn) if pn is not None else np.nan
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("Legacy TT requires finite nonnegative pn")
        clean_noise, _ = project_covariance_to_psd(
            noise_rad2, "direct-angular STRAP noise"
        )
        atmospheric, _ = project_covariance_to_psd(
            measured - (1.0+gain)*clean_noise,
            "direct-angular STRAP residual atmosphere"
        )
        return atmospheric + gain*clean_noise, {
            "method": method,
            "measured_centroid_covariance_rad2": measured.tolist(),
            "sensor_noise_covariance_rad2": noise_rad2.tolist(),
            "atmosphere_covariance_rad2": atmospheric.tolist(),
            "propagated_noise_covariance_rad2": (gain*clean_noise).tolist(),
        }
    if method == "spectral_closed_loop":
        if ttloop is None:
            raise ValueError("Spectral STRAP path requires ttloop")
        from p3.telemetry.ttClosedLoop import physical_tt_covariance_from_telemetry
        centered = (values - values.mean(axis=0))*ARCSEC_TO_RAD
        atmosphere, noise, total, report = physical_tt_covariance_from_telemetry(
            centered, noise_rad2, ttloop, mirror_model=mirror_model,
            nperseg=nperseg,
        )
        return total, {
            "method": method,
            "measured_centroid_covariance_rad2": measured.tolist(),
            "sensor_noise_covariance_rad2": noise_rad2.tolist(),
            "atmosphere_covariance_rad2": atmosphere.tolist(),
            "propagated_noise_covariance_rad2": noise.tolist(),
            "spectral_gain_sensor_to_physical_noise": report[
                "gain_sensor_to_physical_noise"
            ],
        }
    raise ValueError("Unknown STRAP TT method: "+str(method))


def assert_equivalent_angular_covariances(legacy_opd_cov_m2, direct_rad2,
                                          diameter_m, rtol=2e-9):
    """Stop if the recorded-centroid route and OPD route disagree."""
    from p3.aoSystem.tipTiltGeometry import opd_covariance_to_angular
    original = opd_covariance_to_angular(legacy_opd_cov_m2, diameter_m)
    direct = np.asarray(direct_rad2, dtype=float)
    if original.shape != (2, 2) or direct.shape != (2, 2):
        raise ValueError("Expected 2x2 TT covariance matrices")
    if not np.isfinite(original).all() or not np.isfinite(direct).all():
        raise ValueError("Nonfinite TT covariance")
    error = float(np.max(np.abs(original-direct)))
    scale = float(max(np.max(np.abs(original)), np.max(np.abs(direct)), 1e-30))
    if error > float(rtol)*scale:
        raise ValueError(
            "TT unit-chain mismatch from DTTCENTROIDS to angular jitter: "
            f"relative max discrepancy={error/scale:.6e}, tolerance={rtol:.3e}. "
            "Check diameter, centroid unit, D/4 and OPD conversion."
        )
    return {"tt_centroid_to_otf_units_verified": True,
            "angular_route_relative_error": error/scale,
            "angular_covariance_rad2": direct.tolist(),
            "angular_covariance_source": "Keck DTTCENTROIDS arcsec, replayed in rad",
            "calibration_of_instrument_gain_verified": False}


def angular_otf_frequencies(freq, diameter_m, wavelength_m, *,
                            pixel_scale_mas=None):
    """Compute two Cartesian angular OTF frequency arrays [cycles/rad].

    P3's arrays U_ (array ROW) and V_ (array COLUMN) represent [-1,1]
    samples. Its dimensionless pupil separation is sampRef*[U,V]/2.
    nu = (D/lambda)*sampRef*[U,V]/2, independent of Noll Zernikes.

    Pixel-scale check is algebraically independent of r0 or TT covariance:
    nu_step = (kRef_float/kRef_integer)/(Npix*pixel_scale_rad).
    The factor kRef_float/kRef_integer allows PSD-expanded grids.
    """
    a, b = np.asarray(freq.U_, dtype=float), np.asarray(freq.V_, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Square matched angular frequency arrays required")
    D, wl = float(diameter_m), float(wavelength_m)
    sample = float(np.asarray(freq.sampRef).reshape(-1)[0])
    if D <= 0 or wl <= 0 or sample <= 0:
        raise ValueError("Invalid optical wavelength, pupil or sampling")
    factor = (D/wl)*(sample/2.0)
    row = factor*a
    col = factor*b
    if pixel_scale_mas is not None:
        px_rad = float(pixel_scale_mas)*1e-3*ARCSEC_TO_RAD
        n = int(a.shape[0])
        k_integer = int(freq.kRef_)
        k_float = float(freq.kRef_float)
        n_pix = int(freq.nPix)
        if px_rad <= 0 or k_integer < 1 or k_float <= 0 or n != n_pix*k_integer:
            raise ValueError("Invalid NIRC2 pixel scale / Fourier grid dimensions")
        indices = np.arange(n, dtype=float)-n//2
        expected = indices[:, None]*np.ones((1,n)) * (
            k_float/k_integer
        )/(n_pix*px_rad)
        expected_col = expected.T
        # This proves array-axis scaling against the detector pixel scale;
        # orientation and parity to STRAP remain instrument-calibration questions.
        scale = max(float(np.max(np.abs(row))), float(np.max(np.abs(col))), 1.0)
        if (np.max(np.abs(row-expected)) > 3e-12*scale
                or np.max(np.abs(col-expected_col)) > 3e-12*scale):
            raise ValueError("TT angular Fourier grid disagrees with NIRC2 pixel scale")
    return row, col


def gaussian_tt_otf(covariance_rad2, frequency_row, frequency_col):
    """Analytic Fourier transform of a zero-mean Gaussian image-motion PDF."""
    C = np.asarray(covariance_rad2, dtype=float)
    r = np.asarray(frequency_row, dtype=float)
    c = np.asarray(frequency_col, dtype=float)
    if C.shape != (2,2) or r.shape != c.shape or r.ndim != 2:
        raise ValueError("Invalid covariance or OTF coordinate arrays")
    if not (np.isfinite(C).all() and np.isfinite(r).all() and np.isfinite(c).all()):
        raise ValueError("Nonfinite jitter OTF inputs")
    if not np.allclose(C, C.T, rtol=1e-12, atol=1e-30):
        raise ValueError("TT covariance must be symmetric")
    eig = np.linalg.eigvalsh((C+C.T)/2)
    if eig.min() < -1e-12*max(float(np.abs(eig).max()), 1e-30):
        raise ValueError("TT angular covariance is not positive-semidefinite")
    q = C[0,0]*r**2 + 2*C[0,1]*r*c + C[1,1]*c**2
    # Positive semidefinite covariance => q >= 0 in exact arithmetic.
    if q.min() < -1e-12*max(float(np.max(np.abs(q))),1e-30):
        raise ValueError("Negative TT Gaussian exponent")
    q = np.maximum(q, 0)
    dphi = 4*np.pi**2*q
    otf = np.exp(-dphi/2)
    return dphi, otf
