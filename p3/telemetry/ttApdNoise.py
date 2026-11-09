"""Keck STRAP quad-cell APD measurement-noise audit.

The detector is a four-APD quad cell. This module implements the ratio
centroid and its first-order (delta-method) propagation from independent
detector count variances. The four-value APD_IMAT layout was verified
numerically on the Keck 2020-12-02 n0013 telemetry: this is NOT proof
of a separately calibrated 2020 on-sky optical centroid gain.

Notation, for frame k and APD i:
  C_ki = observed raw counter in detector units;
  B_i  = recorded, assumed fixed sky background mean;
  z_ki = C_ki - B_i;   F_k = sum_i(z_ki)
  q_kj = sum_i(A_ji*z_ki)/F_k;
  y_k  = M*q_k, in arcsec according to the archived APD_IMAT convention.
  d q_kj/d C_ki = (A_ji-q_kj)/F_k.
  C_y,k = (M J_k) C_counts,k (M J_k)^T.

The independent-Poisson baseline uses C_counts = diag(C_k).
For a detector calibrated to p effective photoelectrons per recorded count
and APD excess-noise factor F_excess, the first-order *hypothesis* is
  Var(C_ki) = F_excess * C_ki/p + sigma_read,i²
              + sigma_background_estimate,i².
The model assumes the count unit, gain, background and channel independence;
these are not established for the n0013 archive. This estimator is only
a physical *cross-check*, never the default reconstruction noise model.

Sources:
- van Dam et al. (2006), PASP 118, 310, §2.4: Keck 4-APD STRAP;
  highlights Monte Carlo evaluation due to fluctuations of denominator.
- Looze, van Dam & Johansson (2006), Proc. SPIE 6272 62724L,
  Eq. (2): normalized quad-cell reconstruction.
- Keck APD_IMAT 2020 empirical layout: archived n0013 SAV and
  the independent exact-centroid reproduction regression.
"""
from __future__ import annotations

import numpy as np

from p3.aoSystem.tipTiltGeometry import zernike_tilt_opd_per_arcsec


QUADRANT_PATTERNS = np.array([
    [1.0, 1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0, 1.0],
], dtype=float)


def apd_imat_matrix(apd_imat):
    """2020 Keck archived 4-element IMAT -> 2x2 angular response [arcsec]."""
    values = np.asarray(apd_imat, dtype=float)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("APD_IMAT must contain four finite coefficients")
    a, b, c, d = values.reshape(4)
    return np.array([[-b, a], [-d, c]], dtype=float)


def _validate_counts(raw_counts, sky_background):
    raw = np.asarray(raw_counts, dtype=float)
    sky = np.asarray(sky_background, dtype=float).reshape(-1)
    if raw.ndim != 2 or raw.shape[1] != 4 or len(raw) < 2:
        raise ValueError("APDCOUNTS must be a (frames>=2, 4) array")
    if sky.shape != (4,):
        raise ValueError("APD_SKY_BACK must have four detector values")
    if (not np.isfinite(raw).all() or not np.isfinite(sky).all()
            or np.any(raw < 0) or np.any(sky < 0)):
        raise ValueError("APD counts and sky means must be finite/non-negative")
    net_total = np.sum(raw-sky[None, :], axis=1)
    if np.any(net_total <= 0):
        raise ValueError("Background-subtracted total flux must be positive")
    return raw, sky, net_total


def normalized_quad_cell(raw_counts, sky_background):
    """Return normalized two-axis contrasts, net photon-counter flux/frame."""
    raw, sky, flux = _validate_counts(raw_counts, sky_background)
    quadrants = (raw-sky[None, :]) @ QUADRANT_PATTERNS.T
    return quadrants/flux[:, None], flux


def reproduce_keck_centroids(
    raw_counts, sky_background, apd_imat, recorded_centroids_arcsec,
    *, max_rms_error_arcsec=2e-6,
):
    """Fail closed if the archived APD layout cannot reproduce DTTCENTROIDS."""
    quad, flux = normalized_quad_cell(raw_counts, sky_background)
    model = quad @ apd_imat_matrix(apd_imat).T
    recorded = np.asarray(recorded_centroids_arcsec, dtype=float)
    if recorded.shape != model.shape or not np.isfinite(recorded).all():
        raise ValueError("DTTCENTROIDS must be finite and shape (nframes, 2)")
    difference = model-recorded
    rms = np.sqrt(np.mean(difference**2, axis=0))
    if np.any(rms > max_rms_error_arcsec):
        raise ValueError(
            "APD_IMAT/quad order or centroid unit does not reproduce "
            "archived DTTCENTROIDS: RMS [arcsec] = "
            + str(rms.tolist())
        )
    return {
        "centroid_rms_error_arcsec": rms.tolist(),
        "centroid_max_abs_error_arcsec":
            np.max(np.abs(difference), axis=0).tolist(),
        "mean_net_counts_per_frame": float(np.mean(flux)),
        "min_net_counts_per_frame": float(np.min(flux)),
        "centroid_reproduction_valid": True,
    }


def _four_positive_or_zero(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(4, float(array))
    if array.shape != (4,) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be a nonnegative scalar or length-four vector")
    return array


def apd_delta_method_covariance(
    raw_counts, sky_background, apd_imat, *,
    telescope_diameter_m=10.5,
    effective_photoelectrons_per_count=1.0,
    excess_noise_factor=1.0,
    read_noise_variance_counts2=0.0,
    sky_estimation_variance_counts2=0.0,
):
    """Return mean centroid noise covariance (OPD m²) and transparent metadata.

    By default this is an *unverified* ideal Poisson baseline. If 1 raw
    counter is not 1 effective photon, users must supply an independently
    calibrated photoelectrons/count conversion.

    For an APD with non-Poisson statistics, F_excess and read-noise
    parameters are inputs, not coefficients inferred from the science PSF.
    """
    raw, sky, total = _validate_counts(raw_counts, sky_background)
    matrix = apd_imat_matrix(apd_imat)
    p = float(effective_photoelectrons_per_count)
    F = float(excess_noise_factor)
    if not np.isfinite(p) or p <= 0:
        raise ValueError("Effective photoelectrons per count must be positive")
    if not np.isfinite(F) or F < 1:
        raise ValueError("APD excess-noise factor must be finite and >= 1")
    read = _four_positive_or_zero(read_noise_variance_counts2, "Read-noise variance")
    sky_est = _four_positive_or_zero(
        sky_estimation_variance_counts2, "Sky-estimation variance"
    )
    quad, _ = normalized_quad_cell(raw, sky)
    # Jacobian wrt the four raw counter values, shape (nframes, 2, 4).
    jac_quad = (
        QUADRANT_PATTERNS[None, :, :]-quad[:, :, None]
    ) / total[:, None, None]
    jac_angle = np.einsum("ab,nbc->nac", matrix, jac_quad)
    # This is a conditional first-order APD shot/excess/read-noise model.
    counts_var = F*raw/p + read[None, :] + sky_est[None, :]
    per_frame_angle = np.einsum(
        "nai,ni,nbi->nab", jac_angle, counts_var, jac_angle
    )
    covariance_arcsec2 = per_frame_angle.mean(axis=0)
    opd_scale = zernike_tilt_opd_per_arcsec(telescope_diameter_m)
    covariance_opd_m2 = covariance_arcsec2 * opd_scale**2
    return covariance_opd_m2, {
        "method": "apd_delta_poisson_baseline",
        "units": "m² RMS Zernike TT OPD",
        "counts_variance_units": "(APD counter units)²",
        "sample_count": int(len(raw)),
        "telescope_diameter_m": float(telescope_diameter_m),
        "effective_photoelectrons_per_count": p,
        "excess_noise_factor": F,
        "read_noise_variance_counts2": read.tolist(),
        "sky_estimation_variance_counts2": sky_est.tolist(),
        "centroid_covariance_arcsec2": covariance_arcsec2.tolist(),
        "noise_covariance_m2": covariance_opd_m2.tolist(),
        "noise_covariance_eigenvalues_m2":
            np.linalg.eigvalsh(covariance_opd_m2).tolist(),
        "mean_gross_counts_per_frame": float(raw.sum(axis=1).mean()),
        "mean_net_counts_per_frame": float(total.mean()),
        "noise_model_validated": False,
        "apd_gain_independently_calibrated": False,
        "apd_excess_noise_independently_calibrated": False,
        "background_uncertainty_independently_calibrated": False,
        "assumptions": [
            "Four independent APD channels, no cross-channel count covariance",
            "Background means treated as known; extra variance input separately",
            "Poisson-like count variance with specified excess-noise factor",
            "First-order delta method for a random normalized denominator",
            "Archive APD_IMAT layout validated against recorded centroids",
            "Not a calibration of science-camera tip-tilt or optical centroid gain",
        ],
    }


def apd_poisson_monte_carlo(
    raw_counts, sky_background, apd_imat, *,
    telescope_diameter_m=10.5,
    effective_photoelectrons_per_count=1.0,
    trials_per_frame=256,
    seed=0,
    frame_batch=128,
):
    """Monte Carlo of shot-noise + stochastic ratio denominator.

    Pure Poisson *baseline only*: no APD avalanche excess/read noise, no
    uncertain sky estimator. Simulates independent raw detector photon
    counts around the observed per-frame count rates, recenters each
    frame's simulated centroid distribution, then averages its sample
    covariance. This is a plug-in estimate conditional on measured rates.
    """
    raw, sky, flux = _validate_counts(raw_counts, sky_background)
    trials = int(trials_per_frame)
    if trials < 16 or frame_batch < 1:
        raise ValueError("Monte Carlo needs >=16 trials and positive batch size")
    p = float(effective_photoelectrons_per_count)
    if not np.isfinite(p) or p <= 0:
        raise ValueError("Effective photoelectrons per count must be positive")
    transform = apd_imat_matrix(apd_imat)
    rng = np.random.default_rng(seed)
    total_cov = np.zeros((2, 2), dtype=float)
    valid_frames = 0
    invalid_draws = 0
    for start in range(0, len(raw), int(frame_batch)):
        counts = raw[start:start+frame_batch]
        stochastic_raw = rng.poisson(
            counts[:, None, :] * p,
            size=(len(counts), trials, 4),
        ) / p
        net = stochastic_raw - sky[None, None, :]
        flux_draw = np.sum(net, axis=-1)
        valid = flux_draw > 0
        invalid_draws += int(np.count_nonzero(~valid))
        # It is invalid to discard a variable fraction of MC realizations
        # and still claim unbiased Poisson covariance. Require at least
        # 90% draws on *every* frame; otherwise calibration is unreliable.
        fractions = np.mean(valid, axis=1)
        if np.any(fractions < 0.90):
            raise ValueError(
                "More than 10% of Poisson draws have nonpositive net flux "
                "for some frames; normalized APD noise is not identifiable"
            )
        for i in range(len(counts)):
            frame_draws = net[i, valid[i]]
            q = (frame_draws @ QUADRANT_PATTERNS.T) / (
                frame_draws.sum(axis=1)[:, None]
            )
            centroids = q @ transform.T
            centered = centroids - centroids.mean(axis=0)
            total_cov += centered.T @ centered / max(len(centroids)-1, 1)
            valid_frames += 1
    covariance_arcsec2 = total_cov / valid_frames
    scale = zernike_tilt_opd_per_arcsec(telescope_diameter_m)
    covariance_opd_m2 = covariance_arcsec2 * scale**2
    return covariance_opd_m2, {
        "method": "apd_poisson_monte_carlo_baseline",
        "sample_count": int(len(raw)),
        "trials_per_frame": trials,
        "seed": int(seed),
        "nonpositive_denominator_draws": invalid_draws,
        "nonpositive_denominator_fraction":
            invalid_draws/(len(raw)*trials),
        "mean_net_counts_per_frame": float(np.mean(flux)),
        "noise_covariance_m2": covariance_opd_m2.tolist(),
        "noise_covariance_arcsec2": covariance_arcsec2.tolist(),
        "effective_photoelectrons_per_count": p,
        "noise_model_validated": False,
        "apd_gain_independently_calibrated": False,
        "assumptions": [
            "Independent Poisson raw APD counts, conditioned on observed rates",
            "No APD excess noise or read noise included",
            "No stochastic sky-background estimate included",
            "Nonlinear denominator Monte Carlo, not first-order Jacobian",
        ],
    }
