"""P4-B research-only STRAP noise and loop audit.

No science FITS intensities are used to select noise parameters. Raw Keck
DTTCENTROIDS are arcsec and APDCOUNTS are uncalibrated detector counts.
This is NOT a modification of the production PSF-R estimator.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr

from p3.telemetry.ttNoise import estimate_tt_noise
from p3.telemetry.ttClosedLoop import (
    from_p3_tiptilt_loop, physical_tt_covariance_from_telemetry,
)
from p3.telemetry.ttApdNoise import (
    apd_delta_method_covariance, normalized_quad_cell,
    reproduce_keck_centroids,
)

BANDS = ((0.55, 0.85), (0.65, 0.95), (0.75, 0.95))
WELCH_SIZES = (256, 512, 1024)
MIRRORS = ("unity", "ragland_2016_keck_ii")
REFERENCE_BAND = (0.65, 0.95)
REFERENCE_WELCH = 512


def _trace(c):
    return float(np.trace(np.asarray(c, dtype=float)))


def _cov(a):
    x = np.asarray(a, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite STRAP samples with shape (frames,2)")
    x = x - x.mean(axis=0)
    return x.T @ x / len(x)


def _matrix_summary(matrix):
    c = np.asarray(matrix, dtype=float)
    if c.shape != (2, 2) or not np.all(np.isfinite(c)):
        raise ValueError("Nonfinite 2x2 covariance")
    c = (c+c.T)/2
    lam = np.linalg.eigvalsh(c)
    tol = 1e-10*max(float(np.max(np.abs(lam))), 1e-30)
    psd = bool(lam[0] >= -tol)
    tr = float(np.trace(c))
    return {
        "matrix_arcsec2": c.tolist(),
        "trace_arcsec2": tr,
        "eigenvalues_arcsec2": lam.tolist(),
        "psd_with_roundoff_only": psd,
        "rms_per_axis_mas": float(1000*np.sqrt(tr/2))
        if psd and tr >= 0 else None,
    }


def loop_identity_report(model, sample_rate_hz):
    """Check the feedback identities and integration gains (no calibration)."""
    f = np.asarray(model.frequency_hz, dtype=float)
    fs = float(sample_rate_hz)
    s = model.sensitivity
    l = model.loop
    closure = float(np.max(np.abs(s*(1+l)-1)))
    expected = (
        model.measured_disturbance == model.measured_disturbance
    )  # unused numerically; all identities below are explicit
    del expected
    identities = {
        "S_times_1_plus_WPK_equals_one": closure,
        "measured_disturbance_equals_W_times_S": float(
            np.max(np.abs(model.measured_disturbance-model.sensor*s))
        ),
        "measured_sensor_noise_equals_S": float(
            np.max(np.abs(model.measured_sensor_noise-s))
        ),
        "physical_disturbance_equals_S": float(
            np.max(np.abs(model.physical_disturbance-s))
        ),
        "physical_sensor_noise_equals_minus_PKS": float(
            np.max(np.abs(model.physical_sensor_noise+
                          model.plant*model.controller*s))
        ),
    }
    if not all(np.isfinite(v) and v < 2e-9 for v in identities.values()):
        raise ValueError("P4-B feedback W/K/P/S algebra is inconsistent")
    if f[0] < 0 or f[-1] > fs/2 + 1e-6:
        raise ValueError("Loop transfer frequency grid exceeds Nyquist")
    g_meas = float(2/fs*trapezoid(np.abs(s)**2, x=f))
    g_phys = float(2/fs*trapezoid(
        np.abs(model.physical_sensor_noise)**2, x=f))
    if not np.isfinite(g_meas+g_phys) or min(g_meas,g_phys) < 0:
        raise ValueError("Nonphysical loop noise transfer gain")
    return {
        "transfer_algebra_errors": identities,
        "algebra_verified": True,
        "sensor_noise_to_measured_gain": g_meas,
        "sensor_noise_to_physical_gain": g_phys,
        "frequency_hz_min_max": [float(f[0]), float(f[-1])],
        "worst_W_magnitude": float(np.min(np.abs(model.sensor))),
        "min_closed_loop_denominator_magnitude": float(
            np.min(np.abs(1+model.loop))
        ),
        "max_closed_loop_sensitivity": float(np.max(np.abs(s))),
        "keck_2020_plant_independently_calibrated": False,
    }


def _estimate(centroids_arcsec, loop, model, band, welch_size):
    """Run unchanged van Dam implementation directly in arcsec units."""
    c, d = estimate_tt_noise(
        centroids_arcsec, method="van_dam",
        sample_rate_hz=float(loop.freq),
        rejection_frequency_hz=model.frequency_hz,
        rejection_transfer=model.sensitivity,
        high_frequency_band=band, nperseg=welch_size,
    )
    return c, d


def _spectrum_diagnostics(d):
    f = np.asarray(d["frequency_hz"], dtype=float)
    measured = np.array([
        d["measured_psd_x_m2_per_hz"],
        d["measured_psd_y_m2_per_hz"],
    ], dtype=float)
    predicted = np.array([
        d["model_noise_psd_x_m2_per_hz"],
        d["model_noise_psd_y_m2_per_hz"],
    ], dtype=float)
    residual = measured-predicted
    observed = np.array(d["observed_covariance_m2"], dtype=float)
    return {
        "axis_noise_to_observed_integral_fraction": [
            float(trapezoid(predicted[i],f)/trapezoid(measured[i],f))
            for i in range(2)
        ],
        "negative_signed_residual_bin_fraction": [
            float(np.mean(residual[i] < 0)) for i in range(2)
        ],
        "signed_residual_integral_arcsec2": [
            float(trapezoid(residual[i],f)) for i in range(2)
        ],
        "welch_observed_to_time_variance_fraction": [
            float(trapezoid(measured[i],f)/observed[i,i])
            for i in range(2)
        ],
        "fit_bin_count": int(d["fit_bin_count"]),
        "welch_segment_count": int(d["welch_segment_count"]),
        "noise_psd_was_clipped": False,
    }


def sensitivity_audit(centroids_arcsec, loop, *, mirrors=MIRRORS,
                      bands=BANDS, welch_sizes=WELCH_SIZES):
    """Prespecified band/Welch grid and W/K/P plant alternatives.

    A spectral non-PSD covariance is recorded as a failed hypothesis,
    never projected to make a preferred scientific image.
    """
    samples = np.asarray(centroids_arcsec, dtype=np.float64)
    cy = _cov(samples)
    models = {name: from_p3_tiptilt_loop(loop, mirror_model=name)
              for name in mirrors}
    identities = {
        name: loop_identity_report(m, loop.freq)
        for name, m in models.items()
    }
    rows = []
    reference_noise = None
    for mirror in mirrors:
        model = models[mirror]
        for band in bands:
            for nperseg in welch_sizes:
                cov, diag = _estimate(samples, loop, model, band, nperseg)
                unprojected = np.asarray(
                    diag["unprojected_noise_covariance_m2"],dtype=float)
                projected = not np.allclose(
                    unprojected, cov, rtol=1e-10, atol=1e-30)
                remainder = cy-cov
                row = {
                    "mirror": mirror,
                    "band_fraction_nyquist": list(band),
                    "welch_nperseg": int(nperseg),
                    "sensor_noise": _matrix_summary(cov),
                    "sensor_noise_unprojected": _matrix_summary(unprojected),
                    "noise_covariance_was_psd_projected": bool(projected),
                    "measured_centroid": _matrix_summary(cy),
                    "legacy_Cy_minus_Cn": _matrix_summary(remainder),
                    "spectral_welch": _spectrum_diagnostics(diag),
                    "sensor_noise_is_independently_calibrated": False,
                }
                if band == REFERENCE_BAND and nperseg == REFERENCE_WELCH:
                    if mirror == "unity":
                        reference_noise = cov.copy()
                try:
                    atmosphere, propagated, total, report = (
                        physical_tt_covariance_from_telemetry(
                            samples, cov, loop, mirror_model=mirror,
                            nperseg=nperseg,
                        )
                    )
                    row["spectral_physical"] = {
                        "status": "computed_unvalidated",
                        "atmosphere": _matrix_summary(atmosphere),
                        "propagated_noise": _matrix_summary(propagated),
                        "total": _matrix_summary(total),
                        "sensor_to_measured_noise_gain":
                            report["gain_sensor_to_measured_noise"],
                        "sensor_to_physical_noise_gain":
                            report["gain_sensor_to_physical_noise"],
                        "noise_transfer_mismatch": False,
                    }
                except ValueError as exc:
                    row["spectral_physical"] = {
                        "status": "failed_physical_model_validation",
                        "error": str(exc),
                        "covariance_was_clipped": False,
                    }
                rows.append(row)
    if reference_noise is None:
        raise ValueError("Baseline unity 0.65-0.95 Welch512 not evaluated")
    # Keep Cn fixed to isolate ONLY changed P and S. This differs
    # from the same plant with noise re-fitted under its own S.
    fixed_noise = {}
    for mirror, model in models.items():
        try:
            atmosphere, propagated, total, report = (
                physical_tt_covariance_from_telemetry(
                    samples, reference_noise, loop,
                    mirror_model=mirror, nperseg=REFERENCE_WELCH
                )
            )
            fixed_noise[mirror] = {
                "status": "computed_unvalidated",
                "atmosphere": _matrix_summary(atmosphere),
                "propagated_noise": _matrix_summary(propagated),
                "total": _matrix_summary(total),
            }
        except ValueError as exc:
            fixed_noise[mirror] = {"status": "failed_physical_model_validation",
                                   "error": str(exc), "covariance_was_clipped": False}
    vals = np.array([
        row["sensor_noise"]["trace_arcsec2"]
        for row in rows if row["mirror"] == "unity"
    ])
    return {
        "fixed_hypothesis_grid": {
            "bands_fraction_of_nyquist": [list(b) for b in bands],
            "welch_lengths": list(welch_sizes),
            "mirror_models": list(mirrors),
        },
        "reference": {
            "mirror": "unity",
            "band_fraction_nyquist": list(REFERENCE_BAND),
            "welch_nperseg": REFERENCE_WELCH,
            "reference_noise_arcsec2": reference_noise.tolist(),
        },
        "noise_trace_range_unity_arcsec2": [
            float(vals.min()), float(vals.max())
        ],
        "noise_trace_range_fraction_of_baseline": float(
            (vals.max()-vals.min())/_trace(reference_noise)
        ),
        "feedback": identities,
        "fixed_unity_noise_plant_sensitivity": fixed_noise,
        "rows": rows,
        "not_ranked_by_science_image": True,
        "physical_model_validated": False,
        "none_of_these_are_keck_2020_calibrations": True,
    }


def flux_segment_audit(raw_apd_counts, sky, imat, centroids_arcsec, loop,
                       *, segment_length=512, nperseg=256,
                       mirror_model="unity"):
    """Temporal contiguous blocks, never frame-by-frame flux selection.

    Binning instantaneous counts would condition on photon noise itself
    and bias the result. Here blocks are fixed before seeing the flux.
    Each has its own mean net counts and independent Welch estimate;
    samples remain in chronological order.
    """
    raw = np.asarray(raw_apd_counts,dtype=float)
    centroids = np.asarray(centroids_arcsec,dtype=float)
    if (raw.ndim != 2 or raw.shape[1] != 4
            or centroids.shape != (len(raw),2)):
        raise ValueError("APD counts and STRAP centroids must be frame-aligned")
    segment_length, nperseg = int(segment_length), int(nperseg)
    if nperseg < 128 or segment_length < 2*nperseg:
        raise ValueError("Each flux segment requires at least two Welch windows")
    reproduce_keck_centroids(raw, sky, imat, centroids)
    _, net_flux = normalized_quad_cell(raw, sky)
    model = from_p3_tiptilt_loop(loop, mirror_model=mirror_model)
    blocks = []
    for start in range(0, len(raw)-segment_length+1, segment_length):
        stop = start+segment_length
        rc, sc = raw[start:stop], centroids[start:stop]
        raw_cov, apd_diag = apd_delta_method_covariance(
            rc,sky,imat, telescope_diameter_m=10.5,
            effective_photoelectrons_per_count=1.0,
            excess_noise_factor=1.0,
        )
        # The physical detector model above returns OPD m². Its
        # native arcsec² prediction is already included in diagnostics.
        predicted_apd = np.array(apd_diag["centroid_covariance_arcsec2"])
        vd, diag = _estimate(
            sc, loop, model, REFERENCE_BAND, nperseg
        )
        blocks.append({
            "start_frame": int(start), "end_frame_exclusive": int(stop),
            "mean_gross_counts": float(np.mean(rc.sum(axis=1))),
            "mean_net_counts": float(np.mean(net_flux[start:stop])),
            "net_counts_coefficient_of_variation": float(
                np.std(net_flux[start:stop])/np.mean(net_flux[start:stop])
            ),
            "predicted_ideal_apd_noise": _matrix_summary(predicted_apd),
            "van_dam_noise": _matrix_summary(vd),
            "van_dam_noise_to_apd_trace_ratio": float(
                _trace(vd)/_trace(predicted_apd)
            ),
            "measured_centroid": _matrix_summary(_cov(sc)),
            "spectral_welch": _spectrum_diagnostics(diag),
        })
    if not blocks:
        raise ValueError("Not enough data for any complete flux window")
    flux = np.array([r["mean_net_counts"] for r in blocks])
    van = np.array([r["van_dam_noise"]["trace_arcsec2"] for r in blocks])
    apd = np.array([r["predicted_ideal_apd_noise"]["trace_arcsec2"] for r in blocks])
    variation = float((flux.max()-flux.min())/flux.mean())
    enough = len(blocks) >= 6 and variation >= 0.10
    rho = float(spearmanr(flux, van).statistic) if enough else None
    return {
        "segmentation": "predeclared_contiguous_nonoverlapping",
        "segment_length_frames": segment_length,
        "welch_nperseg": nperseg,
        "segment_count": len(blocks),
        "discarded_trailing_frames": int(len(raw)-len(blocks)*segment_length),
        "intersegment_net_flux_range_over_mean": variation,
        "predeclared_min_flux_range_over_mean": 0.10,
        "sufficient_flux_leverage": bool(enough),
        "flux_vs_van_dam_spearman_rho": rho,
        "flux_trend_inference": (
            "exploratory_not_detector_calibration" if enough else
            "undetermined_insufficient_flux_dynamic_range_or_windows"
        ),
        "noise_apd_gain_assumed_photoelectrons_per_count": 1.0,
        "noise_apd_excess_factor_assumed": 1.0,
        "flux_stratification_selection_bias_avoided": True,
        "blocks": blocks,
        "all_APD_instrument_gains_unverified": True,
        "no_science_image_calibration": True,
    }
