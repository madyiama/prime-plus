"""Independent Keck STRAP APD noise audit; no PSF or Strehl optimization.

The 4-APD Poisson and ratio Monte Carlo calculations are conditional
baselines, NOT independently calibrated 2020 Keck detector gains.

Example:
  uv run python scripts/keck_strap_apd_audit.py \
    --sav "/path/n0013_LGS_trs.sav" \
    --fits "/path/n0013.fits" \
    --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
    --output "$PWD/results/n0013_apd_calibration"
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
from scipy.io import readsav

from p3.telemetry.ttApdNoise import (
    apd_delta_method_covariance,
    apd_poisson_monte_carlo,
    reproduce_keck_centroids,
)
from p3.telemetry.ttNoise import estimate_tt_noise
from p3.telemetry.ttClosedLoop import from_p3_tiptilt_loop


def _cov(samples):
    d = np.asarray(samples, dtype=float)
    d = d-d.mean(axis=0)
    return d.T@d/len(d)


def _plot_report(centroid_noise_covariances, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(centroid_noise_covariances)
    values = [np.diag(centroid_noise_covariances[m]) * 1e12 for m in labels]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis in range(2):
        axes[axis].bar(x, np.array(values)[:,axis], alpha=0.8)
        axes[axis].set_xticks(x, labels, rotation=25, ha="right")
        axes[axis].set_ylabel("Sensor-noise variance [10⁻¹² m²]")
        axes[axis].set_title(("STRAP X", "STRAP Y")[axis])
        axes[axis].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Keck STRAP noise methods: APD physical baseline vs telemetry estimators\n"
        "APD gain, avalanche excess noise, and background-error model unvalidated"
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sav", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/n0013_apd_calibration"))
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] /
                "config/keck_tt_noise.toml",
        help="TOML with strap_apd_reference hypotheses (overridden by flags)",
    )
    parser.add_argument(
        "--mirror-model", choices=("unity", "ragland_2016_keck_ii"),
        default="unity", help="Use the same physical W/K/P model for van Dam",
    )
    parser.add_argument(
        "--photoelectrons-per-count", type=float, default=None,
        help="Photoelectron-equivalent gain; default from explicitly unverified TOML",
    )
    parser.add_argument(
        "--apd-excess-noise-factor", type=float, default=None,
        help="Detector shot-noise excess factor; default from unverified TOML",
    )
    parser.add_argument(
        "--read-noise-variance-counts2", type=float, default=None,
        help="Detector read-noise variance in recorded count units squared",
    )
    parser.add_argument(
        "--sky-estimation-variance-counts2", type=float, default=None,
        help="Uncertainty of subtracted sky-background estimate in count units squared",
    )
    parser.add_argument("--mc-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    for path in (args.sav, args.fits, args.calib, args.config):
        if not path.exists():
            parser.error("Missing Keck file or configuration: " + str(path))
    with args.config.open("rb") as handle:
        apd_config = tomllib.load(handle).get("strap_apd_reference", {})
    if apd_config.get("calibration_status") != "unverified_hypothesis":
        parser.error(
            "Expected strap_apd_reference.calibration_status="
            "'unverified_hypothesis'; independent calibration requires "
            "a separately validated instrument profile."
        )
    if args.photoelectrons_per_count is None:
        args.photoelectrons_per_count = apd_config.get(
            "effective_photoelectrons_per_count", 1.0
        )
    if args.apd_excess_noise_factor is None:
        args.apd_excess_noise_factor = apd_config.get("excess_noise_factor", 1.0)
    if args.read_noise_variance_counts2 is None:
        args.read_noise_variance_counts2 = apd_config.get(
            "read_noise_variance_counts2", 0.0
        )
    if args.sky_estimation_variance_counts2 is None:
        args.sky_estimation_variance_counts2 = apd_config.get(
            "sky_estimation_variance_counts2", 0.0
        )
    if args.mc_trials is None:
        args.mc_trials = apd_config.get("monte_carlo_trials", 128)
    if args.seed is None:
        args.seed = apd_config.get("monte_carlo_seed", 2020)
    args.output.mkdir(parents=True, exist_ok=True)

    from p3.telemetry.telemetryKeck import telemetryKeck
    from p3.telemetry.systemDiagnosis import systemDiagnosis

    sav = readsav(str(args.sav))
    if "b" not in sav:
        raise ValueError("This audit requires 4-APD STRAP LGS telemetry")
    raw = np.asarray(sav["b"][0]["APDCOUNTS"], dtype=float)
    recorded = np.asarray(sav["b"][0]["DTTCENTROIDS"], dtype=float)
    sky = np.asarray(sav["apd_sky_back"], dtype=float)
    imat = np.asarray(sav["apd_imat"], dtype=float)
    independent_reproduction = reproduce_keck_centroids(
        raw, sky, imat, recorded
    )
    print("[STRAP] APD->DTTCENTROIDS reconstruction verified:", 
          independent_reproduction["centroid_rms_error_arcsec"], flush=True)

    baseline, apd_audit = apd_delta_method_covariance(
        raw, sky, imat,
        effective_photoelectrons_per_count=args.photoelectrons_per_count,
        excess_noise_factor=args.apd_excess_noise_factor,
        read_noise_variance_counts2=args.read_noise_variance_counts2,
        sky_estimation_variance_counts2=args.sky_estimation_variance_counts2,
    )
    mc_noise, mc_audit = apd_poisson_monte_carlo(
        raw, sky, imat,
        effective_photoelectrons_per_count=args.photoelectrons_per_count,
        trials_per_frame=args.mc_trials, seed=args.seed,
    )

    # Reconstruct the EXACT STRAP transfer functions used by P3 and use
    # the mirror choice consistently for both noise fitting and propagation.
    trs = telemetryKeck(
        str(args.sav.resolve()), str(args.fits.resolve()),
        str(args.calib.resolve()), path_save=str(args.output.resolve()),
        cam_fov=128, nLayer=1, addNCPA=False,
    )
    sd = systemDiagnosis(
        trs, tt_noise_method="van_dam", tt_mirror_model=args.mirror_model,
    )
    van_dam_cov = np.asarray(sd.trs.tipTilt.Cn_tt)
    van_dam_diag = sd.trs.tipTilt.noise_estimation_diagnostics
    gendron_cov, gendron_diag = estimate_tt_noise(
        sd.trs.tipTilt.slopes, method="gendron"
    )
    model = from_p3_tiptilt_loop(
        sd.trs.ttloop, mirror_model=args.mirror_model
    )
    fs = float(sd.trs.ttloop.freq)
    f = model.frequency_hz
    rejection_gain = 2/fs*np.trapezoid(
        np.abs(model.measured_sensor_noise)**2, x=f
    )
    if abs(rejection_gain - van_dam_diag["mean_rejection_power"]) > 1e-3:
        raise ValueError("Van Dam and physical loop use inconsistent rejection gains")
    observed_cov = _cov(sd.trs.tipTilt.slopes)
    methods = {
        "van_dam": van_dam_cov,
        "gendron": gendron_cov,
        "apd_delta_conditional": baseline,
        "apd_poisson_monte_carlo_conditional": mc_noise,
    }

    comparisons = {}
    for label, sensor_cov in methods.items():
        sensor_cov = np.asarray(sensor_cov, dtype=float)
        predicted_sensor_noise_in_centroids = rejection_gain * sensor_cov
        remainder = observed_cov - predicted_sensor_noise_in_centroids
        comparisons[label] = {
            "sensor_noise_covariance_m2": sensor_cov.tolist(),
            "sensor_noise_trace_m2": float(np.trace(sensor_cov)),
            "sensor_noise_min_eigenvalue_m2": float(
                np.linalg.eigvalsh(sensor_cov).min()
            ),
            "predicted_filtered_noise_in_centroids_m2":
                predicted_sensor_noise_in_centroids.tolist(),
            "atmospheric_remainder_in_centroids_m2": remainder.tolist(),
            "atmospheric_remainder_min_eigenvalue_m2": float(
                np.linalg.eigvalsh((remainder+remainder.T)/2).min()
            ),
        }

    result = {
        "dataset": "Keck II NIRC2 2020-12-02 n0013 LGS",
        "tt_mirror_model": args.mirror_model,
        "apd_config_path": str(args.config.resolve()),
        "apd_calibration_status": apd_config["calibration_status"],
        "noise_rejection_mirror_model":
            sd.trs.tipTilt.noise_rejection_mirror_model,
        "rejection_gain_sensor_to_measured_residual":
            float(rejection_gain),
        "observed_centroid_covariance_m2": observed_cov.tolist(),
        "apd_archive_calibration": {
            "imat_raw": imat.tolist(),
            "sky_background_raw_counts": sky.tolist(),
            "apd_threshold_raw": np.asarray(
                sav.get("apd_threshold", []), dtype=float
            ).reshape(-1).tolist(),
            "raw_count_min": float(raw.min()),
            "raw_count_max": float(raw.max()),
            "raw_count_mean_per_channel": raw.mean(axis=0).tolist(),
            "background_subtracted_flux_mean": 
                independent_reproduction["mean_net_counts_per_frame"],
            "independently_calibrated_2020": False,
        },
        "raw_apd_centroid_reproduction": independent_reproduction,
        "apd_first_order": apd_audit,
        "apd_poisson_monte_carlo": mc_audit,
        "van_dam_summary": {
            "method": "van_dam",
            "sensor_noise_covariance_m2": van_dam_cov.tolist(),
            "fit_band_fraction_nyquist": 
                van_dam_diag["high_frequency_band_fraction_nyquist"],
            "rejection_mirror_model":
                van_dam_diag["rejection_mirror_model"],
            "noise_model_validated": False,
        },
        "gendron_summary": gendron_diag,
        "comparison": comparisons,
        "noise_model_validated": False,
        "science_image_used_for_calibration": False,
        "parameters_from_external_keck_apd_calibration": False,
        "limitations": [
            "Raw APD counts may not be photon counts",
            "Effective photoelectrons per counter unit uncalibrated",
            "APD avalanche excess noise and read noise not calibrated",
            "Sky subtraction uncertainty and APD channel correlations uncalibrated",
            "2020 IMAT numerical reconstruction is not optical centroid gain",
            "Baseline Poisson Monte Carlo excludes excess/read noise",
            "Comparisons address covariance at sensor input only, not science-camera jitter",
            "Gendron temporal estimate need not represent sensor-input noise in closed loop",
            "No model fitted to the NIRC2 science PSF",
        ],
    }
    (args.output/"strap_apd_noise_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    _plot_report(
        methods, result, args.output/"strap_apd_noise_comparison.png",
    )
    md = [
        "# STRAP APD independent detector-noise audit",
        "",
        "**Uncalibrated physical baseline; no science-image fitting.**",
        "",
        f"Mirror/rejection model: `{args.mirror_model}`.",
        f"APD centroid reconstruction RMS [arcsec]: "
        f"{independent_reproduction['centroid_rms_error_arcsec']}.",
        f"Mean net background-subtracted counts/frame: "
        f"{independent_reproduction['mean_net_counts_per_frame']:.2f}.",
        f"Predicted STRAP measured-noise loop gain g_S: {rejection_gain:.6f}.",
        "",
        "| Method | Sensor noise trace [m²] | Residual-in-centroids trace [m²] |",
        "|---|---:|---:|",
    ]
    for method, entry in comparisons.items():
        md.append(
            f"| {method} | {entry['sensor_noise_trace_m2']:.6e} | "
            f"{np.trace(entry['atmospheric_remainder_in_centroids_m2']):.6e} |"
        )
    md += [
        "",
        "The APD first-order model assumes a known effective counter gain, "
        "an excess-noise factor and independent channels. None is "
        "instrumentally verified for the 2020 archive.",
        "Monte Carlo evaluates a **pure Poisson baseline only**, not a "
        "calibrated APD avalanche process.",
        "A positive corrected covariance is not evidence of correct calibration.",
        "Do not select a model because it matches the NIRC2 binary.",
    ]
    (args.output/"strap_apd_noise_report.md").write_text(
        "\n".join(md)+"\n", encoding="utf-8",
    )
    print("[STRAP APD] van Dam Cn trace [m²] %.7e" %
          np.trace(van_dam_cov))
    print("[STRAP APD] delta-method baseline Cn trace [m²] %.7e" %
          np.trace(baseline))
    print("[STRAP APD] Poisson Monte Carlo Cn trace [m²] %.7e" %
          np.trace(mc_noise))
    print("[STRAP APD] Monte Carlo invalid-denominator draws:",
          mc_audit["nonpositive_denominator_draws"])
    print("[STRAP APD] baseline model independently calibrated: NO")
    for name in (
        "strap_apd_noise_audit.json", "strap_apd_noise_comparison.png",
        "strap_apd_noise_report.md",
    ):
        print("Result:", args.output/name)


if __name__ == "__main__":
    main()
