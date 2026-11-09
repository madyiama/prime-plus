"""Robustness benchmark for the Keck STRAP van Dam noise estimator.

This module does not change the default estimator or PSF-R. It repeatedly
estimates sensor noise from the same calibrated STRAP samples across spectral
bands, Welch resolutions and *disjoint* temporal blocks.

Block-to-block dispersions are descriptive sensitivity ranges, not formal
confidence intervals: blocks may be non-stationary or mutually dependent.
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np

from p3.telemetry.ttNoise import estimate_tt_noise


DEFAULT_BANDS = (
    (0.50, 0.80),
    (0.60, 0.85),
    (0.65, 0.95),
    (0.70, 0.90),
    (0.80, 0.98),
)
DEFAULT_WELCH = (256, 512, 1024)
DEFAULT_PARTITIONS = (2, 3, 4)


def _check_inputs(samples, sample_rate_hz, frequency_hz, rejection):
    samples = np.asarray(samples, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    rejection = np.asarray(rejection)
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 128:
        raise ValueError("STRAP samples must have shape (N>=128, 2)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("STRAP samples must be finite")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("STRAP sampling rate must be positive")
    if (frequency_hz.ndim != 1 or rejection.ndim != 1
            or len(frequency_hz) != len(rejection) or len(frequency_hz) < 2
            or not np.all(np.isfinite(frequency_hz))
            or not np.all(np.isfinite(rejection))
            or not np.all(np.diff(frequency_hz) > 0)):
        raise ValueError("Invalid loop rejection frequency response")
    return samples, frequency_hz, rejection


def _covariance(samples):
    centered = samples - np.mean(samples, axis=0)
    return centered.T @ centered / len(centered)


def _estimate_one(
    data, sample_rate_hz, frequency_hz, rejection,
    band, nperseg, trim_fraction, pn, family, blocks, index,
    min_segments,
):
    n = len(data)
    nperseg = int(nperseg)
    segments = (1 + (n - nperseg) // (nperseg // 2)
                if n >= nperseg and nperseg >= 2 else 0)
    record = {
        "family": family,
        "blocks": blocks,
        "block_index": index,
        "samples": int(n),
        "band_low": float(band[0]),
        "band_high": float(band[1]),
        "welch_nperseg": nperseg,
        "welch_segments": segments,
        "status": "skipped",
    }
    if segments < min_segments:
        record["reason"] = (
            f"Only {segments} Welch segments; at least {min_segments} required"
        )
        return record

    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", RuntimeWarning)
            noise, diagnostics = estimate_tt_noise(
                data, method="van_dam", sample_rate_hz=sample_rate_hz,
                rejection_frequency_hz=frequency_hz,
                rejection_transfer=rejection,
                high_frequency_band=band, nperseg=nperseg,
                trim_fraction=trim_fraction,
            )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record.update(status="error", reason=str(exc))
        return record

    measured = _covariance(data)
    filtered_noise = np.asarray(
        diagnostics["predicted_observed_noise_covariance_m2"], dtype=float
    )
    # This is the atmospheric contribution *in measured residual centroids*.
    # It is NOT the physical, on-sky jitter covariance of the mirror.
    atmospheric_measured = measured - filtered_noise
    # Retain the historical P3 algebra as a diagnostic only.
    atmospheric_p3 = measured - (1.0 + pn) * noise
    # A P3 negative eigenvalue should be reported, not silently zeroed.
    eig_measured = np.linalg.eigvalsh(
        (atmospheric_measured + atmospheric_measured.T) / 2
    )
    eig_p3 = np.linalg.eigvalsh((atmospheric_p3 + atmospheric_p3.T) / 2)
    record.update(
        status="ok",
        trace_observed_m2=float(np.trace(measured)),
        trace_sensor_noise_m2=float(np.trace(noise)),
        trace_filtered_noise_m2=float(np.trace(filtered_noise)),
        trace_atmos_measured_m2=float(np.trace(atmospheric_measured)),
        trace_atmos_p3_m2=float(np.trace(atmospheric_p3)),
        min_eig_atmos_measured_m2=float(eig_measured[0]),
        min_eig_atmos_p3_m2=float(eig_p3[0]),
        sensor_noise_variance_x_m2=float(noise[0, 0]),
        sensor_noise_variance_y_m2=float(noise[1, 1]),
        sensor_noise_covariance_xy_m2=float(noise[0, 1]),
        filtered_noise_fraction_x=float(
            diagnostics["predicted_observed_noise_fraction_per_axis"][0]
        ),
        filtered_noise_fraction_y=float(
            diagnostics["predicted_observed_noise_fraction_per_axis"][1]
        ),
        fit_bins=int(diagnostics["fit_bin_count"]),
        warnings=[str(w.message) for w in seen],
    )
    return record


def run_robustness_benchmark(
    samples_opd_m, *, sample_rate_hz, rejection_frequency_hz,
    rejection_transfer, noise_transfer_factor,
    bands=DEFAULT_BANDS, welch_segments=DEFAULT_WELCH,
    partitions=DEFAULT_PARTITIONS, nominal_band=(0.65, 0.95),
    nominal_nperseg=512, trim_fraction=0.10, min_segments=3,
):
    """Run full-record fits and non-overlapping temporal-block fits.

    Returns JSON-serializable results. Full-record estimates and disjoint-block
    estimates are kept separate. Dispersion ranges are *not* formal CIs.
    """
    data, frequencies, rejection = _check_inputs(
        samples_opd_m, sample_rate_hz, rejection_frequency_hz,
        rejection_transfer,
    )
    if not np.isfinite(noise_transfer_factor) or noise_transfer_factor < 0:
        raise ValueError("Invalid tip-tilt noise transfer factor")
    bands = tuple(tuple(float(v) for v in band) for band in bands)
    welch_segments = tuple(sorted(set(int(v) for v in welch_segments)))
    partitions = tuple(sorted(set(int(v) for v in partitions)))
    if not bands or not welch_segments:
        raise ValueError("At least one band and Welch length required")
    if any(len(b) != 2 or not (0 < b[0] < b[1] < 1) for b in bands):
        raise ValueError("Bands must satisfy 0 < low < high < 1")
    if any(n < 128 for n in welch_segments):
        raise ValueError("Welch segment lengths must be at least 128")
    if any(p < 2 for p in partitions):
        raise ValueError("Temporal partitions must contain at least two blocks")
    if min_segments < 2:
        raise ValueError("Require at least two Welch segments per estimate")

    cases = []
    for band in bands:
        for length in welch_segments:
            cases.append(_estimate_one(
                data, sample_rate_hz, frequencies, rejection,
                band, length, trim_fraction, noise_transfer_factor,
                "full", 1, 0, min_segments,
            ))
            for block_count in partitions:
                for block_index, block in enumerate(np.array_split(data, block_count)):
                    cases.append(_estimate_one(
                        block, sample_rate_hz, frequencies, rejection,
                        band, length, trim_fraction, noise_transfer_factor,
                        "disjoint_block", block_count, block_index,
                        min_segments,
                    ))

    keys = ("trace_sensor_noise_m2", "trace_atmos_measured_m2",
            "trace_atmos_p3_m2")
    summaries = []
    for band in bands:
        for length in welch_segments:
            for count in partitions:
                group = [
                    c for c in cases
                    if c["family"] == "disjoint_block"
                    and c["welch_nperseg"] == length
                    and c["blocks"] == count
                    and (c["band_low"], c["band_high"]) == band
                    and c["status"] == "ok"
                ]
                if len(group) < 2:
                    continue
                summary = {
                    "band_low": band[0], "band_high": band[1],
                    "welch_nperseg": length, "blocks": count,
                    "valid_block_count": len(group),
                    "uncertainty_type": "inter-block empirical dispersion, not confidence interval",
                }
                for key in keys:
                    values = np.array([g[key] for g in group])
                    summary[key + "_mean"] = float(values.mean())
                    summary[key + "_std"] = float(values.std(ddof=1))
                    summary[key + "_p16"] = float(np.percentile(values, 16))
                    summary[key + "_p84"] = float(np.percentile(values, 84))
                summaries.append(summary)

    ok_full = [c for c in cases if c["family"] == "full"
               and c["status"] == "ok"]
    if not ok_full:
        raise ValueError("No valid full-record van Dam estimates")
    nominal = next(
        (c for c in ok_full
         if (c["band_low"], c["band_high"]) == tuple(nominal_band)
         and c["welch_nperseg"] == nominal_nperseg),
        None,
    )
    return {
        "schema_version": 1,
        "method": "van_dam",
        "noise_model_validated": False,
        "sample_count": len(data),
        "sampling_frequency_hz": float(sample_rate_hz),
        "noise_transfer_factor_pn": float(noise_transfer_factor),
        "sample_units": "metres RMS Zernike tilt OPD",
        "covariance_units": "metres squared OPD",
        "nominal_band": list(nominal_band),
        "nominal_nperseg": nominal_nperseg,
        "nominal_full_record": nominal,
        "bands": [list(b) for b in bands],
        "welch_nperseg_values": list(welch_segments),
        "partitions": list(partitions),
        "min_welch_segments": min_segments,
        "trim_fraction": float(trim_fraction),
        "cases": cases,
        "interblock_summaries": summaries,
        "methodology_limits": [
            "No independent validation of STRAP centroid gain or controller",
            "White-noise high-frequency hypothesis is unverified",
            "Contiguous blocks may be nonstationary or mutually dependent",
            "Inter-block percentiles are sensitivity ranges, not formal confidence intervals",
            "C_y - g_S C_n is atmospheric covariance IN measured centroids, not on-sky jitter",
            "C_y - (1+pn) C_n is reported for legacy comparison, not endorsed",
        ],
    }


def save_robustness_outputs(result, output_dir):
    """Save compact reproducible JSON/CSV, figure and Markdown summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "van_dam_robustness.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    cases = result["cases"]
    # CSV is scalar-only, without nested warnings.
    scalar_rows = [
        {key: val for key, val in record.items()
         if not isinstance(val, (list, dict))}
        for record in cases
    ]
    fields = list(dict.fromkeys(
        key for row in scalar_rows for key in row
    ))
    with (output / "van_dam_robustness.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)

    _plot_robustness(result, output / "van_dam_robustness.png")
    _write_report(result, output / "van_dam_robustness.md")
    return output


def _plot_robustness(result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = result["bands"]
    lengths = result["welch_nperseg_values"]
    lookup = {
        (c["band_low"], c["band_high"], c["welch_nperseg"]): c
        for c in result["cases"]
        if c["family"] == "full" and c["status"] == "ok"
    }
    noise = np.full((len(lengths), len(bands)), np.nan)
    atmos = np.full_like(noise, np.nan)
    for i, length in enumerate(lengths):
        for j, b in enumerate(bands):
            record = lookup.get((b[0], b[1], length))
            if record:
                noise[i, j] = record["trace_sensor_noise_m2"] * 1e12
                atmos[i, j] = record["trace_atmos_p3_m2"] * 1e12
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    names = ["Sensor noise, trace [10⁻¹² m²]",
             "P3 corrected residual, trace [10⁻¹² m²]"]
    for ax, matrix, title in zip(axes, [noise, atmos], names):
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(bands)),
                      [f"{b[0]:.2f}–{b[1]:.2f}" for b in bands],
                      rotation=35, ha="right")
        ax.set_yticks(range(len(lengths)), [str(n) for n in lengths])
        ax.set_xlabel("Frequency band / Nyquist")
        ax.set_ylabel("Welch nperseg")
        ax.set_title(title)
        for i in range(len(lengths)):
            for j in range(len(bands)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}",
                            ha="center", va="center", fontsize=9,
                            color="white")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("STRAP van Dam: sensitivity to frequency band and Welch length\n"
                 "No PSF fit; no independent sensor-noise calibration")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _write_report(result, destination):
    nominal = result["nominal_full_record"]
    full = [c for c in result["cases"] if c["family"] == "full"
            and c["status"] == "ok"]
    text = [
        "# van Dam STRAP — robustness benchmark",
        "",
        f"Samples: {result['sample_count']}, sampling rate: "
        f"{result['sampling_frequency_hz']:.4f} Hz.",
        "",
        "This report compares sensor-noise estimates; it does not validate "
        "the noise model or optimize Strehl on the observed binary.",
        "",
    ]
    if nominal:
        text.extend([
            "## Nominal setting",
            "",
            f"Band: {nominal['band_low']:.2f}–{nominal['band_high']:.2f} "
            f"of Nyquist; Welch length: {nominal['welch_nperseg']}.",
            f"Sensor-noise trace: {nominal['trace_sensor_noise_m2']:.5e} m².",
            f"Filtered-noise trace in measurements: "
            f"{nominal['trace_filtered_noise_m2']:.5e} m².",
            f"Atmospheric trace in measured centroids: "
            f"{nominal['trace_atmos_measured_m2']:.5e} m².",
            f"P3 legacy corrected trace: "
            f"{nominal['trace_atmos_p3_m2']:.5e} m².",
            "",
        ])
    text += [
        "## Full-record sensitivity",
        "",
        "| Band / Nyquist | Welch | Sensor noise trace (10⁻¹² m²) | "
        "P3-corrected trace (10⁻¹² m²) |",
        "|---|---:|---:|---:|",
    ]
    for c in full:
        text.append(
            f"| {c['band_low']:.2f}–{c['band_high']:.2f} | "
            f"{c['welch_nperseg']} | "
            f"{1e12*c['trace_sensor_noise_m2']:.3f} | "
            f"{1e12*c['trace_atmos_p3_m2']:.3f} |"
        )
    text += [
        "",
        "## Statistical interpretation",
        "",
        "Disjoint temporal-block runs and empirical 16–84% dispersion ranges are "
        "available in JSON. These are **not** formal confidence intervals: "
        "the signal is short, blocks may not be stationary, and the loop "
        "model/gain have not been independently calibrated.",
        "",
        "The P3-corrected trace is a **diagnostic of the historical algebra**, "
        "not a validated estimate of on-sky atmospheric jitter.",
        "",
        "## Next scientific gate",
        "",
        "Validate S(f) and noise-to-mirror T(f) in a closed-loop simulator; "
        "then compare van Dam to independent APD counting statistics.",
        "",
    ]
    destination.write_text("\n".join(text), encoding="utf-8")
