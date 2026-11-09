"""Synthetic regression tests for the van Dam robustness benchmark."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from p3.telemetry.ttNoiseRobustness import (
    DEFAULT_BANDS,
    run_robustness_benchmark,
    save_robustness_outputs,
)


def _synthetic_strap(n=8192, noise_sigma=(2e-7, 3e-7), seed=71):
    fs = 125.0
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, 2))*np.asarray(noise_sigma)
    # Constant closed-loop rejection: S(f) = 0.9.
    residuals = 0.9 * white
    # Add a strong, slowly varying atmospheric residual, deliberately
    # outside the van Dam high-frequency fit band.
    time = np.arange(n)/fs
    residuals += np.column_stack((
        9e-7*np.sin(2*np.pi*0.7*time),
        7e-7*np.cos(2*np.pi*0.9*time),
    ))
    frequencies = np.linspace(0, fs/2, 1024)
    return residuals, fs, frequencies, np.full_like(frequencies, 0.9)


def _run(samples, fs, freq, rejection, **overrides):
    kwargs = dict(
        sample_rate_hz=fs,
        rejection_frequency_hz=freq,
        rejection_transfer=rejection,
        noise_transfer_factor=0.05,
        bands=[(0.65, 0.95), (0.8, 0.98)],
        welch_segments=[256, 512],
        partitions=[2, 4],
        nominal_band=(0.65, 0.95),
        nominal_nperseg=512,
    )
    kwargs.update(overrides)
    return run_robustness_benchmark(samples, **kwargs)


def test_robustness_recovers_white_sensor_noise_and_units():
    samples, fs, freq, rejection = _synthetic_strap()
    report = _run(samples, fs, freq, rejection)
    nominal = report["nominal_full_record"]
    assert report["method"] == "van_dam"
    assert nominal["status"] == "ok"
    assert not report["noise_model_validated"]
    assert report["sample_count"] == len(samples)
    assert report["covariance_units"] == "metres squared OPD"
    # Independent synthetic input white-noise covariance is known.
    assert nominal["sensor_noise_variance_x_m2"] == pytest.approx(
        (2e-7)**2, rel=0.30
    )
    assert nominal["sensor_noise_variance_y_m2"] == pytest.approx(
        (3e-7)**2, rel=0.30
    )
    assert nominal["trace_atmos_measured_m2"] > 0


def test_disjoint_block_dispersion_is_not_mislabeled_confidence_interval():
    data, fs, freq, rejection = _synthetic_strap()
    report = _run(data, fs, freq, rejection)
    assert len(report["interblock_summaries"]) > 0
    for group in report["interblock_summaries"]:
        assert group["valid_block_count"] >= 2
        assert "not confidence interval" in group["uncertainty_type"]
        assert group["trace_sensor_noise_m2_std"] >= 0
        assert (group["trace_sensor_noise_m2_p16"] <=
                group["trace_sensor_noise_m2_p84"])
    full = [r for r in report["cases"] if r["family"] == "full"]
    assert len(full) == 4
    assert all(r["status"] == "ok" for r in full)


def test_short_blocks_are_skipped_rather_than_claiming_precision():
    data, fs, freq, rejection = _synthetic_strap(n=4096)
    report = _run(
        data, fs, freq, rejection,
        welch_segments=[512, 1024],
        partitions=[4],
        nominal_nperseg=512,
    )
    short = [
        row for row in report["cases"]
        if row["family"] == "disjoint_block"
        and row["welch_nperseg"] == 1024
    ]
    assert len(short) == 2*4
    assert all(r["status"] == "skipped" for r in short)
    assert all("Welch segments" in r["reason"] for r in short)


def test_benchmark_is_deterministic_and_requires_valid_band():
    data, fs, freq, rejection = _synthetic_strap(n=4096)
    opts = dict(
        bands=[(0.65, 0.95)], welch_segments=[256],
        partitions=[2], nominal_nperseg=256,
    )
    a = _run(data, fs, freq, rejection, **opts)
    b = _run(data, fs, freq, rejection, **opts)
    assert a["nominal_full_record"]["trace_sensor_noise_m2"] == (
        b["nominal_full_record"]["trace_sensor_noise_m2"])
    with pytest.raises(ValueError, match="Bands"):
        _run(data, fs, freq, rejection, bands=[(0.9, 0.5)])
    with pytest.raises(ValueError, match="positive"):
        _run(data, fs, freq, rejection, noise_transfer_factor=-0.1)


def test_outputs_are_reproducible_and_include_scalar_csv(tmp_path):
    data, fs, freq, rejection = _synthetic_strap(n=4096)
    result = _run(
        data, fs, freq, rejection,
        bands=[(0.65, 0.95)],
        welch_segments=[256],
        partitions=[2],
        nominal_nperseg=256,
    )
    save_robustness_outputs(result, tmp_path)
    for filename in [
        "van_dam_robustness.csv", "van_dam_robustness.json",
        "van_dam_robustness.png", "van_dam_robustness.md",
    ]:
        assert (tmp_path / filename).is_file()
        assert (tmp_path / filename).stat().st_size > 0
    raw = json.loads((tmp_path / "van_dam_robustness.json").read_text())
    assert raw["schema_version"] == 1
    assert not raw["noise_model_validated"]
    with (tmp_path / "van_dam_robustness.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(result["cases"])
    assert rows[0]["status"] == "ok"
    assert "warnings" not in rows[0]


def test_default_bands_cover_high_frequency_sensitivity():
    assert DEFAULT_BANDS[2] == (0.65, 0.95)
    assert len(DEFAULT_BANDS) >= 4
