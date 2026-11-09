"""Audit Keck STRAP TT closed-loop physics WITHOUT modifying PSF reconstruction.

For n0013:
  uv run python scripts/keck_tt_loop_audit.py \
    --fits "/path/n0013.fits" --sav "/path/n0013_LGS_trs.sav" \
    --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
    --output "$PWD/results/n0013_tt_loop"

Optional alternative Keck II mirror response:
  --mirror-model ragland_2016_keck_ii

A sensor-noise estimator alone cannot calibrate a Keck TT controller,
command scale, latency or mirror response. These outputs are conditional
MODEL PREDICTIONS, not validated measurements of science-camera jitter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from p3.telemetry.ttClosedLoop import (
    from_p3_tiptilt_loop,
    measured_tiptilt_cospectrum,
    propagate_spectral_covariances,
)


def _covariance(samples):
    centered = samples-np.mean(samples, axis=0)
    return (centered.T @ centered) / len(centered)


def _resample_cospectrum(f_from, data, f_to):
    """Interpolate Welch co-PSD to the P3 transfer grid, retaining units."""
    return np.array([
        [np.interp(f_to, f_from, data[i,j])
         for j in range(2)] for i in range(2)
    ])


def _plot_audit(freq, observed, model, noise_cov, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = 2*freq[-1]
    noise_psd = 2*np.asarray(noise_cov, dtype=float)/fs
    power_measured_noise = np.abs(model.measured_sensor_noise)**2
    fig, axs = plt.subplots(2, 1, figsize=(10, 7),
                            constrained_layout=True, sharex=True)
    for i, ax in enumerate(axs):
        ax.semilogy(freq, np.maximum(observed[i,i], 1e-38),
                    label="Measured STRAP centroid PSD")
        ax.semilogy(freq,
                    np.maximum(power_measured_noise*noise_psd[i,i], 1e-38),
                    label="Predicted filtered sensor noise |S|² Pn")
        ax.set_ylabel(("X", "Y")[i] + " [m²/Hz]")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
    axs[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Keck STRAP linear feedback — sensor-noise transfer (conditional)")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--sav", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/n0013_tt_loop"))
    parser.add_argument("--mirror-model",
                        choices=("unity", "ragland_2016_keck_ii"),
                        default="unity")
    parser.add_argument("--welch-nperseg", type=int, default=512)
    parser.add_argument("--tt-band", type=float, nargs=2,
                        default=(0.65, 0.95), metavar=("LOW", "HIGH"))
    args = parser.parse_args()
    for f in (args.fits, args.sav, args.calib):
        if not f.exists():
            parser.error(f"Missing input: {f}")
    args.output.mkdir(parents=True, exist_ok=True)
    from p3.telemetry.telemetryKeck import telemetryKeck
    from p3.telemetry.systemDiagnosis import systemDiagnosis

    t0 = time.perf_counter()
    trs = telemetryKeck(
        str(args.sav.resolve()), str(args.fits.resolve()),
        str(args.calib.resolve()), path_save=str(args.output.resolve()),
        cam_fov=128, nLayer=1, addNCPA=False,
    )
    print(f"[telemetryKeck] OK ({time.perf_counter()-t0:.1f}s)", flush=True)
    t0 = time.perf_counter()
    sd = systemDiagnosis(
        trs, tt_noise_method="van_dam",
        tt_noise_high_band=args.tt_band,
        tt_noise_nperseg=args.welch_nperseg,
    )
    print(f"[systemDiagnosis] OK ({time.perf_counter()-t0:.1f}s)", flush=True)

    model = from_p3_tiptilt_loop(sd.trs.ttloop, mirror_model=args.mirror_model)
    measurement = sd.trs.tipTilt.slopes
    sensor_noise = sd.trs.tipTilt.Cn_tt
    fs = float(sd.trs.ttloop.freq)
    welch_f, measured_psd = measured_tiptilt_cospectrum(
        measurement, fs, nperseg=args.welch_nperseg,
    )
    on_model_grid = _resample_cospectrum(
        welch_f, measured_psd, model.frequency_hz,
    )
    results = propagate_spectral_covariances(
        model, on_model_grid, sensor_noise, fs,
    )
    observed_time = _covariance(measurement)
    legacy_atmos = observed_time - (
        1+float(sd.trs.ttloop.tf.pn)
    )*sensor_noise
    results.update({
        "instrument": "Keck II STRAP LGS",
        "science_observation": "n0013 (2020-12-02)",
        "tt_noise_method": "van_dam",
        "tt_noise_fit_band_fraction_nyquist": list(args.tt_band),
        "mirror_model": args.mirror_model,
        "mirror_model_verified_for_2020": False,
        "command_units_verified": bool(
            getattr(sd.trs.tipTilt, "command_units_verified", False)
        ),
        "controller_gain_independently_verified": False,
        "observed_centroid_time_cov_m2": observed_time.tolist(),
        "historical_P3_pn": float(sd.trs.ttloop.tf.pn),
        "historical_P3_corrected_atmos_cov_m2": legacy_atmos.tolist(),
        "historical_P3_corrected_min_eig_m2": float(
            np.linalg.eigvalsh(legacy_atmos).min()
        ),
        "welch_nperseg": args.welch_nperseg,
        "welch_covariance_m2": np.trapezoid(
            measured_psd, x=welch_f, axis=-1
        ).tolist(),
        "interpretation": (
            "Spectrally propagated TT residual in RMS Zernike OPD units. "
            "Not a calibrated prediction of science-camera image motion; "
            "no parameters fitted against NIRC2."
        ),
    })
    (args.output/"tt_closed_loop_audit.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    _plot_audit(
        model.frequency_hz, on_model_grid, model, sensor_noise,
        args.output/"tt_closed_loop_audit.png",
    )
    print("[TT physical-noise gain] %.6f" %
          results["gain_sensor_to_physical_noise"])
    print("[P3 old pn] %.6f" % results["historical_P3_pn"])
    print("[TT physical total cov trace m²] %.7e" %
          np.trace(results["physical_total_cov_m2"]))
    print("[TT physical min eigenvalue m²] %.7e" %
          results["physical_total_min_eig_m2"])
    print("[TT model validated for observation] NO")
    print("Files:",
          args.output/"tt_closed_loop_audit.json",
          args.output/"tt_closed_loop_audit.png")


if __name__ == "__main__":
    main()
