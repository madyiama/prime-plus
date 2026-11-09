"""Keck/NIRC2 LGS PSF-R smoke-test with scientifically labelled diagnostic plots.

Run from the repository root with `uv run python scripts/keck_n0013_smoke.py`.
Fits/SAV files remain outside the repository; no uploads occur.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tomllib

import numpy as np


def _plot_comparison(observed, psf_cube, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed = np.asarray(observed, dtype=float)
    psf_cube = np.asarray(psf_cube, dtype=float)
    if psf_cube.ndim == 3:
        psf_model = np.sum(psf_cube, axis=-1)
    elif psf_cube.ndim == 2:
        psf_model = psf_cube
    else:
        raise ValueError("PSF must be 2D or 3D with spectral channels last")

    def normalized(image, subtract_background=False):
        image = image.copy()
        if subtract_background:
            ny, nx = image.shape
            outer = np.ones((ny, nx), dtype=bool)
            outer[ny//4:3*ny//4, nx//4:3*nx//4] = False
            image -= np.nanmedian(image[outer])
        image = np.clip(image, 0, None)
        total = np.sum(image)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Image has no usable positive flux")
        return image / total

    # Uses P3's pre-centred 128-pixel FITS crop.
    observed = normalized(observed, subtract_background=True)
    psf_model = normalized(psf_model)
    assert observed.shape == psf_model.shape, (observed.shape, psf_model.shape)

    fig, ax = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for a, image, title in [
        (ax[0, 0], observed, "NIRC2 — observed binary"),
        (ax[0, 1], psf_model, "PSF-R — single-source model, K-band"),
    ]:
        im = a.imshow(np.log10(np.maximum(image, 1e-7)), vmin=-7, vmax=-1.4,
                      origin="lower", cmap="magma")
        a.set_title(title)
        a.set_xlabel("x [pixels]")
        a.set_ylabel("y [pixels]")
    for a, image, title in [
        (ax[1, 0], observed, "Observed core (48 × 48 px)"),
        (ax[1, 1], psf_model, "Model core (48 × 48 px)"),
    ]:
        ny, nx = image.shape
        cy, cx = np.unravel_index(np.nanargmax(image), image.shape)
        y0, y1 = max(0, cy-24), min(ny, cy+24)
        x0, x1 = max(0, cx-24), min(nx, cx+24)
        a.imshow(np.log10(np.maximum(image[y0:y1, x0:x1], 1e-7)),
                 vmin=-7, vmax=-1.4, origin="lower", cmap="magma")
        a.set_title(title)
        a.set_xlabel("x [pixels]")
        a.set_ylabel("y [pixels]")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.75,
                 label="log10(normalized positive flux per pixel)")
    fig.suptitle("Keck/NIRC2 2020-12-02, LP415-20\n"
                 "Qualitative only: binary image versus telemetry single-star PSF; no source fitting")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _tt_sensitivity(model, observed, destination):
    """Diagnostic only: vary the fraction of estimated TT noise subtracted.

    alpha=1 is the current estimator and alpha=0 is uncorrected centroid jitter.
    Neither endpoint is assumed to be the true on-sky residual covariance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.asarray(model.Ctt, dtype=float)
    noise = np.asarray(model.trs.tipTilt.Cn_tt, dtype=float)
    original_dphi = np.array(model.dphi_tt, copy=True)
    alpha_values = [0.0, 0.75, 0.9, 0.98, 1.0]
    ref_r0 = float(model.ao.atm.r0)
    spectra = []
    records = []

    try:
        for alpha in alpha_values:
            covariance = raw - alpha * noise
            covariance = (covariance + covariance.T) / 2
            eigenvalues = np.linalg.eigvalsh(covariance)
            model.dphi_tt = model.jitter_phase_structure_function(Ctt=covariance)
            cube = np.asarray(model([ref_r0, 1.0, 1.0])).squeeze()
            image = cube.sum(axis=-1) if cube.ndim == 3 else cube
            total = float(np.sum(image))
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f"Invalid model flux at alpha={alpha}")
            image = image / total
            spectra.append(image)
            records.append({
                "alpha": alpha,
                "trace_m2": float(np.trace(covariance)),
                "eigenvalues_m2": eigenvalues.tolist(),
                "covariance_psd": bool(np.all(eigenvalues >= 0)),
                "normalized_psf_peak": float(image.max()),
            })
            label = "PSD" if records[-1]["covariance_psd"] else "NOT PSD"
            print(f"[TT sensitivity] alpha={alpha:.2f}, peak={image.max():.5g}, {label}", flush=True)
    finally:
        model.dphi_tt = original_dphi
        model([ref_r0, 1.0, 1.0])  # Restore original PSF output/state.

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    flat = axes.flat

    observed = np.asarray(observed, dtype=float)
    ny, nx = observed.shape
    mask = np.ones((ny, nx), dtype=bool)
    mask[ny//4:3*ny//4, nx//4:3*nx//4] = False
    observed = np.clip(observed - np.median(observed[mask]), 0, None)
    observed /= observed.sum()
    images = [observed] + spectra
    titles = ["Observed NIRC2 binary (background subtracted)"]
    for row in records:
        validity = "non-PSD!" if not row["covariance_psd"] else "PSD"
        titles.append(f"Noise subtraction: {row['alpha']:.0%} ({validity})")
    for ax, image, title in zip(flat, images, titles):
        ax.imshow(np.log10(np.maximum(image, 1e-7)), vmin=-7, vmax=-1.4,
                  origin="lower", cmap="magma")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Keck n0013 — tip-tilt noise sensitivity (illustrative, no binary fitting)\n"
                 "All other PSF-R terms fixed; alpha scales the TT noise covariance")
    image_path = destination / "tiptilt_sensitivity.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    (destination / "tiptilt_sensitivity.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print("TT sensitivity figure:", image_path, flush=True)


def _plot_tt_noise_spectra(diagnostics, output):
    """Measured STRAP Welch spectra versus fitted loop-shaped noise floors."""
    if diagnostics.get("method") != "van_dam":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq = np.asarray(diagnostics["frequency_hz"])
    rejection = np.asarray(diagnostics["rejection_power"])
    fit_band = diagnostics["high_frequency_band_hz"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             constrained_layout=True)
    for axis, label in enumerate(("x", "y")):
        ax = axes[axis]
        psd = np.asarray(diagnostics[f"measured_psd_{label}_m2_per_hz"])
        fitted = np.asarray(diagnostics[f"model_noise_psd_{label}_m2_per_hz"])
        ax.semilogy(freq, np.maximum(psd, 1e-40), label="Observed STRAP residual PSD")
        ax.semilogy(freq, np.maximum(fitted, 1e-40), label="van Dam fitted noise x |S(f)|²")
        ax.axvspan(fit_band[0], fit_band[1], alpha=0.12, label="Noise-fit band")
        ax.set_ylabel(f"PSD {label.upper()} [m²/Hz]")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Temporal frequency [Hz]")
    fig.suptitle("Keck STRAP — van Dam PSD noise estimate (uncalibrated gain)")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--sav", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/n0013"))
    parser.add_argument("--tt-sensitivity", action="store_true",
                        help="Compare PSFs for different TT noise-subtraction fractions")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "config/keck_tt_noise.toml",
        help="TOML file selecting the STRAP TT noise estimator",
    )
    parser.add_argument("--tt-noise-method",
                        choices=("van_dam", "gendron", "legacy_autocorrelation"),
                        help="Override config's TT-noise estimator")
    parser.add_argument("--tt-noise-band", type=float, nargs=2,
                        help="Override lower/upper fractions of Nyquist for van_dam")
    parser.add_argument("--tt-welch-nperseg", type=int,
                        help="Override Welch segment length for van_dam")
    parser.add_argument(
        "--tt-jitter-model",
        choices=("legacy_psfr", "spectral_closed_loop"),
        help="TT physical covariance used for OTF; overrides TOML",
    )
    parser.add_argument(
        "--tt-mirror-model",
        choices=("unity", "ragland_2016_keck_ii"),
        help="TT actuator model; overrides TOML",
    )
    parser.add_argument(
        "--tt-jitter-welch-nperseg", type=int,
        help="Welch segment length for physical TT spectral covariance",
    )
    parser.add_argument(
        "--disable-tt-jitter", action="store_true",
        help=("Diagnostic ablation only: omit TT jitter from PSF-R OTF, "
              "without changing other components or fitting observations"),
    )
    args = parser.parse_args()
    with args.config.open("rb") as conf_file:
        full_config = tomllib.load(conf_file)
    noise_config = full_config.get("tip_tilt_noise", {})
    tt_method = args.tt_noise_method or noise_config.get("method", "van_dam")
    tt_band = (args.tt_noise_band if args.tt_noise_band is not None
               else noise_config.get("high_frequency_band", [0.65, 0.95]))
    tt_nperseg = (args.tt_welch_nperseg if args.tt_welch_nperseg is not None
                   else noise_config.get("welch_nperseg", 512))
    jitter_config = full_config.get("tip_tilt_jitter", {})
    tt_jitter_model = args.tt_jitter_model or jitter_config.get("method", "legacy_psfr")
    tt_mirror_model = args.tt_mirror_model or jitter_config.get("mirror_model", "unity")
    tt_jitter_nperseg = (
        args.tt_jitter_welch_nperseg if args.tt_jitter_welch_nperseg is not None
        else jitter_config.get("welch_nperseg", tt_nperseg)
    )
    if args.tt_sensitivity and tt_jitter_model != "legacy_psfr":
        parser.error(
            "--tt-sensitivity scales the historical sensor-noise subtraction; "
            "it is only defined for --tt-jitter-model legacy_psfr"
        )
    print(f"[STRAP noise] method={tt_method}, band={tt_band}, Welch={tt_nperseg}",
          flush=True)
    print(f"[TT jitter OTF] method={tt_jitter_model}, mirror={tt_mirror_model}, "
          f"Welch={tt_jitter_nperseg}", flush=True)
    for file in (args.fits, args.sav, args.calib):
        if not file.exists():
            parser.error(f"Not found: {file}")
    args.output.mkdir(parents=True, exist_ok=True)

    from p3.telemetry.telemetryKeck import telemetryKeck
    from p3.telemetry.systemDiagnosis import systemDiagnosis
    from p3.telemetry.configFile import configFile
    from p3.psfr.psfR import psfR

    def stage(name, fn):
        print(f"[{name}] START", flush=True)
        start = time.perf_counter()
        result = fn()
        print(f"[{name}] OK ({time.perf_counter() - start:.1f}s)", flush=True)
        return result

    trs = stage("telemetryKeck", lambda: telemetryKeck(
        str(args.sav.resolve()), str(args.fits.resolve()),
        str(args.calib.resolve()), path_save=str(args.output.resolve()),
        cam_fov=128, nLayer=1, addNCPA=False,
    ))
    sd = stage("systemDiagnosis", lambda: systemDiagnosis(
        trs, tt_noise_method=tt_method, tt_noise_high_band=tt_band,
        tt_noise_nperseg=tt_nperseg,
    ))
    noise_diagnostics = sd.trs.tipTilt.noise_estimation_diagnostics
    (args.output / "tt_noise_diagnostics.json").write_text(
        json.dumps(noise_diagnostics, indent=2), encoding="utf-8",
    )
    if noise_diagnostics.get("method") == "van_dam":
        stage("STRAP noise PSD figure", lambda: _plot_tt_noise_spectra(
            noise_diagnostics, args.output / "tt_noise_psd.png",
        ))
    noise_cov = np.asarray(sd.trs.tipTilt.Cn_tt)
    print(f"[STRAP noise] covariance trace={np.trace(noise_cov):.6e} m²",
          flush=True)
    stage("configFile", lambda: configFile(sd))
    if args.disable_tt_jitter:
        print("[TT jitter OTF] DISABLED for ablation (not a physical model)",
              flush=True)
    model = stage("psfR", lambda: psfR(
        sd.trs, compute_dphi_jitter=not args.disable_tt_jitter,
        tt_covariance_mode=tt_jitter_model,
        tt_mirror_model=tt_mirror_model,
        tt_welch_nperseg=tt_jitter_nperseg,
    ))

    # [r0, gho, gtt] keeps P3's default K-band throughput at all three
    # wavelength bins. Passing [r0, 1, 1, 1, 0, 0, 0, ...] instead assigns
    # the stellar flux to the first bin only.
    psf = np.asarray(stage("psfR forward", lambda: model([
        float(model.ao.atm.r0), 1.0, 1.0,
    ]))).squeeze()
    np.save(args.output / "telemetry_only_psf.npy", psf)
    print("PSF shape:", psf.shape, "min/max:", np.nanmin(psf), np.nanmax(psf))
    print("Error breakdown:")
    for key, value in model.wfe.items():
        print("  ", key, value)

    tt = getattr(model, "tiptilt_diagnostics", {})
    print("Tip-tilt covariance diagnostics:")
    for key, value in tt.items():
        print("  ", key, value)
    (args.output / "tiptilt_diagnostics.json").write_text(
        json.dumps(tt, indent=2), encoding="utf-8",
    )
    stage("comparison figure", lambda: _plot_comparison(
        trs.cam.image, psf, args.output / "comparison_nirc2_psfr.png",
    ))
    print("Figure:", args.output / "comparison_nirc2_psfr.png")
    if args.tt_sensitivity:
        stage("TT noise sensitivity", lambda: _tt_sensitivity(
            model, trs.cam.image, args.output,
        ))
    (args.output / "metadata.json").write_text(json.dumps({
        "input_fits": str(args.fits.resolve()),
        "input_sav": str(args.sav.resolve()),
        "addNCPA": False,
        "tt_noise_method": tt_method,
        "tt_noise_config": str(args.config.resolve()),
        "tt_jitter_covariance_method": tt_jitter_model,
        "tt_mirror_model": tt_mirror_model,
        "tt_jitter_welch_nperseg": tt_jitter_nperseg,
        "tt_jitter_noise_model_validated": False,
        "tt_jitter_otf_enabled": not args.disable_tt_jitter,
        "tt_jitter_ablated_for_diagnostics": bool(args.disable_tt_jitter),
        "target": "LP415-20 (binary)",
        "psf_shape": list(psf.shape),
        "note": "No astrometric registration or binary fit performed",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
