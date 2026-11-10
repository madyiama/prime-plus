"""Keck/NIRC2 PSF-R smoke-test (LGS and NGS) with labelled diagnostics.

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


def _central_crop(image, width=48):
    """Fixed geometric centre crop: do not chase stars/hot pixels."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Central crop expects a two-dimensional image")
    ny, nx = image.shape
    height = min(int(width), ny)
    width = min(int(width), nx)
    y0 = (ny - height) // 2
    x0 = (nx - width) // 2
    return image[y0:y0 + height, x0:x0 + width]


def _plot_comparison(observed, psf_cube, destination, *, dataset_label="Keck/NIRC2"):
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
        (ax[0, 0], observed, "NIRC2 — observed field"),
        (ax[0, 1], psf_model, "PSF-R — single-source model, K-band"),
    ]:
        im = a.imshow(np.log10(np.maximum(image, 1e-7)), vmin=-7, vmax=-1.4,
                      origin="lower", cmap="magma")
        a.set_title(title)
        a.set_xlabel("x [pixels]")
        a.set_ylabel("y [pixels]")
    for a, image, title in [
        (ax[1, 0], observed, "Observed — central 48 × 48 px"),
        (ax[1, 1], psf_model, "Model — central 48 × 48 px"),
    ]:
        core = _central_crop(image, width=48)
        a.imshow(np.log10(np.maximum(core, 1e-7)),
                 vmin=-7, vmax=-1.4, origin="lower", cmap="magma")
        a.set_title(title)
        a.set_xlabel("x [pixels]")
        a.set_ylabel("y [pixels]")
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.75,
                 label="log10(normalized positive flux per pixel)")
    fig.suptitle(f"{dataset_label}\n"
                 "Qualitative only: observed field versus single-star PSF; "
                 "no source fitting or astrometric registration")
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
    titles = ["Observed NIRC2 field (background subtracted)"]
    for row in records:
        validity = "non-PSD!" if not row["covariance_psd"] else "PSD"
        titles.append(f"Noise subtraction: {row['alpha']:.0%} ({validity})")
    for ax, image, title in zip(flat, images, titles):
        ax.imshow(np.log10(np.maximum(image, 1e-7)), vmin=-7, vmax=-1.4,
                  origin="lower", cmap="magma")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Keck NIRC2 — tip-tilt noise sensitivity (illustrative, no image fitting)\n"
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


def select_tt_jitter_mode(ao_mode, requested_mode, *, explicitly_requested=False):
    """Select TT covariance mode by guide-star sensor, without silent fallback."""
    if str(ao_mode).upper() == "NGS" and requested_mode == "spectral_closed_loop":
        if explicitly_requested:
            raise ValueError(
                "spectral_closed_loop requires Keck LGS/STRAP telemetry. "
                "For this NGS file use --tt-jitter-model legacy_psfr "
                "(NGS TT units uncalibrated), or --disable-tt-jitter "
                "for a high-order-only control."
            )
        return (
            "legacy_psfr",
            "NGS spectral TT is unsupported; using legacy_psfr for an "
            "exploratory, uncalibrated NGS reconstruction",
        )
    return requested_mode, None


def _ho_noise_endpoint_audit(trs):
    """Compare finite-record and circular autocorrelation HO-noise estimates.

    Runs before psfR initialization, so diagnostics survive PSD rejection.
    These are noise *proxies*, not independent instrumental calibrations.
    """
    from p3.telemetry.systemDiagnosis import autocorrelation_noise_covariance

    commands = np.asarray(trs.dm.com, dtype=float)
    rate_gain = float(np.asarray(trs.holoop.gain).reshape(-1)[0])
    if commands.ndim != 2 or len(commands) < 3 or not np.isfinite(rate_gain) or rate_gain == 0:
        raise ValueError("Cannot diagnose HO covariance: invalid DM command array or loop gain")
    delta = np.diff(commands, axis=0) / rate_gain
    command_difference_cov = delta.T @ delta / len(delta)
    periodic = autocorrelation_noise_covariance(commands, periodic=True)
    finite = autocorrelation_noise_covariance(commands, periodic=False)
    actual = np.asarray(trs.wfs.Cn_ao, dtype=float)

    def spectrum(noise):
        centered = command_difference_cov - noise
        centered = 0.5 * (centered + centered.T)
        eigenvalues = np.linalg.eigvalsh(centered)
        neg = -eigenvalues[eigenvalues < -1e-12 * max(
            float(np.max(np.abs(eigenvalues))), 1e-30
        )]
        positive = eigenvalues[eigenvalues > 0]
        return {
            "min_eigenvalue_m2": float(eigenvalues.min()),
            "max_eigenvalue_m2": float(eigenvalues.max()),
            "negative_count": int(len(neg)),
            "max_negative_to_max_positive": float(
                np.max(neg) / np.max(positive)
            ) if len(neg) and len(positive) else 0.0,
            "negative_mass_to_positive_trace": float(
                np.sum(neg) / np.sum(positive)
            ) if len(neg) and len(positive) else 0.0,
        }

    n = len(commands)
    closing_jump = commands[-1] - commands[0]
    return {
        "ao_mode": str(trs.aoMode),
        "frames": n,
        "ho_loop_gain": rate_gain,
        "units": "m2 OPD covariance before phase-radian scaling",
        "noise_estimator_warning": (
            "Both estimates come from command first-differences, "
            "not an independent calibrated detector-noise measurement"
        ),
        "cyclic_closing_jump_opd_norm_m": float(np.linalg.norm(closing_jump)),
        "closing_jump_covariance_trace_m2": float(
            (closing_jump @ closing_jump) / (2 * n)
        ),
        "noise_trace_periodic_m2": float(np.trace(periodic)),
        "noise_trace_nonperiodic_m2": float(np.trace(finite)),
        "noise_trace_used_m2": float(np.trace(actual)),
        "covariance_after_subtraction_periodic": spectrum(periodic),
        "covariance_after_subtraction_nonperiodic": spectrum(finite),
        "covariance_after_subtraction_used": spectrum(actual),
    }


def _ho_covariance_preflight(trs):
    """Analyze HO subtraction before psfR, including on strict-mode failures.

    Eigenvalue ratios are invariant to the positive (2*pi/lambda)^2 factor.
    Diagnostics use actuator OPD m^2, not phase rad^2.
    """
    commands = np.asarray(trs.dm.com, dtype=float)
    noise = np.asarray(trs.wfs.Cn_ao, dtype=float)
    gain = float(np.asarray(trs.holoop.gain).reshape(-1)[0])
    if not np.isfinite(gain) or gain == 0 or commands.ndim != 2 or len(commands) < 2:
        raise ValueError("Cannot evaluate HO covariance: invalid loop gain or commands")
    diffs = np.diff(commands, axis=0) / gain
    measured = diffs.T @ diffs / len(diffs)
    subtracted = 0.5 * (measured - noise + (measured - noise).T)
    eig = np.linalg.eigvalsh(subtracted)
    tolerance = max(1e-12 * float(np.max(np.abs(eig))), 1e-24)
    negatives = -eig[eig < -tolerance]
    positives = eig[eig > 0]
    largest = float(positives.max()) if len(positives) else 0.0
    pos_mass = float(np.sum(positives))
    worst = float(np.max(negatives)) / largest if len(negatives) and largest > 0 else 0.0
    mass = float(np.sum(negatives)) / pos_mass if pos_mass > 0 else 0.0
    return {
        "ao_mode": str(trs.aoMode),
        "units": "OPD m^2 (not phase radians^2)",
        "commands_frames": int(len(commands)),
        "ho_loop_gain": gain,
        "command_covariance_trace_m2": float(np.trace(measured)),
        "noise_covariance_trace_m2": float(np.trace(noise)),
        "subtracted_covariance_trace_m2": float(np.trace(subtracted)),
        "lambda_min_m2": float(eig.min()),
        "lambda_max_m2": float(eig.max()),
        "negative_mode_count": int(len(negatives)),
        "negative_mode_fraction_of_lambda_max": worst,
        "negative_trace_fraction_of_positive_mass": mass,
        "tiny_negative_policy_would_pass": bool(
            worst <= 1e-5 and mass <= 1e-5
        ),
        "scientific_status": (
            "Diagnostic of finite-sample covariance subtraction, not an "
            "independent validation of the WFS noise estimator"
        ),
    }


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
        "--ho-covariance-policy",
        choices=("strict", "tiny_negative", "small_negative"),
        default="strict",
        help=(
            "HO modal covariance: strict (default) rejects significant negative "
            "eigenvalues; tiny_negative permits explicitly warned projection "
            "if both maximum and cumulative negative spectral fractions are "
            "<= 0.001% (LGS/NGS); small_negative permits <= 0.5% (NGS only). "
            "Neither policy validates the sensor-noise estimator."
        ),
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
    print(f"[TT noise request] method={tt_method}, band={tt_band}, Welch={tt_nperseg}",
          flush=True)
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
    filter_profile_diagnostics = getattr(
        trs.cam, "filter_profile_diagnostics", {}
    )
    if filter_profile_diagnostics:
        (args.output / "filter_profile_diagnostics.json").write_text(
            json.dumps(filter_profile_diagnostics, indent=2),
            encoding="utf-8",
        )
        print("[NIRC2 filter]", filter_profile_diagnostics, flush=True)
    try:
        tt_jitter_model, model_policy_note = select_tt_jitter_mode(
            trs.aoMode, tt_jitter_model,
            explicitly_requested=(
                args.tt_jitter_model == "spectral_closed_loop"
                and not args.disable_tt_jitter
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if model_policy_note:
        print(f"[TT jitter OTF] {model_policy_note}", flush=True)
    if args.tt_sensitivity and tt_jitter_model != "legacy_psfr":
        parser.error(
            "--tt-sensitivity scales the historical sensor-noise subtraction; "
            "it is only defined for --tt-jitter-model legacy_psfr"
        )
    if str(trs.aoMode).upper() == "NGS":
        print("[NGS TT] The STRAP noise option is not used; systemDiagnosis "
              "uses its historical NGS autocorrelation estimator. "
              "NGS RESIDUALWAVEFRONT TT units are not independently verified.",
              flush=True)
    print(f"[TT jitter OTF] mode={tt_jitter_model}, mirror={tt_mirror_model}, "
          f"Welch={tt_jitter_nperseg}", flush=True)
    # The noise estimator and the physical jitter must share a single
    # W/K/P rejection model. For legacy PSF-R, keep the historical unity
    # transfer regardless of an unused TT mirror selection.
    noise_mirror = (
        tt_mirror_model if tt_jitter_model == "spectral_closed_loop"
        else "unity"
    )
    print(f"[STRAP van Dam] rejection mirror={noise_mirror}", flush=True)
    sd = stage("systemDiagnosis", lambda: systemDiagnosis(
        trs, tt_noise_method=tt_method, tt_noise_high_band=tt_band,
        tt_noise_nperseg=tt_nperseg, tt_mirror_model=noise_mirror,
    ))
    noise_diagnostics = sd.trs.tipTilt.noise_estimation_diagnostics
    effective_tt_noise_method = sd.trs.tipTilt.noise_estimation_method
    print(f"[TT noise effective] ao_mode={trs.aoMode}, "
          f"estimator={effective_tt_noise_method}", flush=True)
    (args.output / "tt_noise_diagnostics.json").write_text(
        json.dumps(noise_diagnostics, indent=2), encoding="utf-8",
    )
    if noise_diagnostics.get("method") == "van_dam":
        stage("STRAP noise PSD figure", lambda: _plot_tt_noise_spectra(
            noise_diagnostics, args.output / "tt_noise_psd.png",
        ))
    noise_cov = np.asarray(sd.trs.tipTilt.Cn_tt)
    print(f"[TT noise] covariance trace={np.trace(noise_cov):.6e} m²",
          flush=True)
    stage("configFile", lambda: configFile(sd))
    ho_preflight = _ho_covariance_preflight(sd.trs)
    (args.output / "ho_covariance_preflight.json").write_text(
        json.dumps(ho_preflight, indent=2), encoding="utf-8"
    )
    print("[HO covariance preflight] worst negative fraction=",
          ho_preflight["negative_mode_fraction_of_lambda_max"],
          "negative trace fraction=",
          ho_preflight["negative_trace_fraction_of_positive_mass"], flush=True)
    if str(trs.aoMode).upper() == "NGS":
        ho_endpoint = _ho_noise_endpoint_audit(sd.trs)
        (args.output / "ho_noise_endpoint_audit.json").write_text(
            json.dumps(ho_endpoint, indent=2), encoding="utf-8"
        )
        print("[NGS HO noise endpoint audit]",
              ho_endpoint["covariance_after_subtraction_periodic"],
              "vs finite",
              ho_endpoint["covariance_after_subtraction_nonperiodic"],
              flush=True)
    if args.ho_covariance_policy == "small_negative" and str(trs.aoMode).upper() != "NGS":
        parser.error(
            "--ho-covariance-policy small_negative is NGS-only; "
            "use tiny_negative for a tighter LGS/NGS projection"
        )
    print(f"[HO covariance] policy={args.ho_covariance_policy}", flush=True)
    if args.disable_tt_jitter:
        print("[TT jitter OTF] DISABLED for ablation (not a physical model)",
              flush=True)
    model = stage("psfR", lambda: psfR(
        sd.trs, compute_dphi_jitter=not args.disable_tt_jitter,
        tt_covariance_mode=tt_jitter_model,
        tt_mirror_model=tt_mirror_model,
        tt_welch_nperseg=tt_jitter_nperseg,
        ho_covariance_policy=args.ho_covariance_policy,
    ))

    # Reference-wavelength / Fourier-grid / angular-centroid audit.
    # The physical image-motion covariance must be independent of lambda;
    # only the OTF sampling and diffraction kernel depend on wavelength.
    rad2mas = 180 * 3600 * 1000 / np.pi
    diameter = float(model.ao.tel.D)
    ref_wavelength = float(model.freq.wvlRef)
    pupil_sample = float(model.freq.sampRef)
    k_ref = int(model.freq.kRef_)
    declared_pixel_mas = float(np.asarray(model.ao.cam.psInMas).reshape(-1)[0])
    implied_pixel_mas = (
        ref_wavelength * rad2mas * k_ref / (diameter * pupil_sample)
    )
    expected_psd_step = 1.0 / (diameter * pupil_sample)
    original_tilt_gain = float(sd.trs.tipTilt.tilt2meter)
    expected_tilt_gain = diameter * np.pi / (180 * 3600) / 4
    slope_measured = np.asarray(sd.trs.tipTilt.slopes, dtype=float)
    slope_covariance = slope_measured.T @ slope_measured / len(slope_measured)
    is_ngs = str(trs.aoMode).upper() == "NGS"
    if is_ngs:
        print("[NGS TT geometry] Historical RESIDUALWAVEFRONT conversion is "
              f"{original_tilt_gain / expected_tilt_gain:.3g} times the LGS "
              "Noll-angle conversion; raw NGS units are unverified. "
              "No angular mas RMS is certified.", flush=True)
    spatial_diagnostics = {
        "reference_wavelength_m": ref_wavelength,
        "science_wavelength_bins_m": np.asarray(model.freq.wvl_, dtype=float).tolist(),
        "input_atmosphere_wavelength_m": float(sd.trs.atm.wvl),
        "rescaled_atmosphere_wavelength_m": float(model.ao.atm.wvl),
        "rescaled_r0_m": float(model.ao.atm.r0),
        "telescope_diameter_m": diameter,
        "otf_grid_size": int(model.freq.nOtf),
        "camera_pixels": int(model.ao.cam.fovInPix),
        "otf_sampling_sampRef": pupil_sample,
        "integer_oversampling_kRef": k_ref,
        "pixel_scale_declared_mas_per_pixel": declared_pixel_mas,
        "pixel_scale_implied_mas_per_pixel": implied_pixel_mas,
        "pixel_scale_relative_error": implied_pixel_mas / declared_pixel_mas - 1.0,
        "psd_frequency_step_inv_m": float(model.freq.PSDstep),
        "psd_frequency_step_expected_inv_m": expected_psd_step,
        "dm_cutoff_max_inv_m": float(model.freq.kcMax_),
        "tt_signal_origin": (
            "RESIDUALWAVEFRONT" if is_ngs else "DTTCENTROIDS"
        ),
        "straptiptilt_opd_gain_m_per_arcsec": original_tilt_gain,
        "noll_tilt_opd_gain_expected_m_per_arcsec": expected_tilt_gain,
        "ngs_tt_scale_ratio_to_lgs_noll": (
            float(original_tilt_gain / expected_tilt_gain) if is_ngs else None
        ),
        "straptiptilt_raw_measured_covariance_trace_m2": float(
            np.trace(slope_covariance)
        ),
        "straptiptilt_raw_rms_per_axis_mas": (
            None if is_ngs else float(
                1000 * np.sqrt(np.trace(slope_covariance) / 2)
                / expected_tilt_gain
            )
        ),
        "tt_wavelength_model": (
            "Angular covariance independent of wavelength; "
            "Gaussian OTF phase uses reference wavelength and pupil frequency"
        ),
    }
    (args.output / "frequency_grid_diagnostics.json").write_text(
        json.dumps(spatial_diagnostics, indent=2), encoding="utf-8"
    )
    print("[Frequency/units]", spatial_diagnostics, flush=True)

    # Record the noise-subtracted high-order covariance that actually fed
    # modes2Otf. This is an internal model diagnostic, not a fit to NIRC2.
    ho_cov = (2*np.pi/model.freq.wvlRef)**2 * (
        np.asarray(model.Cao, dtype=float)
        - np.asarray(model.trs.wfs.Cn_ao, dtype=float)
    )
    ho_cov = 0.5 * (ho_cov + ho_cov.T)
    ho_eig = np.linalg.eigvalsh(ho_cov)
    ho_tolerance = max(1e-12 * float(np.max(np.abs(ho_eig))), 1e-24)
    ho_diagnostics = {
        "source": "noise-subtracted HO actuator covariance used by modes2Otf",
        "units": "rad2",
        "lambda_min": float(ho_eig.min()),
        "lambda_max": float(ho_eig.max()),
        "negative_eigenvalue_count": int(np.count_nonzero(ho_eig < -ho_tolerance)),
        "negative_tolerance": ho_tolerance,
        "trace": float(np.trace(ho_cov)),
        "positive_semidefinite_within_tolerance":
            bool(np.all(ho_eig >= -ho_tolerance)),
    }
    ho_diagnostics.update(getattr(model, "ho_covariance_diagnostics", {}))
    (args.output / "ho_covariance_diagnostics.json").write_text(
        json.dumps(ho_diagnostics, indent=2), encoding="utf-8"
    )
    print("[HO covariance]", ho_diagnostics, flush=True)

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
        dataset_label=f"Keck/NIRC2 {args.fits.name} — {trs.aoMode}",
    ))
    print("Figure:", args.output / "comparison_nirc2_psfr.png")
    if args.tt_sensitivity:
        stage("TT noise sensitivity", lambda: _tt_sensitivity(
            model, trs.cam.image, args.output,
        ))
    (args.output / "metadata.json").write_text(json.dumps({
        "input_fits": str(args.fits.resolve()),
        "input_sav": str(args.sav.resolve()),
        "filter_profile_diagnostics": filter_profile_diagnostics,
        "addNCPA": False,
        "ao_mode": str(trs.aoMode),
        "ho_covariance_policy": args.ho_covariance_policy,
        "ho_covariance_projection_applied": bool(
            getattr(model, "ho_covariance_diagnostics", {}).get(
                "projection_applied", False
            )
        ),
        "tt_noise_method_requested": tt_method,
        "tt_noise_method": effective_tt_noise_method,
        "tt_jitter_model_requested": args.tt_jitter_model or jitter_config.get(
            "method", "legacy_psfr"
        ),
        "tt_jitter_policy_note": model_policy_note,
        "ngs_tt_units_independently_verified": (
            False if str(trs.aoMode).upper() == "NGS" else None
        ),
        "tt_noise_config": str(args.config.resolve()),
        "tt_jitter_covariance_method": tt_jitter_model,
        "tt_mirror_model": tt_mirror_model,
        "tt_noise_rejection_mirror_model": noise_mirror,
        "tt_jitter_welch_nperseg": tt_jitter_nperseg,
        "tt_jitter_noise_model_validated": False,
        "tt_jitter_otf_enabled": not args.disable_tt_jitter,
        "tt_jitter_ablated_for_diagnostics": bool(args.disable_tt_jitter),
        "target": args.fits.name + " (observed field; multiplicity not inferred)",
        "psf_shape": list(psf.shape),
        "psf_spectral_model": (
            "Single reference-wavelength PSF shared across the three "
            "science passband bins with different throughput weights; "
            "not three independently propagated chromatic PSFs"
        ),
        "psf_reference_wavelength_m": float(model.freq.wvlRef),
        "note": "No source fitting or astrometric registration performed",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
