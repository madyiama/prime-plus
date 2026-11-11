"""Keck/NIRC2 PSF-R smoke-test (LGS and NGS) with labelled diagnostics.

Run from the repository root with `uv run python scripts/keck_n0013_smoke.py`.
Fits/SAV files remain outside the repository; no uploads occur.
"""
from __future__ import annotations

import argparse
import json
import subprocess
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


def _plot_comparison(
    observed, psf_cube, destination, *, dataset_label="Keck/NIRC2",
    science_filter=None, reference_wavelength_um=None,
):
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
        (ax[0, 1], psf_model,
         "PSF-R — single-source, "
         + (str(science_filter) if science_filter else "science filter")
         + (
             f", PSF at {reference_wavelength_um:.3f} µm"
             if reference_wavelength_um is not None else ""
         )),
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


def _validate_canonical_tt_geometry(diameter_m):
    """Detect leftover manual 4/D -> 1/D edits before expensive PSF-R.

    The historical no-16 comparison is selected with the explicit
    tt_kernel_convention option, NOT by editing the shared OPD->angle
    geometry. Doing both would suppress the TT SF by 256, not 16.
    """
    from p3.aoSystem.tipTiltGeometry import opd_covariance_to_angular

    diameter_m = float(diameter_m)
    test_covariance = np.eye(2, dtype=float)
    result = opd_covariance_to_angular(test_covariance, diameter_m)
    expected = (4.0/diameter_m)**2 * test_covariance
    if not np.allclose(result, expected, rtol=1e-13, atol=0.0):
        raise ValueError(
            "Noncanonical TT OPD-to-angle conversion detected. "
            "tipTiltGeometry.opd_covariance_to_angular must use (4/D)**2 "
            "for RMS Noll tilts. Restore any local (1/D)**2 edit; "
            "use --tt-kernel-convention legacy_unscaled_opd for the "
            "controlled historical 1/16 OTF comparison."
        )


def _structure_function_wfe_bridge(dphi, otf_dl, wavelength_m):
    """Relate a pupil-averaged SF to two *distinct* phase-variance estimates.

    For P(rho)=OTF_DL(rho)/sum(OTF_DL) (assuming nonnegative real
    pupil autocorrelation and a valid pupil-averaged structure function):

      piston_removed_phase_variance = 0.5 * sum(P(rho)*Dphi(rho))
      Strehl_factor = sum(P(rho)*exp(-Dphi(rho)/2))
      Strehl_equivalent_phase_variance = -log(Strehl_factor)

    The first formula is an exact second-moment identity for the same
    ensemble and aperture; the second is nonlinear (Jensen's inequality
    implies Strehl-equivalent variance <= pupil variance). Neither
    quantity is *automatically* equal to Hardy's Kolmogorov cone result
    if finite L0, pupil, filtering or geometry assumptions differ.
    """
    dphi = np.asarray(dphi, dtype=float)
    otf_dl = np.asarray(otf_dl, dtype=float)
    wavelength_m = float(wavelength_m)
    if (dphi.ndim != 2 or dphi.shape != otf_dl.shape
            or not np.all(np.isfinite(dphi))
            or not np.all(np.isfinite(otf_dl))
            or wavelength_m <= 0 or not np.isfinite(wavelength_m)):
        raise ValueError("SF bridge needs matching finite 2-D arrays and lambda>0")
    dc = float(dphi[dphi.shape[0]//2, dphi.shape[1]//2])
    if abs(dc) > 1e-9:
        raise ValueError(
            f"SF bridge requires Dphi(0)=0; got {dc:g} rad^2"
        )
    # Spline interpolation of an otherwise nonnegative pupil OTF can
    # introduce tiny negative sidelobes. The weighted-variance/Jensen
    # identity requires nonnegative weights. Clip them ONLY in the
    # diagnostic and record the removed negative mass. Never change
    # the forward model's diffraction OTF or tune the underlying SF.
    positive_mass = float(np.maximum(otf_dl, 0.0).sum())
    negative_mass = -float(np.minimum(otf_dl, 0.0).sum())
    negative_mass_fraction = negative_mass / max(positive_mass, 1e-30)
    if negative_mass_fraction > 1e-3:
        import warnings
        warnings.warn(
            "Diffraction OTF has >0.1% negative weight mass. "
            "Pupil-SF bridge clips negative weights for the mathematical "
            "second-moment identity; this changes the diagnostic weighting "
            "relative to the current forward OTF, which must be audited.",
            RuntimeWarning, stacklevel=2,
        )
    weights = np.maximum(otf_dl, 0.0)
    area = float(weights.sum())
    if area <= 0:
        raise ValueError("Pupil OTF has zero total support")
    weights /= area
    pupil_phase_variance = 0.5 * float(np.sum(weights * dphi))
    strehl = float(np.sum(weights * np.exp(-0.5*dphi)))
    if strehl <= 0 or not np.isfinite(strehl):
        raise ValueError("SF bridge gave nonphysical Strehl")
    strehl_phase_variance = -float(np.log(strehl))
    if pupil_phase_variance < -1e-9:
        raise ValueError("Pupil-weighted structure function has negative variance")
    if strehl_phase_variance > pupil_phase_variance + 1e-9:
        raise ValueError(
            "Jensen inequality violation: inconsistent SF or OTF normalization"
        )
    nm_per_rad = wavelength_m * 1e9 / (2*np.pi)
    return {
        "reference_wavelength_m": wavelength_m,
        "sf_min_rad2": float(np.min(dphi)),
        "sf_max_rad2": float(np.max(dphi)),
        "sf_zero_lag_rad2": dc,
        "diffraction_otf_negative_weight_fraction": negative_mass_fraction,
        "diffraction_otf_weights_clipped_for_diagnostic": bool(
            negative_mass > 0
        ),
        "pupil_piston_removed_phase_variance_rad2": pupil_phase_variance,
        "pupil_piston_removed_wfe_nm": float(
            np.sqrt(max(pupil_phase_variance, 0.0))*nm_per_rad
        ),
        "strehl_equivalent_phase_variance_rad2": strehl_phase_variance,
        "strehl_equivalent_wfe_nm": float(
            np.sqrt(max(strehl_phase_variance, 0.0))*nm_per_rad
        ),
        "anisoplanatism_only_strehl_factor": strehl,
        "jensen_gap_phase_variance_rad2": (
            pupil_phase_variance - strehl_phase_variance
        ),
        "jensen_ratio_strehl_to_pupil_variance": (
            strehl_phase_variance/pupil_phase_variance
            if pupil_phase_variance > 0 else None
        ),
        "note": (
            "Pupil-weighted SF moment is a linear phase-variance identity; "
            "minus log Strehl is nonlinear. Both use the same SF, unlike "
            "the independent infinite-L0 Hardy analytic expression."
        ),
    }


def _telemetry_forward_parameters(model, gho=1.0, gtt=1.0):
    """Build PSF-R's on-axis forward vector for 1 or multiple turbulence layers.

    FourierUtils.sort_params_from_labels expects [r0, gho, gtt] for a
    single-layer model, but [w_1*r0**(-5/3), ..., w_n*r0**(-5/3),
    gho, gtt] for a multi-layer model with multiple reconstructed layers.
    Each w_i is the dimensionless turbulence-integral fraction (sum=1),
    and the first n entries have units m**(-5/3), NOT Cn2 in m**(-2/3).
    The r0 used here is the model-atmosphere r0 at model.ao.atm.wvl,
    already rescaled from the original 500-nm estimate if required.
    This is a forward evaluation, not an r0/Cn2 fit to the observed PSF.
    """
    atm = model.ao.atm
    r0 = float(atm.r0)
    if not np.isfinite(r0) or r0 <= 0:
        raise ValueError(f"PSF-R requires a finite positive r0, got {r0!r}")
    if not np.isfinite(gho) or not np.isfinite(gtt):
        raise ValueError("PSF-R gains must be finite")
    n_layer = int(atm.nL)
    n_rec = int(model.ao.dms.nRecLayers)
    if n_layer > 1 and n_rec > 1:
        weights = np.asarray(atm.weights, dtype=float).reshape(-1)
        if (weights.size != n_layer or not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0, atol=1e-6, rtol=0)):
            raise ValueError(
                "Multi-layer PSF-R needs one finite nonnegative turbulence "
                f"weight per layer, normalized to one; nL={n_layer}, "
                f"weights={weights.tolist()}"
            )
        strengths = weights * r0**(-5.0/3.0)
        return strengths.tolist() + [float(gho), float(gtt)]
    return [r0, float(gho), float(gtt)]


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
    forward_parameters = _telemetry_forward_parameters(model)
    spectra = []
    records = []

    try:
        for alpha in alpha_values:
            covariance = raw - alpha * noise
            covariance = (covariance + covariance.T) / 2
            eigenvalues = np.linalg.eigvalsh(covariance)
            model.dphi_tt = model.jitter_phase_structure_function(Ctt=covariance)
            cube = np.asarray(model(forward_parameters)).squeeze()
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
        model(forward_parameters)  # Restore original PSF output/state.

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


def _git_code_revision():
    """Record the exact model source state, including uncommitted edits."""
    root = Path(__file__).resolve().parents[1]

    def git(*argv):
        try:
            return subprocess.run(
                ["git", *argv], cwd=root, capture_output=True, text=True,
                check=False, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    rev = git("rev-parse", "HEAD")
    dirty = git("diff", "--quiet", "--", "p3", "scripts")
    return {
        "git_revision": rev.stdout.strip() if rev and rev.returncode == 0 else None,
        "tracked_physics_code_has_uncommitted_edits": (
            dirty.returncode == 1 if dirty is not None else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--sav", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/n0013"))
    parser.add_argument(
        "--n-layer", type=int, choices=(1, 3, 7), default=1,
        help="Equivalent atmospheric layers (1=historical baseline; "
             "7=retain all MASS-DIMM altitude bins).",
    )
    parser.add_argument(
        "--p3-pupil-resampling", choices=("legacy", "physical_binary"),
        default="legacy",
        help=("P3 optical pupil sampling: legacy spline as baseline, "
              "or subpixel area-averaged physically bounded Keck binary mask."),
    )
    parser.add_argument(
        "--p3-pupil-parity", choices=("native", "flip_x", "flip_y"),
        default="native",
        help=("FITS pupil array parity hypothesis. Non-native parity requires "
              "--p3-pupil-resampling physical_binary; verify against science "
              "diffraction, never by blind PSF fitting."),
    )
    parser.add_argument(
        "--p3-pupil-rotation",
        choices=("header", "as_archived", "reverse_sign", "plus_90", "minus_90"),
        default="header",
        help=("P3 uncalibrated sensitivity comparison: header is KAON nominal; "
              "alternatives change pupil+TT axes and must not be selected by "
              "science-PSF fit."),
    )
    parser.add_argument(
        "--tt-slope-source",
        choices=("centroids", "centroids_aligned", "command_increment"),
        default="centroids",
        help=(
            "P4-F controlled TT-only input source: centroids (original "
            "unmodified baseline), centroids_aligned (drop first STRAP "
            "sample for exact A/B), or command_increment (replace TT "
            "slopes by diff(DTTCOMMANDS)/DT_SERVO gain, then drop first "
            "sample in all TT channels). Re-estimates van Dam/Gendron "
            "noise from the selected signal; NOT a sky-angle calibration."
        ),
    )
    parser.add_argument("--tt-sensitivity", action="store_true",
                        help="Compare PSFs for different TT noise-subtraction fractions")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "config/keck_tt_noise.toml",
        help="TOML file selecting the STRAP TT noise estimator",
    )
    parser.add_argument(
        "--tt-profile",
        choices=("reduced_jitter", "spectral_comparison", "gendron_comparison"),
        default=None,
        help=(
            "P4-C fixed STRAP recipe: reduced_jitter (van Dam + legacy "
            "covariance, provisional default), spectral_comparison "
            "(van Dam + W/K/P/S spectral), or gendron_comparison "
            "(time-lag noise + legacy covariance). When omitted, use TOML."
        ),
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
        "--tt-kernel-convention",
        choices=("noll_angular", "legacy_unscaled_opd"),
        default="noll_angular",
        help=(
            "Documented TT kernel normalization. noll_angular uses a=D*theta/4 "
            "and physical angular OTF; legacy_unscaled_opd removes factor 16 "
            "in Dphi for a controlled historical comparison ONLY."
        ),
    )
    parser.add_argument(
        "--tt-axis-angle-deg", type=float, default=None,
        help=("P4 STRAP-axis to camera-frame rotation hypothesis in degrees. "
              "If omitted, reproduce historical coupling to pupilAngle. "
              "Not an instrument calibration."),
    )
    parser.add_argument(
        "--tt-axis-parity", choices=("native", "flip_x", "flip_y"),
        default="native",
        help=("P4 STRAP sensor-axis parity hypothesis, independent of "
              "the NIRC2 pupil-mask parity."),
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
            "at most 0.001 percent (LGS/NGS); small_negative permits "
            "at most 0.5 percent (NGS only). "
            "Neither policy validates the sensor-noise estimator."
        ),
    )
    parser.add_argument(
        "--disable-tt-jitter", action="store_true",
        help=("Diagnostic ablation only: omit TT jitter from PSF-R OTF, "
              "without changing other components or fitting observations"),
    )
    parser.add_argument(
        "--p1-only", action="store_true",
        help=(
            "After systemDiagnosis export P1 modal-noise and Zernike variance "
            "diagnostics, then stop BEFORE constructing a PSF. "
            "Production atmospheric estimator remains unchanged."
        ),
    )
    parser.add_argument(
        "--r0-method", choices=("zernike", "jolissaint", "both"),
        default="zernike",
        help=(
            "P1-only unified seeing estimator: production Zernike, "
            "experimental Jolissaint modified DM modal basis, or both. "
            "Production PSF-R always retains the legacy estimator."
        ),
    )
    parser.add_argument(
        "--jolissaint-noise", choices=("spectral", "none"),
        default="spectral",
        help="Experimental HF PSD/closed-loop noise correction, or no subtraction"
    )
    parser.add_argument(
        "--jolissaint-modes", choices=("idl_2018", "all"),
        default="idl_2018",
        help="Published Keck 2018 LGS exclusions or all valid Noll 7..120"
    )
    parser.add_argument(
        "--jolissaint-alias", action="store_true",
        help=(
            "Apply the published Keck 2018 empirical WFS alias polynomial; "
            "NOT independently calibrated for n0120"
        ),
    )
    parser.add_argument(
        "--jolissaint-grid-sizes", default=None,
        help=(
            "P1-only spatial convergence: comma-separated pixel axes, "
            "e.g. '100,201'. The native 100x100 DM IF is used at 100, "
            "higher resolutions REGENERATE the calibrated Keck IF. "
            "261 corresponds to 13 samples per actuator pitch."
        ),
    )
    args = parser.parse_args()
    if args.p3_pupil_parity != "native" and args.p3_pupil_resampling != "physical_binary":
        parser.error("--p3-pupil-parity flip_x/flip_y requires physical_binary")
    if args.r0_method != "zernike" and not args.p1_only:
        parser.error("--r0-method jolissaint/both requires --p1-only")
    with args.config.open("rb") as conf_file:
        full_config = tomllib.load(conf_file)
    from p3.telemetry.ttSelection import resolve_tt_selection
    try:
        tt_selection = resolve_tt_selection(
            full_config,
            profile=args.tt_profile,
            overrides={
                "noise_method": args.tt_noise_method,
                "noise_band": args.tt_noise_band,
                "noise_welch": args.tt_welch_nperseg,
                "covariance_method": args.tt_jitter_model,
                "mirror_model": args.tt_mirror_model,
                "jitter_welch": args.tt_jitter_welch_nperseg,
            },
        )
    except ValueError as exc:
        parser.error(str(exc))
    tt_method = tt_selection.noise_method
    tt_band = tt_selection.noise_band
    tt_nperseg = tt_selection.noise_welch
    tt_jitter_model = tt_selection.covariance_method
    tt_mirror_model = tt_selection.mirror_model
    tt_jitter_nperseg = tt_selection.jitter_welch
    print(
        "[TT selection] source=", tt_selection.source,
        ", noise=", tt_method, ", jitter=", tt_jitter_model,
        ", mirror=", tt_mirror_model, ", band=", tt_band,
        ", Welch=", tt_nperseg, sep="", flush=True,
    )
    if args.tt_kernel_convention == "legacy_unscaled_opd":
        print(
            "[TT WARNING] /16 kernel is a nonphysical historical ablation, "
            "not a calibrated science-camera jitter model.",
            flush=True,
        )
    for file in (args.fits, args.sav, args.calib):
        if not file.exists():
            parser.error(f"Not found: {file}")
    args.output.mkdir(parents=True, exist_ok=True)
    tt_selection_report = tt_selection.report()
    tt_selection_report.update({
        "tt_kernel_convention": args.tt_kernel_convention,
        "tt_kernel_is_physical_noll": (
            args.tt_kernel_convention == "noll_angular"
        ),
        "r0_l0_reestimated_for_tt_profile": False,
        "ho_noise_estimator_changed_for_tt_profile": False,
    })
    if args.tt_kernel_convention != "noll_angular":
        # An explicitly selected /16 kernel cannot be labelled the
        # recommended provisional science framework even when the
        # covariance/noise profile otherwise matches A1.
        tt_selection_report["recommended_for_framework_default"] = False
        tt_selection_report["method_status"] = (
            "nonphysical_historical_kernel_ablation"
        )
    (args.output / "tt_selection.json").write_text(
        json.dumps(tt_selection_report, indent=2), encoding="utf-8",
    )

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
        cam_fov=128, nLayer=args.n_layer, addNCPA=False,
    ))
    # P4-F: substitute TT slopes BEFORE systemDiagnosis (noise estimator)
    # and psfR (jitter covariance). HO telemetry and atmospheric r0 stay
    # untouched. The aligned baseline crops the same frame as replay.
    from p3.telemetry.ttCommandReplay import apply_tt_slope_source
    tt_command_replay_report = apply_tt_slope_source(
        trs,
        list(np.asarray(trs.ttloop.tf.num,dtype=float).reshape(-1))
        +list(np.asarray(trs.ttloop.tf.den,dtype=float).reshape(-1)),
        args.tt_slope_source,
    )
    (args.output / "tt_command_replay.json").write_text(
        json.dumps(tt_command_replay_report,indent=2,allow_nan=False),
        encoding="utf-8",
    )
    print("[TT source]",args.tt_slope_source,
          "frames=",trs.tipTilt.nExp,
          "replay relative error=",
          tt_command_replay_report.get("relative_rms_error_vs_centroids"),
          flush=True)
    # P3: explicit pupil hypotheses without changing the default legacy path.
    pupil_diagnostics = dict(getattr(trs.tel, "pupil_orientation", {}))
    if args.p3_pupil_resampling == "physical_binary":
        from astropy.io import fits
        from p3.telemetry.keckPupilAudit import (
            pupil_orientation_report, read_mask,
        )
        p3_header = fits.getheader(args.fits, ignore_missing_end=True)
        p3_report = pupil_orientation_report(p3_header)
        mask_path, _ = read_mask(args.calib, p3_report)
        # The historical telemetry loader silently falls back to OPEN for
        # unsupported masks. The physical P3 path must never do that.
        trs.tel.path_pupil = str(mask_path.resolve())
        pupil_diagnostics["p3_mask_inventory"] = p3_report
    nominal_angle = float(trs.tel.pupilAngle)
    p3_angles = {
        "header": nominal_angle,
        "as_archived": 0.0,
        "reverse_sign": -nominal_angle,
        "plus_90": nominal_angle + 90.0,
        "minus_90": nominal_angle - 90.0,
    }
    trs.tel.pupilAngle = float(p3_angles[args.p3_pupil_rotation])
    trs.tel.pupil_resampling = args.p3_pupil_resampling
    trs.tel.pupil_parity = args.p3_pupil_parity
    pupil_diagnostics["p3_rotation_hypothesis"] = args.p3_pupil_rotation
    pupil_diagnostics["p3_array_rotation_applied_deg"] = float(trs.tel.pupilAngle)
    pupil_diagnostics["p3_pupil_sampling"] = args.p3_pupil_resampling
    pupil_diagnostics["p3_input_array_parity"] = args.p3_pupil_parity
    pupil_diagnostics["p3_rotation_zero_calibrated"] = False
    pupil_diagnostics["p3_rotation_tt_axes_coupled"] = True
    trs.tel.pupil_orientation = pupil_diagnostics
    massdimm_diagnostics = getattr(trs.atm, "massdimm_diagnostics", {})
    (args.output / "pupil_orientation_diagnostics.json").write_text(
        json.dumps(pupil_diagnostics, indent=2), encoding="utf-8"
    )
    (args.output / "massdimm_diagnostics.json").write_text(
        json.dumps(massdimm_diagnostics, indent=2), encoding="utf-8"
    )
    print("[Pupil orientation]", pupil_diagnostics, flush=True)
    print("[MASS-DIMM source]", massdimm_diagnostics, flush=True)
    # This must hold even for a legacy OTF comparison: the physical
    # angle conversion remains canonical, with /16 applied only once by
    # --tt-kernel-convention legacy_unscaled_opd.
    _validate_canonical_tt_geometry(trs.tel.D)
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
    atmosphere_audit = {
        "r0_tel_m_at_atmosphere_wavelength": float(sd.trs.atm.r0_tel),
        "atmosphere_wavelength_m": float(sd.trs.atm.wvl),
        "seeing_tel_arcsec": float(sd.trs.atm.seeing),
        "seeing_dimm_arcsec": (
            float(sd.trs.atm.seeing_dimm)
            if sd.trs.atm.seeing_dimm > 0 else None
        ),
        "airmass_nirc2": float(sd.trs.tel.airmass),
        "dimm_seeing_reference_assumption": (
            "UNKNOWN; zenith-projection below is conditional, "
            "not an independently verified MKWC calibration"
        ),
        "seeing_dimm_los_if_zenith_arcsec": (
            float(sd.trs.atm.seeing_dimm)
            * float(sd.trs.tel.airmass)**(3.0/5.0)
            if sd.trs.atm.seeing_dimm > 0 else None
        ),
        "seeing_mass_arcsec": (
            float(sd.trs.atm.seeing_mass)
            if sd.trs.atm.seeing_mass > 0 else None
        ),
        "fitted_L0_m": float(sd.trs.atm.L0_tel),
        "r0_fit_uncertainty_m": float(sd.trs.atm.dr0_tel),
        "fitted_L0_uncertainty_m": float(sd.trs.atm.dL0_tel),
        "r0_fit_success": bool(sd.res_r0.success),
        "r0_fit_cost": float(sd.res_r0.cost),
        "r0_fit_nfev": int(sd.res_r0.nfev),
        "r0_fit_active_parameter_bounds": (
            np.asarray(sd.res_r0.active_mask, dtype=int).tolist()
        ),
        "r0_fit_L0_at_upper_bound": bool(
            np.isclose(float(sd.trs.atm.L0_tel), 100.0, rtol=0, atol=1e-4)
        ),
        "r0_tel_over_dimm": (
            float(sd.trs.atm.r0_tel) / (
                0.976 * (180*3600/np.pi) * float(sd.trs.atm.wvl)
                / float(sd.trs.atm.seeing_dimm)
            )
            if sd.trs.atm.seeing_dimm > 0 else None
        ),
        "r0_tel_over_dimm_los_if_zenith": (
            float(sd.trs.atm.r0_tel) / (
                (0.976 * (180*3600/np.pi) * float(sd.trs.atm.wvl)
                 / float(sd.trs.atm.seeing_dimm))
                * float(sd.trs.tel.airmass)**(-3.0/5.0)
            )
            if sd.trs.atm.seeing_dimm > 0 else None
        ),
        "pseudo_open_loop": {
            "delay_frames": float(sd.trs.dm.pseudo_ol_delay_frames),
            "frames_used": int(sd.trs.dm.pseudo_ol_frames_used),
            "frames_dropped": int(sd.trs.dm.pseudo_ol_frames_dropped),
            "method": sd.trs.dm.pseudo_ol_alignment,
            "sign_independently_verified": bool(
                sd.trs.dm.pseudo_ol_sign_independently_verified
            ),
        },
        "fried_coefficient": 0.976,
        "r0_wavelength_power": 1.2,
        "atmosphere_layer_count": len(sd.trs.atm.Cn2Weights),
        "atmosphere_layer_weights": list(map(float, sd.trs.atm.Cn2Weights)),
        "atmosphere_layer_heights_m": list(map(float, sd.trs.atm.Cn2Heights)),
        "full_massdimm_layer_weights": list(map(
            float, sd.trs.atm.Cn2Weights_full
        )),
        "full_massdimm_layer_heights_m": list(map(
            float, sd.trs.atm.Cn2Heights_full
        )),
        "massdimm_provenance": massdimm_diagnostics,
        "tt_kernel_convention_requested": args.tt_kernel_convention,
    }
    (args.output / "atmosphere_reconstruction_audit.json").write_text(
        json.dumps(atmosphere_audit, indent=2), encoding="utf-8"
    )
    if args.p1_only:
        # P1 research is deliberately isolated from the forward PSF-R.
        # The archived n0120 1/7-layer P2 outputs remain untouched if
        # the user supplies a dedicated --output folder.
        from keck_p1_modal_noise_audit import analyse_diagnosis, write_audit
        audit = stage(
            "P1 modal/Zernike/noise audit",
            lambda: analyse_diagnosis(sd)
        )
        write_audit(audit, args.output)
        from p3.telemetry.seeingEstimator import estimate_r0
        methods = (
            ("zernike", "jolissaint") if args.r0_method == "both"
            else (args.r0_method,)
        )
        seeing = {}
        native_resolution = int(sd.trs.tel.resolution)
        grid_sizes = (
            [native_resolution] if args.jolissaint_grid_sizes is None
            else [int(s.strip()) for s in
                  args.jolissaint_grid_sizes.split(",")]
        )
        if (not grid_sizes or len(grid_sizes) != len(set(grid_sizes))
                or any(n < 51 or n > 301 or
                       (n % 2 != 1 and n != native_resolution)
                       for n in grid_sizes)):
            parser.error(
                "--jolissaint-grid-sizes must be unique odd axes "
                "in [51,301], or native resolution 100"
            )
        for name in methods:
            resolutions = (
                grid_sizes if name == "jolissaint" else [None]
            )
            for n in resolutions:
                key = (
                    name if n is None or n == native_resolution
                    else f"jolissaint_grid_{n}"
                )
                seeing[key] = stage(
                    f"P1 r0 estimator [{name}, grid={n}]",
                    lambda m=name, res=n: estimate_r0(
                        sd, method=m, spatial_resolution=res,
                        noise=args.jolissaint_noise,
                        mode_selection=args.jolissaint_modes,
                        alias_correction=args.jolissaint_alias,
                    ),
                )
                print(
                    f"[P1 r0 {key}] "
                    f"r0={seeing[key]['r0_m']:.6g}m, "
                    f"L0={seeing[key]['L0_m']:.6g}m",
                    flush=True,
                )
        (args.output / "p1_r0_estimator_comparison.json").write_text(
            json.dumps(seeing, indent=2), encoding="utf-8"
        )
        from p3.telemetry.seeingEstimator import (
            assess_jolissaint_spatial_convergence
        )
        convergence = assess_jolissaint_spatial_convergence(seeing)
        (args.output / "p1_r0_spatial_convergence.json").write_text(
            json.dumps(convergence, indent=2), encoding="utf-8"
        )
        print(
            f"[P1 Jolissaint spatial] "
            f"status={convergence['status']}, "
            f"relative r0 range={convergence.get('r0_relative_range')}, "
            f"L0 boundaries={convergence.get('L0_at_grid_boundary')}",
            flush=True,
        )
        print(
            "[P1-only] Complete; no PSF reconstruction, no fitting of "
            "science image, no changes to production r0/L0 estimator. "
            f"flags={audit['quality_flags']}",
            flush=True,
        )
        return
    print("[Atmosphere / r0]", {
        k: atmosphere_audit[k] for k in (
            "r0_tel_m_at_atmosphere_wavelength",
            "seeing_tel_arcsec", "seeing_dimm_arcsec", "fitted_L0_m",
            "atmosphere_layer_count",
        )
    }, flush=True)
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
        tt_kernel_convention=args.tt_kernel_convention,
        tt_axis_angle_deg=args.tt_axis_angle_deg,
        tt_axis_parity=args.tt_axis_parity,
    ))

    # Telescope applies PupilResampling to the archived pupil at OTF build
    # time. Record its *effective* transmission, not just the input CLI flag.
    p3_model_pupil = getattr(model.ao.tel, "pupil_resampling_diagnostics", {})
    (args.output / "p3_model_pupil_diagnostics.json").write_text(
        json.dumps(p3_model_pupil, indent=2), encoding="utf-8",
    )
    if args.p3_pupil_resampling == "physical_binary":
        p3_pupil_array = np.asarray(model.ao.tel.pupil, dtype=float)
        if np.min(p3_pupil_array) < 0 or np.max(p3_pupil_array) > 1:
            raise ValueError("P3 telescope pupil contains nonphysical amplitudes")
        print("[P3 pupil] bounded mask",
              p3_model_pupil.get("output_amplitude_min"),
              p3_model_pupil.get("output_amplitude_max"), flush=True)

    # The WFS fit is already a line-of-sight r0. Its chromatic rescaling
    # must not have another airmass factor hidden in the INI parser.
    expected_los_r0 = float(sd.trs.atm.r0_tel) * (
        float(model.ao.atm.wvl) / float(sd.trs.atm.wvl)
    )**(6.0/5.0)
    actual_los_r0 = float(model.ao.atm.r0)
    if not np.isclose(actual_los_r0, expected_los_r0, rtol=2e-7, atol=1e-12):
        raise ValueError(
            "Keck model atmospheric r0 does not match WFS line-of-sight "
            "r0 after lambda^(6/5) scaling: "
            f"actual={actual_los_r0:g}m, expected={expected_los_r0:g}m, "
            f"ratio={actual_los_r0/expected_los_r0:g}. "
            "Inspect SeeingReference in the generated INI and the "
            "aoSystem airmass conversion. No science PSF was fitted."
        )
    atmosphere_audit.update({
        "input_seeing_reference": model.ao.atmosphere_seeing_reference,
        "model_los_r0_m": actual_los_r0,
        "model_los_r0_expected_from_wfs_m": expected_los_r0,
        "r0_model_over_expected": actual_los_r0 / expected_los_r0,
        "airmass_correction_applied_to_wfs_r0": False,
    })
    (args.output / "atmosphere_reconstruction_audit.json").write_text(
        json.dumps(atmosphere_audit, indent=2), encoding="utf-8"
    )
    print("[Atmosphere r0 LOS] passed:", actual_los_r0, "m at",
          model.ao.atm.wvl, "m", flush=True)

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
        "r0_expected_wfs_los_m": expected_los_r0,
        "r0_model_over_expected": actual_los_r0/expected_los_r0,
        "seeing_reference": model.ao.atmosphere_seeing_reference,
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

    # For nLayer=1 this is [r0, gho, gtt]; for nLayer>1 and a
    # multi-layer reconstructor it is [weights*r0**(-5/3), gho, gtt].
    # Deliberately omit stellar fitting parameters in both cases: the
    # Fourier model supplies default throughput in every spectral bin.
    forward_parameters = _telemetry_forward_parameters(model)
    from p3.aoSystem.FourierUtils import sort_params_from_labels
    forward_profile, recovered_r0, forward_gains, _, _, _ = (
        sort_params_from_labels(model, forward_parameters)
    )
    if not np.isclose(recovered_r0, model.ao.atm.r0, rtol=1e-12):
        raise ValueError(
            "Multi-layer forward vector changed the atmospheric r0: "
            f"model={model.ao.atm.r0}, parsed={recovered_r0}"
        )
    if len(forward_gains) != 2 or not np.allclose(forward_gains, [1.0, 1.0]):
        raise ValueError(
            "PSF-R forward vector did not preserve both HO and TT gains: "
            f"{forward_gains}"
        )
    forward_audit = {
        "mode": (
            "layer_strengths_r0_to_minus_5_over_3"
            if forward_profile is not None else "single_layer_r0"
        ),
        "n_atmospheric_parameters": (
            len(forward_profile) if forward_profile is not None else 1
        ),
        "atmosphere_layers": int(model.ao.atm.nL),
        "reconstructed_layers": int(model.ao.dms.nRecLayers),
        "r0_model_m": float(model.ao.atm.r0),
        "r0_model_wavelength_m": float(model.ao.atm.wvl),
        "r0_roundtrip_m": float(recovered_r0),
        "layer_strengths_m_to_minus_5_over_3": (
            np.asarray(forward_profile, dtype=float).tolist()
            if forward_profile is not None else None
        ),
        "gho_gtt": list(map(float, forward_gains)),
        "stellar_parameters": "default throughput; no source fitting",
    }
    (args.output / "forward_parameters_diagnostics.json").write_text(
        json.dumps(forward_audit, indent=2), encoding="utf-8"
    )
    print("[Forward model parameters]", forward_audit, flush=True)
    psf = np.asarray(stage("psfR forward",
                           lambda: model(forward_parameters))).squeeze()
    np.save(args.output / "telemetry_only_psf.npy", psf)
    # P4 compares the same model-only outputs for all TT ablations. A
    # three-bin spectral cube is collapsed by sum, as for the smoke figure.
    from p3.aoSystem.ttFrame import (
        psf_aperture_metrics, angular_covariance_metrics,
    )
    p4_psf = psf.sum(axis=-1) if psf.ndim == 3 else psf
    p4_psf_metrics = psf_aperture_metrics(p4_psf)
    p4_ttcov = getattr(model, 'tt_model_cov', None)
    p4_angle = (
        float(model.trs.tel.pupilAngle) if args.tt_axis_angle_deg is None
        else float(args.tt_axis_angle_deg)
    )
    p4_cov_metrics = (
        angular_covariance_metrics(p4_ttcov, float(model.ao.tel.D),
                                   p4_angle, args.tt_axis_parity)
        if (p4_ttcov is not None and not args.disable_tt_jitter) else None
    )
    p4_metrics = {
        'source': 'telemetry-only PSF and conditional STRAP covariance',
        'psf': p4_psf_metrics,
        'tt_jitter_enabled': bool(not args.disable_tt_jitter),
        'tt_model': tt_jitter_model,
        'tt_kernel_convention': args.tt_kernel_convention,
        'tt_axis_angle_deg': p4_angle,
        'tt_axis_parity': args.tt_axis_parity,
        'tt_axis_angle_uses_legacy_pupil': args.tt_axis_angle_deg is None,
        'camera_angular_covariance': p4_cov_metrics,
        'no_science_image_fitting': True,
    }
    (args.output / "p4_tt_metrics.json").write_text(
        json.dumps(p4_metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("PSF shape:", psf.shape, "min/max:", np.nanmin(psf), np.nanmax(psf))
    print("Error breakdown:")
    for key, value in model.wfe.items():
        print("  ", key, value)
    # Separate the analytical Kolmogorov cone WFE from the OTF-based
    # Maréchal-equivalent WFE. They are NOT the same observable.
    anisoplanatism_audit = {
        "reference": (
            "Hardy (1998) Eq. 16 / PRIME (2019) Eq. 12: "
            "0.5*(h/H)^(5/3)-0.452*(h/H)^2, normalized by 0.423"
        ),
        "comparison_note": (
            "Analytical piston-removed full Kolmogorov cone variance and "
            "diffraction-weighted OTF equivalent WFE are distinct; "
            "finite L0 and DM filtering are not in the Hardy formula"
        ),
        "reference_wavelength_m": float(model.freq.wvlRef),
        "atmosphere_r0_m": float(model.ao.atm.r0),
        "atmosphere_layers": int(model.ao.atm.nL),
        "layer_weights": list(map(float, model.ao.atm.weights)),
        "layer_heights_los_m": list(map(float, model.ao.atm.heights)),
        "lgs_height_los_m": (
            float(np.asarray(model.ao.lgs.height).reshape(-1)[0])
            if model.ao.lgs is not None else None
        ),
        "otf_equivalent_total_anisoplanatism_wfe_nm": float(
            model.wfe.get("TOTAL ANISOPLANATISM", 0)
        ),
        "otf_equivalent_focal_anisoplanatism_wfe_nm": float(
            model.wfe.get("FOCAL ANISOPLANATISM", 0)
        ),
    }
    if model.ao.lgs is not None and model.freq.isAniso:
        from p3.aoSystem.anisoplanatismModel import (
            focal_anisoplanatism_phase_variance,
            focal_anisoplanatism_wfe,
        )
        anisoplanatism_audit.update({
            "hardy_cone_phase_variance_rad2": float(
                focal_anisoplanatism_phase_variance(
                    model.ao.tel, model.ao.atm, model.ao.lgs
                )
            ),
            "hardy_cone_wfe_nm": float(
                focal_anisoplanatism_wfe(
                    model.ao.tel, model.ao.atm, model.ao.lgs
                )
            ),
        })
        dani = np.asarray(model.dani, dtype=float)
        if dani.ndim == 3:
            dani = dani[:, :, 0]
        otf_dl = np.asarray(model.freq.otfDL, dtype=float)
        anisoplanatism_only_strehl = (
            np.sum(otf_dl * np.exp(-0.5*dani)) / np.sum(otf_dl)
        )
        anisoplanatism_audit.update({
            "pupil_sf_bridge_total_anisoplanatism": (
                _structure_function_wfe_bridge(
                    dani, otf_dl, float(model.freq.wvlRef)
                )
            ),
            "anisoplanatism_dphi_min_rad2": float(np.min(dani)),
            "anisoplanatism_dphi_max_rad2": float(np.max(dani)),
            "anisoplanatism_dphi_center_rad2": float(
                dani[dani.shape[0]//2, dani.shape[1]//2]
            ),
            "anisoplanatism_dphi_center_before_normalization_rad2": float(
                np.asarray(model.dani_raw_zero_lag_rad2).reshape(-1)[0]
            ),
            "otf_dc_multiplier_before_normalization": float(
                np.exp(-0.5 *
                       np.asarray(model.dani_raw_zero_lag_rad2).reshape(-1)[0])
            ),
            "zero_lag_normalization": model.dani_zero_lag_reference,
            "zero_lag_physical_note": (
                "A phase structure function must vanish at zero OTF "
                "separation. Subtracting its interpolated DC offset "
                "preserves the flux-normalized PSF while removing a "
                "spurious Strehl/WFE scale factor."
            ),
            "otf_only_anisoplanatism_strehl_factor": float(
                anisoplanatism_only_strehl
            ),
            "otf_only_total_wfe_nm": float(
                np.sqrt(-np.log(anisoplanatism_only_strehl))
                * model.freq.wvlRef * 1e9/(2*np.pi)
            ),
        })
        # Compare the *same focal-only* kernel underlying the published
        # 'FOCAL ANISOPLANATISM' error budget, excluding field angular
        # and NGS anisokinetism contributions. The Hardy expression is
        # focal-only and must not be compared with a combined kernel.
        from p3.psfr.psfR import _anisoplanatism_zero_lag
        cone_layers = (
            np.asarray(model.dani_focang[0], dtype=float)
            - np.asarray(model.dani_ang[0], dtype=float)
        )
        cn2_science = (
            np.asarray(model.ao.atm.weights, dtype=float)
            * float(model.ao.atm.r0)**(-5.0/3.0)
            * (float(model.ao.atm.wvl)/float(model.freq.wvlRef))**2
        )
        cone_sf = _anisoplanatism_zero_lag(np.einsum(
            "lxy,l->xy", cone_layers, cn2_science
        ))
        bridge_cone = _structure_function_wfe_bridge(
            cone_sf, otf_dl, float(model.freq.wvlRef)
        )
        reported_focal = anisoplanatism_audit[
            "otf_equivalent_focal_anisoplanatism_wfe_nm"
        ]
        focal_budget_agreement = bool(np.isclose(
            bridge_cone["strehl_equivalent_wfe_nm"],
            reported_focal, rtol=2e-5, atol=5e-4,
        ))
        if not focal_budget_agreement:
            import warnings
            warnings.warn(
                "Focal-only SF bridge differs from the psfR error budget: "
                f"{bridge_cone['strehl_equivalent_wfe_nm']:.9g} vs "
                f"{reported_focal:.9g} nm. Check any negative diffraction "
                "OTF weights and the angular/anisokinetic decomposition.",
                RuntimeWarning, stacklevel=2,
            )
        anisoplanatism_audit["focal_bridge_matches_reported_wfe"] = (
            focal_budget_agreement
        )
        # Persist the underlying arrays for cheap offline checks, rather
        # than requiring repeated ~2-minute Keck forward reconstructions.
        np.save(args.output / "focal_anisoplanatism_sf.npy", cone_sf)
        np.save(args.output / "diffraction_otf.npy", otf_dl)
        anisoplanatism_audit["pupil_sf_bridge_focal_only"] = bridge_cone
        anisoplanatism_audit["focal_hardy_to_sf_pupil_wfe_ratio"] = (
            anisoplanatism_audit["hardy_cone_wfe_nm"]
            / bridge_cone["pupil_piston_removed_wfe_nm"]
            if bridge_cone["pupil_piston_removed_wfe_nm"] > 0 else None
        )
        anisoplanatism_audit["focal_model_conditions"] = {
            "von_karman_outer_scale_m": float(model.ao.atm.L0),
            "pupil": "NIRC2 numerical pupil via OTF_DL",
            "dm_filter_requested_in_psfr": not np.isscalar(model.Hfilter),
            "dm_filter_forwarded_to_focal_kernel": False,
            "dm_filter_warning": (
                "LGS/SLAO frequencyDomain does not currently forward "
                "self.Hfilter to focal_angular_function; do not use "
                "DM filtering to explain a discrepancy before auditing."
            ),
            "hardy_reference": "Kolmogorov infinite L0, analytic pupil model",
        }
    (args.output / "anisoplanatism_diagnostics.json").write_text(
        json.dumps(anisoplanatism_audit, indent=2), encoding="utf-8"
    )
    print("[Focal anisoplanatism P2]", anisoplanatism_audit, flush=True)

    # Save the decomposition, not just stdout: critical for 1-vs-7
    # comparisons where a similar PSF can hide an anisoplanatism change.
    error_budget = {
        str(name): float(np.asarray(value).reshape(-1)[0])
        for name, value in model.wfe.items()
    }
    (args.output / "error_breakdown_diagnostics.json").write_text(
        json.dumps(error_budget, indent=2), encoding="utf-8"
    )

    tt = getattr(model, "tiptilt_diagnostics", {})
    print("Tip-tilt covariance diagnostics:")
    for key, value in tt.items():
        print("  ", key, value)
    (args.output / "tiptilt_diagnostics.json").write_text(
        json.dumps(tt, indent=2), encoding="utf-8",
    )
    # P4: persist the independent angular replay and exact TT OTF
    # separately from noise-model goodness. This is a numerical/unit
    # validation, NOT an optical-gain or closed-loop calibration.
    angular_path = tt.get("tt_source_angle_replay")
    if (trs.aoMode == "LGS" and not args.disable_tt_jitter
            and (not angular_path
                 or not angular_path.get("tt_centroid_to_otf_units_verified"))):
        raise ValueError(
            "P4 LGS TT lacks validated DTTCENTROIDS -> angular covariance path"
        )
    angular_report = {
        "source": (
            "Keck B.DTTCENTROIDS in nominal arcsec"
            if args.tt_slope_source != "command_increment"
            else "Keck diff(B.DTTCOMMANDS)/DT_SERVO[0], "
                 "nominal arcsec-equivalent; not independent"
        ),
        "otf_domain": "angular spatial frequencies [cycles/radian]",
        "covariance_domain": "angular image-motion covariance [radian^2]",
        "tt_model": tt_jitter_model,
        "tt_kernel_convention": args.tt_kernel_convention,
        "tt_kernel_multiplier": model.tt_kernel_dphi_multiplier_vs_noll,
        "tt_source_angle_replay": angular_path,
        "otf_math_route_relative_error":
            tt.get("tt_otf_direct_angular_noll_relative_error"),
        "otf_zero_frequency":
            tt.get("tt_otf_zero_lag"),
        "otf_transmission_min": tt.get("tt_otf_min"),
        "otf_transmission_max": tt.get("tt_otf_max"),
        "sensor_gain_independently_calibrated": False,
        "loop_plant_independently_calibrated": False,
        "status": (
            "numerical_chain_verified_instrument_uncalibrated"
            if angular_path is not None else "tt_ablated_or_unavailable"
        ),
    }
    (args.output / "p4_tt_angular_chain.json").write_text(
        json.dumps(angular_report, indent=2, allow_nan=False), encoding="utf-8"
    )
    if getattr(model, "otf_tt", None) is not None:
        otf = np.asarray(model.otf_tt, dtype=float)
        if (not np.isfinite(otf).all() or otf.min() < 0
                or otf.max() > 1+1e-12
                or abs(otf[otf.shape[0]//2,otf.shape[1]//2]-1) > 1e-12):
            raise ValueError("Nonphysical TT OTF: violates DC/positivity invariants")
        # The error-budget routine evaluates separate noise/atmospheric
        # kernels, but diagnostics must save the TOTAL science OTF from
        # self.dphi_tt. Checking only OTF(0)=1 would miss component reuse.
        if not args.disable_tt_jitter:
            expected_full = np.exp(
                -0.5*np.asarray(model.dphi_tt, dtype=float)
            )
            if not np.allclose(otf, expected_full, rtol=1e-12, atol=1e-14):
                raise ValueError(
                    "P4 saved OTF_TT differs from the full jitter term "
                    "used by the science PSF; inspect diagnostic side effects"
                )
        np.save(args.output / "p4_otf_tt.npy", otf)
    stage("comparison figure", lambda: _plot_comparison(
        trs.cam.image, psf, args.output / "comparison_nirc2_psfr.png",
        dataset_label=(
            f"Keck/NIRC2 {args.fits.name} — {trs.aoMode}"
            f" — TT {args.tt_slope_source}"
        ),
        science_filter=filter_profile_diagnostics.get("filter_name"),
        reference_wavelength_um=1e6 * float(model.freq.wvlRef),
    ))
    print("Figure:", args.output / "comparison_nirc2_psfr.png")
    if args.tt_sensitivity:
        stage("TT noise sensitivity", lambda: _tt_sensitivity(
            model, trs.cam.image, args.output,
        ))
    (args.output / "metadata.json").write_text(json.dumps({
        "input_fits": str(args.fits.resolve()),
        "input_sav": str(args.sav.resolve()),
        "code_revision": _git_code_revision(),
        "filter_profile_diagnostics": filter_profile_diagnostics,
        "addNCPA": False,
        "ao_mode": str(trs.aoMode),
        "atmosphere_reconstruction_audit": atmosphere_audit,
        "forward_parameters_diagnostics": forward_audit,
        "error_breakdown_diagnostics": error_budget,
        "anisoplanatism_diagnostics": anisoplanatism_audit,
        "pupil_orientation_diagnostics": pupil_diagnostics,
        "p3_pupil_resampling": args.p3_pupil_resampling,
        "p3_pupil_parity": args.p3_pupil_parity,
        "p3_pupil_rotation_hypothesis": args.p3_pupil_rotation,
        "p3_array_rotation_applied_deg": float(trs.tel.pupilAngle),
        "p3_model_pupil_diagnostics": p3_model_pupil,
        "massdimm_diagnostics": massdimm_diagnostics,
        "tt_kernel_convention": model.tt_kernel_convention,
        "p4_tt_axis_angle_deg": p4_angle,
        "p4_tt_axis_parity": args.tt_axis_parity,
        "p4_tt_axis_uses_legacy_pupil": args.tt_axis_angle_deg is None,
        "p4_tt_metrics": p4_metrics,
        "tt_kernel_phase_structure_multiplier_vs_noll": (
            model.tt_kernel_dphi_multiplier_vs_noll
        ),
        "ho_covariance_policy": args.ho_covariance_policy,
        "ho_covariance_projection_applied": bool(
            getattr(model, "ho_covariance_diagnostics", {}).get(
                "projection_applied", False
            )
        ),
        "tt_noise_method_requested": tt_method,
        "tt_noise_method": effective_tt_noise_method,
        "tt_slope_source": args.tt_slope_source,
        "tt_command_replay": tt_command_replay_report,
        "tt_source_command_units_independently_verified": False,
        "tt_selection": tt_selection_report,
        "tt_selection_profile": tt_selection.profile,
        "tt_selection_source": tt_selection.source,
        "tt_kernel_is_physical_noll": model.tt_kernel_convention == "noll_angular",
        "tt_jitter_model_requested": tt_selection.covariance_method,
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
