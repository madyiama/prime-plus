"""Keck/NIRC2 LGS PSF-R smoke-test with scientifically labelled diagnostic plots.

Run from the repository root with `uv run python scripts/keck_n0013_smoke.py`.
Fits/SAV files remain outside the repository; no uploads occur.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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
    alpha_values = [0.0, 0.5, 0.9, 0.98, 1.0]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--sav", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/n0013"))
    parser.add_argument("--tt-sensitivity", action="store_true",
                        help="Compare PSFs for different TT noise-subtraction fractions")
    args = parser.parse_args()
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
    sd = stage("systemDiagnosis", lambda: systemDiagnosis(trs))
    stage("configFile", lambda: configFile(sd))
    model = stage("psfR", lambda: psfR(sd.trs))

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
        "target": "LP415-20 (binary)",
        "psf_shape": list(psf.shape),
        "note": "No astrometric registration or binary fit performed",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
