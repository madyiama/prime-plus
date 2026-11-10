"""Compare 1- and 7-layer Keck telemetry-only PSFs at fixed AO parameters.

No PSF fitting, source recentering or external FITS access. Both input
directories must come from scripts/keck_n0013_smoke.py with the same
FITS/SAV, physics switches and fitted r0; only --n-layer should differ.

Example
-------
uv run python scripts/keck_compare_layer_counts.py \\
  --one results/n0120_1layer --seven results/n0120_7layers_v2 \\
  --output results/n0120_layer_comparison
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np


def _read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_run(path):
    root = Path(path)
    meta = _read_json(root / "metadata.json")
    if not meta:
        raise FileNotFoundError(f"metadata.json is missing in {root}")
    atmosphere = _read_json(root / "atmosphere_reconstruction_audit.json")
    if not atmosphere:
        atmosphere = meta.get("atmosphere_reconstruction_audit") or {}
    budget = _read_json(root / "error_breakdown_diagnostics.json")
    if not budget:
        budget = meta.get("error_breakdown_diagnostics") or {}
    anisoplanatism = _read_json(root / "anisoplanatism_diagnostics.json")
    if not anisoplanatism:
        anisoplanatism = meta.get("anisoplanatism_diagnostics") or {}
    if not atmosphere:
        raise ValueError(f"No atmosphere provenance in {root}")
    cube = np.load(root / "telemetry_only_psf.npy", allow_pickle=False)
    return meta, atmosphere, budget, anisoplanatism, cube


def _normalized_image(cube):
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim == 3:
        image = np.sum(cube, axis=2)
    elif cube.ndim == 2:
        image = cube
    else:
        raise ValueError("PSF must be 2D or (rows, cols, spectral_bins)")
    if (image.shape[0] != image.shape[1]
            or not np.all(np.isfinite(image)) or np.min(image) < -1e-12):
        raise ValueError("Invalid PSF image or non-square support")
    image = np.maximum(image, 0)
    norm = float(np.sum(image))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("PSF has no positive total flux")
    return image / norm


def _radii(image):
    yy, xx = np.indices(image.shape)
    center_y, center_x = np.array(image.shape) // 2
    return np.hypot(xx - center_x, yy - center_y)


def _radial_profile(image):
    radius = np.floor(_radii(image) + 0.5).astype(int)
    num = np.bincount(radius.ravel(), weights=image.ravel())
    den = np.bincount(radius.ravel())
    return num / np.maximum(den, 1)


def _encircled_energy(image, radii=(2, 4, 8, 16, 32)):
    dist = _radii(image)
    return {str(r): float(np.sum(image[dist <= r])) for r in radii}


def compare(one, seven, *, allow_mismatch=False):
    meta_1, atm_1, err_1, aniso_1, cube_1 = _read_run(one)
    meta_7, atm_7, err_7, aniso_7, cube_7 = _read_run(seven)
    flags = []
    comparisons = {
        "input_fits": (
            meta_1.get("input_fits"), meta_7.get("input_fits")
        ),
        "input_sav": (
            meta_1.get("input_sav"), meta_7.get("input_sav")
        ),
        "ao_mode": (meta_1.get("ao_mode"), meta_7.get("ao_mode")),
        "tt_kernel_convention": (
            meta_1.get("tt_kernel_convention"), meta_7.get("tt_kernel_convention")
        ),
        "tt_jitter_covariance_method": (
            meta_1.get("tt_jitter_covariance_method"),
            meta_7.get("tt_jitter_covariance_method")
        ),
        "tt_noise_method": (
            meta_1.get("tt_noise_method"), meta_7.get("tt_noise_method")
        ),
        "tt_mirror_model": (
            meta_1.get("tt_mirror_model"), meta_7.get("tt_mirror_model")
        ),
        "ho_covariance_policy": (
            meta_1.get("ho_covariance_policy"),
            meta_7.get("ho_covariance_policy")
        ),
        "psf_reference_wavelength_m": (
            meta_1.get("psf_reference_wavelength_m"),
            meta_7.get("psf_reference_wavelength_m")
        ),
        "pupil_angle_nirc2_deg": (
            (meta_1.get("pupil_orientation_diagnostics") or {}).get(
                "nirc2_pupil_angle_deg"
            ),
            (meta_7.get("pupil_orientation_diagnostics") or {}).get(
                "nirc2_pupil_angle_deg"
            ),
        ),
        "addNCPA": (meta_1.get("addNCPA"), meta_7.get("addNCPA")),
    }
    mismatches = []
    for key, (lhs, rhs) in comparisons.items():
        if lhs != rhs:
            mismatches.append(f"{key}: one={lhs!r}, seven={rhs!r}")

    layers_1 = atm_1.get("atmosphere_layer_count")
    layers_7 = atm_7.get("atmosphere_layer_count")
    if (layers_1, layers_7) != (1, 7):
        mismatches.append(f"layer_counts: one={layers_1!r}, seven={layers_7!r}")

    for key in ["r0_tel_m_at_atmosphere_wavelength", "fitted_L0_m"]:
        lhs, rhs = atm_1.get(key), atm_7.get(key)
        if lhs is None or rhs is None or not np.isclose(
            float(lhs), float(rhs), rtol=1e-8, atol=1e-12
        ):
            mismatches.append(f"{key}: one={lhs!r}, seven={rhs!r}")

    if mismatches and not allow_mismatch:
        raise ValueError(
            "Cannot attribute changes to atmospheric layer count when "
            "other parameters differ.\n" + "\n".join(mismatches)
            + "\nUse --allow-mismatch only for exploratory, not controlled tests."
        )
    if mismatches:
        flags.append("DIFFERENT_PHYSICAL_PARAMETERS")
        warnings.warn("; ".join(mismatches), RuntimeWarning, stacklevel=2)

    sha_1 = (meta_1.get("code_revision") or {}).get("git_revision")
    sha_7 = (meta_7.get("code_revision") or {}).get("git_revision")
    if not err_1 or not err_7:
        flags.append("INCOMPLETE_WFE_ERROR_BUDGET")
    if not aniso_1 or not aniso_7:
        flags.append("INCOMPLETE_ANISOPLANATISM_DIAGNOSTICS")
    if sha_1 != sha_7:
        flags.append("DIFFERENT_CODE_REVISIONS")
        warnings.warn(
            "The runs have different Git revisions; check physics changes "
            "before interpreting the difference.", RuntimeWarning, stacklevel=2
        )
    img1, img7 = _normalized_image(cube_1), _normalized_image(cube_7)
    if img1.shape != img7.shape:
        raise ValueError(f"Incompatible PSF shapes {img1.shape} and {img7.shape}")
    metrics = {
        "one_layer_directory": str(Path(one).resolve()),
        "seven_layer_directory": str(Path(seven).resolve()),
        "code_revisions": [sha_1, sha_7],
        "comparison_flags": flags,
        "parameter_mismatches": mismatches,
        "peak_fraction_one": float(img1.max()),
        "peak_fraction_seven": float(img7.max()),
        "relative_peak_change_seven_minus_one": (
            float(img7.max()/img1.max()-1)
        ),
        "total_variation_distance": float(0.5 * np.sum(np.abs(img7-img1))),
        "encircled_energy_one_by_pixel_radius": _encircled_energy(img1),
        "encircled_energy_seven_by_pixel_radius": _encircled_energy(img7),
        "anisoplanatism_one": aniso_1,
        "anisoplanatism_seven": aniso_7,
        "anisoplanatism_difference_seven_minus_one": {
            k: float(aniso_7[k])-float(aniso_1[k])
            for k in aniso_1.keys() & aniso_7.keys()
            if k.endswith("_wfe_nm")
            and isinstance(aniso_1[k], (float, int))
            and isinstance(aniso_7[k], (float, int))
        },
        "error_budget_one": err_1,
        "error_budget_seven": err_7,
        "error_budget_difference_seven_minus_one": {
            k: float(err_7[k]) - float(err_1[k])
            for k in err_1.keys() & err_7.keys()
            if isinstance(err_1[k], (float, int))
            and isinstance(err_7[k], (float, int))
        },
    }
    return metrics, img1, img7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", type=Path, required=True)
    parser.add_argument("--seven", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()
    metrics, im1, im7 = compare(
        args.one, args.seven, allow_mismatch=args.allow_mismatch
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "layer_comparison.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = im7 - im1
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, image, name in [
        (axes[0, 0], im1, "1 equivalent layer"),
        (axes[0, 1], im7, "7 MASS-DIMM layers"),
    ]:
        ax.imshow(
            np.log10(np.maximum(image, 1e-8)), origin="lower",
            vmin=-7, vmax=-1.4, cmap="magma"
        )
        ax.set_title(name)
    vlim = np.max(np.abs(delta))
    axes[1, 0].imshow(
        delta, vmin=-vlim, vmax=vlim,
        cmap="RdBu_r", origin="lower",
    )
    axes[1, 0].set_title("PSF (7 layers) − PSF (1 layer)")
    profile_1 = _radial_profile(im1)
    profile_7 = _radial_profile(im7)
    rad = np.arange(min(len(profile_1), len(profile_7)))
    axes[1, 1].semilogy(
        rad, np.maximum(profile_1[:len(rad)], 1e-12), label="1 layer"
    )
    axes[1, 1].semilogy(
        rad, np.maximum(profile_7[:len(rad)], 1e-12), label="7 layers"
    )
    axes[1, 1].set_xlim(0, min(50, len(rad)-1))
    axes[1, 1].set_xlabel("Radius from fixed PSF center [pixels]")
    axes[1, 1].set_ylabel("Mean normalized PSF/pixel")
    axes[1, 1].set_title("Radial profiles: no fitting or recentering")
    axes[1, 1].legend()
    fig.suptitle(
        f"Keck PSF-R layer-count test, normalized peak ratio = "
        f"{metrics['peak_fraction_seven']/metrics['peak_fraction_one']:.4f}"
    )
    fig.savefig(args.output / "layer_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
