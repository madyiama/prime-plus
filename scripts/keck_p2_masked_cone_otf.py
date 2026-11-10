"""P2 Keck prototype: exact masked, nonstationary LGS cone OTF with modal projection.

This is NOT the PSF-R production model and never fits an observed star.
It independently builds the cone-residual covariance on a sampled pupil,
removes weighted piston or weighted piston+x/y tilt, then evaluates

  OTF_cone(rho) = Σ_(i,j: r_i-r_j=rho) A_i A_j
                    exp[-Var(phi_i-phi_j)/2] / Σ_i A_i**2

without assuming stationarity or using FourierUtils.matrix_to_map().

A_i is the AMPLITUDE transmission of the same rotated Keck pupil
that telescope() loads for the science model, including any apodizer.
The weighted modal fit uses intensity weights A_i**2; OTF pairs use
A_i A_j. A separate comparison uses the *same* illuminated pupil pairs
but averages the SF before exponentiation. This isolates the Jensen
stationarisation error, not pupil geometry.

The native FFT PSFs here are low-resolution *focal-only* diagnostics.
They are not full AO PSFs, do not include HO/TT/anisokinetism/NCPA,
and are not suitable to compare directly with an observed science star.

Examples:
  uv run python scripts/keck_p2_masked_cone_otf.py \
    --case results/n0120_P2_1layers --pupil keck \
    --grid-sizes 15,21,31 --output results/n0120_P2_keck_cone_otf

  uv run python scripts/keck_p2_masked_cone_otf.py \
    --case results/n0120_P2_7layers --pupil circle \
    --outer-scales inf --grid-sizes 21,31 \
    --output results/n0120_P2_circle_7layers

Default L0 list = measured L0 from the case and inf (Kolmogorov).
The finite L0 structure function uses the McDonald von Karman kernel
normalized to Dphi = 6.88 (rho/r0)^(5/3) as L0 -> infinity.
"""

from __future__ import annotations

import argparse
import ast
import csv
from configparser import ConfigParser
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import gamma, kv

# Allow direct "uv run python scripts/..." on macOS, not only module imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.keck_l0_cone_sweep import load_geometry


def load_case_geometry(case: Path):
    """Load a completed P2 smoke run, including its analytic Hardy reference.

    The L0 sweep loader returns atmospheric/OTF geometry but deliberately
    does not flatten Hardy WFE. This script needs the same reference for
    output metrics, so transfer it explicitly and validate its units.
    """
    g = load_geometry(Path(case))
    raw = g["aniso"].get("hardy_cone_wfe_nm")
    if raw is None:
        raise ValueError(
            "Missing hardy_cone_wfe_nm in anisoplanatism_diagnostics.json; "
            "regenerate the P2 diagnostics before running this prototype"
        )
    hardy_nm = float(raw)
    if not np.isfinite(hardy_nm) or hardy_nm <= 0:
        raise ValueError(
            "Invalid Hardy cone WFE [nm] in anisoplanatism diagnostics"
        )
    g["hardy_wfe_nm"] = hardy_nm
    return g


def _numeric_ini_value(cfg, section, key, fallback=None):
    if not cfg.has_option(section, key):
        return fallback
    raw = cfg.get(section, key)
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def load_keck_science_pupil(case: Path, diameter_m: float):
    """Use the exact rotated Keck telescope pupil and apodizer from run INI.

    The 'PathPupil' input must exist: silently substituting an annulus
    would invalidate this experiment. telescope() performs the same
    FITS loading, image rotation and image-size interpolation as aoSystem.
    """
    ini_files = sorted(Path(case).glob("*.ini"))
    if len(ini_files) != 1:
        raise ValueError(
            f"Expected exactly one generated Keck INI in {case}; "
            f"found {len(ini_files)}"
        )
    cfg = ConfigParser()
    cfg.optionxform = str
    cfg.read(ini_files[0])
    D = float(_numeric_ini_value(cfg, "telescope", "TelescopeDiameter"))
    if not np.isclose(D, diameter_m, rtol=1e-10):
        raise ValueError("Case telescope diameter differs from science INI")
    filename = _numeric_ini_value(cfg, "telescope", "PathPupil")
    if not filename:
        raise ValueError("Keck INI has no PathPupil; do not use an annulus here")
    from p3.aoSystem import resolve_config_path
    root = str(Path(__file__).resolve().parents[1] / "p3")
    resolved = Path(resolve_config_path(str(filename), "", root))
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Keck calibrated pupil FITS missing: {resolved}"
        )
    apod = _numeric_ini_value(cfg, "telescope", "PathApodizer", "")
    resolved_apod = None
    if apod:
        resolved_apod = Path(resolve_config_path(str(apod), "", root))
        if not resolved_apod.is_file():
            raise FileNotFoundError(
                f"Calibrated Keck apodizer FITS missing: {resolved_apod}"
            )

    from p3.aoSystem.telescope import telescope
    tel = telescope(
        D,
        int(_numeric_ini_value(cfg, "telescope", "Resolution")),
        zenith_angle=float(_numeric_ini_value(
            cfg, "telescope", "ZenithAngle", 0
        )),
        obsRatio=float(_numeric_ini_value(
            cfg, "telescope", "ObscurationRatio", 0
        )),
        pupilAngle=float(_numeric_ini_value(
            cfg, "telescope", "PupilAngle", 0
        )),
        path_pupil=str(resolved),
        path_apodizer=(
            str(resolved_apod) if resolved_apod is not None else None
        ),
        verbose=False,
    )
    pupil = np.asarray(tel.pupil, dtype=np.float64)
    apo = np.asarray(tel.apodizer, dtype=np.float64)
    amplitude = pupil*apo
    if pupil.ndim != 2 or pupil.shape[0] != pupil.shape[1]:
        raise ValueError("Keck science pupil must be a square array")
    if not np.all(np.isfinite(amplitude)):
        raise ValueError("Nonfinite Keck pupil amplitude")
    # Rotation/interpolation of binary aperture FITS can cause very
    # small cubic overshoot; count it for audit rather than ignore it.
    overshoot = int(np.count_nonzero((amplitude < 0) | (amplitude > 1)))
    amplitude = np.clip(amplitude, 0, 1)
    return amplitude, {
        "source": str(resolved),
        "apodizer": (
            str(resolved_apod) if resolved_apod is not None else None
        ),
        "resolution": list(amplitude.shape),
        "pupil_angle_deg": float(tel.pupilAngle),
        "rotation_or_interpolation_overshoot_pixels": overshoot,
        "pupil_amplitude_sum": float(amplitude.sum()),
        "pupil_intensity_sum": float(np.sum(amplitude**2)),
        "source_type": "telescope.pupil times telescope.apodizer",
    }


def _sample_pupil(D: float, n: int, kind: str, full_pupil=None):
    if n < 5 or n % 2 != 1:
        raise ValueError("Pupil grid size must be odd and >=5")
    if kind not in ("circle", "keck"):
        raise ValueError("Pupil kind must be circle or keck")
    # telescope.py's coordinate convention for pixel centers.
    coords = (np.arange(n)-(n-1)/2)*D/n
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    if kind == "circle":
        amp = (xx*xx + yy*yy <= (D/2)**2).astype(float)
    else:
        if full_pupil is None:
            raise ValueError("Real Keck science-pupil map must be supplied")
        N = int(full_pupil.shape[0])
        grid_coords = (
            (np.arange(n)-(n-1)/2)*N/n + (N-1)/2
        )
        iy, ix = np.meshgrid(grid_coords, grid_coords, indexing="ij")
        amp = map_coordinates(
            np.asarray(full_pupil, dtype=float), [iy, ix],
            order=1, mode="constant", cval=0.0
        )
        amp = np.clip(amp, 0, 1)
    good = amp > 1e-6
    if good.sum() < 12:
        raise ValueError("Sampled pupil has too few illuminated points")
    index_y, index_x = np.nonzero(good)
    return {
        "amplitude": amp[good],
        "grid_pupil": amp,
        "x": xx[good],
        "y": yy[good],
        "ix": index_x,
        "iy": index_y,
        "n": n,
        "D": D,
        "sample_pitch_m": float(D/n),
        "illuminated_points": int(good.sum()),
    }


def phase_structure_function_von_karman(
    rho_m, r0_m: float, L0_m: float,
):
    """Isotropic phase SF in radians² at r0's wavelength.

    Normalize the von Karman kernel to the *exact* Fried/Kolmogorov
    coefficient 6.88, rather than P3's rounded 6.905 multiplier.
    Use the analytic Kolmogorov power law for L0=infinity to avoid
    cancellation of nearly equal McDonald-function terms.
    """
    rho = np.asarray(rho_m, dtype=float)
    if r0_m <= 0 or (not (L0_m > 0 or np.isposinf(L0_m))):
        raise ValueError("Invalid r0 or outer scale")
    if np.isposinf(L0_m):
        return 6.88*(rho/r0_m)**(5.0/3.0)
    nu = 5.0/6.0
    a = -gamma(-nu)/(2**(2*nu+1)*gamma(1+nu))
    z = 2*np.pi*rho/L0_m
    B = np.full_like(z, 3.0/5.0, dtype=float)
    nz = z > 0
    B[nz] = (
        z[nz]**nu * kv(nu, z[nz])
        / (2**nu * gamma(1+nu))
    )
    sf = (
        6.88*(L0_m/r0_m)**(5.0/3.0)
        * (3.0/5.0-B)/(a*(2*np.pi)**(5.0/3.0))
    )
    return np.maximum(sf, 0.0)


def _cone_covariance(geometry, sampled, L0):
    """Full LGS cone-residual covariance on the sampled pupil [rad²]."""
    x, y = sampled["x"], sampled["y"]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    C = np.zeros((x.size, x.size), dtype=np.float64)
    r0 = float(geometry["r0_m"])
    H = float(geometry["lgs_height_m"])
    for weight, height in zip(geometry["weights"], geometry["heights"]):
        if float(weight) <= 0 or float(height) == 0:
            continue
        g = float(height)/H
        sf = lambda a, b: phase_structure_function_von_karman(
            np.hypot(a, b), r0, float(L0)
        )
        C += 0.5 * float(weight) * (
            sf(x[:, None]-(1-g)*x[None, :],
               y[:, None]-(1-g)*y[None, :])
            + sf((1-g)*x[:, None]-x[None, :],
                 (1-g)*y[:, None]-y[None, :])
            - sf(dx, dy)
            - sf((1-g)*dx, (1-g)*dy)
        )
    C = 0.5*(C+C.T)
    if not np.all(np.isfinite(C)):
        raise ValueError("Nonfinite LGS cone covariance")
    return C


def _project_modes(C, pupil, remove_tilt):
    """Weighted piston/tilt covariance projection; low-rank form.

    R = I - F (Fᵀ W F)^-1 Fᵀ W; C_projected = R C Rᵀ.
    W is normalized pupil intensity A², and F contains real piston,
    x tilt, y tilt shapes. No scalar WFE subtraction is performed.
    """
    amp, x, y, D = (
        pupil["amplitude"], pupil["x"], pupil["y"], pupil["D"]
    )
    w = amp**2/np.sum(amp**2)
    cols = [np.ones_like(x)]
    if remove_tilt:
        cols.extend((x/D, y/D))
    F = np.column_stack(cols)
    G = F.T@(w[:, None]*F)
    H = np.linalg.solve(G, F.T*w[None, :])
    HC = H@C
    CG = HC@H.T
    Cp = C - F@HC - (F@HC).T + F@CG@F.T
    Cp = 0.5*(Cp+Cp.T)
    # Weighted phase-variance identity: trace(W Cp).
    variance = float(np.dot(w, np.diag(Cp)))
    if variance < -1e-8:
        raise ValueError("Projected cone phase variance negative")
    return Cp, max(0., variance)


def _nonstationary_pupil_otf(Cp, pupil):
    """Exact Gaussian ensemble OTF from all illuminated point pairs.

    Pair lags are integer multiples of the sampled telescope pupil grid.
    The zero-lag OTF is exactly unity; pupil autocorrelation weights
    are coherent A_i*A_j and modal-fit weights were A_i**2.
    """
    amp = pupil["amplitude"]
    d = np.diag(Cp)
    phase_sf = d[:, None] + d[None, :] - 2*Cp
    scale = max(float(np.max(np.abs(phase_sf))), 1.0)
    if np.min(phase_sf) < -1e-8*scale:
        raise ValueError(
            "Negative phase structure function after modal projection"
        )
    phase_sf = np.maximum(phase_sf, 0.0)
    n = pupil["n"]
    bins = (
        (pupil["iy"][:, None] - pupil["iy"][None, :] + (n-1))
        * (2*n-1)
        + (pupil["ix"][:, None] - pupil["ix"][None, :] + (n-1))
    ).ravel()
    weights = (amp[:, None]*amp[None, :]).ravel()
    denom = float(np.dot(amp, amp))
    count = (2*n-1)**2
    raw_dl = np.bincount(
        bins, weights=weights, minlength=count
    ).reshape(2*n-1, 2*n-1)
    raw_cone = np.bincount(
        bins, weights=weights*np.exp(-0.5*phase_sf.ravel()),
        minlength=count
    ).reshape(2*n-1, 2*n-1)
    raw_average = np.bincount(
        bins, weights=weights*phase_sf.ravel(),
        minlength=count
    ).reshape(2*n-1, 2*n-1)
    dl = raw_dl/denom
    exact = raw_cone/denom
    mean_sf = np.divide(
        raw_average, raw_dl,
        out=np.zeros_like(raw_average), where=raw_dl > 0
    )
    stationary = dl*np.exp(-0.5*mean_sf)
    centre = n-1
    if not (np.isclose(dl[centre, centre], 1.0, atol=1e-12)
            and np.isclose(exact[centre, centre], 1.0, atol=1e-12)
            and np.isclose(stationary[centre, centre], 1.0, atol=1e-12)):
        raise ValueError("Cone/diffraction OTF must have DC=1")
    if not (np.allclose(exact, exact[::-1, ::-1], atol=2e-12)
            and np.allclose(stationary, stationary[::-1, ::-1], atol=2e-12)):
        raise ValueError("Cone OTF is not Hermitian-even")
    if np.max(stationary-exact) > 1e-9:
        raise ValueError("Jensen inequality failed: average before exp")
    return dl, exact, stationary, mean_sf


def _otf_psf_metrics(dl, otf):
    """Native-grid *focal-only* PSF and Strehl, independent of HO/TT."""
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(otf))).real
    psf_dl = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dl))).real
    s = float(psf.sum())
    s_dl = float(psf_dl.sum())
    if s <= 0 or s_dl <= 0:
        raise ValueError("Nonpositive focal-only PSF flux")
    psf /= s
    psf_dl /= s_dl
    return {
        "otf_dc": float(otf[otf.shape[0]//2, otf.shape[1]//2]),
        "strehl_from_otf_integral": float(otf.sum()/dl.sum()),
        "strehl_from_psf_peak_ratio": float(psf.max()/psf_dl.max()),
        "peak_fraction_native": float(psf.max()),
        "negative_psf_flux_fraction": float(
            np.maximum(-psf, 0).sum()
            / max(np.maximum(psf, 0).sum(), 1e-30)
        ),
    }, psf


def evaluate_configuration(geometry, pupil, L0):
    """Compute piston-only and piston+tilt nonstationary Keck cone OTF."""
    C = _cone_covariance(geometry, pupil, L0)
    n = pupil["n"]
    result = {
        "outer_scale_m": "infinity" if np.isposinf(L0) else float(L0),
        "pupil_grid_n": n,
        "illuminated_points": pupil["illuminated_points"],
        "pupil_source": geometry.get("pupil_source", "unknown"),
        "pupil_sampling_pitch_m": pupil["sample_pitch_m"],
        "pupil_intensity_sum": float(np.sum(pupil["amplitude"]**2)),
        "r0_m": float(geometry["r0_m"]),
        "wavelength_m": float(geometry["wavelength_m"]),
        "modal_conventions": [
            "piston_removed", "piston_and_tilt_removed"
        ],
    }
    saved = {}
    nm_per_rad = geometry["wavelength_m"]*1e9/(2*np.pi)
    for name, has_tilt in (
        ("piston_removed", False),
        ("piston_and_tilt_removed", True),
    ):
        Cp, phasevar = _project_modes(C, pupil, has_tilt)
        dl, otf, stationary, _ = _nonstationary_pupil_otf(Cp, pupil)
        metrics, psf = _otf_psf_metrics(dl, otf)
        metric_approx, _ = _otf_psf_metrics(dl, stationary)
        metrics.update({
            "pupil_projected_phase_variance_rad2": phasevar,
            "pupil_projected_wfe_nm": float(
                np.sqrt(phasevar)*nm_per_rad
            ),
            "stationarized_otf_strehl": metric_approx[
                "strehl_from_otf_integral"
            ],
            "difference_exact_minus_stationarized_strehl": (
                metrics["strehl_from_otf_integral"]
                - metric_approx["strehl_from_otf_integral"]
            ),
            "hardy_reference_wfe_nm": (
                float(geometry["hardy_wfe_nm"])
                if has_tilt else None
            ),
            "hardy_relative_error_piston_tilt_only": (
                np.sqrt(phasevar)*nm_per_rad/geometry["hardy_wfe_nm"]-1
                if has_tilt else None
            ),
        })
        result[name] = metrics
        saved[name] = {
            "otf": otf, "otf_dl": dl,
            "stationary": stationary, "psf": psf,
        }
    return result, saved


def _write_outputs(summary, saved, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/"masked_cone_otf_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    rows = []
    for e in summary["evaluations"]:
        for key in ("piston_removed", "piston_and_tilt_removed"):
            row = {
                "L0_m": e["outer_scale_m"],
                "grid_n": e["pupil_grid_n"],
                "pupil_kind": e["pupil_source"],
                "modal_convention": key,
            }
            row.update(e[key])
            rows.append(row)
    with (output/"masked_cone_otf_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [
        f"{e['pupil_grid_n']} / {e['outer_scale_m']}"
        for e in summary["evaluations"]
    ]
    xx = np.arange(len(labels))
    for key, name in (
        ("piston_removed", "piston only"),
        ("piston_and_tilt_removed", "piston + tilt removed"),
    ):
        axes[0].plot(
            xx, [e[key]["pupil_projected_wfe_nm"]
                 for e in summary["evaluations"]],
            "o-", label=name
        )
        axes[1].plot(
            xx, [e[key]["strehl_from_otf_integral"]
                 for e in summary["evaluations"]],
            "o-", label=name
        )
    axes[0].plot(
        xx, [e["piston_and_tilt_removed"]["hardy_reference_wfe_nm"]
             for e in summary["evaluations"]],
        "k--", label="Hardy: circular piston+tilt, infinite L0"
    )
    for ax in axes:
        ax.set_xticks(xx, labels, rotation=25, ha="right")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Pupil piston/tilt-removed RMS OPD [nm]")
    axes[1].set_ylabel("Focal-only OTF Strehl (not full PSF-R SR)")
    fig.suptitle("P2 nonstationary LGS cone on sampled pupil")
    fig.tight_layout()
    fig.savefig(output/"masked_cone_otf_audit.png", dpi=150)
    plt.close(fig)
    # Native arrays are deliberately saved per grid/L0, not as a single
    # "corrected PSF-R" file, to avoid confusing the scientific scope.
    for label, bundle in saved.items():
        for key, data in bundle.items():
            np.save(output/f"{label}_{key}.npy", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pupil", choices=("keck", "circle"), default="keck"
    )
    parser.add_argument(
        "--grid-sizes", default="15,21,31",
        help="Odd axis sizes. n=43 can be slow, especially with 7 layers."
    )
    parser.add_argument(
        "--outer-scales", default=None,
        help="Comma-separated L0 in m; default measured and inf."
    )
    args = parser.parse_args()
    g = load_case_geometry(args.case)
    sizes = [int(s.strip()) for s in args.grid_sizes.split(",")]
    if not sizes or any(n < 5 or n%2 == 0 for n in sizes):
        parser.error("Grid axis sizes must all be odd integers >=5")
    outer = (
        [float(v.strip()) for v in args.outer_scales.split(",")]
        if args.outer_scales
        else [float(g["aniso"]["focal_model_conditions"][
            "von_karman_outer_scale_m"
        ]), np.inf]
    )
    if any(not (v > 0 and (np.isfinite(v) or np.isposinf(v))) for v in outer):
        parser.error("Outer scales must be positive or inf")
    case = Path(args.case)
    full_pupil, provenance = (
        load_keck_science_pupil(case, g["diameter_m"])
        if args.pupil == "keck"
        else (None, {
            "source": "synthetic_ideal_circle", "pupil_angle_deg": 0
        })
    )
    g["pupil_source"] = args.pupil
    summaries, last = [], {}
    for n in sizes:
        sampled = _sample_pupil(
            g["diameter_m"], n, args.pupil, full_pupil
        )
        for L0 in outer:
            print(
                f"[P2 masked cone] pupil={args.pupil}, n={n}, "
                f"illuminated={sampled['illuminated_points']}, L0={L0:g}, "
                f"layers={len(g['weights'])}",
                flush=True
            )
            metrics, products = evaluate_configuration(g, sampled, L0)
            summaries.append(metrics)
            suffix = "inf" if np.isposinf(L0) else f"{L0:g}".replace(".", "p")
            for name, bundle in products.items():
                for key, arr in bundle.items():
                    last[f"n{n}_L0{suffix}_{name}_{key}"] = arr
            print(
                "  cone RMS piston = "
                f"{metrics['piston_removed']['pupil_projected_wfe_nm']:.2f}nm; "
                "piston+tilt = "
                f"{metrics['piston_and_tilt_removed']['pupil_projected_wfe_nm']:.2f}nm; "
                "Hardy = "
                f"{metrics['piston_and_tilt_removed']['hardy_reference_wfe_nm']:.2f}nm",
                flush=True,
            )
    # This is a benchmark against the existing P3 *piston-only*
    # result, not a claim that piston+tilt should be inserted into the
    # full closed-loop OTF before calibrating the HO/STRAP modal split.
    legacy = g["aniso"].get("pupil_sf_bridge_focal_only", {})
    summary = {
        "source_case": str(case.resolve()),
        "legacy_p3_focal_only_reference": {
            "model": "P3 focal SF, historical square-grid averaging",
            "pupil_sf_wfe_nm": legacy.get(
                "pupil_piston_removed_wfe_nm"
            ),
            "strehl_equivalent_wfe_nm": legacy.get(
                "strehl_equivalent_wfe_nm"
            ),
            "comparability_warning": (
                "P3 historical focal SF is not the same modal/pupil "
                "definition as the new piston+tilt removed Keck model."
            ),
        },
        "pupil_provenance": provenance,
        "pupil_sampling": (
            "grid centers span [-D/2,D/2] with pitch D/n, "
            "same metric pixel-center definition as telescope.py"
        ),
        "physical_assumptions": {
            "cone_covariance": "independent 2-point von Karman or Kolmogorov",
            "pupil_weights_modal_fit": "A_i^2",
            "otf_pair_weights": "A_i*A_j",
            "otf_nonstationary": "mean[exp(-Dpair/2)] at each lag",
            "otf_stationary_approx": "exp[-mean(Dpair)/2]",
            "tilt_basis": "weighted least-squares x,y removed in pupil",
            "tt_noise_model": "NOT included; not replacing STRAP term",
            "ho_model": "NOT included; not replacing PSF-R AO residual",
            "ncpa": "NOT included; no source fitting",
            "finite_L0_normalization": (
                "continuous von Karman normalized to 6.88 Kolmogorov limit"
            ),
            "pupil_orientation": (
                "telescope() uses Keck science INI PupilAngle and PathPupil "
                "without arbitrary rotation offset"
            ),
        },
        "geometry": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in g.items() if k not in ("aniso", "otf")
        },
        "evaluations": summaries,
        "warning": (
            "Prototype only. Tilt-removed LGS cone must not replace a "
            "production term until Keck HO/STRAP modal split is independently "
            "verified. Native coarse-grid PSFs are not science PSF-R."
        ),
    }
    _write_outputs(summary, last, Path(args.output))


if __name__ == "__main__":
    main()
