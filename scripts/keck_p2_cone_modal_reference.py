"""Independent Kolmogorov LGS cone covariance and tip-tilt modal audit.

Uses no matrix_to_map(), Fourier interpolation, WFS reconstruction, or
science source fitting. Independently evaluates the stationary 6.88
Kolmogorov phase structure function for the cone residual

    delta_phi(r) = phi(r) - phi((1-h/H)*r),

and projects the resulting 2-point phase covariance onto weighted
piston and x/y tilt. Compares the resulting piston+tilt-removed RMS
with Hardy/Sasiela's analytical focal-cone formula (0.452).

Example:
  uv run python scripts/keck_p2_cone_modal_reference.py \
      --case results/n0120_P2_1layers \
      --grid-sizes 21,43,61 --include-square \
      --output results/n0120_P2_modal_reference

Only the Kolmogorov L0=infinity case is implemented. The finite L0
comparison is a separate experiment (keck_p2_spatial_sampling_audit.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _geometry_from_case(folder: Path) -> dict:
    source = folder / "anisoplanatism_diagnostics.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}")
    d = json.loads(source.read_text(encoding="utf-8"))
    g = {
        "r0_m": float(d["atmosphere_r0_m"]),
        "wavelength_m": float(d["reference_wavelength_m"]),
        "lgs_height_m": float(d["lgs_height_los_m"]),
        "weights": np.asarray(d["layer_weights"], dtype=float),
        "heights_m": np.asarray(d["layer_heights_los_m"], dtype=float),
        "hardy_wfe_nm": float(d["hardy_cone_wfe_nm"]),
        "diameter_m": 10.5,
    }
    grid_file = folder / "frequency_grid_diagnostics.json"
    if grid_file.is_file():
        f = json.loads(grid_file.read_text(encoding="utf-8"))
        g["diameter_m"] = float(f["telescope_diameter_m"])
    if (g["weights"].size != g["heights_m"].size
            or not np.all(np.isfinite(g["weights"]))
            or not np.all(np.isfinite(g["heights_m"]))
            or np.any(g["weights"] < 0)
            or np.any(g["heights_m"] < 0)
            or np.any(g["heights_m"] >= g["lgs_height_m"])
            or not np.isclose(g["weights"].sum(), 1.0, atol=1e-6)
            or min(g["r0_m"], g["wavelength_m"], g["diameter_m"],
                   g["lgs_height_m"]) <= 0):
        raise ValueError("Invalid turbulence weights, heights, r0 or LGS range")
    return g


def _pupil_coordinates(D: float, n: int, shape: str):
    if n < 5 or n % 2 == 0:
        raise ValueError("Grid axis must have an odd number of >=5 points")
    if shape not in ("circle", "square"):
        raise ValueError("Pupil shape must be circle or square")
    u = np.linspace(-D/2, D/2, n)
    x, y = np.meshgrid(u, u)
    mask = (x*x + y*y <= (D/2)**2 + 1e-12) if shape == "circle" else np.ones_like(x, bool)
    return x[mask].ravel(), y[mask].ravel()


def _cone_covariance_block(xi, yi, xj, yj, g: float, prefactor: float):
    """Stationary phase covariance of phi(r)-phi((1-g)r).

    The 4 Kolmogorov differences cancel the unknown phase piston
    covariance exactly, so this stays finite in the L0=infinity limit.
    """
    p = lambda a, b: np.hypot(a, b)**(5.0/3.0)
    return 0.5*prefactor*(
        p(xi-(1-g)*xj, yi-(1-g)*yj)
        + p((1-g)*xi-xj, (1-g)*yi-yj)
        - p(xi-xj, yi-yj)
        - p((1-g)*(xi-xj), (1-g)*(yi-yj))
    )


def independent_modal_budget(geometry, n_grid: int, pupil="circle",
                             block_size: int = 128):
    """Return piston-only and piston+tilt-removed cone variance in rad^2."""
    x, y = _pupil_coordinates(geometry["diameter_m"], n_grid, pupil)
    n = x.size
    w = np.full(n, 1.0/n)
    # Weighted least-squares piston, x tilt and y tilt. Modal design
    # coordinates are dimensionless; the span does not affect projection.
    F = np.column_stack((
        np.ones(n), x/geometry["diameter_m"], y/geometry["diameter_m"]
    ))
    WF = w[:, None]*F
    G = F.T @ WF
    coeff_cov = np.zeros((3, 3), float)
    total = 0.0
    pref = 6.88*geometry["r0_m"]**(-5.0/3.0)
    for strength, height in zip(geometry["weights"], geometry["heights_m"]):
        if strength == 0 or height == 0:
            continue
        g = height/geometry["lgs_height_m"]
        total += float(strength*pref*np.dot(
            w, np.hypot(g*x, g*y)**(5.0/3.0)
        ))
        for lo in range(0, n, block_size):
            hi = min(lo+block_size, n)
            C = _cone_covariance_block(
                x[lo:hi, None], y[lo:hi, None],
                x[None, :], y[None, :], g, pref*strength,
            )
            coeff_cov += WF[lo:hi].T @ (C @ WF)
    piston_removed = total - coeff_cov[0, 0]
    piston_tilt_removed = total - np.trace(np.linalg.solve(G, coeff_cov))
    tilt_contribution = piston_removed-piston_tilt_removed
    if min(piston_removed, piston_tilt_removed, tilt_contribution) < -1e-8:
        raise ValueError("Negative modal variance: recheck covariance/projection")
    nm_per_rad = geometry["wavelength_m"]*1e9/(2*np.pi)
    f = lambda variance: float(np.sqrt(max(variance, 0.0))*nm_per_rad)
    hardy = geometry["hardy_wfe_nm"]
    h_ratio = geometry["heights_m"]/geometry["lgs_height_m"]
    # Sasiela 2007 Eq. 6.196: piston removed only.
    # Eq. 6.197 removes piston AND tilt and reproduces Hardy's 0.452.
    analytic_piston_var = (
        (geometry["diameter_m"]/geometry["r0_m"])**(5.0/3.0)
        * np.dot(geometry["weights"],
                 (0.5*h_ratio**(5.0/3.0)-0.0835*h_ratio**2)/0.423)
    )
    return {
        "pupil": pupil,
        "grid_axis_points": int(n_grid),
        "illuminated_grid_points": int(n),
        "Kolmogorov_Dphi_prefactor": 6.88,
        "piston_removed_variance_rad2": float(piston_removed),
        "piston_removed_wfe_nm": f(piston_removed),
        "tilt_contribution_variance_rad2": float(tilt_contribution),
        "tilt_contribution_wfe_nm": f(tilt_contribution),
        "piston_and_tilt_removed_variance_rad2": float(piston_tilt_removed),
        "piston_and_tilt_removed_wfe_nm": f(piston_tilt_removed),
        "sasiela_analytic_piston_only_wfe_nm": f(analytic_piston_var),
        "relative_difference_vs_sasiela_piston_only": (
            f(piston_removed)/f(analytic_piston_var)-1.0
            if analytic_piston_var > 0 else None
        ),
        "hardy_piston_and_tilt_removed_wfe_nm": hardy,
        "relative_difference_vs_hardy": f(piston_tilt_removed)/hardy-1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-sizes", default="21,43")
    parser.add_argument("--include-square", action="store_true",
                        help="Also show unphysical full-square pupil averages")
    args = parser.parse_args()
    g = _geometry_from_case(args.case)
    sizes = [int(v.strip()) for v in args.grid_sizes.split(",")]
    rows = []
    for size in sizes:
        for shape in (("circle", "square") if args.include_square else ("circle",)):
            print(f"[Cone modal P2] n={size}, pupil={shape}", flush=True)
            row = independent_modal_budget(g, size, pupil=shape)
            rows.append(row)
            print("  piston only = {piston_removed_wfe_nm:.3f} nm; "
                  "piston+tilt removed = "
                  "{piston_and_tilt_removed_wfe_nm:.3f} nm; "
                  "Hardy = {hardy_piston_and_tilt_removed_wfe_nm:.3f} nm"
                  .format(**row), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": ("Hardy/Sasiela coefficient 0.452 corresponds "
                      "to piston AND tip-tilt removed cone WFE"),
        "L0": "infinity",
        "source_case": str(args.case.resolve()),
        "geometry": {k:(v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in g.items()},
        "results": rows,
        "note": "No science fitting, TT sensor gain, NIRC2 mask, or empirical scaling."
    }
    (args.output/"cone_modal_reference.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output/"cone_modal_reference.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
