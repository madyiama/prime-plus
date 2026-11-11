"""P1-B: read-only WFS/DM modal noise and Zernike-fit audit.

Run from the live systemDiagnosis object AFTER its original r0/L0 fit,
BEFORE the expensive PSF-R construction. Does NOT modify the production
noise estimator, the sign/timing of pseudo-open-loop, or r0/L0.

Outputs:
- per-mode and per-radial-order measured/noise-subtracted variances,
  theoretical fit, noise fractions and negative corrected modes;
- autocorrelation lag-noise PROXIES from DM commands, reconstructed WFS
  residual, and raw centroids, with physical units and limitations;
- estimated fixed-L0 r0 sensitivity to scaling noise subtraction;
- projection matrix conditioning and selected Noll-index ranges.

Important: a difference of successive CLOSED-LOOP DM commands is NOT
an unbiased measurement of WFS sensor noise. An 'alternative estimate'
below is only a test of sensitivity, not calibrated noise.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def lag_difference_variances(samples, lag=1, *, periodic=False):
    """Half the channel-wise squared sample difference, no input mutation."""
    a = np.asarray(samples, dtype=np.float64)
    if (a.ndim != 2 or len(a) < 3 or not np.all(np.isfinite(a))
            or not isinstance(lag, (int, np.integer))
            or lag < 1 or lag >= len(a)):
        raise ValueError("Expected finite (frames,channels), valid lag")
    if periodic:
        difference = a - np.roll(a, -int(lag), axis=0)
    else:
        difference = a[:-lag] - a[lag:]
    return 0.5*np.mean(difference*difference, axis=0)


def project_lag_variance(commands, indices, reconstructor, lag=1,
                         *, periodic=False):
    """Propagate time-difference covariance through *identical* u2z."""
    cmds = np.asarray(commands, dtype=float)
    idx = np.asarray(indices)
    R = np.asarray(reconstructor, dtype=float)
    if idx.dtype.kind == "b":
        idx = np.flatnonzero(idx)
    else:
        idx = idx.astype(int)
    if idx.ndim != 1 or len(idx) != R.shape[1]:
        raise ValueError("Selected actuators and u2z projector disagree")
    if cmds.ndim != 2 or np.any(idx < 0) or np.any(idx >= cmds.shape[1]):
        raise ValueError("Invalid actuator channel indices")
    # A projection of differences is equivalent to R C_delta R.T.
    projected = cmds[:, idx] @ R.T
    return lag_difference_variances(projected, lag, periodic=periodic)


def _radial_median(values, radial_orders):
    v = np.asarray(values, dtype=float)
    orders = np.asarray(radial_orders, dtype=int)
    if v.ndim != 1 or orders.shape != v.shape:
        raise ValueError("Radial orders and modal variances incompatible")
    unique = np.unique(orders)
    return unique, np.array([
        np.median(v[orders == order]) for order in unique
    ])


def fixed_l0_noise_sensitivity(
    raw_modal_opd2, estimated_noise_opd2, model_modal_rad2,
    radial_orders, wavelength_m, fitted_r0_m, multipliers=(0, 0.5, 1, 1.5),
):
    """Re-estimate only the turbulence AMPLITUDE with frozen L0.

    At fixed L0 the theoretical Zernike variance is proportional to
    r0^(-5/3). This is an analytical one-parameter diagnostic around
    the existing fitted (r0,L0), NOT a joint refit and NOT a validation
    of the detector noise covariance.
    """
    raw = np.asarray(raw_modal_opd2, dtype=float)
    noise = np.asarray(estimated_noise_opd2, dtype=float)
    model = np.asarray(model_modal_rad2, dtype=float)
    if (raw.shape != noise.shape or raw.shape != model.shape
            or np.any(~np.isfinite(raw)) or np.any(~np.isfinite(noise))
            or np.any(~np.isfinite(model)) or wavelength_m <= 0
            or fitted_r0_m <= 0):
        raise ValueError("Invalid modal variance arrays or units")
    _, ref_model = _radial_median(model, radial_orders)
    gain = (2*np.pi/wavelength_m)**2
    denom = float(ref_model@ref_model)
    if denom <= 0:
        raise ValueError("Theoretical modal variance model is zero")
    rows = []
    for eta in multipliers:
        eta = float(eta)
        if eta < 0:
            raise ValueError("Noise multiplier must be nonnegative")
        corrected = raw-eta*noise
        _, measured = _radial_median(gain*corrected, radial_orders)
        ratio = float(np.dot(ref_model, measured)/denom)
        rows.append({
            "noise_subtraction_multiplier": eta,
            "fixed_L0_best_variance_amplitude": ratio,
            "fixed_L0_r0_m": (
                float(fitted_r0_m*ratio**(-3.0/5.0))
                if ratio > 0 else None
            ),
            "negative_corrected_modes": int(np.count_nonzero(corrected < 0)),
            "negative_radial_medians": int(np.count_nonzero(measured < 0)),
            "mean_noise_to_raw_variance_ratio": float(
                np.sum(eta*noise)/max(float(np.sum(raw)), 1e-40)
            ),
            "relative_radial_fit_residual": float(
                np.linalg.norm(measured-ratio*ref_model)
                / max(float(np.linalg.norm(measured)), 1e-30)
            ),
        })
    return rows


def analyse_diagnosis(sd):
    """Extract P1 noise/mode diagnostics without touching the input object."""
    trs = sd.trs
    raw_var = np.asarray(trs.wfs.var_meas, dtype=float).reshape(-1)
    noise_var = np.asarray(trs.wfs.var_noise_zer, dtype=float).reshape(-1)
    model = np.asarray(trs.wfs.coeffs_mod, dtype=float).reshape(-1)
    noll = np.asarray(trs.wfs.jIndex, dtype=int)
    radial = np.asarray(sd.z.n, dtype=int)
    u2z = np.asarray(trs.mat.u2z, dtype=float)
    commands = np.asarray(trs.dm.com, dtype=float)
    residual = np.asarray(trs.rec.res, dtype=float)
    if (raw_var.shape != noise_var.shape or raw_var.shape != model.shape
            or raw_var.shape != noll.shape or raw_var.shape != radial.shape
            or u2z.shape[0] != len(noll)
            or commands.shape != residual.shape
            or np.any(~np.isfinite(raw_var))
            or np.any(~np.isfinite(noise_var))
            or np.any(~np.isfinite(model))):
        raise ValueError("Inconsistent Keck modal reconstruction arrays")
    idx = np.asarray(sd.actuInPupil)
    if idx.dtype.kind == "b":
        idx = np.flatnonzero(idx)
    else:
        idx = idx.astype(int)
    if u2z.shape[1] != len(idx):
        raise ValueError("u2z column count mismatches selected actuators")
    if noll.min() < 4:
        raise ValueError(
            "P1 high-order atmospheric fit must exclude Noll piston and tilt"
        )
    wl = float(trs.atm.wvl)
    r0 = float(trs.atm.r0_tel)
    l0 = float(trs.atm.L0_tel)
    rad_per_opd = (2*np.pi/wl)**2
    corrected = raw_var-noise_var
    mode_rows = []
    for j in range(len(noll)):
        mode_rows.append({
            "Noll": int(noll[j]),
            "radial_order": int(radial[j]),
            "variance_measured_OPD_m2": float(raw_var[j]),
            "variance_noise_proxy_OPD_m2": float(noise_var[j]),
            "variance_corrected_OPD_m2": float(corrected[j]),
            "variance_corrected_rad2": float(rad_per_opd*corrected[j]),
            "theory_at_fit_rad2": float(model[j]),
            "noise_fraction_of_measured": (
                float(noise_var[j]/raw_var[j])
                if raw_var[j] > 0 else None
            ),
        })
    nr, data_raw = _radial_median(raw_var*rad_per_opd, radial)
    _, data_noise = _radial_median(noise_var*rad_per_opd, radial)
    _, data_corrected = _radial_median(corrected*rad_per_opd, radial)
    _, theory_radial = _radial_median(model, radial)
    # Fit code uses median(after subtraction), not
    # median(raw)-median(noise): preserve that noncommutativity here.
    radial_rows = [{
        "radial_order": int(n),
        "median_measured_rad2": float(x),
        "median_noise_proxy_rad2": float(y),
        "median_corrected_rad2": float(z),
        "median_theory_fit_rad2": float(t),
        "corrected_minus_model_rad2": float(z-t),
    } for n, x, y, z, t in zip(
        nr, data_raw, data_noise, data_corrected, theory_radial
    )]
    command_proxy = (
        project_lag_variance(commands, idx, u2z, 1, periodic=True)
    )
    finite_command_proxy = (
        project_lag_variance(commands, idx, u2z, 1, periodic=False)
    )
    residual_proxy = (
        project_lag_variance(residual, idx, u2z, 1, periodic=False)
    )
    wfs_slope_sample = np.asarray(trs.wfs.slopes, dtype=float)
    if wfs_slope_sample.ndim != 2:
        raise ValueError("Raw WFS centroids must have shape (frames,channels)")
    finite_slope_channels = np.all(np.isfinite(wfs_slope_sample), axis=0)
    if not np.any(finite_slope_channels):
        raise ValueError("All raw WFS centroid channels are nonfinite")
    slopes_proxy = lag_difference_variances(
        wfs_slope_sample[:, finite_slope_channels], 1, periodic=False
    )
    if not np.all(np.isfinite(slopes_proxy)):
        raise ValueError("Nonfinite raw WFS slope difference proxy")
    # A constant command across actuators is NOT necessarily a pure
    # wavefront piston after finite DM influence-function projection.
    # Expose the sensitivity of the HO reconstructor to the uniform
    # actuator gauge; do not interpret it as a calibrated tilt leak.
    uniform_actuator_to_ho = u2z @ np.ones(len(idx))
    uniform_actuator_leak_norm = float(
        np.linalg.norm(uniform_actuator_to_ho)
        / max(np.linalg.norm(u2z, ord="fro")*np.sqrt(len(idx)), 1e-30)
    )

    g = np.asarray(sd.mZZ, dtype=float)
    if g.shape != (len(noll), len(noll)):
        raise ValueError("Zernike normal Gram matrix shape mismatch")
    e = np.linalg.eigvalsh(0.5*(g+g.T))
    if e[-1] <= 0:
        raise ValueError("Zernike projector Gram must be nonzero")
    gram_condition = float(e[-1]/max(e[0], 1e-30))
    legacy_expected_match = float(np.linalg.norm(
        command_proxy-noise_var
    )/max(np.linalg.norm(noise_var), 1e-30))
    # Because the legacy noise estimator is run BEFORE the reconstructed
    # pseudo-OL, its DM command series was only demeaned in time.
    # Temporal differencing is invariant to this demeaning.
    expected_method = getattr(sd, "noiseMethod", "unknown")
    quality_flags = []
    if expected_method == "autocorrelation" and str(trs.aoMode) == "LGS":
        if legacy_expected_match > 1e-5:
            quality_flags.append("LEGACY_LGS_NOISE_PROXY_REPRODUCTION_MISMATCH")
    if np.any(corrected <= 0):
        quality_flags.append("NEGATIVE_OR_ZERO_NOISE_SUBTRACTED_MODE")
    if np.sum(noise_var)/max(np.sum(raw_var), 1e-30) > 0.25:
        quality_flags.append("NOISE_PROXY_MORE_THAN_25_PERCENT_MODAL_VARIANCE")
    if gram_condition > 1e6:
        quality_flags.append("ZERNIKE_GRAM_POORLY_CONDITIONED")
    if len(np.unique(radial)) < 4:
        quality_flags.append("TOO_FEW_RADIAL_ORDERS_FOR_R0_L0_DIAGNOSIS")
    if not getattr(trs.dm, "pseudo_ol_sign_independently_verified", False):
        quality_flags.append("PSEUDO_OL_RTC_SIGN_UNVERIFIED")
    if not np.all(finite_slope_channels):
        quality_flags.append("NONFINITE_WFS_SLOPE_CHANNELS_EXCLUDED")
    return {
        "physics_scope": (
            "read-only diagnosis of current LGS HO pseudo-OL in actuator "
            "OPD basis; not a new slope noise estimate"
        ),
        "method_description": {
            "input_wfs_centroids": "OFFSETCENTROID, pixel; not directly used in current r0 fit",
            "input_residual": "RESIDUALWAVEFRONT reconstructed into DM actuator OPD metres",
            "input_commands": "DMCOMMAND in OPD metres",
            "current_noise_source": (
                "lag-1 autocorrelation proxy of CLOSED-LOOP DMCOMMAND; "
                "periodic wrapping in LGS, despite finite record"
            ),
            "current_noise_subtraction": "Cz(signal)-u2z*Cdelta(DM)*u2z.T, diagonal for Zernike fit",
            "zernike_aperture": (
                "ideal circular or optional annulus generated by zernike(), "
                "not actual segmented NIRC2/Keck science pupil"
            ),
            "zernike_tt_convention": "Noll >=4; piston and tip-tilt excluded nominally",
            "lag_proxy_caution": (
                "command and residual lag differences include atmospheric "
                "temporal variations and servo dynamics; not isolated WFS noise"
            ),
            "raw_slope_noise_caution": (
                "raw OFFSETCENTROID is in pixel and cannot be compared to "
                "DM/OPD noise without WFS reconstruction and optical gain"
            ),
            "r0_sensitivity": "fixed fitted L0; refit variance amplitude only, not joint r0/L0",
            "production_code_changed": False,
        },
        "fit": {
            "r0_wfs_los_m": r0,
            "L0_m": l0,
            "wavelength_m": wl,
            "joint_fit_success": bool(sd.res_r0.success),
            "joint_fit_cost": float(sd.res_r0.cost),
            "radial_orders": nr.astype(int).tolist(),
            "n_modes": len(noll),
            "n_valid_actuators": len(idx),
            "n_frames_pseudo_ol": int(trs.dm.pseudo_ol_frames_used),
            "noise_subtraction_multiplier_production": 1,
            "negative_corrected_modes": int(np.count_nonzero(corrected <= 0)),
            "noise_proxy_fraction_of_total_modal_variance": float(
                noise_var.sum()/max(raw_var.sum(), 1e-30)
            ),
            "rms_radial_fit_relative_residual": float(
                np.linalg.norm(data_corrected-theory_radial)
                / max(float(np.linalg.norm(data_corrected)), 1e-30)
            ),
        },
        "projection": {
            "u2z_shape": list(u2z.shape),
            "Noll_first": int(noll.min()),
            "Noll_last": int(noll.max()),
            "Gram_min_eigenvalue": float(e[0]),
            "Gram_max_eigenvalue": float(e[-1]),
            "Gram_condition_number": gram_condition,
            "nominal_tip_tilt_excluded": True,
            "actual_keck_mask_weighting_in_zernike_projection": False,
            "measured_tilt_to_HO_leakage": "not measured; next P1-C experiment",
            "uniform_actuator_gauge_to_HO_fraction": uniform_actuator_leak_norm,
            "uniform_actuator_gauge_to_HO_vector_OPD_per_command_m": (
                uniform_actuator_to_ho.tolist()
            ),
        },
        "noise_proxies": {
            "legacy_noise_mode_variance_reproduction_relative_error": (
                legacy_expected_match
            ),
            "source_noise_method": expected_method,
            "DM_command_lag1_periodic_modal_trace_m2": float(
                command_proxy.sum()
            ),
            "DM_command_lag1_finite_modal_trace_m2": float(
                finite_command_proxy.sum()
            ),
            "WFS_residual_lag1_finite_modal_trace_m2": float(
                residual_proxy.sum()
            ),
            "DM_legacy_production_noise_modal_trace_m2": float(noise_var.sum()),
            "raw_centroid_lag1_finite_variance_median_pixel2": float(
                np.median(slopes_proxy)
            ),
            "raw_centroid_num_channels": int(wfs_slope_sample.shape[1]),
            "raw_centroid_nonfinite_channels_excluded": int(
                np.count_nonzero(~finite_slope_channels)
            ),
            "no_calibrated_sensor_noise_available": True,
        },
        "noise_scale_sensitivity": fixed_l0_noise_sensitivity(
            raw_var, noise_var, model, radial, wl, r0
        ),
        "modes": mode_rows,
        "radial": radial_rows,
        "quality_flags": quality_flags,
    }


def write_audit(d, output: Path):
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest/"p1_modal_noise_audit.json").write_text(
        json.dumps(d, indent=2), encoding="utf-8"
    )
    for rows, filename in [
        (d["modes"], "p1_modal_noise_modes.csv"),
        (d["radial"], "p1_modal_noise_radial.csv"),
        (d["noise_scale_sensitivity"], "p1_modal_noise_sensitivity.csv"),
    ]:
        if rows:
            with (dest/filename).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    radial = d["radial"]
    order = [e["radial_order"] for e in radial]
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in [
        ("median_measured_rad2", "Measured radial median"),
        ("median_corrected_rad2", "Measured - DM lag-noise proxy"),
        ("median_theory_fit_rad2", "Current r0/L0 theoretical model"),
    ]:
        ax.plot(order, [v[key] for v in radial], "o-", label=label)
    ax.set_xlabel("Zernike radial order, Noll >=4 (no nominal TT)")
    ax.set_ylabel("Phase variance [rad²] at WFS atmosphere wavelength")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("P1: empirical, noise-corrected and fitted Zernike variances")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest/"p1_modal_noise_radial.png", dpi=150)
    plt.close(fig)


def main():
    raise SystemExit(
        "P1 modal/noise audit needs the live systemDiagnosis instance. "
        "Run scripts/keck_n0013_smoke.py with --p1-only and the same "
        "--fits/--sav/--calib as your validated n0120 baseline."
    )


if __name__ == "__main__":
    main()
