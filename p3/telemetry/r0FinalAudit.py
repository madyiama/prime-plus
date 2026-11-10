"""Read-only P1 final three-gate audit on the *existing* Keck diagnosis.

This module deliberately never updates sd.trs.atm, sd.trs.wfs, or the
production r0 fit. It flags method identifiability, not absolute truth.
A sign/lag choice requires independent RTC timing/calibration evidence.
"""
from __future__ import annotations

import numpy as np


def sample_shifted_residual(residual, frame_indices, lag_frames):
    """Linear fractional frame interpolation without actuator/time wrapping."""
    values = np.asarray(residual, dtype=float)
    frames = np.asarray(frame_indices, dtype=np.int64)
    lag = float(lag_frames)
    if (values.ndim != 2 or frames.ndim != 1 or len(frames) < 2
            or not np.isfinite(values).all() or not np.isfinite(lag)):
        raise ValueError("Finite residual (frames,actuators) and indices needed")
    pos = frames.astype(float) + lag
    lo = np.floor(pos).astype(np.int64)
    fraction = pos - lo
    # Treat exact integers as single samples. No artificial extra frame.
    fraction[np.isclose(fraction, 0.0, rtol=0, atol=1e-12)] = 0.0
    hi = lo + (fraction > 0).astype(np.int64)
    if np.any(lo < 0) or np.any(hi >= len(values)):
        raise ValueError("RTC lag would read outside telemetry record")
    return ((1-fraction[:, None])*values[lo]
            + fraction[:, None]*values[hi])


def fixed_l0_modal_fit(observed, theoretical, radial_orders, r0_ref_m,
                       low, high):
    """Compare absolute-variance and log-variance fits at identical L0.

    Uses medians per radial order to mirror the legacy fit data
    reduction. This is a sensitivity diagnostic, NOT calibrated r0.
    """
    obs = np.asarray(observed, dtype=float)
    ref = np.asarray(theoretical, dtype=float)
    orders = np.asarray(radial_orders, dtype=int)
    if (obs.shape != ref.shape or obs.shape != orders.shape
            or obs.ndim != 1 or not np.all(np.isfinite(obs))
            or not np.all(np.isfinite(ref)) or r0_ref_m <= 0
            or low > high):
        raise ValueError("Incompatible modal arrays / r0 or radial bounds")
    mask = (orders >= low) & (orders <= high)
    radials = np.unique(orders[mask])
    if len(radials) < 2:
        raise ValueError("Need at least two radial orders")
    y = np.array([np.median(obs[orders == n]) for n in radials])
    m = np.array([np.median(ref[orders == n]) for n in radials])
    if np.any(y <= 0) or np.any(m <= 0):
        return {
            "radial_first": int(low), "radial_last": int(high),
            "n_radial_orders": int(len(radials)),
            "status": "nonpositive_modal_variance",
            "r0_absolute_loss_m": None, "r0_log_loss_m": None,
            "shape_rms_log_residual": None,
        }
    scale = float(np.dot(y, m) / np.dot(m, m))
    log_ratio = np.log(y / m)
    mean_log = float(np.mean(log_ratio))
    return {
        "radial_first": int(low),
        "radial_last": int(high),
        "n_radial_orders": int(len(radials)),
        "n_noll_modes": int(np.count_nonzero(mask)),
        "status": "diagnostic_only",
        "r0_absolute_loss_m": float(r0_ref_m*scale**(-3/5)),
        "r0_log_loss_m": float(r0_ref_m*np.exp(-3/5*mean_log)),
        "shape_rms_log_residual": float(
            np.sqrt(np.mean((log_ratio-mean_log)**2))
        ),
        "median_observed_over_theory": float(np.median(y / m)),
        "radial_orders": radials.tolist(),
        "observed_over_theory_by_radial_order": (y/m).tolist(),
    }


def _selected_indices(actuators, n_cols):
    indices = np.asarray(actuators)
    if indices.dtype.kind == "b":
        indices = np.flatnonzero(indices.ravel())
    else:
        indices = indices.astype(int).ravel()
    if np.any(indices < 0) or np.any(indices >= n_cols):
        raise ValueError("Invalid selected actuator indices")
    return indices


def sign_lag_diagnostic(dm_commands, residual, u2z, actuator_indices,
                        nominal_lag_frames, theoretical_variance_rad2,
                        radial_orders, r0_reference_m, wavelength_m,
                        *, frame_limit=4096):
    """Compare +/- pseudo-OL sign and nominal lag +/-1 frame.

    No criterion selects a 'correct' sign from the lowest r0-fit cost:
    a closed-loop turbulence realization cannot independently prove RTC
    polarity or timing. All 6 trials reuse the same subsampled frames.
    """
    commands = np.asarray(dm_commands, dtype=float)
    residual = np.asarray(residual, dtype=float)
    U = np.asarray(u2z, dtype=float)
    model = np.asarray(theoretical_variance_rad2, dtype=float)
    radial = np.asarray(radial_orders, dtype=int)
    if (commands.shape != residual.shape or commands.ndim != 2
            or U.ndim != 2 or len(model) != len(radial)
            or not (len(model) == U.shape[0])
            or wavelength_m <= 0 or frame_limit < 64
            or not np.isfinite(commands).all()
            or not np.isfinite(residual).all()
            or not np.isfinite(U).all()):
        raise ValueError("Incompatible time series, modal projector or wavelength")
    indices = _selected_indices(actuator_indices, commands.shape[1])
    if U.shape[1] != len(indices):
        raise ValueError("Zernike projector and DM actuator mask disagree")
    offsets = (-1.0, 0.0, 1.0)
    lags = [float(nominal_lag_frames)+off for off in offsets]
    pad = int(np.ceil(max(abs(x) for x in lags))) + 2
    if 2*pad + 64 >= len(commands):
        raise ValueError("Insufficient common RTC samples for three lag offsets")
    core = np.arange(pad, len(commands)-pad, dtype=int)
    if len(core) > frame_limit:
        core = core[np.linspace(0, len(core)-1, frame_limit).astype(int)]
    selected_modes = (radial >= 3) & (radial <= 14)
    if selected_modes.sum() < 10:
        raise ValueError("Need >=10 Noll modes between radial orders 3 and 14")
    P = U[selected_modes, :].T
    obs_dm = commands[core] - np.mean(commands[core], axis=1, keepdims=True)
    modal_dm = obs_dm[:, indices] @ P
    model_selected = model[selected_modes]
    radial_selected = radial[selected_modes]
    output = []
    for shift in lags:
        res = sample_shifted_residual(residual, core, shift)
        res -= np.mean(res, axis=1, keepdims=True)
        modal_res = res[:, indices] @ P
        for sign in (-1, +1):
            poly = modal_dm + sign*modal_res
            poly -= poly.mean(axis=0, keepdims=True)
            modal_variance = (2*np.pi/wavelength_m)**2 * np.var(
                poly, axis=0
            )
            fit = fixed_l0_modal_fit(
                modal_variance, model_selected, radial_selected,
                r0_reference_m, 3, 14
            )
            output.append({
                "residual_sign": int(sign), "residual_lag_frames": float(shift),
                "n_time_samples": int(len(core)),
                "median_variance_rad2": float(np.median(modal_variance)),
                "r0_log_m_at_frozen_L0_without_noise": fit["r0_log_loss_m"],
                "r0_abs_m_at_frozen_L0_without_noise": fit["r0_absolute_loss_m"],
                "model_shape_rms_log_residual": fit["shape_rms_log_residual"],
            })
    return {
        "nominal_lag_frames": float(nominal_lag_frames),
        "n_total_frames": int(len(commands)),
        "n_common_frames": int(len(core)),
        "frame_selection": "deterministic_uniform_subsample_whole_exposure",
        "fit_radial_orders": [3, 14],
        "subtraction_of_noise_applied": False,
        "results": output,
        "rtc_sign_independently_validated": False,
        "rtc_latency_independently_validated": False,
        "warning": (
            "DO NOT choose sign or lag from fit quality alone; cross-check "
            "the Keck RTC timestamp, control-law and interaction-matrix conventions."
        ),
    }


def gain_hypothesis_diagnostic(r0_m, present_opd_m_per_volt,
                               alternatives=(0.6e-6,)):
    """Show high-leverage gain bias *if both DM/residual scales change*."""
    if r0_m <= 0 or present_opd_m_per_volt <= 0:
        raise ValueError("Valid r0 and calibration required")
    hypotheses = []
    for candidate in alternatives:
        if candidate <= 0:
            raise ValueError("Positive DM calibration alternative required")
        ratio = float(candidate/present_opd_m_per_volt)
        hypotheses.append({
            "candidate_opd_m_per_volt": float(candidate),
            "gain_ratio": ratio,
            "r0_m_if_all_OPD_scales_were_changed": float(
                r0_m*ratio**(-6/5)
            ),
        })
    return {
        "current_opd_m_per_volt": float(present_opd_m_per_volt),
        "current_r0_m": float(r0_m),
        "alternatives": hypotheses,
        "gain_source_2017_kaon_m_per_volt": 0.6e-6,
        "conversion_interpretation_verified": False,
        "warning": (
            "KAON 2017 and historical P3 differ; verify whether each "
            "factor refers to OPD or surface and which Keck configuration "
            "and epoch, before changing a single code constant."
        ),
    }


def analyse_final_three_gates(sd, *, frame_limit=4096):
    """Make one JSON-safe scientific report without changing diagnosis."""
    trs = sd.trs
    radial = np.asarray(sd.z.n, dtype=int)
    model = np.asarray(trs.wfs.coeffs_mod, dtype=float)
    wavelength = float(trs.atm.wvl)
    r0 = float(trs.atm.r0_tel)
    if wavelength <= 0 or r0 <= 0:
        raise ValueError("Invalid reference wavelength or legacy r0")
    measured = (2*np.pi/wavelength)**2 * (
        np.asarray(trs.wfs.var_meas, dtype=float)
        - np.asarray(trs.wfs.var_noise_zer, dtype=float)
    )
    ranges = [(3, 10), (3, 12), (3, 14), (15, 22), (2, 22)]
    modes = [fixed_l0_modal_fit(
        measured, model, radial, r0, low, high
    ) for low, high in ranges]

    timing = sign_lag_diagnostic(
        trs.dm.com, trs.rec.res, trs.mat.u2z, sd.actuInPupil,
        float(trs.holoop.lat * trs.holoop.freq),
        model, radial, r0, wavelength, frame_limit=frame_limit
    )

    U = np.asarray(trs.mat.u2z)
    if_mask = np.asarray(trs.mat.dmIF)
    n_pix = int(np.sqrt(if_mask.shape[0]))
    gram = np.asarray(sd.mZZ)
    current_gain = float(trs.dm.volt2meter)
    noise_opd = np.asarray(trs.wfs.var_noise_zer, dtype=float)
    measured_opd = np.asarray(trs.wfs.var_meas, dtype=float)
    diagnostics = {
        "method": "historical_zernike_pseudo_OL",
        "code_result_is_validated": False,
        "reference_r0_m": r0,
        "reference_L0_m": float(trs.atm.L0_tel),
        "reference_wavelength_m": wavelength,
        "radial_mode_sensitivity": modes,
        "rtc_sign_lag_sensitivity": timing,
        "units": gain_hypothesis_diagnostic(r0, current_gain),
        "projection": {
            "n_projection_modes": int(U.shape[0]),
            "n_selected_actuators": int(U.shape[1]),
            "pupil_grid_pixels": n_pix,
            "dm_influence_support_m": float(
                (int(np.asarray(trs.dm.nActuators).ravel()[0])-1)
                * np.asarray(trs.dm.pitch).ravel()[0]
            ),
            "theory_diameter_m": float(trs.tel.D),
            "gram_condition_number": float(np.linalg.cond(gram)),
            "uniform_DM_command_piston_leak_opd": (
                U @ np.ones(U.shape[1])
            ).tolist(),
            "P1_wfs_geometry_validated": False,
            "warning": (
                "Historical Zernike basis uses an idealized pupil and "
                "is not the same annular DM-modified covariance used by Jolissaint."
            ),
        },
        "noise": {
            "subtraction_proxy": "lag_one_closed_loop_DM_command_difference",
            "LGS_periodic_last_to_first_boundary": (
                str(trs.aoMode).upper() == "LGS"
            ),
            "noise_trace_over_measured_trace": float(
                np.sum(noise_opd)/max(np.sum(measured_opd), 1e-60)
            ),
            "noise_WFS_independently_calibrated": False,
        },
        "decision": (
            "Diagnostic research only; no changes to production r0/L0. "
            "Independent RTC polarity/latency and DM OPD gain remain gating "
            "calibrations for a physically certified estimate."
        ),
    }
    return diagnostics
