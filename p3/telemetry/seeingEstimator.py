"""Unified Keck telemetry seeing estimator.

estimate_r0(diagnosis, method="zernike"|"jolissaint") is the ONLY
public r0 estimation entrypoint for research comparisons.

The historical Zernike option reports the original estimator verbatim.
The Jolissaint option is an EXPERIMENTAL, physics-derived adaptation of
the user's 2019 IDL seeingMonitor/seeing.pro, get_seeing.pro,
dm_noise_covmat.pro and actuator_to_zernike_matrices.pro. The 2018 IDL
precomputed .sav covariances are absent, hence we recompute modified
Zernikes and turbulence covariance using actual Keck DM influence
functions and the phase structure function. This is NOT a bitwise
IDL port and must not silently replace the production r0.

Jolissaint uses closed-loop DMCOMMAND, not the pseudo-open-loop
commands of the legacy PRIME+ estimate. Both explicitly exclude LGS
piston and tip-tilt. Values are physical OPD metres and phase radians
at trs.atm.wvl; no DIMM calibration is applied.
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.special import gamma, kv
from scipy.signal import fftconvolve


# Exact LGS bad-mode list in get_seeing.pro (2018 code; calibrated for
# that Keck dataset, NOT automatically validated for NIRC2 n0120).
_IDL_LGS_BAD = frozenset([
    7, 8, 11, 16, 17, 18, 22, 29, 30, 37, 38, 44, 46, 47,
    56, 60, 62, 64, 66, 67, 68, 79, 81, 82, 84, 86, 88,
    90, 92, 93, 94, 95, 106, 110, 112, 114, 116, 118, 120,
    *range(121, 172),
])


def _phase_sf(rho, r0, outer_scale):
    """von Karman SF; exact Kolmogorov 6.88 limit, phase rad²."""
    rho = np.asarray(rho, dtype=np.float64)
    if r0 <= 0 or not (outer_scale > 0):
        raise ValueError("r0 and outer scale must be positive")
    if np.isposinf(outer_scale):
        return 6.88*(rho/r0)**(5.0/3.0)
    nu = 5.0/6.0
    z = 2*np.pi*rho/outer_scale
    B = np.full_like(z, 3.0/5.0)
    positive = z > 0
    B[positive] = (
        z[positive]**nu * kv(nu, z[positive])
        / (2**nu*gamma(1+nu))
    )
    a = -gamma(-nu)/(2**(2*nu+1)*gamma(1+nu))
    out = 6.88*(outer_scale/r0)**(5/3) * (
        (3/5-B)/(a*(2*np.pi)**(5/3))
    )
    return np.maximum(out, 0)


def modified_dm_projection(
    influence_matrix, diameter_m, resolution, *,
    annular_outer_m=9.0, annular_inner_m=2.65,
    j_start=7, j_stop=120, pinv_rcond=1e-4,
):
    """Return DM->modified-Zernike operator and zero-piston phase probes.

    Modes are fitted to the *Keck DM influence functions on the
    9.0/2.65m annulus*, then reprojected through their own Gram matrix.
    No Keck science-image pupil is involved. Pupil quadrature uses
    square pixels of width D/resolution and a uniform annular weight.
    """
    from p3.aoSystem.zernike import zernike

    IF = np.asarray(influence_matrix, dtype=float)
    n = int(resolution)
    if (n < 15 or IF.ndim != 2 or IF.shape[0] != n*n
            or not np.all(np.isfinite(IF))):
        raise ValueError("Invalid Keck DM influence-function matrix")
    if not 0 < annular_inner_m < annular_outer_m < diameter_m:
        raise ValueError("Invalid analysis annulus")
    if j_start < 4 or j_stop < j_start:
        raise ValueError("LGS piston / tip-tilt must not be selected")
    # Same NIR Keck DM actuator peaks as in Keck calibration: use
    # a DM-region selection, not the science image pupil mask.
    pixel_centres = (np.arange(n)-(n-1)/2)*diameter_m/n
    y, x = np.meshgrid(pixel_centres, pixel_centres, indexing="ij")
    r = np.hypot(x, y)
    ann = ((r <= annular_outer_m/2) &
           (r >= annular_inner_m/2))
    nmodes = j_stop-j_start+1
    # The IDL uses 320 of 349 actuators, selected by DM location
    # 1.2*pitch < r < 10.05*pitch. Use IF peak positions, preserving
    # raw indices and flagging if selection differs from 320.
    peaks = np.argmax(np.abs(IF), axis=0)
    peak_r = r.ravel()[peaks]
    selected = np.flatnonzero(
        (peak_r > 1.2*0.5625) &
        (peak_r < 10.05*0.5625)
    )
    if selected.size < nmodes+3:
        raise ValueError(
            "Insufficient DM actuators for modified Zernike projection"
        )
    A = IF[ann.ravel(), :][:, selected]
    # Zernikes on a physical 9m disk: rescale the telescope radius.
    z = zernike(list(range(j_start, j_stop+1)), n)
    z.radius = r/(annular_outer_m/2)
    z.angle = np.arctan2(y, x)
    z.pupil = ann
    Z = np.asarray(z.polynomials(), dtype=float)[:, ann].T
    # Projection of ideal annular Zernikes into DM space. A truncated
    # pseudoinverse avoids amplifying near-unobservable actuator modes.
    dm_coeff = np.linalg.pinv(A, rcond=pinv_rcond) @ Z
    Zapp = A @ dm_coeff
    G = Zapp.T @ Zapp / len(Zapp)
    # The 2018 IDL takes inverse(matZZ) @ matZI, with matZZ the
    # modified, not ideal, Zernike spatial covariance.
    inv_G = np.linalg.pinv(G, rcond=pinv_rcond)
    Q = inv_G @ Zapp.T / len(Zapp)   # phase -> coeff
    # An unknown Kolmogorov piston must not leak into the j>=7 probes.
    Q = Q - Q.sum(axis=1, keepdims=True)/Q.shape[1]
    U = Q @ A                       # DM actuator -> coeff
    fullQ = np.zeros((nmodes, n*n), dtype=np.float64)
    fullQ[:, ann.ravel()] = Q
    return {
        "dm_to_modal": U, "phase_probes": fullQ.reshape(nmodes, n, n),
        "noll": np.arange(j_start, j_stop+1, dtype=int),
        "radial_order": np.asarray(z.n, dtype=int),
        "actuator_indices": selected, "annulus_pixel_count": int(ann.sum()),
        "modified_gram_condition": float(np.linalg.cond(G)),
        "analysis_outer_diameter_m": float(annular_outer_m),
        "analysis_inner_diameter_m": float(annular_inner_m),
        "telescope_diameter_m": float(diameter_m),
        "pupil_resolution": int(n),
    }


def _model_modal_variance(probes, diameter_m, outer_scale, telescope_diameter_m):
    """Deterministic phase-covariance quadrature for (D/r0)^5/3=1.

    A zero-piston projection q gives:
      Var[<q,phi>] = -0.5 sum_{p,k} q(p) q(k) Dphi(p-k).
    The stationary 2D kernel is evaluated by FFT linear convolution,
    not square/circular wrapping or random Monte Carlo.
    """
    q = np.asarray(probes, dtype=float)
    nmod, n, n2 = q.shape
    if n != n2:
        raise ValueError("Phase probes must be square")
    step = telescope_diameter_m/n
    diff = np.arange(-n+1, n)*step
    dy, dx = np.meshgrid(diff, diff)
    kernel = -0.5*_phase_sf(
        np.hypot(dx, dy), diameter_m, outer_scale
    )
    conv = fftconvolve(
        q, kernel[None, :, :],
        mode="full", axes=(-2, -1)
    )[:, n-1:n-1+n, n-1:n-1+n]
    variance = np.sum(q*conv, axis=(1, 2))
    if not np.all(np.isfinite(variance)):
        raise ValueError("Nonfinite theoretical modified-Zernike variances")
    if np.any(variance < -1e-7*max(float(np.max(variance)), 1e-30)):
        raise ValueError("Nonphysical negative model variance")
    return np.maximum(variance, 0)


def spectral_command_noise_variance(modal_commands, fs, freq_tf, transfer,
                                    high_band=(0.75, 0.95)):
    """PSD plateau deconvolution / reconvolution, IDL-inspired.

    Only the *diagonal* modal noise covariance is required for the
    variance fit. Welch one-sided density PSD and one-sided integral
    have consistent physical units; no arbitrary FFT normalization.
    Must not be interpreted as an independently calibrated WFS noise
    when atmospheric power remains in the chosen HF plateau.
    """
    command = np.asarray(modal_commands, dtype=float)
    if command.ndim != 2 or command.shape[0] < 64 or fs <= 0:
        raise ValueError("Need >=64 frames of modal DM commands")
    f, pxx = signal.welch(
        command, fs=fs, axis=0, nperseg=min(2048, len(command)),
        window="hann", detrend="constant", scaling="density",
        average="mean",
    )
    f_tf = np.asarray(freq_tf, dtype=float).reshape(-1)
    tf = np.asarray(transfer).reshape(-1)
    if (len(f_tf) != len(tf) or len(tf) < 3
            or np.any(~np.isfinite(f_tf))
            or np.any(~np.isfinite(tf))
            or f_tf[0] < 0 or not np.all(np.diff(f_tf) > 0)):
        raise ValueError("Invalid Keck AO closed-loop transfer function")
    psd_tf = np.interp(
        f, f_tf, np.abs(tf)**2,
        left=float(np.abs(tf[0])**2),
        right=float(np.abs(tf[-1])**2),
    )
    hf = ((f >= high_band[0]*fs/2)
          & (f <= high_band[1]*fs/2)
          & (psd_tf > 1e-5*max(float(psd_tf.max()), 1e-30)))
    if np.count_nonzero(hf) < 8:
        raise ValueError("Too little valid high-frequency WFS noise PSD")
    whitened = pxx[hf, :] / psd_tf[hf, None]
    plateau = np.median(whitened, axis=0)
    noise = plateau * trapezoid(psd_tf, f)
    return np.maximum(noise, 0), {
        "high_band_fraction_nyquist": list(high_band),
        "welch_nperseg": int(min(2048, len(command))),
        "selected_frequency_bins": int(np.count_nonzero(hf)),
        "plateau_one_sided_psd_rad2_per_hz": plateau.tolist(),
        "transfer_power_band_min_max": [
            float(np.min(psd_tf[hf])), float(np.max(psd_tf[hf]))
        ],
    }


def _fit_amplitude(observed, models, noll, radial, diameter, L0_grid):
    """Joint amplitude and L0 grid fit, equal weight in log variance."""
    logy = np.log(observed)
    records = []
    for outer, ref in zip(L0_grid, models):
        if np.any(ref <= 0):
            raise ValueError("Theoretical modal variance must be positive")
        delta = logy-np.log(ref)
        logamp = float(np.median(delta))
        cost = float(np.mean((delta-logamp)**2))
        records.append({
            "L0_m": "infinity" if np.isinf(outer) else float(outer),
            "r0_m": float(diameter*np.exp(-3/5*logamp)),
            "log_variance_shape_cost": cost,
            "amplitude_D_over_r0_to_5_3": float(np.exp(logamp)),
            "relative_residual_median": float(
                np.median(np.exp(delta-logamp)-1)
            ),
        })
    winner = int(np.argmin([v["log_variance_shape_cost"] for v in records]))
    best = records[winner]
    other = sorted(v["log_variance_shape_cost"] for v in records)
    delta = (other[1]-other[0]) if len(other)>1 else 0
    return best, records, {
        "L0_identifiability": "not_independently_validated",
        "L0_best_on_grid_boundary": bool(winner in (0, len(records)-1)),
        "next_best_shape_cost_difference": float(delta),
        "model_selection_warning": (
            "L0 is inferred without LGS tip-tilt and may be weakly "
            "identifiable. Grid fit is not a confidence interval."
        ),
        "Noll_selected": noll.tolist(),
        "radial_selected": radial.tolist(),
    }


def estimate_r0(
    diagnosis, *,
    method: Literal["zernike", "jolissaint"] = "zernike",
    mode_selection: Literal["idl_2018", "all"] = "idl_2018",
    noise: Literal["spectral", "none"] = "spectral",
    alias_correction: bool = False,
    L0_grid=(10, 15, 22, 31, 45, 65, 100, 150, 250, np.inf),
    j_start=7, j_stop=171,
):
    """Estimate atmospheric r0 and L0 using ONE selectable API.

    diagnosis: constructed Keck systemDiagnosis instance. The Zernike
    branch returns its preexisting result unmodified; the Jolissaint
    branch derives a separate DM-command-based result (no mutation).

    Experimental method DOES NOT import the missing archived IDL .sav
    calibration matrices and is not an exact binary reproduction of
    Jolissaint et al. (2018). It computes its own atmospheric reference
    on the modified DM modes, and optionally applies the empirically
    published Keck alias correction only for Noll 7..120.
    """
    trs = diagnosis.trs
    if method == "zernike":
        return {
            "method": "zernike",
            "r0_m": float(trs.atm.r0_tel),
            "L0_m": float(trs.atm.L0_tel),
            "r0_reference_wavelength_m": float(trs.atm.wvl),
            "line_of_sight": True,
            "noll_first": min(trs.wfs.jIndex),
            "noll_last": max(trs.wfs.jIndex),
            "noise_method": str(diagnosis.noiseMethod),
            "status": "production_legacy_existing_fit",
        }
    if method != "jolissaint":
        raise ValueError(f"Unknown r0 estimation method: {method}")
    if str(trs.aoMode).upper() != "LGS":
        raise ValueError("Jolissaint LGS variant requires Keck LGS telemetry")

    IF = np.asarray(trs.mat.dmIF, dtype=float)
    com = np.asarray(trs.dm.com, dtype=float)
    if IF.ndim != 2 or com.ndim != 2 or IF.shape[1] != com.shape[1]:
        raise ValueError("Keck influence/DM command actuator layout mismatch")
    diameter = float(trs.tel.D)
    calib = modified_dm_projection(
        IF, diameter, int(trs.tel.resolution),
        j_start=j_start, j_stop=j_stop,
    )
    act = calib["actuator_indices"]
    U = calib["dm_to_modal"]
    commands = (com[:, act] - np.mean(com[:, act], axis=0)) @ U.T
    # Commands and IF must be OPD METRES, not actuator volts.
    wavelength = float(trs.atm.wvl)
    if wavelength <= 0:
        raise ValueError("Invalid reference wavelength")
    commands *= 2*np.pi/wavelength
    measured = np.mean(commands**2, axis=0)
    if noise == "spectral":
        tf = trs.holoop.tf
        transfer = np.asarray(tf.ctf)/np.asarray(tf.wfs)
        noise_var, noise_diag = spectral_command_noise_variance(
            commands, float(trs.holoop.freq),
            tf.freq, transfer,
        )
    elif noise == "none":
        noise_var = np.zeros_like(measured)
        noise_diag = {"note": "No noise subtraction; diagnostic only"}
    else:
        raise ValueError("noise must be 'spectral' or 'none'")
    correct = measured-noise_var
    j = calib["noll"]
    radial = calib["radial_order"]
    # The IDL constructs all 165 modified modes Noll 7..171,
    # then retains the calibrated subset within Noll 7..120.
    # The omitted 121..171 modes still participate in the Gram
    # inversion and thus in the definition of the low-order modes.
    selectable = (j >= 7)&(j <= min(j_stop, 120))&(correct > 0)
    if mode_selection == "idl_2018":
        selectable &= ~np.isin(j, list(_IDL_LGS_BAD))
    elif mode_selection != "all":
        raise ValueError("mode_selection must be idl_2018 or all")
    if np.count_nonzero(selectable) < 12:
        raise ValueError("Too few usable modified Zernike LGS modes")
    if alias_correction:
        if (np.max(j[selectable]) > 120
                or np.min(j[selectable]) < 7):
            raise ValueError("IDL alias fit is only validated in Noll 7..120")
        pp = np.array([
            0.99893514401310313, 0.00077066823200766521,
            -0.00018233364176012401, 1.9037601227012146e-5,
            -2.7620673576089771e-7,
        ])
        alias_multiplier = np.ones(len(j), dtype=float)
        alias_multiplier[selectable] = np.polynomial.polynomial.polyval(
            radial[selectable], pp
        )
        if np.any(alias_multiplier[selectable] <= 0):
            raise ValueError("Invalid published Keck alias multiplier")
        correct = correct/alias_multiplier
    L0_grid = tuple(float(v) for v in L0_grid)
    if len(L0_grid) < 3 or any(v <= 0 for v in L0_grid):
        raise ValueError("Use >=3 positive L0 grid values in metres")
    reference = [
        _model_modal_variance(
            calib["phase_probes"][selectable],
            calib["analysis_outer_diameter_m"], outer, diameter
        ) for outer in L0_grid
    ]
    best, profile, fit_diag = _fit_amplitude(
        correct[selectable], reference, j[selectable],
        radial[selectable], calib["analysis_outer_diameter_m"], L0_grid
    )
    return {
        "method": "jolissaint",
        "status": "experimental_recomputed_modified_covariance",
        "r0_m": best["r0_m"], "L0_m": best["L0_m"],
        "r0_reference_wavelength_m": wavelength,
        "line_of_sight": True,
        "r0_is_validated": False,
        "L0_is_validated": False,
        "input_signal": "closed_loop_DMCOMMAND_OPD_m",
        "annular_outer_diameter_m": calib["analysis_outer_diameter_m"],
        "annular_inner_diameter_m": calib["analysis_inner_diameter_m"],
        "actuator_count": int(len(act)),
        "gram_condition": calib["modified_gram_condition"],
        "selected_Noll_indices": j[selectable].tolist(),
        "excluded_Noll_indices": j[~selectable].tolist(),
        "n_modes_used": int(np.count_nonzero(selectable)),
        "measured_variance_rad2": measured.tolist(),
        "noise_variance_rad2": noise_var.tolist(),
        "corrected_variance_rad2": correct.tolist(),
        "noll_all": j.tolist(),
        "noise": noise_diag, "noise_method": noise,
        "mode_selection": mode_selection,
        "alias_correction": (
            "published_2018_empirical" if alias_correction
            else "not_applied_unvalidated_for_n0120"
        ),
        "L0_profile": profile, "fit_diagnostics": fit_diag,
        "provenance_warning": (
            "2018 IDL .sav covariance calibrations absent; recomputed "
            "from current Keck DM influence functions. Original mode "
            "blacklist and alias polynomial have not been independently "
            "validated for n0120. This is not a production result."
        ),
    }
