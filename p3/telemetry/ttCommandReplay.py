"""P4-F controlled Keck LGS tip-tilt slope-source ablation.

Compare the archived STRAP centroid series to the exact digital
integrator replay: (DTTCOMMANDS[k]-DTTCOMMANDS[k-1])/DT_SERVO[0].
Both series are in the same *nominal recorded arcsec* scale; the
comparison is NOT an independent sky-angle calibration.

Apply before systemDiagnosis so the sensor noise and PSF-R consume
the same substituted TT signal. Baseline and replay always exclude
frame zero, which has no preceding command in the archived window.
"""
from __future__ import annotations

import numpy as np

from p3.telemetry.ttLongExposureAudit import require_integrator
from p3.aoSystem.tipTiltGeometry import zernike_tilt_opd_per_arcsec


SOURCE_OPTIONS = ("centroids", "centroids_aligned", "command_increment")


def prepare_tt_command_replay(centroids_arcsec, commands, servo, *,
                              max_relative_rms_mismatch=1e-4):
    """Return the two aligned (N-1,2) signals and unit-consistency audit.

    If the command/centroid relation disagrees substantially, **refuse**
    to label the increment as an arcsecond equivalent; neither a
    telescope angle nor a calibrated optical gain can then be inferred.
    """
    y=np.asarray(centroids_arcsec,dtype=np.float64)
    u=np.asarray(commands,dtype=np.float64)
    if (y.ndim!=2 or y.shape!=u.shape or y.shape[1]!=2
        or len(y)<1024 or not np.isfinite(y).all() or
        not np.isfinite(u).all()):
        raise ValueError("Need finite matched (N>=1024,2) Keck TT centroids/commands")
    gain=require_integrator(servo)
    if not np.isfinite(max_relative_rms_mismatch) or not 0<max_relative_rms_mismatch<0.1:
        raise ValueError("Invalid maximum relative replay mismatch")
    # IMPORTANT: conversion before OPD and before noise estimation.
    command_centroids=np.diff(u,axis=0)/gain
    reference=y[1:].copy()
    difference=command_centroids-reference
    denom=float(np.sqrt(np.mean((reference-reference.mean(axis=0))**2)))
    if denom<=0:
        raise ValueError("No centroid fluctuations to compare")
    mismatch=float(np.sqrt(np.mean(difference**2))/denom)
    max_abs=float(np.max(np.abs(difference)))
    if mismatch>max_relative_rms_mismatch:
        raise ValueError(
            "Keck command-increment replay is NOT equivalent to the "
            "calibrated centroid stream: relative RMS mismatch "
            f"{mismatch:.6g} exceeds {max_relative_rms_mismatch:.6g}. "
            "Do not guess command units or a calibration factor."
        )
    before=np.cov(reference,rowvar=False,bias=True)
    after=np.cov(command_centroids,rowvar=False,bias=True)
    return reference,command_centroids,{
        "source":"Keck STRAP TT command increment replay",
        "servo_gain":float(gain),
        "relation":"delta(DTTCOMMANDS[k])/gain = DTTCENTROIDS[k]",
        "alignment":"same Keck B record index k, drop first archived frame",
        "input_frame_count":int(len(y)),
        "aligned_frame_count":int(len(reference)),
        "relative_rms_error_vs_centroids":mismatch,
        "maximum_absolute_error_nominal_arcsec":max_abs,
        "centroid_covariance_nominal_arcsec2":before.tolist(),
        "increment_covariance_nominal_arcsec2":after.tolist(),
        "covariance_relative_frobenius_error":float(
            np.linalg.norm(after-before)/max(np.linalg.norm(before),1e-30)
        ),
        "command_units_independently_calibrated_to_sky":False,
        "strap_optical_gain_independently_calibrated":False,
        "physical_tt_jitter_calibrated":False,
        "passes_nominal_integrator_replay":True,
        "note":(
            "This controlled ablation cannot create information absent "
            "from DTTCENTROIDS; an identical PSF is the expected result "
            "if P3 is internally consistent."
        ),
    }


def apply_tt_slope_source(trs,servo,source, *,max_relative_rms_mismatch=1e-4):
    """Switch a Keck LGS telemetry object to a TT-only A/B sample source.

    Do not change HO telemetry, r0, pupil, lenslet calibration or the
    controller; and never apply a further 0.576 or optical-gain factor.
    Baseline `centroids` is a true no-op for compatibility. Both
    experimental variants discard the first frame to align samples.
    """
    if source not in SOURCE_OPTIONS:
        raise ValueError("Invalid Keck TT slope source: "+str(source))
    if source=="centroids":
        return {
            "source":source,
            "first_frame_removed":False,
            "experimental_replay":False,
            "command_units_independently_calibrated_to_sky":False,
            "strap_optical_gain_independently_calibrated":False,
        }
    if str(trs.aoMode).upper()!="LGS":
        raise ValueError("Keck DTT replay requires LGS STRAP B telemetry")
    tilt=trs.tipTilt
    reference,derived,audit=prepare_tt_command_replay(
        tilt.slopes_arcsec,tilt.com_raw,servo,
        max_relative_rms_mismatch=max_relative_rms_mismatch,
    )
    chosen=reference if source=="centroids_aligned" else derived
    gain=zernike_tilt_opd_per_arcsec(float(trs.tel.D))
    stored_gain=float(tilt.tilt2meter)
    if not np.isclose(gain,stored_gain,rtol=1e-12):
        raise ValueError("P3 STRAP tilt angle-to-OPD scale has changed")
    tilt.slopes_arcsec=chosen.copy()
    opd=chosen*gain
    tilt.slopes=opd-opd.mean(axis=0)
    tilt.nExp=int(len(chosen))
    # Preserve index alignment for ancillary TT and turbulence-command
    # diagnostics even though they do not calibrate the DTT on sky.
    for attr in ("com_raw","com","intensity"):
        if hasattr(tilt,attr):
            values=np.asarray(getattr(tilt,attr))
            if values.ndim>=1 and len(values)==len(reference)+1:
                setattr(tilt,attr,np.array(values[1:],copy=True))
    return {
        **audit,
        "source":source,
        "first_frame_removed":True,
        "experimental_replay":True,
        "chosen_tt_frame_count":int(tilt.nExp),
        "chosen_centroid_covariance_nominal_arcsec2":
            np.cov(chosen,rowvar=False,bias=True).tolist(),
        "science_tt_slope_conversion":"a_Zernike = D*theta_rad/4",
        "noise_estimation_uses_chosen_tt_source":True,
        "tiptilt_noise_recalculated_with_same_method":True,
        "ho_telemetry_unmodified":True,
    }
