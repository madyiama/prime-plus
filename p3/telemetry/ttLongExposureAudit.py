"""P4-E: statistical Keck STRAP / integrator-command consistency audit.

Purpose
-------
Validate the known digital TT controller using the *recorded* centroid
and command streams, and describe sensitivity of van-Dam SENSOR noise
estimates to a hypothesized effective optical gain.

Neither an increment of a command nor an output from this module is
an independent angle calibration.  No scalar gain is identified and
no change to the production physical TT covariance is made.
"""
from __future__ import annotations

import warnings
import numpy as np

from p3.telemetry.ttMechanicalAudit import timestamp_quality
from p3.telemetry.ttNoise import estimate_tt_noise


def require_integrator(servo, *, rtol=1e-6):
    """Require the exact archive form [gain,0,0,0,-1,0,0].

    Keck uses [b0..b3,a1..a3] with denominator 1+a1*z^-1+... .
    An unsupported controller is not silently approximated.
    """
    coeff=np.asarray(servo,dtype=float).reshape(-1)
    if coeff.shape!=(7,) or not np.isfinite(coeff).all():
        raise ValueError("DT_SERVO must be 7 finite coefficients")
    gain=float(coeff[0])
    expected=np.array([gain,0,0,0,-1,0,0],dtype=float)
    if not 0<gain<2 or not np.allclose(coeff,expected,rtol=rtol,atol=1e-7):
        raise ValueError(
            "Only the verified discrete integrator DT_SERVO form "
            "[gain,0,0,0,-1,0,0] is supported; do not assume it for "
            "another controller configuration"
        )
    return gain


def _shifted_increments(y,u,bad,lag,split):
    """Align du[k]=u[k]-u[k-1] with y[k-lag].

    lag is an archived stream offset, not the physical loop latency.
    If timestamps show an anomalous interval, reject pairs spanning it.
    """
    n=len(u)
    k=np.arange(1,n)
    j=k-lag
    mask=(j>=0)&(j<n)&(~bad[k-1])
    jclipped=np.clip(j,0,n-1)
    lo=np.minimum(jclipped,k)
    hi=np.maximum(jclipped,k)
    prefix=np.concatenate([[0],np.cumsum(bad,dtype=int)])
    mask &= (prefix[hi]-prefix[lo]==0)
    x=y[jclipped]
    du=u[k]-u[k-1]
    split_index=int(split*n)
    train=mask&(k<split_index)
    test=mask&(k>=split_index)
    return x,du,train,test


def _score(reference,prediction):
    denom=float(np.sum((reference-reference.mean(axis=0))**2))
    err=float(np.sum((reference-prediction)**2))
    return 1-err/denom if denom>0 else None


def audit_command_increments(centroids,commands,timestamps,servo, *,
                             max_lag=10,train_fraction=0.7):
    y=np.asarray(centroids,dtype=float)
    u=np.asarray(commands,dtype=float)
    if (y.shape!=u.shape or y.ndim!=2 or y.shape[1]!=2 or
            len(y)<256 or not np.isfinite(y).all() or
            not np.isfinite(u).all()):
        raise ValueError("Centroids/commands must be finite paired (N>=256,2)")
    if not 0.55<=train_fraction<=0.85:
        raise ValueError("train_fraction must be in [0.55,0.85]")
    if not 1<=max_lag<=100:
        raise ValueError("max_lag must be in [1,100]")
    gain=require_integrator(servo)
    bad,runs,timing=timestamp_quality(timestamps)
    if len(bad)!=len(y)-1:
        raise ValueError("TT timestamp length mismatch")
    scans=[]
    for lag in range(-max_lag,max_lag+1):
        x,du,tr,te=_shifted_increments(y,u,bad,lag,train_fraction)
        if tr.sum()<128 or te.sum()<64:
            continue
        X=np.column_stack([np.ones(len(x)),x])
        if np.linalg.matrix_rank(X[tr])<3:
            continue
        fit=np.linalg.lstsq(X[tr],du[tr],rcond=None)[0]
        ideal=gain*x
        val={
            "lag_frames":lag,
            "train_count":int(tr.sum()),
            "test_count":int(te.sum()),
            "r2_free_train":_score(du[tr],X[tr]@fit),
            "r2_free_test":_score(du[te],X[te]@fit),
            "r2_expected_controller_test":_score(du[te],ideal[te]),
            "coefficient_matrix_2_by_2_command_per_recorded_arcsec":
                fit[1:].tolist(),
            "intercept_command_units_per_frame":fit[0].tolist(),
            "expected_controller_matrix_2_by_2":
                (np.eye(2)*gain).tolist(),
            "relative_matrix_deviation_from_expected":float(
                np.linalg.norm(fit[1:]-gain*np.eye(2))/
                (gain*np.sqrt(2))
            ),
            "relative_expected_controller_rms_error_test":float(
                np.sqrt(np.mean((du[te]-ideal[te])**2))/
                max(np.sqrt(np.mean(du[te]**2)),1e-30)
            ),
            "measured_centroid_covariance_arcsec2":
                np.cov(x[te],rowvar=False,bias=True).tolist(),
            "observed_command_increment_covariance_archived_units2":
                np.cov(du[te],rowvar=False,bias=True).tolist(),
            "expected_command_increment_covariance_from_centroids":
                (gain**2*np.cov(x[te],rowvar=False,bias=True)).tolist(),
        }
        scans.append(val)
    if not scans:
        raise ValueError("Insufficient excitation or valid timestamp frames")
    best=max(scans,key=lambda a:a["r2_free_train"])
    result={
        "status":"controller_signal_consistency_only",
        "frame_count":len(y),
        "servo_coefficients":np.asarray(servo,dtype=float).ravel().tolist(),
        "integrator_gain":gain,
        "timebase":timing,
        "best_lag_frames_train":best["lag_frames"],
        "best_alignment":best,
        "lag_scan":scans,
        "interpretation":{
            "controller_increments_are_independent_optical_measurement":False,
            "integrator_difference_equals_recorded_centroids_under_model":True,
            "archive_command_unit_verified_as_true_sky_arcsec":False,
            "lag_is_not_physical_mirror_latency":True,
            "optical_gain_identified":False,
            "physical_jitter_covariance_estimated":False,
            "warning":(
                "If the controller relation holds, command increments carry "
                "the same calibrated centroid signal; their covariance "
                "cannot provide an independent correction to STRAP optical "
                "gain. A discrepancy is evidence for timing, units, control "
                "law or feedforward requiring investigation, NOT a fitted "
                "optical gain."
            ),
        },
    }
    return result


def ideal_integrator_sensor_noise_response(frequencies_hz,sampling_hz, *,
                                           loop_gain,optical_gain):
    """S_n for an ideal one-sample-delay mirror and instantaneous WFS.

    Measured y[k]=g*(d[k]-u[k-1])+n[k]
    u[k]=u[k-1]+k_loop*y[k].
    Then y_noise=S_n*n, S_n=(1-q)/(1+(g*k_loop-1)*q).
    This response is a *hypothesis*, not the measured 2020 Keck plant.
    """
    f=np.asarray(frequencies_hz,dtype=float)
    fs=float(sampling_hz); g=float(optical_gain)
    if (f.ndim!=1 or not np.all(np.isfinite(f)) or np.any(f<0)
        or np.any(f>fs/2) or not 0<loop_gain<2 or
        not 0<g or not 0<loop_gain*g<2):
        raise ValueError("Unphysical frequency, loop gain, or optical gain")
    q=np.exp(-2j*np.pi*f/fs)
    return (1-q)/(1+(loop_gain*g-1)*q)


def van_dam_gain_sensitivity(centroids,timestamps,servo, *,
                            optical_gains=(0.5,0.75,1.0,1.25,1.5,2.0),
                            noise_bands=((0.65,0.95),(0.55,0.85),(0.75,0.95)),
                            nperseg=512):
    """Van Dam sensor-noise estimates under *assumed* gain hypotheses.

    Does not optimize any gain. The noise level is re-fitted for EACH
    assumed optical gain and frequency band; therefore this scan cannot
    constitute an independent optical-gain measurement. Outputs are
    centroids in *recorded arcsec*, not true sky angle.
    """
    x=np.asarray(centroids,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<nperseg:
        raise ValueError("Need N>=nperseg paired STRAP centroids")
    gain=require_integrator(servo)
    bad,runs,timing=timestamp_quality(timestamps)
    fs=timing["sampling_hz_median"]
    if len(x)!=len(timestamps):
        raise ValueError("STRAP samples and timestamps differ")
    # Estimate on the longest uniform run, do not concatenate disjoint
    # time slices (which would invent a discontinuous stationary signal).
    a,b=runs[0]
    if b-a<nperseg:
        raise ValueError("No sufficiently long contiguous STRAP run")
    samples=x[a:b]
    grid=np.linspace(0,fs/2,2049)
    trials=[]
    for g in optical_gains:
        s=ideal_integrator_sensor_noise_response(
            grid,fs,loop_gain=gain,optical_gain=g
        )
        for band in noise_bands:
            with warnings.catch_warnings(record=True) as warn:
                warnings.simplefilter("always",RuntimeWarning)
                cov,diag=estimate_tt_noise(
                    samples,method="van_dam",sample_rate_hz=fs,
                    rejection_frequency_hz=grid,
                    rejection_transfer=s,
                    high_frequency_band=band,nperseg=nperseg,
                )
            # Preserve the *observed centroids* interpretation.
            atmospheric_measured=(
                np.asarray(diag["observed_covariance_m2"],dtype=float)-
                np.asarray(
                    diag["predicted_observed_noise_covariance_m2"],
                    dtype=float
                )
            )
            trial={
                "assumed_effective_optical_gain":float(g),
                "band_fraction_of_nyquist":list(band),
                "sensor_noise_covariance_recorded_arcsec2":cov.tolist(),
                "filtered_sensor_noise_covariance_recorded_arcsec2":
                    diag["predicted_observed_noise_covariance_m2"],
                "observed_centroid_covariance_recorded_arcsec2":
                    diag["observed_covariance_m2"],
                "measured_centroid_atmospheric_component_arcsec2":
                    atmospheric_measured.tolist(),
                "measured_component_min_eigenvalue_arcsec2":
                    float(np.linalg.eigvalsh(atmospheric_measured).min()),
                "predicted_filtered_noise_fraction_per_axis":
                    diag["predicted_observed_noise_fraction_per_axis"],
                "warnings":[str(w.message) for w in warn],
                "fit_bins":diag["fit_bin_count"],
            }
            trials.append(trial)
    return {
        "status":"conditional_noise_sensitivity_not_optical_gain_fit",
        "source":"DTTCENTROIDS in *recorded* arcsec",
        "sample_rate_hz":fs,
        "time_interval_frames":[a,b],
        "hypothetical_loop_model":{
            "sensor":"unity instantaneous WFS; neglect integration",
            "controller":"exact archived IIR 0.1/(1-z^-1)",
            "mirror":"unit angular plant delayed by one sample (unverified)",
            "gain_convention":"recorded centroid / physical residual sky angle",
            "controller_command_units_verified":False,
            "physical_mirror_response_verified":False,
        },
        "nperseg":int(nperseg),
        "trials":trials,
        "selection":{
            "optical_gain_estimated":False,
            "no_gain_ranking_from_sensor_noise_floors":True,
            "scientific_psf_used":False,
            "physical_jitter_covariance_estimated":False,
            "note":(
                "The white noise variance is re-estimated for each assumed "
                "gain; low-frequency atmospheric PSD is unconstrained. "
                "Without independent physical command calibration or "
                "atmospheric spectral prior the gain is not identifiable "
                "from command increments and centroid PSD alone."
            ),
        },
    }
