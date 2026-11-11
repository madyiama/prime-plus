"""P4-D: Keck DTT command/strain-gauge mechanical telemetry audit.

Pure telemetry analysis. No assumption that gauges are angles or that
DTTCOMMANDS are calibrated in arcsec. No modification of PSF-R, sensor
noise, gain or actuator transfer. In particular, statistical command-
gauge correlation is NOT a measurement of the residual sky tip-tilt.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch, csd


def timestamp_quality(raw_timestamp, *, tick_seconds=1e-7, gap_ratio=1.5):
    """Use integer tick differences to avoid float64 loss at ~1e18 ticks."""
    ticks = np.asarray(raw_timestamp).reshape(-1)
    if len(ticks) < 32 or not np.issubdtype(ticks.dtype, np.integer):
        raise ValueError("Expected >=32 integer Keck RTC timestamps")
    ticks = ticks.astype(np.int64)
    dt = np.diff(ticks)
    if np.any(dt <= 0):
        raise ValueError("Non-increasing/duplicate Keck RTC timestamps")
    median = float(np.median(dt))
    if median <= 0 or tick_seconds <= 0:
        raise ValueError("Invalid RTC time calibration")
    bad = (dt > median*gap_ratio) | (dt < median/gap_ratio)
    boundaries = [0, *(np.flatnonzero(bad)+1).tolist(), len(ticks)]
    runs = [
        (int(a), int(b))
        for a, b in zip(boundaries[:-1], boundaries[1:])
        if b > a
    ]
    runs.sort(key=lambda x: x[1]-x[0], reverse=True)
    return bad, runs, {
        "tick_seconds_assumed": float(tick_seconds),
        "sampling_hz_median": float(1/(median*tick_seconds)),
        "sampling_hz_mean": float(1/(np.mean(dt)*tick_seconds)),
        "timestamp_median_ticks": median,
        "timestamp_mean_ticks": float(np.mean(dt)),
        "timestamp_interval_p01_p99_ticks": np.percentile(dt,[1,99]).tolist(),
        "nonuniform_interval_count": int(np.count_nonzero(bad)),
        "nonuniform_interval_indices_first_20":
            np.flatnonzero(bad)[:20].astype(int).tolist(),
        "longest_uniform_run_frames": int(runs[0][1]-runs[0][0]),
        "longest_uniform_run_start": int(runs[0][0]),
        "cadence_consistency_relative":
            float(abs(np.mean(dt)-median)/median),
        "timebase_uniform": not bool(np.any(bad)),
    }


def _differential_gauge(g):
    g = np.asarray(g, dtype=float)
    if g.ndim != 2 or g.shape[1] != 3 or not np.isfinite(g).all():
        raise ValueError("DTTSTRAINGAUGE must be finite (frames, 3)")
    return g - g.mean(axis=1, keepdims=True)


def _aligned(u, gd, bad_step, lag, *, difference, split=0.7):
    """Align gauge[k+lag] with command[k], excluding timestamp gaps."""
    n = len(u)
    if gd.shape != (n,3) or bad_step.shape != (n-1,):
        raise ValueError("Signal dimensions/timebase mismatch")
    k = np.arange(n, dtype=int)
    command_idx = k[max(0,-lag):min(n,n-lag)]
    gauge_idx = command_idx+lag
    prefix = np.concatenate([[0],np.cumsum(bad_step.astype(int))])
    lo,hi = np.minimum(command_idx,gauge_idx),np.maximum(command_idx,gauge_idx)
    mask = (prefix[hi]-prefix[lo]) == 0
    if difference:
        mask &= (command_idx>0) & (gauge_idx>0)
        mask &= ~bad_step[np.maximum(command_idx-1,0)]
        mask &= ~bad_step[np.maximum(gauge_idx-1,0)]
        x = u[command_idx] - u[np.maximum(command_idx-1,0)]
        y = gd[gauge_idx] - gd[np.maximum(gauge_idx-1,0)]
    else:
        x = u[command_idx]
        y = gd[gauge_idx]
    train_cut=int(n*split)
    train = mask & (hi < train_cut)
    test = mask & (lo >= train_cut)
    return x,y,train,test


def _linear_fit(x,y, train, test, *, difference=False):
    if train.sum() < 128 or test.sum() < 64:
        raise ValueError("Not enough independent train/test valid frames")
    X = x if difference else np.column_stack([np.ones(len(x)),x])
    train_X = X[train]
    if np.linalg.matrix_rank(train_X) < train_X.shape[1]:
        raise ValueError("Command variation has deficient rank")
    beta=np.linalg.lstsq(train_X, y[train],rcond=None)[0]
    result={}
    for label,idx in (("train",train),("test",test)):
        base=np.zeros(3) if difference else y[train].mean(axis=0)
        denom=float(np.sum((y[idx]-base)**2))
        mse=float(np.sum((y[idx]-X[idx]@beta)**2))
        result["r2_"+label]=float(1-mse/denom) if denom>0 else None
        result["rms_error_"+label+"_gauge_units"]=float(
            np.sqrt(np.mean((y[idx]-X[idx]@beta)**2))
        )
        result["count_"+label]=int(idx.sum())
    result["matrix_2_by_3"]=beta.tolist() if difference else beta[1:].tolist()
    if not difference:
        result["intercept_gauge_units"]=beta[0].tolist()
    return result,beta


def _lag_scan(u, gd, bad, *, max_lag=10, difference=False, split=0.7):
    scans=[]
    for lag in range(-max_lag,max_lag+1):
        x,y,train,test=_aligned(u,gd,bad,lag,difference=difference,split=split)
        try:
            report,beta=_linear_fit(x,y,train,test,difference=difference)
        except ValueError as exc:
            scans.append({"lag_frames":lag,"valid":False,"reason":str(exc)})
            continue
        scans.append({"lag_frames":int(lag),"valid":True,**report})
    valid=[s for s in scans if s["valid"] and s["r2_train"] is not None]
    if not valid:
        raise ValueError("No valid candidate delay for command/strain-gauge data")
    best=max(valid,key=lambda s:s["r2_train"])
    return scans,best


def _conf_id_segments(ids):
    change=np.flatnonzero(ids[1:]!=ids[:-1])+1
    boundaries=[0,*change.tolist(),len(ids)]
    return [(int(a),int(b),int(ids[a]))
            for a,b in zip(boundaries[:-1],boundaries[1:])]


def _configuration_stability(u,gd,ids,bad,*,lag,global_matrix,
                             min_frames=128,cond_max=1e4):
    segments=[]
    for a,b,cid in _conf_id_segments(ids):
        item={"conf_id":cid,"start":a,"stop":b,"frames":b-a}
        if b-a < min_frames or np.any(bad[a:max(a,b-1)]):
            item["status"]="skipped_short_or_timestamp_gap"
            segments.append(item)
            continue
        k=np.arange(a+max(0,-lag),b-max(0,lag))
        t=k+lag
        if len(k)<min_frames:
            item["status"]="skipped_insufficient_aligned_frames"
            segments.append(item)
            continue
        x=u[k];y=gd[t]
        xc=x-x.mean(axis=0)
        singular=np.linalg.svd(xc,compute_uv=False)
        cond=float(singular[0]/singular[-1]) if singular[-1]>0 else np.inf
        item["command_excitation_condition"]=cond if np.isfinite(cond) else None
        if not np.isfinite(cond) or cond>cond_max:
            item["status"]="skipped_unidentifiable_command_directions"
            segments.append(item)
            continue
        X=np.column_stack([np.ones(len(x)),x])
        beta=np.linalg.lstsq(X,y,rcond=None)[0]
        a_local=beta[1:]
        resid=y-X@beta
        denom=float(np.sum((y-y.mean(axis=0))**2))
        item.update({
            "status":"fit",
            "r2_in_segment":float(1-np.sum(resid**2)/denom) if denom>0 else None,
            "matrix_change_relative_frobenius":float(
                np.linalg.norm(a_local-global_matrix)/
                max(np.linalg.norm(global_matrix),1e-30)
            ),
            "command_std_archived_units":x.std(axis=0).tolist(),
            "matrix_2_by_3":a_local.tolist(),
        })
        segments.append(item)
    return {
        "distinct_configuration_ids":len(np.unique(ids)),
        "configuration_id_transitions":int(np.count_nonzero(np.diff(ids))),
        "segments":segments,
        "fit_segment_count":sum(s["status"]=="fit" for s in segments),
        "note":"CONF_ID values are IDs, not physical gains or calibration epochs",
    }


def _spectral(u,gd,bad,runs,*,matrix,intercept,lag,fs,nperseg=1024,split=0.7):
    """Estimate transfer/coherence by Welch on timestamp-contiguous blocks.

    Pool *spectra*, never blindly concatenate timestamp-disjoint blocks.
    Welch cross-spectra need multiple segments: with only one, measured
    coherence is identically 1 and is scientifically meaningless.
    The gauge projection comes from TRAIN-only level regression.
    This is descriptive single-axis coherence, not a calibrated MIMO TF.
    """
    if np.linalg.matrix_rank(matrix)<2:
        raise ValueError("Gauge mixing matrix has rank <2")
    m=(gd-intercept)@np.linalg.pinv(matrix)  # command-equivalent units
    split_at=int(len(u)*split)
    original=int(nperseg)
    candidates=sorted(set([original,512,256,128]),reverse=True)
    candidates=[x for x in candidates if x<=original and x>=128]
    choice=None
    for nseg in candidates:
        for role in ("held_out_validation","all_frames_not_independent"):
            candidates_runs=[
                (max(a,split_at) if role=="held_out_validation" else a,b)
                for a,b in runs
            ]
            candidates_runs=[(a,b) for a,b in candidates_runs if b-a>=nseg]
            segment_counts=[
                1+(b-a-nseg)//(nseg//2)
                for a,b in candidates_runs
            ]
            if sum(segment_counts)>=6:
                choice=(nseg,role,candidates_runs,segment_counts)
                break
        if choice is not None:
            break
    if choice is None:
        return {
            "status":"insufficient_contiguous_welch_segments",
            "requested_nperseg":original,
            "longest_run_frames":int(max(b-a for a,b in runs)),
            "required_segment_count":6,
        }
    nseg,role,selected,segment_counts=choice
    out={
        "status":"ok",
        "segment_role":role,
        "nperseg_requested":original,
        "nperseg_effective":nseg,
        "sample_rate_hz":float(fs),
        "timestamp_contiguous_block_count":len(selected),
        "total_independent_welch_windows":sum(segment_counts),
        "axes":[],
    }
    for axis in range(2):
        psd_x=psd_y=cspec=None
        f=None
        n_total=0
        kwargs=dict(fs=fs,nperseg=nseg,noverlap=nseg//2,
                    detrend="linear",scaling="density")
        for (a,b),count in zip(selected,segment_counts):
            x=u[a:b,axis]-np.mean(u[a:b,axis])
            y=m[a:b,axis]-np.mean(m[a:b,axis])
            f,p_x=welch(x,**kwargs)
            _,p_y=welch(y,**kwargs)
            _,p_xy=csd(x,y,**kwargs)
            if psd_x is None:
                psd_x=np.zeros_like(p_x)
                psd_y=np.zeros_like(p_y)
                cspec=np.zeros_like(p_xy)
            psd_x+=p_x*count
            psd_y+=p_y*count
            cspec+=p_xy*count
            n_total+=count
        psd_x/=n_total
        psd_y/=n_total
        cspec/=n_total
        H=cspec/np.maximum(psd_x,np.finfo(float).tiny)
        coh=np.abs(cspec)**2/np.maximum(
            psd_x*psd_y,np.finfo(float).tiny
        )
        coh=np.clip(coh,0,1)
        fmax=min(80.,0.3*fs)
        candidate=(f>=3.)&(f<=fmax)
        fit=candidate&(coh>=0.8)&(
            psd_x>=np.percentile(psd_x[candidate],25)
        )
        phase_delay=None
        phase_error=None
        if np.sum(fit)>=6:
            ff=f[fit];hh=H[fit]
            # Circular phase residual, allowing frequency response to
            # deviate from a pure delay. Requires both strong excitation
            # and coherent transfer bins.
            tau=np.arange(0.,(max(abs(lag),1)+6)/fs,0.0001)
            angular=np.angle(hh[None,:]*np.exp(
                2j*np.pi*tau[:,None]*ff[None,:]
            ))
            weights=coh[fit]*psd_x[fit]
            weights/=weights.sum()
            loss=np.sum(weights[None,:]*angular**2,axis=1)
            j=int(np.argmin(loss))
            phase_delay=float(tau[j]*1000.)
            phase_error=float(np.sqrt(loss[j])*180/np.pi)
        bands=[]
        for lo,hi in [(0,5),(5,20),(20,50),(50,100),(100,200)]:
            mask=(f>=lo)&(f<hi)
            if np.any(mask):
                bands.append({
                    "hz":[lo,hi],
                    "bin_count":int(np.sum(mask)),
                    "median_coherence":float(np.median(coh[mask])),
                    "median_command_equivalent_gain":float(np.median(np.abs(H[mask]))),
                })
        out["axes"].append({
            "axis":axis,
            "coherence_bands":bands,
            "phase_delay_fit_ms":phase_delay,
            "phase_rms_error_deg":phase_error,
            "phase_fit_bins":int(np.sum(fit)),
            "frequency_hz":f.tolist(),
            "coherence":coh.tolist(),
            "transfer_gain":np.abs(H).tolist(),
            "transfer_phase_deg":np.angle(H,deg=True).tolist(),
        })
    out["warning"]=(
        "Scalar axis-pair coherence after projection; controller axes "
        "may be correlated. These are archived-command-equivalent units. "
        "No gauge-to-sky angular calibration or optical gain identified."
    )
    return out


def audit_dtt_mechanics(commands, gauges, timestamps, conf_ids, *,
                        max_lag=10, nperseg=1024, split=0.7,
                        tick_seconds=1e-7):
    """Report level and differentiated delays, spectral, CONF_ID checks."""
    u=np.asarray(commands,dtype=float)
    if u.ndim!=2 or u.shape[1]!=2 or not np.isfinite(u).all():
        raise ValueError("DTTCOMMANDS must be finite (frames,2)")
    gd=_differential_gauge(gauges)
    ids=np.asarray(conf_ids).reshape(-1)
    if len(u)!=len(gd) or len(u)!=len(ids) or len(u)!=len(timestamps):
        raise ValueError("All DTT telemetry arrays must have same frame count")
    if len(u)<max(2*nperseg,1024):
        raise ValueError("Need at least twice nperseg frames")
    bad,runs,timing=timestamp_quality(timestamps,tick_seconds=tick_seconds)
    levels,baseline=_lag_scan(u,gd,bad,max_lag=max_lag,split=split)
    deltas,best=_lag_scan(u,gd,bad,max_lag=max_lag,split=split,
                          difference=True)
    # Do not assume that the differenced optimum is the true actuator
    # delay; identify level response at the same lag, on train only.
    x,y,tr,te=_aligned(u,gd,bad,best["lag_frames"],difference=False,split=split)
    level_at_delay,beta=_linear_fit(x,y,tr,te,difference=False)
    conf=_configuration_stability(
        u,gd,ids,bad,lag=best["lag_frames"],global_matrix=beta[1:]
    )
    spectral=_spectral(
        u,gd,bad,runs,matrix=beta[1:],intercept=beta[0],
        lag=best["lag_frames"],fs=timing["sampling_hz_median"],
        nperseg=nperseg,split=split
    )
    warnings=[]
    if timing["nonuniform_interval_count"]:
        warnings.append(
            "Nonuniform timestamps: spectral estimate uses a single "
            "contiguous segment; average sample rate is NOT valid for PSD"
        )
    if best["r2_test"] is None or best["r2_test"] < 0.5:
        warnings.append(
            "Differenced transfer does not generalize strongly: delay "
            "is not reliably established from these telemetry samples"
        )
    if spectral["status"]!="ok":
        warnings.append("Spectral response unavailable: "+spectral["status"])
    elif spectral["segment_role"]!="held_out_validation":
        warnings.append("Frequency diagnostics not on held-out frames")
    return {
        "status":"telemetry_mechanical_relative_audit_only",
        "frame_count":int(len(u)),
        "duration_seconds":float(
            (np.asarray(timestamps,dtype=np.int64)[-1]
             -np.asarray(timestamps,dtype=np.int64)[0])*tick_seconds
        ),
        "timing":timing,
        "level_delay_scan":levels,
        "best_level_delay_train":baseline,
        "difference_delay_scan":deltas,
        "best_difference_delay_train":best,
        "level_response_at_difference_delay":level_at_delay,
        "spectral_response":spectral,
        "configuration_stability":conf,
        "warnings":warnings,
        "calibration_status":{
            "gauge_to_command_relative_mapping_estimated":True,
            "gauge_to_sky_angle_independently_calibrated":False,
            "strap_optical_gain_identified":False,
            "residual_sky_jitter_measured":False,
            "real_mechanical_delay_confirmed":False,
        },
    }
