"""P4-D synthetic DTT mechanical audit: no Keck SAV required.

These tests establish relative transfer identification, not an absolute
Keck STRAP optical or mirror-sky angular gain.
"""
from __future__ import annotations

import numpy as np
import pytest

from p3.telemetry.ttMechanicalAudit import (
    audit_dtt_mechanics, timestamp_quality,
)


def _case(n=8192, *, lag=2, gap=False, change_matrix=False):
    rng=np.random.default_rng(162)
    raw=rng.normal(0,1,size=(n,2))
    u=np.zeros_like(raw)
    for j in range(1,n):
        u[j]=0.65*u[j-1]+raw[j]*0.06
    a=np.array([
        [-1.37777355e-5,-6.05289928e-5,7.43067283e-5],
        [8.70645909e-5,-2.88770046e-5,-5.81875863e-5],
    ])
    gauges=np.empty((n,3))
    for j in range(n):
        effective=a*(1.17 if change_matrix and j>=n//2 else 1.)
        gauges[j]=u[max(j-lag,0)]@effective
    common=2e-5+3e-5*np.sin(2*np.pi*np.arange(n)/n)
    gauges+=common[:,None]
    gauges+=rng.normal(0,1.0e-7,size=gauges.shape)
    conf_ids=np.repeat(np.arange(1000,1000+(n+511)//512),512)[:n]
    time=np.int64(1441442726994439373)+np.arange(n,dtype=np.int64)*20000
    if gap:
        time[n//2:]+=40000
    return u,gauges,time,conf_ids


def test_integer_timestamp_differences_resolve_keck_nanosecond_epoch():
    u,g,t,ids=_case(gap=True)
    bad,runs,report=timestamp_quality(t.astype(np.uint64))
    assert report["sampling_hz_median"]==pytest.approx(500.)
    assert report["nonuniform_interval_count"]==1
    assert report["sampling_hz_mean"] < report["sampling_hz_median"]
    assert len(runs)==2
    assert bad.sum()==1
    assert max(b-a for a,b in runs)==len(t)//2


def test_recovers_two_frame_delay_on_differences_and_validation():
    u,g,t,ids=_case()
    result=audit_dtt_mechanics(u,g,t,ids,nperseg=512)
    assert result["best_difference_delay_train"]["lag_frames"]==2
    assert result["best_difference_delay_train"]["r2_test"]>0.98
    assert result["best_level_delay_train"]["lag_frames"]==2
    assert result["level_response_at_difference_delay"]["r2_test"]>0.99
    assert result["configuration_stability"]["fit_segment_count"]>=15
    assert result["spectral_response"]["status"]=="ok"
    assert result["calibration_status"]["strap_optical_gain_identified"] is False
    assert result["calibration_status"]["gauge_to_sky_angle_independently_calibrated"] is False


def test_coherence_and_phase_recover_time_delay_without_sky_gain():
    u,g,t,ids=_case()
    result=audit_dtt_mechanics(u,g,t,ids,nperseg=512)
    spectral=result["spectral_response"]
    assert spectral["total_independent_welch_windows"]>=6
    for axis in spectral["axes"]:
        assert axis["phase_fit_bins"]>=6
        assert axis["phase_delay_fit_ms"]==pytest.approx(4.,abs=1.0)
        band=next(b for b in axis["coherence_bands"] if b["hz"]==[5,20])
        assert band["median_coherence"]>0.95


def test_timestamp_gap_never_joined_across_welch_and_config_epochs():
    u,g,t,ids=_case(gap=True)
    result=audit_dtt_mechanics(u,g,t,ids,nperseg=512)
    assert result["timing"]["nonuniform_interval_count"]==1
    assert result["best_difference_delay_train"]["lag_frames"]==2
    assert result["spectral_response"]["status"]=="ok"
    assert result["configuration_stability"]["fit_segment_count"]>=14
    assert any("Nonuniform timestamps" in w for w in result["warnings"])


def test_configuration_drift_is_flagged_not_interpreted_as_gain():
    u,g,t,ids=_case(change_matrix=True)
    result=audit_dtt_mechanics(u,g,t,ids,nperseg=512)
    segs=[
        s for s in result["configuration_stability"]["segments"]
        if s["status"]=="fit"
    ]
    assert len(segs)>=15
    first=[s["matrix_change_relative_frobenius"]
           for s in segs if s["start"]<len(u)//2]
    second=[s["matrix_change_relative_frobenius"]
            for s in segs if s["start"]>=len(u)//2]
    assert np.median(second)>np.median(first)+0.08
    assert result["calibration_status"]["residual_sky_jitter_measured"] is False


def test_degenerate_or_invalid_inputs_fail_closed():
    u,g,t,ids=_case(n=4096)
    with pytest.raises(ValueError,match="Non-increasing"):
        timestamp_quality(np.array([1]*128,dtype=np.int64))
    with pytest.raises(ValueError,match="same frame count"):
        audit_dtt_mechanics(u,g[:-1],t,ids,nperseg=512)
    with pytest.raises(ValueError,match="No valid candidate delay"):
        audit_dtt_mechanics(np.zeros_like(u),g,t,ids,nperseg=512)
