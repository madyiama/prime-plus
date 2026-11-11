"""Synthetic P4-F Keck STRAP slope replay tests.

No Keck real SAV is needed to prove exact integrator invariance. The
real SAV must still be run on the Mac for science-PSF image comparison.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.tipTiltGeometry import zernike_tilt_opd_per_arcsec
from p3.telemetry.ttCommandReplay import (
    SOURCE_OPTIONS,apply_tt_slope_source,prepare_tt_command_replay,
)
from p3.telemetry.ttNoise import estimate_tt_noise


SERVO=np.array([.1,0,0,0,-1,0,0])
D=10.5
GAIN=zernike_tilt_opd_per_arcsec(D)


def _fixtures(n=4096,seed=7):
    rng=np.random.default_rng(seed)
    frames=np.arange(n)
    # Colored atmosphere with independent white STRAP detector noise.
    y=np.column_stack([
        0.04*np.sin(2*np.pi*frames/97),
        0.03*np.cos(2*np.pi*frames/83),
    ])+rng.normal(0,0.01,size=(n,2))
    u=np.zeros_like(y)
    for k in range(1,n):
        u[k]=u[k-1]+0.1*y[k]
    return y,u


def _trs(y,u):
    tilt=NS(
        slopes_arcsec=y.copy(),
        slopes=(y-y.mean(axis=0))*GAIN,
        com_raw=u.copy(),
        com=(u-u.mean(axis=0))*GAIN,
        intensity=np.ones((len(y),4)),
        nExp=len(y),
        tilt2meter=GAIN,
    )
    return NS(tel=NS(D=D),aoMode="LGS",tipTilt=tilt)


def test_exact_integrator_yields_identical_aligned_tt_series():
    y,u=_fixtures()
    ref,replay,audit=prepare_tt_command_replay(y,u,SERVO)
    np.testing.assert_allclose(replay,ref,rtol=1e-7,atol=2e-13)
    assert audit["relative_rms_error_vs_centroids"]<1e-10
    assert audit["input_frame_count"]==4096
    assert audit["aligned_frame_count"]==4095
    assert not audit["command_units_independently_calibrated_to_sky"]


def test_both_ab_variants_preserve_same_tt_covariance_and_noise():
    y,u=_fixtures()
    a,b=_trs(y,u),_trs(y,u)
    ra=apply_tt_slope_source(a,SERVO,"centroids_aligned")
    rb=apply_tt_slope_source(b,SERVO,"command_increment")
    assert ra["first_frame_removed"] and rb["first_frame_removed"]
    assert a.tipTilt.nExp==b.tipTilt.nExp==len(y)-1
    for name in ("slopes_arcsec","slopes","com_raw","com","intensity"):
        np.testing.assert_allclose(getattr(a.tipTilt,name),
                                   getattr(b.tipTilt,name),
                                   rtol=1e-9,atol=1e-12)
    ca=np.cov(a.tipTilt.slopes_arcsec,rowvar=False,bias=True)
    cb=np.cov(b.tipTilt.slopes_arcsec,rowvar=False,bias=True)
    np.testing.assert_allclose(ca,cb,rtol=1e-10,atol=1e-14)
    fs=500.
    f=np.linspace(0,fs/2,1001)
    rt=np.ones_like(f)
    an,_=estimate_tt_noise(
        a.tipTilt.slopes_arcsec,method="van_dam",
        sample_rate_hz=fs,rejection_frequency_hz=f,
        rejection_transfer=rt,nperseg=512,
    )
    bn,_=estimate_tt_noise(
        b.tipTilt.slopes_arcsec,method="van_dam",
        sample_rate_hz=fs,rejection_frequency_hz=f,
        rejection_transfer=rt,nperseg=512,
    )
    np.testing.assert_allclose(an,bn,rtol=1e-9,atol=1e-14)
    cov_a=ca-an
    cov_b=cb-bn
    np.testing.assert_allclose(cov_a,cov_b,rtol=1e-9,atol=1e-14)


def test_original_centroids_mode_is_complete_no_op():
    y,u=_fixtures()
    x=_trs(y,u)
    old=x.tipTilt.slopes.copy()
    report=apply_tt_slope_source(x,SERVO,"centroids")
    assert not report["experimental_replay"]
    assert not report["first_frame_removed"]
    assert x.tipTilt.nExp==len(y)
    np.testing.assert_array_equal(x.tipTilt.slopes,old)
    np.testing.assert_array_equal(x.tipTilt.slopes_arcsec,y)


def test_replay_refuses_wrong_controller_or_unexplained_angle_factor():
    y,u=_fixtures()
    with pytest.raises(ValueError,match="integrator"):
        prepare_tt_command_replay(y,u,[0.1,.1,0,0,-1,0,0])
    with pytest.raises(ValueError,match="NOT equivalent"):
        prepare_tt_command_replay(y,u*0.576,SERVO)
    with pytest.raises(ValueError,match="NOT equivalent"):
        prepare_tt_command_replay(y,u,SERVO,
                                  max_relative_rms_mismatch=1e-15)
    assert SOURCE_OPTIONS==(
        "centroids","centroids_aligned","command_increment"
    )


def test_no_ngs_tt_or_independent_gain_claim():
    y,u=_fixtures()
    x=_trs(y,u)
    x.aoMode="NGS"
    with pytest.raises(ValueError,match="requires LGS"):
        apply_tt_slope_source(x,SERVO,"command_increment")
    x.aoMode="LGS"
    out=apply_tt_slope_source(x,SERVO,"command_increment")
    assert not out["strap_optical_gain_independently_calibrated"]
    assert not out["physical_tt_jitter_calibrated"]
    assert out["noise_estimation_uses_chosen_tt_source"]
    assert out["ho_telemetry_unmodified"]


def test_smoke_cli_exposes_tt_slope_source():
    import subprocess
    import sys
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/"scripts"/"keck_n0013_smoke.py"
    result=subprocess.run([sys.executable,str(script),"--help"],
                          capture_output=True,text=True,check=False,timeout=30)
    assert result.returncode==0,result.stderr
    assert "--tt-slope-source" in result.stdout
    assert "command_increment" in result.stdout
