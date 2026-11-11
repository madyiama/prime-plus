"""Synthetic P4-G G1–G5 scientific invariants, no Keck archive required.

P4-G is an independent HO-prior consistency calculation. It must never
add the separately modelled anisokinetic covariance to science jitter.
"""
from __future__ import annotations
import numpy as np
import pytest

from p3.telemetry.ttHoPrior import (
    ARCSEC_TO_RAD,
    kernel_from_pupil,circular_vk_tilt_variance,
    incident_and_residual_tilt,measured_strap_residual_atmosphere,
    scalar_consistency,
)


def _circular_kernel(n=64,pad=512):
    D=10.5
    x=(np.arange(n)-(n-1)/2)*D/n
    yy,xx=np.meshgrid(x,x,indexing="ij")
    mask=(xx**2+yy**2<=(D/2)**2).astype(float)
    return kernel_from_pupil(mask,D,pad=pad,n_rad=250,n_az=32)


def _model(k,*,r0=.12,L0=25,loop=None,offset=None,
           weights=(.6,.4),heights=(0.,8000.)):
    return incident_and_residual_tilt(
        k,r0_m=r0,L0_m=L0,wavelength_m=500e-9,
        weights=weights,wind_speeds_m_s=(7.,15.),
        wind_dirs_deg=(0.,40.),
        loop_frequency_hz=np.array([0.,1.,300.]) if loop is not None else None,
        loop_sensitivity_power=np.array([loop,loop,loop])
            if loop is not None else None,
        low_band_hz=0.,high_band_hz=250.,
        science_offset_pupil_arcsec=offset,
        altitudes_m=heights if offset is not None else None,
    )


def test_g1_matches_independent_von_karman_circular_bessel_reference():
    kernel=_circular_kernel()
    result=_model(kernel)
    predicted=result["incident_ngs_cov_rad2"]
    reference=circular_vk_tilt_variance(.12,25.,10.5)
    np.testing.assert_allclose(predicted[0,0],reference,rtol=.06)
    np.testing.assert_allclose(predicted[1,1],reference,rtol=.06)
    assert abs(predicted[0,1])<.02*reference
    assert np.linalg.eigvalsh(predicted).min()>0


def test_g1_r0_scaling_and_outer_scale_reduce_tilt():
    kernel=_circular_kernel()
    a=_model(kernel,r0=.12,L0=25.)["incident_ngs_cov_rad2"]
    b=_model(kernel,r0=.24,L0=25.)["incident_ngs_cov_rad2"]
    c=_model(kernel,r0=.12,L0=100.)["incident_ngs_cov_rad2"]
    np.testing.assert_allclose(a/b,2.**(5./3.),rtol=1e-10,atol=1e-12)
    assert np.trace(c)>np.trace(a)


def test_g2_equal_to_incident_for_unfiltered_loop_and_less_with_rejection():
    k=_circular_kernel()
    incident=_model(k)["incident_ngs_cov_rad2"]
    out=_model(k,loop=1.)["residual_ngs_atmos_full_cov_rad2"]
    np.testing.assert_allclose(incident,out,rtol=1e-12,atol=1e-24)
    reduced=_model(k,loop=.1)["residual_ngs_atmos_full_cov_rad2"]
    assert 0<np.trace(reduced)<np.trace(incident)


def test_g5_is_zero_on_axis_separate_and_positive_offaxis():
    k=_circular_kernel()
    zero=_model(k,loop=.1,offset=(0.,0.))
    nonzero=_model(k,loop=.1,offset=(6.,0.))
    np.testing.assert_allclose(zero["anisokinetism_ngs_science_cov_rad2"],
                               np.zeros((2,2)),atol=1e-26)
    diff=nonzero["anisokinetism_ngs_science_cov_rad2"]
    assert np.linalg.eigvalsh(diff).min()>=-1e-25
    assert np.trace(diff)>0
    np.testing.assert_allclose(
        zero["residual_ngs_atmos_full_cov_rad2"],
        nonzero["residual_ngs_atmos_full_cov_rad2"],
        rtol=0.,atol=1e-28,
    )


def test_g5_ground_layer_only_has_no_directional_anisokinetism():
    k=_circular_kernel()
    a=_model(k,offset=(90.,90.),weights=(1.,0.),
             heights=(0.,8000.))
    np.testing.assert_allclose(
        a["anisokinetism_ngs_science_cov_rad2"],
        np.zeros((2,2)),atol=1e-25
    )


def test_g3_measured_strap_atmosphere_divides_by_gain_squared():
    rng=np.random.default_rng(55)
    y=rng.normal(0,.05,size=(4096,2))
    fgrid=np.array([0.,250.])
    def measured(g):
        return measured_strap_residual_atmosphere(
            y,np.zeros((2,2)),sample_rate_hz=500.,
            loop_frequency_hz=fgrid,
            loop_sensitivity_power=np.ones(2),
            loop_sensor_power=np.ones(2),
            gain=g,nperseg=512,
        )["inferred_physical_residual_atm_band_cov_arcsec2"]
    np.testing.assert_allclose(measured(1.),4*measured(2.),
                               rtol=1e-12,atol=1e-19)
    cov=measured(1.)
    assert np.linalg.eigvalsh(cov).min()>0


def test_g4_ratio_is_not_an_optical_gain_and_invalid_psd_is_not_clipped():
    C=np.eye(2)*1e-14
    strap=C/(ARCSEC_TO_RAD**2)
    good=scalar_consistency(C,strap)
    assert good["status"]=="conditional_covariance_trace_ratio"
    assert good["alpha"]==pytest.approx(1.)
    assert not good["physical_optical_gain_identified"]
    bad=scalar_consistency(C,np.diag([-0.001,.00001]))
    assert bad["alpha"] is None
    assert bad["status"]=="invalid_STRAP_residual_covariance"


def test_fail_closed_on_unphysical_hypotheses():
    k=_circular_kernel()
    with pytest.raises(ValueError,match="finite positive"):
        _model(k,r0=-.1)
    with pytest.raises(ValueError,match="G5 anisokinetism needs"):
        incident_and_residual_tilt(
            k,r0_m=.12,L0_m=25.,wavelength_m=500e-9,
            weights=[1],wind_speeds_m_s=[7],
            wind_dirs_deg=[0],
            science_offset_pupil_arcsec=(3,4),
        )


def test_cli_exposes_separated_g5_and_keck_inputs():
    from pathlib import Path
    import subprocess,sys
    script=Path(__file__).resolve().parents[1]/"scripts"/"keck_p4g_ho_to_tt.py"
    out=subprocess.run([sys.executable,str(script),"--help"],
                       capture_output=True,text=True,check=False,timeout=30)
    assert out.returncode==0,out.stderr
    assert "--science-offset-pupil-arcsec" in out.stdout
    assert "--mirror-model" in out.stdout
    assert "--gain-scenarios" in out.stdout
