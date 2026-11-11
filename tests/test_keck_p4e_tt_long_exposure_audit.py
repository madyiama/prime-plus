"""Synthetic P4-E tests: actual controller gain is known; optical gain is not.

No test may infer a physical sky jitter or an optical gain from the
algebraically redundant command increments.
"""
import numpy as np
import pytest

from scipy.integrate import trapezoid

from p3.telemetry.ttConditionalCovariance import conditional_residual_covariance
from p3.telemetry.ttLongExposureAudit import (
    audit_command_increments,
    ideal_integrator_sensor_noise_response,
    require_integrator,
    van_dam_gain_sensitivity,
)


SERVO=np.array([0.1,0,0,0,-1,0,0],dtype=float)


def _sample(n=6144,lag=2,*,bad_gain=False):
    rng=np.random.default_rng(814)
    y=rng.normal(0,0.03,size=(n,2))
    # Colored residual plus white detector noise, in recorded arcsec.
    residual=np.zeros_like(y)
    for k in range(1,n):
        residual[k]=0.92*residual[k-1]+y[k]*0.2
    y=y+residual
    u=np.zeros_like(y)
    controller_gain=0.15 if bad_gain else 0.1
    for k in range(1,n):
        u[k]=u[k-1]+controller_gain*y[max(0,k-lag)]
    ticks=(np.int64(1441442726994439373)+
           np.arange(n,dtype=np.int64)*20000).astype(np.uint64)
    return y,u,ticks


def test_integrator_coefficients_match_exact_archive_and_fail_closed():
    assert require_integrator(SERVO)==pytest.approx(0.1)
    with pytest.raises(ValueError,match="Only the verified"):
        require_integrator([.1,0,.1,0,-1,0,0])
    with pytest.raises(ValueError,match="7 finite"):
        require_integrator([.1,0,0,0,-1])


def test_exact_controller_increments_recover_delay_without_optical_gain():
    y,u,t=_sample()
    result=audit_command_increments(y,u,t,SERVO)
    best=result["best_alignment"]
    assert result["best_lag_frames_train"]==2
    assert best["r2_free_test"]>0.999999
    assert best["r2_expected_controller_test"]>0.999999
    assert best["relative_matrix_deviation_from_expected"]<1e-6
    assert best["relative_expected_controller_rms_error_test"]<1e-8
    np.testing.assert_allclose(
        best["coefficient_matrix_2_by_2_command_per_recorded_arcsec"],
        np.eye(2)*0.1,rtol=1e-6,atol=1e-9,
    )
    assert not result["interpretation"]["optical_gain_identified"]
    assert not result["interpretation"]["physical_jitter_covariance_estimated"]
    assert not result["interpretation"][
        "controller_increments_are_independent_optical_measurement"
    ]


def test_wrong_controller_scale_is_detected_not_mislabeled_optical_gain():
    y,u,t=_sample(bad_gain=True)
    result=audit_command_increments(y,u,t,SERVO)
    best=result["best_alignment"]
    assert best["r2_free_test"]>0.999999
    assert best["r2_expected_controller_test"]<0.9
    assert best["relative_matrix_deviation_from_expected"]>0.4
    assert not result["interpretation"]["optical_gain_identified"]


def test_integrator_sensor_noise_sensitivity_matches_known_equations():
    fs=500.
    f=np.array([0.,10.,100.,200.,fs/2])
    s=ideal_integrator_sensor_noise_response(
        f,fs,loop_gain=.1,optical_gain=1.)
    q=np.exp(-2j*np.pi*f/fs)
    np.testing.assert_allclose(s,(1-q)/(1-0.9*q),rtol=1e-14)
    assert s[0]==0
    with pytest.raises(ValueError,match="Unphysical"):
        ideal_integrator_sensor_noise_response(
            f,fs,loop_gain=.1,optical_gain=0
        )


def test_van_dam_reestimated_for_each_gain_and_band_without_gain_fit():
    y,_,t=_sample()
    result=van_dam_gain_sensitivity(
        y,t,SERVO,optical_gains=(.7,1.0,1.4),
        noise_bands=((.65,.95),(.75,.95)),nperseg=512
    )
    assert len(result["trials"])==6
    assert result["status"]=="conditional_noise_sensitivity_not_optical_gain_fit"
    assert not result["selection"]["optical_gain_estimated"]
    assert not result["selection"]["physical_jitter_covariance_estimated"]
    assert not result["hypothetical_loop_model"][
        "physical_mirror_response_verified"
    ]
    for trial in result["trials"]:
        vals=np.asarray(trial["sensor_noise_covariance_recorded_arcsec2"])
        assert vals.shape==(2,2)
        assert np.isfinite(vals).all()
        assert np.linalg.eigvalsh(vals).min()>-1e-12
        conditional=trial["conditional_jitter_band_sensitivity"]
        assert conditional["status"]==(
            "conditional_band_limited_unverified_instrument_plant"
        )
        assert not conditional["filter_plant_and_command_absolute_scale_calibrated"]
        assert conditional["welch_low_frequency_leakage_not_included"] is True
        assert conditional["frequency_interval_hz"][0]>0


def test_mismatched_input_or_timestamps_fail_not_give_jitter():
    y,u,t=_sample(n=2048)
    with pytest.raises(ValueError,match="paired"):
        audit_command_increments(y,u[:-1],t,SERVO)
    with pytest.raises(ValueError,match="length mismatch"):
        audit_command_increments(y,u,t[:-1],SERVO)
    with pytest.raises(ValueError,match="max_lag"):
        audit_command_increments(y,u,t,SERVO,max_lag=0)


def test_conditional_covariance_integral_matches_independent_discrete_psd():
    """No hidden g² factors, no noise double subtraction, no PSD repair."""
    fs=500.
    f=np.linspace(0,fs/2,513)
    g=0.8
    k=0.1
    cn=np.array([[0.0001,0.00002],[0.00002,0.00014]])
    q=np.exp(-2j*np.pi*f/fs)
    den=1+(g*k-1)*q
    S=(1-q)/den
    T=-q*k/den
    pn=2*cn/fs
    # A valid physical 2x2 incoming-tilt spectrum.
    cd=np.array([[0.0008,0.0001],[0.0001,0.0006]])
    pd=cd[:,:,None]/(1+(f/15)**2)[None,None,:]
    py=g*g*np.abs(S)[None,None,:]**2*pd+(
        np.abs(S)[None,None,:]**2*pn[:,:,None]
    )
    expected_pe=np.abs(S)[None,None,:]**2*pd+(
        np.abs(T)[None,None,:]**2*pn[:,:,None]
    )
    result=conditional_residual_covariance(
        f,py,cn,sampling_hz=fs,controller_gain=k,
        optical_gain=g
    )
    fit=f>=f[1]
    expected=trapezoid(expected_pe[:,:,fit],x=f[fit],axis=-1)
    np.testing.assert_allclose(
        result["residual_band_covariance_conditional_true_arcsec2"],
        expected,rtol=1e-12,atol=1e-18
    )
    assert result["band_covariance_psd"]
    assert result["atmospheric_measured_spectral_bins_negative"]==0
    assert result["no_psd_projection"]
    assert not result["filter_plant_and_command_absolute_scale_calibrated"]


def test_conditional_covariance_rejects_invalid_sensor_noise():
    f=np.linspace(0,250,513)
    py=np.zeros((2,2,len(f)))
    with pytest.raises(ValueError,match="Sensor noise must be PSD"):
        conditional_residual_covariance(
            f,py,np.diag([0.1,-0.01]),
            sampling_hz=500.,controller_gain=.1,optical_gain=1.
        )
    with pytest.raises(ValueError,match="Unstable"):
        conditional_residual_covariance(
            f,py,np.eye(2)*0.01,
            sampling_hz=500.,controller_gain=.1,optical_gain=0.
        )
