"""Batched Vii Fourier transform parity, geometry, covariance and safety."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from p3.psfr.hoBatched import accumulate_vii_batched,validate_fft_options
from p3.psfr.psfrUtils import modes2Otf
from p3.psfr.ho import HOStatistics,reconstruct_ho_otf
from p3.psfr.psfR import psfR


def _inputs(n=12,nm=13,seed=230):
    rng=np.random.default_rng(seed)
    yy,xx=np.indices((n,n))
    # deliberately asymmetric binary pupil to exercise rotated NIRC2
    pupil=((xx-n*.44)**2/(n*.38)**2+
           (yy-n*.55)**2/(n*.39)**2<1).astype(float)
    pupil[(xx<n//2)&(yy>n//2)]=0
    modes=rng.normal(scale=.35,size=(n*n,nm))
    R=rng.normal(size=(nm,nm))
    cov=R@R.T/nm
    return cov,modes,pupil


@pytest.mark.parametrize("n_modes",[1,3,5,17,33])
@pytest.mark.parametrize("batch_size",[1,4,16])
def test_vii_batched_matches_legacy_with_uneven_final_batch(n_modes,batch_size):
    cov,modes,pupil=_inputs(n=10,nm=n_modes)
    orig_cov=cov.copy()
    orig_modes=modes.copy()
    orig_pupil=pupil.copy()
    baseline_otf,baseline_dphi=modes2Otf(
        cov,modes,pupil,20,covariance_psd_policy="strict",
    )
    d={}
    new_otf,new_dphi=modes2Otf(
        cov,modes,pupil,20,covariance_psd_policy="strict",
        fft_mode="batched",fft_batch_size=batch_size,
        fft_workers=1,fft_diagnostics=d,
    )
    np.testing.assert_allclose(new_dphi,baseline_dphi,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(new_otf,baseline_otf,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(orig_cov,cov)
    np.testing.assert_array_equal(orig_modes,modes)
    np.testing.assert_array_equal(orig_pupil,pupil)
    assert d["mode"]=="batched"
    assert d["mode_count"]==n_modes
    assert d["batch_count"]==(n_modes+batch_size-1)//batch_size
    assert d["pupil_modes_or_ffts_cached"] is False


@pytest.mark.parametrize("workers",[1,2])
def test_batched_multithreaded_and_rotating_pupil(workers):
    cov,modes,pupil=_inputs(n=14,nm=21)
    for rotated in [pupil,np.rot90(pupil),np.rot90(pupil,2)]:
        baseline=modes2Otf(cov,modes,rotated,28)
        candidate=modes2Otf(
            cov,modes,rotated,28,fft_mode="batched",
            fft_batch_size=7,fft_workers=workers,
        )
        for new,old in zip(candidate,baseline):
            np.testing.assert_allclose(new,old,rtol=1e-12,atol=1e-12)
    assert np.max(np.abs(modes2Otf(cov,modes,pupil,28)[1]
                         -modes2Otf(cov,modes,np.rot90(pupil),28)[1]))>1e-5


def test_ho_functional_boundary_and_legacy_selection():
    cov,modes,pupil=_inputs(n=10,nm=13)
    stats=HOStatistics(
        empirical_covariance_m2=np.eye(13),
        modal_covariance_rad2=cov,
        n_samples=100,method="dm-based",
        reference_wavelength_m=2.055e-6,
    )
    legacy=reconstruct_ho_otf(stats,modes,pupil,20)
    baseline=modes2Otf(cov,modes,pupil,20)
    for a,b in zip(legacy,baseline):
        np.testing.assert_array_equal(a,b)
    batched=reconstruct_ho_otf(
        stats,modes,pupil,20,fft_mode="batched",
        fft_batch_size=5,fft_workers=1,
    )
    for a,b in zip(batched,baseline):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize("mode,batch,workers",[
    ("bad",16,1),("legacy",17,1),("legacy",16,2),
    ("batched",0,1),("batched",-1,1),
    ("batched",True,1),("batched",16,0),
    ("batched",16,-1),("batched",16,False),
])
def test_invalid_fft_options_are_explicit(mode,batch,workers):
    with pytest.raises(ValueError,match="HO FFT"):
        validate_fft_options(mode,batch,workers)


def test_batched_rejects_uij_without_silent_legacy_fallback():
    cov,modes,pupil=_inputs(n=8,nm=3)
    with pytest.raises(ValueError,match="only supports the Vii"):
        modes2Otf(cov,modes,pupil,16,basis="Uij",fft_mode="batched")


def test_existing_strict_negative_covariance_failure_is_unchanged():
    cov,modes,pupil=_inputs(n=8,nm=4)
    cov[1,1]=-1.
    for mode in ["legacy","batched"]:
        with pytest.raises(ValueError,match="positive-semidefinite"):
            modes2Otf(cov,modes,pupil,16,fft_mode=mode)


def test_diagnostic_tight_memory_guard():
    # The guard operates before attempting to allocate the massive FFT slab.
    with pytest.raises(ValueError,match="256 MiB"):
        accumulate_vii_batched(
            np.ones((200,200,2)),np.ones(2),
            np.ones((200,200)),batch_size=1000,workers=1,
        )


def test_psfr_ho_adapter_forwards_batch_to_existing_kernel():
    rng=np.random.default_rng(81)
    _,modes,pupil=_inputs(n=8,nm=4)
    com=rng.normal(scale=1.e-8,size=(120,4))
    noise=np.zeros((4,4))
    model=psfR.__new__(psfR)
    model.ao=NS(rtc=NS(holoop={"gain":.8}),tel=NS(pupil=pupil))
    model.freq=NS(wvlRef=2.055e-6,nOtf=16,sampRef=2.)
    model.trs=NS(dm=NS(com=com),rec=NS(res=com),
                 mat=NS(dmIF=modes),wfs=NS(Cn_ao=noise))
    model.ho_covariance_policy="strict"
    model.ho_covariance_diagnostics={}
    model.ho_fft_mode="legacy"
    expected=model.residual_phase_structure_function().copy()
    model.ho_fft_mode="batched"
    model.ho_fft_batch_size=2
    model.ho_fft_workers=1
    model.ho_fft_diagnostics={}
    actual=model.residual_phase_structure_function()
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    assert model.ho_fft_diagnostics["batch_count"]==2
    assert model.ho_fft_diagnostics["mode"]=="batched"
