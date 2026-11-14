"""Real one-sided Vii FFT: preserve pupil physics and science outputs.

The full complex factorization is the relevant physics reference. These
are small synthetic acceptance cases only; real NIRC2 PSF parity is
required independently before adopting the accelerated path.
"""
from __future__ import annotations
from types import SimpleNamespace as NS
import numpy as np
import pytest

from p3.aoSystem import FourierUtils
from p3.psfr.hoFactorized import accumulate_vii_factorized_real
from p3.psfr.psfrUtils import modes2Otf
from p3.psfr.ho import HOStatistics, reconstruct_ho_otf
from p3.psfr.psfR import psfR


def case(n=12,nm=9,seed=124):
    rng=np.random.default_rng(seed+n+nm)
    yy,xx=np.indices((n,n))
    aperture=(
        ((xx-.48*n)/(.44*n))**2
        +((yy-.53*n)/(.41*n))**2 < 1
    )
    aperture[(xx < n*.28)&(yy > n*.52)] = False
    pupil=aperture.astype(float)
    modes=rng.normal(scale=.055,size=(n*n,nm))
    factor=rng.normal(size=(nm,nm))
    covariance=(factor@factor.T)/nm
    return covariance,modes,pupil


@pytest.mark.parametrize("n,n_modes",[(8,1),(9,3),(12,7),(13,9),(16,17)])
def test_real_fft_matches_full_complex_vii_for_even_odd_symmetric_and_rotated_pupil(
    n,n_modes,
):
    covariance,modes,pupil=case(n,n_modes)
    for rotated in [pupil,np.rot90(pupil),np.rot90(pupil,2)]:
        full=modes2Otf(
            covariance,modes,rotated,n*2,
            vii_method="factorized",
        )
        diag={}
        real=modes2Otf(
            covariance,modes,rotated,n*2,
            vii_method="factorized_real",vii_diagnostics=diag,
        )
        np.testing.assert_allclose(real[0],full[0],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(real[1],full[1],rtol=1e-12,atol=1e-12)
        assert diag["fft_backend"]=="numpy"
        assert diag["method"]=="factorized_real"
        padded=FourierUtils.enlargeSupport(rotated,2)
        assert diag["half_spectrum_shape"]==[
            padded.shape[0],padded.shape[1]//2+1,
        ]
        assert diag["mode_count"]==n_modes
        assert diag["workers"]==1
        assert diag["pupil_or_ho_otf_cached"] is False


def test_one_sided_hermitian_cross_spectrum_matches_full_real_ifft():
    rng=np.random.default_rng(907)
    for ny,nx in ((15,16),(16,15),(20,20),(21,19)):
        pupil=rng.normal(size=(ny,nx))
        modes=rng.normal(size=(ny,nx,5))
        eig=rng.uniform(.01,1.,size=5)
        Fp=np.fft.fft2(pupil)
        half=accumulate_vii_factorized_real(
            modes,eig,np.conjugate(Fp)
        )
        # Independent full-complex spectral expression; loop order kept
        # the same to isolate transform symmetry.
        A=np.zeros((ny,nx))
        P=np.zeros((ny,nx))
        for k in range(5):
            A=A+eig[k]*modes[:,:,k]**2
            S=np.fft.fft2(modes[:,:,k])
            P=P+eig[k]*np.abs(S)**2
        full=np.real(np.fft.ifft2(
            2*(np.real(np.fft.fft2(A)*np.conjugate(Fp))-P)
        ))
        np.testing.assert_allclose(half,full,rtol=1e-12,atol=1e-10)


def test_rotating_pupil_is_current_observation_not_reused():
    covariance,modes,pupil=case(n=12,nm=6)
    first=modes2Otf(covariance,modes,pupil,24,vii_method="factorized_real")
    rotated=modes2Otf(
        covariance,modes,np.rot90(pupil),24,vii_method="factorized_real"
    )
    assert not np.array_equal(first[1],rotated[1])
    expected=modes2Otf(covariance,modes,np.rot90(pupil),24,vii_method="factorized")
    for one,two in zip(rotated,expected):
        np.testing.assert_allclose(one,two,rtol=1e-12,atol=1e-12)


def test_unmodified_psd_guard_zero_covariance_and_uij_rejection():
    cov,modes,pupil=case(n=10,nm=5)
    cov[0,0]=-100
    for method in ("factorized","factorized_real"):
        with pytest.raises(ValueError,match="positive-semidefinite"):
            modes2Otf(cov,modes,pupil,20,vii_method=method)
    zero=np.zeros((5,5))
    for method in ("legacy","factorized","factorized_real"):
        _,sf=modes2Otf(zero,modes,pupil,20,vii_method=method)
        np.testing.assert_array_equal(sf,np.zeros((20,20)))
    with pytest.raises(ValueError,match="basis Vii"):
        modes2Otf(np.eye(5),modes,pupil,20,basis="Uij",
                  vii_method="factorized_real")


def test_real_ho_adapter_passes_diagnostics_and_matches_complex():
    cov,modes,pupil=case(n=10,nm=5)
    stats=HOStatistics(
        empirical_covariance_m2=np.eye(5),
        modal_covariance_rad2=cov,
        n_samples=200,
        method="dm-based",
        reference_wavelength_m=2.055e-6,
    )
    complex_result=reconstruct_ho_otf(
        stats,modes,pupil,20,vii_method="factorized",
    )
    d={}
    real_result=reconstruct_ho_otf(
        stats,modes,pupil,20,vii_method="factorized_real",
        vii_diagnostics=d,
    )
    for a,b in zip(complex_result,real_result):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    assert d["method"]=="factorized_real"


def test_psfr_residual_sf_pipeline_real_vs_complex_keeps_current_pupil():
    _,modes,pupil=case(n=8,nm=5)
    rng=np.random.default_rng(611)
    commands=rng.normal(scale=1e-8,size=(100,5))
    model=psfR.__new__(psfR)
    model.ao=NS(rtc=NS(holoop={"gain":.8}),tel=NS(pupil=pupil))
    model.freq=NS(wvlRef=2.055e-6,nOtf=16,sampRef=2.)
    model.trs=NS(
        dm=NS(com=commands),
        rec=NS(res=commands),
        mat=NS(dmIF=modes),
        wfs=NS(Cn_ao=np.zeros((5,5))),
    )
    model.ho_covariance_policy="strict"
    model.ho_covariance_diagnostics={}
    model.ho_vii_diagnostics={}
    model.ho_vii_method="factorized"
    old=model.residual_phase_structure_function()
    model.ho_vii_method="factorized_real"
    new=model.residual_phase_structure_function()
    np.testing.assert_allclose(new,old,rtol=1e-12,atol=1e-12)
    assert model.ho_vii_diagnostics["method"]=="factorized_real"


def test_nonfinite_and_mismatched_kernel_inputs_fail_explicitly():
    modes=np.ones((8,8,2))
    weights=np.ones(2)
    fp=np.ones((8,8),dtype=complex)
    with pytest.raises(ValueError,match="dimensions"):
        accumulate_vii_factorized_real(modes,np.ones(3),fp)
    modes[0,0,0]=np.nan
    with pytest.raises(ValueError,match="finite"):
        accumulate_vii_factorized_real(modes,weights,fp)
