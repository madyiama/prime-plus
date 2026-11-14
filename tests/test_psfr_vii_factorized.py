"""Scientific Vii factorization exploration, strict acceptance stays at 1e-12."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from p3.psfr.psfrUtils import modes2Otf
from p3.psfr.ho import HOStatistics, reconstruct_ho_otf
from p3.psfr.psfR import psfR
from scripts.compare_keck_vii_factorized import _compare


def scene(*,n=10,n_modes=5,seed=19):
    rng=np.random.default_rng(seed)
    y,x=np.indices((n,n))
    pupil=(((x-.47*n)/(n*.38))**2+((y-.53*n)/(n*.36))**2<1).astype(float)
    pupil[(y>n*.65)&(x<n*.43)]=0
    modes=rng.normal(scale=.1,size=(n*n,n_modes))
    cov=rng.normal(size=(n_modes,n_modes))
    cov=cov@cov.T/n_modes
    return cov,modes,pupil


@pytest.mark.parametrize("n_modes",[1,3,5,7])
def test_factorized_form_matches_legacy_for_small_psd_scene(n_modes):
    cov,modes,pupil=scene(n_modes=n_modes)
    old_otf,old_sf=modes2Otf(cov,modes,pupil,20)
    diag={}
    new_otf,new_sf=modes2Otf(
        cov,modes,pupil,20,vii_method="factorized",
        vii_diagnostics=diag,
    )
    np.testing.assert_allclose(new_sf,old_sf,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(new_otf,old_otf,rtol=1e-12,atol=1e-12)
    assert diag["fft2_count_nominal"]==n_modes+1
    assert diag["square_mode_fft2_eliminated"]==n_modes-1
    assert diag["numerical_reordering"] is True


def test_default_historical_vii_remains_bitwise_the_same():
    cov,modes,pupil=scene(n_modes=6)
    old=modes2Otf(cov,modes,pupil,20)
    diag={}
    new=modes2Otf(cov,modes,pupil,20,vii_method="legacy",vii_diagnostics=diag)
    for left,right in zip(old,new):
        np.testing.assert_array_equal(left,right)
    assert diag["method"]=="legacy"
    assert diag["numerical_reordering"] is False


def test_rotating_keck_style_asymmetric_pupil_changes_ho_physics():
    cov,modes,pupil=scene(n=12,n_modes=5)
    first=modes2Otf(cov,modes,pupil,24)[1]
    rotated_pupil=np.rot90(pupil)
    second=modes2Otf(cov,modes,rotated_pupil,24)[1]
    assert not np.array_equal(first,second)
    for rotation in [pupil,rotated_pupil]:
        old=modes2Otf(cov,modes,rotation,24)
        new=modes2Otf(cov,modes,rotation,24,vii_method="factorized")
        for a,b in zip(old,new):
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)


def test_negative_covariance_policy_and_uij_contract_unchanged():
    cov,modes,pupil=scene(n=10,n_modes=4)
    cov[0,0]=-5.
    for method in ["legacy","factorized"]:
        with pytest.raises(ValueError,match="positive-semidefinite"):
            modes2Otf(cov,modes,pupil,20,vii_method=method)
    with pytest.raises(ValueError,match="only available for basis Vii"):
        modes2Otf(np.eye(4),modes,pupil,20,basis="Uij",
                  vii_method="factorized")
    with pytest.raises(ValueError,match="Unknown Vii"):
        modes2Otf(np.eye(4),modes,pupil,20,vii_method="none")


def test_pure_ho_adapter_forwards_factorization():
    cov,modes,pupil=scene(n=10,n_modes=4)
    stats=HOStatistics(
        empirical_covariance_m2=np.eye(4),
        modal_covariance_rad2=cov,n_samples=150,
        method="dm-based",reference_wavelength_m=2.055e-6,
    )
    reference=reconstruct_ho_otf(stats,modes,pupil,20)
    diagnostics={}
    candidate=reconstruct_ho_otf(
        stats,modes,pupil,20,vii_method="factorized",
        vii_diagnostics=diagnostics,
    )
    for a,b in zip(reference,candidate):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    assert diagnostics["method"]=="factorized"


def test_psfr_adapter_factorized_uses_current_pupil():
    cov,modes,pupil=scene(n=10,n_modes=4)
    rng=np.random.default_rng(42)
    commands=rng.normal(scale=1e-8,size=(80,4))
    obj=psfR.__new__(psfR)
    obj.ao=NS(rtc=NS(holoop={"gain":.9}),tel=NS(pupil=pupil))
    obj.freq=NS(wvlRef=2.055e-6,nOtf=20,sampRef=2.)
    obj.trs=NS(dm=NS(com=commands),rec=NS(res=commands),
               mat=NS(dmIF=modes),wfs=NS(Cn_ao=np.zeros((4,4))))
    obj.ho_covariance_policy="strict"
    obj.ho_covariance_diagnostics={}
    obj.ho_vii_diagnostics={}
    obj.ho_vii_method="legacy"
    reference=obj.residual_phase_structure_function()
    obj.ho_vii_method="factorized"
    trial=obj.residual_phase_structure_function()
    np.testing.assert_allclose(trial,reference,rtol=1e-12,atol=1e-12)


def test_comparator_never_accepts_only_final_science_psf_parity():
    full=np.ones((8,8,3),dtype=float)
    strict=_compare(full,full.copy(),rtol=1e-12,atol=1e-12)
    assert strict["parity_passed"]
    ho=np.ones((8,8),dtype=float)
    bad=ho.copy()
    bad[3,2]+=1e-6
    assert not _compare(ho,bad,rtol=1e-12,atol=1e-12)["parity_passed"]
