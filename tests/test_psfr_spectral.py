"""Synthetic tests for measured NIRC2 profiles and spectral quadrature."""
import numpy as np
import pytest

from p3.psfr.spectral import (
    FilterProfile, read_kai_filter, photon_quadrature,
    integrate_monochromatic_psfs,
)


def profile(w,t):
    return FilterProfile(np.asarray(w,float),np.asarray(t,float),"synthetic",{})


def test_kai_duplicate_negative_and_source_units(tmp_path):
    path=tmp_path/"Kp.dat"
    path.write_text(
        "#Wavelength Transmission\n"
        "2.1 80\n2.0 -0.1\n2.2 50\n2.1 60\n2.3 0\n"
    )
    with pytest.warns(RuntimeWarning,match="clipped"):
        data=read_kai_filter(path)
    np.testing.assert_array_equal(data.wavelength_um,[2.,2.1,2.2,2.3])
    np.testing.assert_allclose(data.transmission,[0,.7,.5,0])
    assert data.audit["duplicate_wavelength_samples"]==1
    assert data.audit["clipped_negative_samples"]==1
    assert data.audit["complete_system_throughput"] is False
    with pytest.raises(ValueError,match="duplicate"):
        read_kai_filter(path,merge_duplicates=False)
    with pytest.raises(ValueError,match="negative"):
        read_kai_filter(path,clip_negative=False)


def test_flat_photon_filter_has_uniform_integrated_weights():
    q=photon_quadrature(profile([1,1.5,2],[1,1,1]),n_bins=4)
    np.testing.assert_allclose(q.wavelength_m,
        [1.125e-6,1.375e-6,1.625e-6,1.875e-6])
    np.testing.assert_allclose(q.weights,[.25]*4,atol=1e-14)
    assert q.total_photon_integral==pytest.approx(1)
    assert q.effective_wavelength_m==pytest.approx(1.5e-6)


def test_photon_sed_atmosphere_and_bin_width_applied_once():
    p=profile([1,2,3],[1,1,1])
    q=photon_quadrature(
        p,n_bins=2,photon_sed=[1,2,3],atmosphere=[.5,.5,.5]
    )
    assert q.total_photon_integral==pytest.approx(2)
    np.testing.assert_allclose(q.weights,[.375,.625],rtol=1e-14)


def test_integrate_two_monochromatic_psfs_with_unit_flux():
    cube=np.zeros((4,4,2))
    cube[1,1,0]=1
    cube[2,2,1]=1
    image=integrate_monochromatic_psfs(cube,[.3,.7])
    assert image.sum()==pytest.approx(1)
    assert image[1,1]==pytest.approx(.3)
    assert image[2,2]==pytest.approx(.7)
    two_sources=np.stack([cube,np.roll(cube,1,axis=0)],axis=2)
    result=integrate_monochromatic_psfs(two_sources,[.3,.7])
    assert result.shape==(4,4,2)
    np.testing.assert_allclose(result.sum(axis=(0,1)),[1,1])


def test_refuse_bad_sed_and_hidden_double_weighting():
    p=profile([1,2,3],[1,1,1])
    with pytest.raises(ValueError,match="n_bins"):
        photon_quadrature(p,n_bins=0)
    with pytest.raises(ValueError,match="photon_sed"):
        photon_quadrature(p,n_bins=2,photon_sed=[1,-1,1])
    with pytest.raises(ValueError,match="zero"):
        photon_quadrature(p,n_bins=2,photon_sed=[0,0,0])
    with pytest.raises(ValueError,match="atmosphere"):
        photon_quadrature(p,n_bins=2,atmosphere=[1,1.1,1])
    cube=np.ones((4,4,2))
    with pytest.raises(ValueError,match="already have unit flux"):
        integrate_monochromatic_psfs(cube,[.5,.5])
    cube=np.zeros((4,4,2))
    cube[1,1,:]=1
    with pytest.raises(ValueError,match="sum to one"):
        integrate_monochromatic_psfs(cube,[.4,.4])
    with pytest.raises(ValueError,match="match wavelength"):
        integrate_monochromatic_psfs(cube,[1.])
