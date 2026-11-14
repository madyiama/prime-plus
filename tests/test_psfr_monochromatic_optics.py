"""Analytic and numerical contracts of standalone chromatic optical kernel.

No Keck FITS/SAV and no claim that an old P3 nOtf SF is already sampled
on these physical pupil lags. The Keck V1 renderer is NOT changed here.
"""
import numpy as np
import pytest

from p3.psfr.chromatic import (
    prepare_pupil_lag_grid, detector_monochromatic_psf,
)
from p3.psfr.spectral import integrate_monochromatic_psfs


def pupil(n=96):
    yy,xx=np.indices((n,n))
    return (((xx-(n-1)/2)**2+(yy-(n-1)/2)**2)<(n*.47)**2).astype(float)


def render(geo, lam, **kw):
    return detector_monochromatic_psf(
        geo,wavelength_m=lam,n_pixels=128,
        pixel_scale_rad=9.942e-3/206265,
        **kw,
    )


def test_diffraction_peak_and_width_change_with_wavelength():
    geo=prepare_pupil_lag_grid(pupil(),diameter_m=10.5)
    # Longer lambda broadens the diffraction core at fixed NIRC2 pixel scale.
    blue=render(geo,2.0e-6)
    red=render(geo,2.4e-6)
    assert red.psf.max()<blue.psf.max()
    assert red.psf[64,64]<blue.psf[64,64]
    assert blue.psf.shape==red.psf.shape==(128,128)
    np.testing.assert_allclose([blue.psf.sum(),red.psf.sum()],[1.,1.],atol=1e-12)
    np.testing.assert_allclose(
        [blue.otf[64,64],red.otf[64,64]],[1.,1.],atol=1e-12,
    )
    # Physical mask is unchanged by wavelength-specific propagation.
    np.testing.assert_array_equal(geo.amplitude,pupil())


def test_identical_wavelength_and_mono_band_are_same_psf():
    geo=prepare_pupil_lag_grid(pupil(),diameter_m=10.5)
    a=render(geo,2.2e-6)
    b=render(geo,2.2e-6)
    np.testing.assert_array_equal(a.psf,b.psf)
    poly=integrate_monochromatic_psfs(a.psf[:,:,None],[1.])
    np.testing.assert_array_equal(poly,a.psf)


def test_gaussian_phase_sf_scales_by_reference_wavelength_squared():
    geo=prepare_pupil_lag_grid(pupil(),diameter_m=10.5)
    n=geo.amplitude.shape[0]
    y,x=np.mgrid[-(n-1):(n-1)+1,-(n-1):(n-1)+1]
    rho2=(x*x+y*y)*geo.lag_step_m**2
    # Gaussian covariance C(rho)=sigma^2 exp(-rho^2/(2l^2))
    # has Dphi(rho)=2*(C(0)-C(rho)), which is nonnegative.
    dphi=2*0.25*(1-np.exp(-rho2/(2*2.0**2)))
    base=render(geo,2.2e-6)
    with_phase=render(
        geo,2.2e-6,phase_sf_reference_rad2=dphi,
        reference_wavelength_m=1.65e-6,
    )
    assert with_phase.psf.max()<base.psf.max()
    # Sampled OTF is multiplied by exp(-1/2 * Dphi * lambda_ref^2/lambda^2)
    # on the *same detector angular-frequency grid*.
    oy,ox=64,74
    measured=with_phase.otf[oy,ox]/base.otf[oy,ox]
    df=9.942e-3/206265
    spatial=10/(128*df)
    baseline=2.2e-6*spatial
    expected_dphi=2*.25*(1-np.exp(-baseline**2/(2*2.0**2)))
    np.testing.assert_allclose(
        measured,np.exp(-.5*(1.65/2.2)**2*expected_dphi),
        rtol=5e-3,atol=1e-4,
    )


def test_real_static_opd_must_match_pupil_grid():
    geo=prepare_pupil_lag_grid(pupil(),diameter_m=10.5)
    with pytest.raises(ValueError,match="Static OPD"):
        render(geo,2.2e-6,static_opd_m=np.zeros((4,4)))
    with pytest.raises(ValueError,match="zero lag"):
        render(
            geo,2.2e-6,
            phase_sf_reference_rad2=np.ones(geo.otf_lag.shape),
            reference_wavelength_m=2.2e-6,
        )


def test_bad_pupil_and_unphysical_phase_sf_rejected():
    with pytest.raises(ValueError,match="Physical pupil"):
        prepare_pupil_lag_grid(np.ones((8,6)),diameter_m=10.5)
    with pytest.raises(ValueError,match="Physical pupil"):
        prepare_pupil_lag_grid(-np.ones((8,8)),diameter_m=10.5)
    geo=prepare_pupil_lag_grid(pupil(),diameter_m=10.5)
    with pytest.raises(ValueError,match="Reference wavelength"):
        sf=np.zeros(geo.otf_lag.shape)
        render(geo,2.2e-6,phase_sf_reference_rad2=sf)
    with pytest.raises(ValueError,match="phase SF"):
        sf=np.zeros(geo.otf_lag.shape)
        sf[0,0]=-1.
        render(
            geo,2.2e-6,phase_sf_reference_rad2=sf,
            reference_wavelength_m=2.2e-6
        )


def test_wavelength_dependent_psfs_combine_with_photon_weights():
    geo=prepare_pupil_lag_grid(pupil(),diameter_m=10.5)
    a=render(geo,2.0e-6).psf
    b=render(geo,2.4e-6).psf
    mixed=integrate_monochromatic_psfs(np.stack([a,b],axis=-1),[.35,.65])
    np.testing.assert_allclose(mixed,.35*a+.65*b,rtol=0,atol=1e-14)
    assert mixed.sum()==pytest.approx(1.)
    assert b.max()<mixed.max()<a.max()
