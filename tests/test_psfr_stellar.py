"""Regression and recovery tests for flux/position/sky fitting."""
import numpy as np
import pytest
from p3.psfr.template import template_from_psfr
from p3.psfr.stellar import fit_stellar_scene, render_stellar_scene


def template(n=40):
    yy,xx=np.indices((n,n))
    psf=np.exp(-((xx-n//2)**2+(yy-n//2)**2)/(2*2.3**2))
    return template_from_psfr(psf,pixel_scale_mas=9.942)


def test_single_source_recovers_flux_subpixel_position_and_background():
    p=template()
    exact=[[19.35,20.64]]
    observation=render_stellar_scene(
        p,positions_xy_px=exact,fluxes=[13200.],background=13.4
    )
    old=observation.copy()
    fit=fit_stellar_scene(
        observation,p,initial_positions_xy_px=[[19.8,20.1]],
        initial_background=10.,max_position_offset_px=3.
    )
    assert fit.converged,fit.message
    np.testing.assert_allclose(fit.positions_xy_px,exact,atol=1e-3)
    np.testing.assert_allclose(fit.fluxes,[13200.],rtol=1e-4)
    assert fit.background==pytest.approx(13.4,abs=1e-3)
    assert fit.rms<1e-5
    np.testing.assert_array_equal(observation,old)
    np.testing.assert_allclose(fit.model,observation,atol=1e-4)


def test_two_stars_simultaneous_fit_with_nonnegative_flux():
    p=template(48)
    sources=[[21.7,23.35],[31.4,29.8]]
    obs=render_stellar_scene(p,positions_xy_px=sources,
                             fluxes=[5000.,2700.],background=5.2)
    fit=fit_stellar_scene(
        obs,p,initial_positions_xy_px=[[22,23],[31,30]],
        initial_fluxes=[4500.,3000.],max_position_offset_px=2.
    )
    assert fit.converged,fit.message
    np.testing.assert_allclose(fit.positions_xy_px,sources,atol=2e-3)
    np.testing.assert_allclose(fit.fluxes,[5000.,2700.],rtol=2e-4)
    assert fit.background==pytest.approx(5.2,abs=.01)


def test_invalid_masked_pixel_and_explicit_sigma_only_weight_valid():
    p=template()
    observed=render_stellar_scene(
        p,positions_xy_px=[[20.4,19.7]],
        fluxes=[1000.],background=2.
    )
    observed[1,1]=np.nan
    observed[30,30]=1e9   # saturated / cosmic ray explicitly masked
    good=np.ones(observed.shape,dtype=bool)
    good[30,30]=False
    sigma=np.full(observed.shape,3.)
    sigma[1,1]=np.nan
    fit=fit_stellar_scene(
        observed,p,initial_positions_xy_px=[[20,20]],
        valid_mask=good,sigma=sigma,max_position_offset_px=3.
    )
    assert fit.converged
    assert not fit.valid_pixels[1,1]
    assert not fit.valid_pixels[30,30]
    np.testing.assert_allclose(fit.positions_xy_px,[[20.4,19.7]],atol=5e-3)
    assert fit.reduced_chi2 is not None
    assert fit.weighted_chi2 is not None
    assert fit.rms < 1e-3


def test_no_implicit_noise_model_and_zero_background_allowed():
    p=template()
    observed=render_stellar_scene(
        p,positions_xy_px=[[20.,20.]],fluxes=[500.],background=0.
    )
    fit=fit_stellar_scene(observed,p)
    assert fit.converged
    assert fit.weighted_chi2 is None
    assert fit.reduced_chi2 is None
    assert fit.fluxes[0]>=0


def test_validation_and_geometry_contract():
    p=template()
    im=np.ones_like(p.image)
    with pytest.raises(ValueError,match="identical"):
        fit_stellar_scene(im[:12],p)
    with pytest.raises(ValueError,match="valid_mask"):
        fit_stellar_scene(im,p,valid_mask=np.ones_like(im))
    with pytest.raises(ValueError,match="sigma"):
        fit_stellar_scene(im,p,sigma=np.ones((3,3)))
    with pytest.raises(ValueError,match="radius_px"):
        fit_stellar_scene(im,p,radius_px=2.)
    with pytest.raises(ValueError,match="Initial centres"):
        fit_stellar_scene(im,p,initial_positions_xy_px=[[-1,20]])
    with pytest.raises(ValueError,match="max_position_offset_px"):
        fit_stellar_scene(im,p,max_position_offset_px=0.)
    with pytest.raises(ValueError,match="nonnegative"):
        render_stellar_scene(
            p,positions_xy_px=[[20,20]],fluxes=[-1]
        )
    with pytest.raises(ValueError,match="inside"):
        render_stellar_scene(
            p,positions_xy_px=[[50,20]],fluxes=[1]
        )
    with pytest.raises(ValueError,match="Not enough valid"):
        fit_stellar_scene(
            im,p,valid_mask=(np.indices(im.shape)[0]==0)&
            (np.indices(im.shape)[1]<2)
        )


def test_crowded_field_roi_restricts_pixels_to_stars():
    p=template(64)
    obs=render_stellar_scene(
        p,positions_xy_px=[[23.2,21.7]],fluxes=[1000.],background=8.
    )
    fit=fit_stellar_scene(
        obs,p,initial_positions_xy_px=[[23,22]],radius_px=11.
    )
    assert fit.converged
    assert fit.valid_pixels.sum()<obs.size
    np.testing.assert_allclose(fit.positions_xy_px,[[23.2,21.7]],atol=2e-2)
