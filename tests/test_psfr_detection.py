"""Detection tests: PSF stars vs hot pixels, noise, masks and count limit."""
import numpy as np
import pytest

from p3.psfr.detection import detect_stellar_sources
from p3.psfr.stellar import render_stellar_scene
from p3.psfr.template import template_from_psfr


def template(n=80, width=2.8):
    y,x=np.indices((n,n))
    core=np.exp(-((y-n//2)**2+(x-n//2)**2)/(2*width**2))
    return template_from_psfr(core,pixel_scale_mas=9.942)


def test_two_sources_are_detected_despite_brighter_hot_pixels():
    t=template()
    rng=np.random.default_rng(14)
    actual=np.asarray([[18.3,27.5],[58.3,52.4]])
    image=render_stellar_scene(
        t,positions_xy_px=actual,fluxes=[4500,2400],background=20.
    )
    image+=rng.normal(0,1,size=image.shape)
    image[15,62]+=8000       # isolated detector artifact brighter than stars
    image[67,12]+=1200
    original=image.copy()
    result=detect_stellar_sources(
        image,t,max_sources=5,core_radius_px=9,min_snr=6,
        min_halo_snr=4,min_separation_px=5,
    )
    assert len(result.accepted)==2
    assert len(result.positions_xy_px)==2
    np.testing.assert_allclose(result.positions_xy_px,actual,atol=.35)
    assert {x.seed_xy_px for x in result.rejected}.issuperset({(62,15),(12,67)})
    assert not result.fit_valid_pixels[15,62]
    assert not result.fit_valid_pixels[67,12]
    assert result.fit_valid_pixels[27,18]
    assert result.noise_source.startswith("empirical")
    assert result.stopped_reason=="below_matched_snr"
    np.testing.assert_array_equal(image,original)


def test_single_hot_pixel_with_no_stars_is_not_detected():
    t=template()
    image=np.full((80,80),10.)
    rng=np.random.default_rng(4)
    image+=rng.normal(0,1.,size=image.shape)
    image[39,43]+=10000
    result=detect_stellar_sources(
        image,t,max_sources=3,core_radius_px=9,
    )
    assert len(result.accepted)==0
    assert any(x.classification=="hot_pixel_like" for x in result.rejected)
    assert not result.fit_valid_pixels[39,43]


def test_sigma_mask_and_saturation_veto_do_not_invent_a_source():
    t=template()
    rng=np.random.default_rng(21)
    image=render_stellar_scene(
        t,positions_xy_px=[[34.6,31.3]],fluxes=[2600],background=8.
    )
    image+=rng.normal(0,2,size=image.shape)
    image[5,4]=np.nan
    image[23,55]=4000
    mask=np.ones(image.shape,dtype=bool)
    mask[23,55]=False
    sigma=np.full(image.shape,2.)
    result=detect_stellar_sources(
        image,t,sigma=sigma,valid_mask=mask,max_sources=3,
        min_snr=6,min_halo_snr=4,
    )
    assert len(result.accepted)==1
    np.testing.assert_allclose(result.positions_xy_px,[[34.6,31.3]],atol=.4)
    assert result.noise_source=="user_supplied_sigma"
    assert result.noise_sigma is None
    assert not result.fit_valid_pixels[5,4]
    assert not result.fit_valid_pixels[23,55]


def test_maximum_source_count_is_bounded():
    t=template(80)
    image=render_stellar_scene(
        t,positions_xy_px=[[20,20],[60,60]],
        fluxes=[10000,8000],background=1.
    )
    result=detect_stellar_sources(
        image,t,max_sources=1,core_radius_px=8
    )
    assert len(result.accepted)==1
    assert result.stopped_reason=="max_sources"


def test_delta_like_psf_is_declared_unresolvable():
    n=32
    image=np.zeros((n,n))
    image[n//2,n//2]=1.
    t=template_from_psfr(image,pixel_scale_mas=9.942)
    with pytest.raises(ValueError,match="unresolved"):
        detect_stellar_sources(
            np.ones((n,n)),t,core_radius_px=3,
        )


def test_validation_is_strict():
    t=template()
    with pytest.raises(ValueError,match="identical"):
        detect_stellar_sources(np.ones((16,16)),t)
    with pytest.raises(ValueError,match="max_sources"):
        detect_stellar_sources(np.ones((80,80)),t,max_sources=0)
    with pytest.raises(ValueError,match="max_examined"):
        detect_stellar_sources(np.ones((80,80)),t,max_examined=2)
    with pytest.raises(ValueError,match="core_radius_px"):
        detect_stellar_sources(np.ones((80,80)),t,core_radius_px=0)
    with pytest.raises(ValueError,match="valid_mask"):
        detect_stellar_sources(np.ones((80,80)),t,valid_mask=np.ones((80,80)))
    with pytest.raises(ValueError,match="sigma"):
        detect_stellar_sources(np.ones((80,80)),t,sigma=np.ones((12,12)))
    with pytest.raises(ValueError,match="Too few"):
        mask=np.zeros((80,80),dtype=bool)
        mask[2,2]=True
        detect_stellar_sources(np.ones((80,80)),t,valid_mask=mask)
