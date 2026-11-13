"""Synthetic frozen Keck SF basis and constrained AO fit regression tests."""
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest

from p3.psfr.ao_basis import (
    FrozenAOBasis,frozen_basis_from_model,render_ao_psf,
    save_ao_basis,load_ao_basis,validate_ao_basis_parity,SCHEMA,
)
from p3.psfr.ao_fit import fit_hybrid_ao
from p3.psfr.stellar import render_stellar_scene
from p3.psfr.template import template_from_psfr


def make_basis(n=64):
    c=n//2
    yy,xx=np.indices((n,n))
    r2=(xx-c)**2+(yy-c)**2
    # Build an explicitly positive pixel-grid intensity kernel first,
    # then convert it into an exact centred detector OTF.
    psf=np.exp(-r2/(2*2.5**2))+1e-4
    otf=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    otf=otf/otf[c,c]
    sf=np.maximum(0.,1-np.exp(-r2/(2*9**2)))
    kwargs=dict(
        dphi_ao=.5*sf,dphi_tt=.22*sf,
        dphi_fit=.055*sf,dphi_alias=.025*sf,
        dphi_ani=np.empty((0,)),
        layer_weights=np.array([1.]),
        static_pixel_otf=otf,diffraction_otf=np.ones((n,n)),
        reference_psf=np.ones((n//2,n//2))/(n//2)**2,
        r0_m=.3,atmosphere_wavelength_m=500e-9,
        reference_wavelength_m=2.055e-6,
        k_ref=2,n_pixels=n//2,camera_fov=n//2,
        is_anisoplanatic=False,
        provenance={"schema":SCHEMA},
    )
    provisional=FrozenAOBasis(**kwargs)
    correct=render_ao_psf(provisional)
    return replace(provisional,reference_psf=correct)


def test_nominal_ao_forward_parity_and_reusable_archive(tmp_path):
    basis=make_basis()
    parity=validate_ao_basis_parity(basis)
    assert parity["nominal_parity_passed"]
    assert parity["max_absolute_difference"]==0.
    output=save_ao_basis(basis,tmp_path/"ao_fit_basis.npz")
    assert output.exists()
    restored=load_ao_basis(output)
    np.testing.assert_array_equal(
        restored.reference_psf,basis.reference_psf
    )
    np.testing.assert_allclose(
        render_ao_psf(restored),basis.reference_psf,atol=1e-12
    )
    assert not np.allclose(
        render_ao_psf(restored,gho=1.15),restored.reference_psf,
        atol=1e-8,
    )


def test_frozen_basis_from_p3_like_model_is_parity_exact():
    b=make_basis()
    n=b.dphi_ao.shape[0]
    fake=NS(
        ao=NS(
            src=NS(nSrc=1),
            cam=NS(fovInPix=b.camera_fov),
            atm=NS(r0=b.r0_m,wvl=b.atmosphere_wavelength_m,
                   weights=b.layer_weights),
        ),
        freq=NS(
            nOtf=n,kRef_=b.k_ref,wvlRef=b.reference_wavelength_m,
            isAniso=False,otfNCPA=b.static_pixel_otf,
            otfDL=b.diffraction_otf,
        ),
        otfPixel=1.,
        dphi_ao=b.dphi_ao,dphi_tt=b.dphi_tt,
        dphi_fit=b.dphi_fit,dphi_alias=b.dphi_alias,
    )
    reread=frozen_basis_from_model(fake,b.reference_psf)
    assert validate_ao_basis_parity(reread)["nominal_parity_passed"]


def test_ao_fit_reduces_residual_without_fitting_tt():
    b=make_basis()
    template=template_from_psfr(b.reference_psf,pixel_scale_mas=9.942)
    true_r0=.26
    actual_psf=render_ao_psf(b,r0_m=true_r0)
    star=template_from_psfr(actual_psf,pixel_scale_mas=9.942)
    observed=render_stellar_scene(
        star,positions_xy_px=[[15.5,16.3]],
        fluxes=[16000.],background=12.,
    )
    result=fit_hybrid_ao(
        observed,b,template,
        initial_positions_xy_px=[[15.5,16.3]],
        initial_fluxes=[15500.],initial_background=11.,
        sigma=np.full(observed.shape,.2),
        active_parameters=("r0",),n_alternations=2,
        max_nfev=35,stellar_max_nfev=150,
    )
    assert result.nominal_parity["nominal_parity_passed"]
    assert result.ao_parameters["gtt"]==1.
    assert result.ao_parameters["gho"]==1.
    assert result.ao_parameters["r0"]<b.r0_m
    assert abs(result.ao_parameters["r0"]-true_r0)<.025
    assert result.final_rms<result.initial_rms
    assert result.noise_source=="user_supplied_sigma"


def test_nonidentifiable_tt_requires_opt_in_and_parity_precedes_fit():
    b=make_basis()
    t=template_from_psfr(b.reference_psf,pixel_scale_mas=9.942)
    observed=np.ones_like(b.reference_psf)
    with pytest.raises(ValueError,match="gtt requires explicit"):
        fit_hybrid_ao(
            observed,b,t,initial_positions_xy_px=[[15,15]],
            active_parameters=("gtt",),
        )
    altered=replace(b,reference_psf=np.roll(b.reference_psf,2,axis=0))
    with pytest.raises(ValueError,match="numerical parity"):
        fit_hybrid_ao(
            observed,altered,t,initial_positions_xy_px=[[15,15]],
        )
    with pytest.raises(ValueError,match="distinct"):
        fit_hybrid_ao(
            observed,b,t,initial_positions_xy_px=[[15,15]],
            active_parameters=("r0","r0"),
        )


def test_invalid_shape_fails_explicitly():
    b=make_basis()
    broken=replace(b,static_pixel_otf=np.ones((12,12)))
    with pytest.raises(ValueError,match="finite"):
        render_ao_psf(broken)
    with pytest.raises(ValueError,match="nonnegative"):
        render_ao_psf(b,gho=-1.)



def test_anisoplanatism_r0_reweighting_preserves_nominal_and_changes_psf():
    b=make_basis()
    n=b.dphi_ao.shape[0]
    yy,xx=np.indices((n,n))
    r2=(xx-n//2)**2+(yy-n//2)**2
    # Keck-style [source, layer, y, x] SF geometry; nonzero raw
    # centre is intentionally normalized by the same P3 convention.
    kern=np.stack([
        .15*(1-np.exp(-r2/(2*12**2)))+.02,
        .09*(1-np.exp(-r2/(2*7**2)))+.01,
    ],axis=0)[None,...]
    provisional=replace(
        b,dphi_ani=kern,layer_weights=np.array([.3,.7]),
        is_anisoplanatic=True,
    )
    nominal=render_ao_psf(provisional)
    physical=replace(provisional,reference_psf=nominal)
    assert validate_ao_basis_parity(physical)["nominal_parity_passed"]
    changed=render_ao_psf(physical,r0_m=.25)
    assert np.linalg.norm(changed-nominal)>1e-6
    assert changed.shape==nominal.shape


def test_exploratory_gtt_sensitivity_is_explicit_and_reported():
    b=make_basis()
    t=template_from_psfr(b.reference_psf,pixel_scale_mas=9.942)
    actual=template_from_psfr(
        render_ao_psf(b,gtt=1.07),pixel_scale_mas=9.942,
    )
    data=render_stellar_scene(
        actual,positions_xy_px=[[15.2,15.8]],
        fluxes=[20000.],background=3.,
    )
    fitted=fit_hybrid_ao(
        data,b,t,initial_positions_xy_px=[[15.2,15.8]],
        active_parameters=("gtt",),allow_exploratory_gtt=True,
        sigma=np.full(data.shape,.2),n_alternations=1,
    )
    assert fitted.ao_parameters["r0"]==b.r0_m
    assert fitted.ao_parameters["gho"]==1.
    assert .85<=fitted.ao_parameters["gtt"]<=1.15
    assert any("NOT independent STRAP calibration" in x for x in fitted.warnings)
