"""Synthetic math parity for phase structure function assembly."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from p3.psfr.structure import assemble_phase_structure_function
from p3.psfr.psfR import psfR, _anisoplanatism_zero_lag


def kernels():
    yy, xx = np.indices((8, 8))
    r2 = (xx-4)**2 + (yy-4)**2
    return tuple(x*r2 for x in (.03,.07,.11,.13))


def test_isoplanatic_legacy_equivalence_and_n_sources():
    ho, tt, fit, alias = kernels()
    a = assemble_phase_structure_function(
        r0_m=.31,gains=[1.2,.8],dphi_ao=ho,dphi_tt=tt,
        dphi_fit=fit,dphi_alias=alias,n_sources=3
    )
    legacy = 1.2*ho+.8*tt+.31**(-5/3)*(fit+alias)
    assert a.dphi_rad2.shape == (8,8,3)
    for i in range(3):
        np.testing.assert_array_equal(a.dphi_rad2[:,:,i],legacy)
    assert a.raw_zero_lag_rad2 is None


@pytest.mark.parametrize("layers", [1,3,7])
def test_anisoplanatic_legacy_formula_and_zero_lag(layers):
    ho,tt,fit,alias = kernels()
    rng = np.random.default_rng(layers)
    dani_kernel = rng.uniform(.01,.2,size=(2,layers,8,8))
    old_kernel = dani_kernel.copy()
    cn2 = rng.uniform(.2,.8,size=layers)
    atm,sci=500e-9,2.055e-6
    a = assemble_phase_structure_function(
        r0_m=.4,gains=[.9,1.1],dphi_ao=ho,dphi_tt=tt,
        dphi_fit=fit,dphi_alias=alias,n_sources=2,
        anisoplanatism_kernel=dani_kernel,layer_strengths=cn2,
        atmosphere_wavelength_m=atm,science_wavelength_m=sci
    )
    raw=np.einsum('slxy,l->xys',dani_kernel,cn2*(atm/sci)**2)
    normalized=raw-raw[4,4]
    expected=(.9*ho+1.1*tt+.4**(-5/3)*(fit+alias))[:,:,None]+normalized
    np.testing.assert_array_equal(a.dphi_rad2,expected)
    np.testing.assert_array_equal(a.anisoplanatism_rad2,normalized)
    np.testing.assert_array_equal(a.raw_zero_lag_rad2,raw[4,4])
    np.testing.assert_array_equal(a.anisoplanatism_rad2[4,4],np.zeros(2))
    np.testing.assert_array_equal(dani_kernel,old_kernel)
    assert a.zero_lag_reference == "central_otf_sample"


def test_weights_fallback_equals_explicit_strengths():
    ho,tt,fit,alias=kernels()
    opts=dict(
        r0_m=.3,gains=[1.,1.],dphi_ao=ho,dphi_tt=tt,
        dphi_fit=fit,dphi_alias=alias,n_sources=1,
        anisoplanatism_kernel=np.ones((1,2,8,8)),
        atmosphere_wavelength_m=500e-9,science_wavelength_m=2e-6,
    )
    a=assemble_phase_structure_function(**opts,layer_weights=[.4,.6])
    b=assemble_phase_structure_function(
        **opts,layer_strengths=np.array([.4,.6])*.3**(-5/3)
    )
    np.testing.assert_array_equal(a.dphi_rad2,b.dphi_rad2)
    np.testing.assert_array_equal(a.raw_zero_lag_rad2,b.raw_zero_lag_rad2)


@pytest.mark.parametrize("aniso",[False,True])
def test_legacy_psfr_adapter_and_diagnostics(aniso):
    ho,tt,fit,alias=kernels()
    m=psfR.__new__(psfR)
    m.dphi_ao,m.dphi_tt,m.dphi_fit,m.dphi_alias=ho,tt,fit,alias
    m.freq=NS(isAniso=aniso,wvlRef=2.055e-6)
    m.ao=NS(src=NS(nSrc=2),atm=NS(
        nL=2,weights=np.array([.4,.6]),wvl=500e-9
    ))
    if aniso:
        m.dphi_ani=np.ones((2,2,8,8))*.2
    sf=m.phase_structure_function(.31,[1.,1.])
    expected=assemble_phase_structure_function(
        r0_m=.31,gains=[1.,1.],dphi_ao=ho,dphi_tt=tt,
        dphi_fit=fit,dphi_alias=alias,n_sources=2,
        anisoplanatism_kernel=m.dphi_ani if aniso else None,
        layer_weights=m.ao.atm.weights if aniso else None,
        atmosphere_wavelength_m=500e-9 if aniso else None,
        science_wavelength_m=2.055e-6 if aniso else None,
    )
    np.testing.assert_array_equal(sf,expected.dphi_rad2)
    if aniso:
        np.testing.assert_array_equal(m.dani,expected.anisoplanatism_rad2)
        np.testing.assert_array_equal(m.dani_raw_zero_lag_rad2,expected.raw_zero_lag_rad2)
        assert m.dani_zero_lag_reference=="central_otf_sample"


def test_legacy_helper_and_error_behaviour():
    array=np.array([[1.,2.],[3.,4.]])
    np.testing.assert_array_equal(_anisoplanatism_zero_lag(array),array-array[1,1])
    ho,tt,fit,alias=kernels()
    p=dict(r0_m=.3,gains=[1.,1.],dphi_ao=ho,dphi_tt=tt,
           dphi_fit=fit,dphi_alias=alias,n_sources=1)
    with pytest.raises(ValueError,match="two finite HO/TT gains"):
        assemble_phase_structure_function(**{**p,"gains":[]})
    with pytest.raises(ValueError,match="positive r0"):
        assemble_phase_structure_function(**{**p,"r0_m":0.})
    with pytest.raises(ValueError,match="share"):
        assemble_phase_structure_function(**{**p,"dphi_tt":tt[:4]})
    with pytest.raises(ValueError,match="one value per layer"):
        assemble_phase_structure_function(
            **p,anisoplanatism_kernel=np.ones((1,2,8,8)),
            layer_strengths=[1.,2.,3.],
            atmosphere_wavelength_m=500e-9,science_wavelength_m=2e-6,
        )
