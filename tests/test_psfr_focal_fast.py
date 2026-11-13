"""Regression: accelerated Flicker focal covariance preserves original physics."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.FourierUtils import matrix_to_map
from p3.aoSystem.focalFast import grouped_matrix_to_map
from p3.aoSystem.anisoplanatismModel import focal_angular_function


def _separations(n):
    x,y=np.meshgrid(np.arange(n),np.arange(n))
    X=np.ones((n*n,1))*x.T.reshape(-1)
    Y=np.tile(y,[n,n])
    sx=(X.T-x.T.reshape(-1)).T
    sy=Y-y.T.reshape(-1)
    return sx,sy


@pytest.mark.parametrize("n",[3,5,9,13])
def test_grouped_baseline_averages_match_original_asymmetric_matrix(n):
    rng=np.random.default_rng(n)
    data=rng.normal(size=(n*n,n*n))
    sx,sy=_separations(n)
    reference=matrix_to_map(data,sep_x=sx,sep_y=sy)
    accelerated=grouped_matrix_to_map(data,sx,sy)
    np.testing.assert_allclose(accelerated,reference,rtol=4e-13,atol=5e-15)
    assert accelerated.shape==(2*n-1,2*n-1)
    # Preserve the legacy mirroring convention exactly, even for
    # NON-centrosymmetric covariance input: no physical model change.
    np.testing.assert_array_equal(
        accelerated[n:, :], np.flipud(accelerated[:n-1,:])
    )


def _fake_system(n_layers=4,l0=25.):
    telescope=NS(D=10.5)
    heights=np.linspace(0,14000,n_layers)
    atmosphere=NS(nL=n_layers,heights=heights,L0=l0)
    source=NS(
        nSrc=2,direction=np.array([[0.,1.1e-5],[0.,-.8e-5]])
    )
    beacon=NS(
        height=90000.,
        direction=np.array([[0.],[0.]])
    )
    return telescope,atmosphere,source,beacon


@pytest.mark.parametrize("outer_scale",[25.,40.,np.inf])
@pytest.mark.parametrize("output_grid",["native","otf"])
@pytest.mark.parametrize("pupil_grid_mode",["legacy","physical_diameter"])
def test_symmetry_preserves_focal_six_terms_multilayer_and_off_axis(
    outer_scale,output_grid,pupil_grid_mode,
):
    tel,atm,src,gs=_fake_system(n_layers=4,l0=outer_scale)
    args=(tel,atm,src,gs,4,48,2.)
    kw=dict(
        output_grid=output_grid,pupil_grid_mode=pupil_grid_mode,
    )
    before=focal_angular_function(*args,covariance_algorithm="legacy",**kw)
    after=focal_angular_function(*args,covariance_algorithm="symmetry",**kw)
    np.testing.assert_allclose(
        after,before,rtol=1e-12,atol=2e-11,
        err_msg=f"L0={outer_scale}; output={output_grid}; pupil={pupil_grid_mode}",
    )
    # Default must remain numerically BITWISE historical.
    default=focal_angular_function(*args,**kw)
    np.testing.assert_array_equal(default,before)
    assert np.isfinite(after).all()


def test_symmetry_retains_pupil_filter_and_zero_height():
    tel,atm,src,gs=_fake_system(n_layers=3,l0=30.)
    nph=2*3+1
    filt=np.diag(np.linspace(.8,1.,nph*nph))
    args=(tel,atm,src,gs,3,32,2.)
    old=focal_angular_function(
        *args,h_filt=filt,covariance_algorithm="legacy",
        output_grid="native",
    )
    new=focal_angular_function(
        *args,h_filt=filt,covariance_algorithm="symmetry",
        output_grid="native",
    )
    np.testing.assert_allclose(new,old,rtol=1e-12,atol=1e-11)
    np.testing.assert_array_equal(old[:,0],np.zeros_like(old[:,0]))


def test_invalid_algorithm_and_bad_baseline_rejected():
    tel,atm,src,gs=_fake_system()
    with pytest.raises(ValueError,match="covariance algorithm"):
        focal_angular_function(
            tel,atm,src,gs,3,32,2.,covariance_algorithm="not_real",
        )
    sx,sy=_separations(5)
    with pytest.raises(ValueError,match="integer-coordinate"):
        grouped_matrix_to_map(np.ones((25,25)),sx+0.25,sy)



def test_angular_flicker_mode_is_independent_of_focal_acceleration():
    """Regression for an unintended reference to focal-only algorithm state."""
    from p3.aoSystem.anisoplanatismModel import angular_function
    tel,atm,src,gs=_fake_system(n_layers=3,l0=25.)
    tel.airmass=1.
    # Unlike the Fourier default, this tests the angular Flicker
    # alternative, which must never read the focal covariance switch.
    result=angular_function(
        tel,atm,src,gs,nOtf=16,samp=2.,method="Flicker",
    )
    assert result.shape==(src.nSrc,atm.nL,16,16)
    assert np.isfinite(result).all()


def test_keck_like_actuator_array_on_fast_path_is_finite():
    """20+1 actuators maps to n_ph=43, beyond previous small-matrix cases.

    Do not run legacy matrix-to-map here: its n_ph**6 masks consume
    substantial memory. Focus on the new branch's geometry and shapes.
    """
    tel,atm,src,gs=_fake_system(n_layers=2,l0=25.)
    # Only one science direction and one nonzero layer for runtime.
    src.nSrc=1
    src.direction=np.array([[0.], [0.]])
    result=focal_angular_function(
        tel,atm,src,gs,
        nActu=np.array([21]),nOtf=64,samp=2.,
        covariance_algorithm="symmetry",
    )
    assert result.shape==(1,2,64,64)
    np.testing.assert_array_equal(result[:,0],0.)
    assert np.isfinite(result).all()


def test_lgs_dispatch_passes_algorithm_without_affecting_angular_terms(
    monkeypatch,
):
    from p3.aoSystem import anisoplanatismModel as aniso
    tel,atm,src,gs=_fake_system(n_layers=3,l0=25.)
    tel.airmass=1.
    gs.zenith=np.array([0.])
    gs.azimuth=np.array([0.])
    ngs=NS(height=0.,direction=np.array([[0.],[0.]]))
    zeros=np.zeros((src.nSrc,atm.nL,20,20))
    monkeypatch.setattr(aniso,"angular_function",lambda *a,**kw: zeros.copy())
    monkeypatch.setattr(
        aniso,"anisokinetism_function",lambda *a,**kw: zeros.copy()
    )
    legacy=aniso.anisoplanatism_structure_function(
        tel,atm,src,gs,ngs,nOtf=20,samp=2.,nActu=np.array([3]),
        focal_algorithm="legacy",
    )
    symmetry=aniso.anisoplanatism_structure_function(
        tel,atm,src,gs,ngs,nOtf=20,samp=2.,nActu=np.array([3]),
        focal_algorithm="symmetry",
    )
    np.testing.assert_allclose(
        symmetry[0],legacy[0],rtol=1e-12,atol=2e-11
    )



@pytest.mark.parametrize("shape",[(1,), (1,1)])
def test_numpy_singleton_lgs_height_matches_scalar_legacy_and_symmetry(shape):
    """Reproduce n0120 failure: Keck stores gs.height as a NumPy array.

    Both pathways must give the same layer SF as the scalar-height
    calculation, without modifying the physical height or direction.
    """
    tel,atm,src,gs=_fake_system(n_layers=3,l0=25.)
    baseline=focal_angular_function(
        tel,atm,src,gs,4,48,2.,covariance_algorithm="legacy",
    )
    gs.height=np.full(shape,90000.,dtype=float)
    legacy=focal_angular_function(
        tel,atm,src,gs,4,48,2.,covariance_algorithm="legacy",
    )
    accelerated=focal_angular_function(
        tel,atm,src,gs,4,48,2.,covariance_algorithm="symmetry",
    )
    np.testing.assert_allclose(legacy,baseline,rtol=1e-12,atol=2e-11)
    np.testing.assert_allclose(accelerated,legacy,rtol=1e-12,atol=2e-11)


def test_multivalued_lgs_height_does_not_silently_choose_a_beacon():
    from p3.aoSystem.focalFast import _single_real_scalar
    with pytest.raises(ValueError,match="exactly one"):
        _single_real_scalar(np.array([85000.,90000.]),name="cone_fraction")
    with pytest.raises(ValueError,match="exactly one"):
        _single_real_scalar(np.empty((0,)),name="theta_shift_x_m")
    assert _single_real_scalar(np.array([[.2]]),name="cone_fraction")==.2
