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
