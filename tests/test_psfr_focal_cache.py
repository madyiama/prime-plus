"""Geometric cache contract: NEVER cache rotated Keck pupil/OTF products."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.anisoplanatismModel import focal_angular_function
from p3.aoSystem.focalCache import key_for_focal
from p3.psfr.chromatic import prepare_pupil_lag_grid


def make_case():
    tel=NS(D=10.5,pupilAngle=0.0)
    atm=NS(nL=3,heights=np.array([0.,4500.,12000.]),
           L0=25.,r0=.4,weights=np.array([.2,.3,.5]))
    src=NS(nSrc=1,direction=np.array([[1.e-5],[-2.e-5]]))
    gs=NS(height=np.array([90000.]),direction=np.array([[0.],[0.]]))
    return tel,atm,src,gs


def calculate(tel,atm,src,gs,path,*,filter_=1,**kwargs):
    diag={}
    image=focal_angular_function(
        tel,atm,src,gs,np.array([3]),32,2.,
        h_filt=filter_,covariance_algorithm="symmetry",
        focal_cache_dir=path,focal_diagnostics=diag,
        **kwargs,
    )
    return image,diag


def test_cache_cold_warm_legacy_parity_and_independent_copies(tmp_path):
    tel,atm,src,gs=make_case()
    legacy=focal_angular_function(
        tel,atm,src,gs,np.array([3]),32,2.,
        covariance_algorithm="legacy",
    )
    cold,d1=calculate(tel,atm,src,gs,tmp_path)
    assert d1["cache_status"]=="miss"
    assert d1["n_computed_layers"]==2
    assert d1["covariance_ms"]>=0
    assert d1["grouping_ms"]>=0
    assert d1["pupil_rotation_or_optical_otf_cached"] is False
    np.testing.assert_allclose(cold,legacy,rtol=1e-12,atol=2e-11)
    warm,d2=calculate(tel,atm,src,gs,tmp_path)
    assert d2["cache_status"]=="hit"
    assert d2["n_computed_layers"]==0
    np.testing.assert_array_equal(warm,cold)
    # The cached file is not a mutable alias shared with the caller.
    warm[0,1,5,5]=123456.
    again,d3=calculate(tel,atm,src,gs,tmp_path)
    assert d3["cache_status"]=="hit"
    np.testing.assert_array_equal(again,cold)
    assert len(list(tmp_path.glob("focal_symmetric_*.npz")))==1


def test_rotating_or_changing_pupil_does_not_reuse_optical_otf(tmp_path):
    tel,atm,src,gs=make_case()
    first,d1=calculate(tel,atm,src,gs,tmp_path)
    # Deliberately asymmetric physical pupil; two orientations produce
    # different diffraction OTFs, even though unfiltered Flicker focal
    # SF depends only on D, AO geometry and field angles.
    pupil=np.zeros((32,32))
    pupil[6:24,5:12]=1.
    pupil[17:22,12:24]=1.
    otf0=prepare_pupil_lag_grid(pupil,diameter_m=10.5).otf_lag
    tel.pupilAngle=90.
    pupil90=np.rot90(pupil)
    otf90=prepare_pupil_lag_grid(pupil90,diameter_m=10.5).otf_lag
    assert np.max(np.abs(otf90-otf0))>1e-3
    tel.pupil=pupil90
    second,d2=calculate(tel,atm,src,gs,tmp_path)
    assert d2["cache_status"]=="hit"
    np.testing.assert_array_equal(first,second)
    # This cache stores only a 4-D geometrical focal layer SF.
    with np.load(next(tmp_path.glob("*.npz")),allow_pickle=False) as data:
        assert set(data.files)=={"schema","key","checksum","layers"}
        assert data["layers"].ndim==4


@pytest.mark.parametrize("change",[
    "telescope_diameter","outer_scale","layer_height",
    "source_direction","laser_direction","laser_height",
])
def test_physically_changed_geometry_invalidates_cache_key(tmp_path,change):
    tel,atm,src,gs=make_case()
    initial,diag=calculate(tel,atm,src,gs,tmp_path)
    key=diag["cache_key"]
    if change=="telescope_diameter": tel.D+=0.1
    elif change=="outer_scale": atm.L0+=1.
    elif change=="layer_height": atm.heights[1]+=5.
    elif change=="source_direction": src.direction[0,0]+=1e-7
    elif change=="laser_direction": gs.direction[0,0]+=1e-7
    elif change=="laser_height": gs.height[0]+=100.
    changed=key_for_focal(
        tel,atm,src,gs,np.array([3]),32,2.,
        pupil_grid_mode="legacy",output_grid="otf",
    )
    assert changed!=key
    second,report=calculate(tel,atm,src,gs,tmp_path)
    assert report["cache_status"]=="miss"
    assert report["cache_key"]==changed
    assert np.isfinite(second).all()
    assert len(list(tmp_path.glob("*.npz")))==2


def test_r0_and_cn2_weights_are_reapplied_after_geometric_cache_hit(tmp_path):
    tel,atm,src,gs=make_case()
    unweighted,report=calculate(tel,atm,src,gs,tmp_path)
    baseline=(unweighted*atm.weights[None,:,None,None]
              *atm.r0**(-5/3)).sum(axis=1)
    atm.r0=.31
    atm.weights=np.array([.1,.4,.5])
    reused,report2=calculate(tel,atm,src,gs,tmp_path)
    assert report2["cache_status"]=="hit"
    np.testing.assert_array_equal(unweighted,reused)
    new=(reused*atm.weights[None,:,None,None]
         *atm.r0**(-5/3)).sum(axis=1)
    assert not np.allclose(new,baseline)


def test_filtered_pupil_matrix_never_uses_unfiltered_cache(tmp_path):
    tel,atm,src,gs=make_case()
    unfiltered,diag=calculate(tel,atm,src,gs,tmp_path)
    files=list(tmp_path.glob("*.npz"))
    nph=7
    # Nonidentity matrix simulates a pupil-dependent spatial filter.
    filter_=np.eye(nph*nph)*.9
    filtered,report=calculate(tel,atm,src,gs,tmp_path,filter_=filter_)
    assert report["cache_status"]=="bypass_filtered"
    assert len(list(tmp_path.glob("*.npz")))==len(files)
    assert not np.allclose(filtered,unfiltered)
    legacy=focal_angular_function(
        tel,atm,src,gs,np.array([3]),32,2.,h_filt=filter_,
        covariance_algorithm="legacy",
    )
    np.testing.assert_allclose(filtered,legacy,rtol=1e-12,atol=2e-11)


def test_corrupt_cache_never_reused_and_is_recomputed(tmp_path):
    tel,atm,src,gs=make_case()
    correct,_=calculate(tel,atm,src,gs,tmp_path)
    file=next(tmp_path.glob("*.npz"))
    # Invalid ZIP is not a valid physical cache hit.
    file.write_bytes(b"not-an-npz")
    repaired,diag=calculate(tel,atm,src,gs,tmp_path)
    assert diag["cache_status"]=="invalidated"
    np.testing.assert_array_equal(repaired,correct)
    reread,diag2=calculate(tel,atm,src,gs,tmp_path)
    assert diag2["cache_status"]=="hit"
    np.testing.assert_array_equal(reread,correct)


def test_disabled_cache_has_no_disk_side_effect(tmp_path):
    tel,atm,src,gs=make_case()
    output,report=calculate(tel,atm,src,gs,None)
    assert report["cache_status"]=="disabled"
    assert report["n_computed_layers"]==2
    assert np.isfinite(output).all()
    assert not list(tmp_path.iterdir())
