"""Scientific Keck factorization A/B comparator refuses misleading results."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.compare_keck_vii_factorized import compare


def _write(folder,method,*,pupil="header",cache="disabled",sf_delta=0.,psf_delta=0.):
    folder=Path(folder)
    folder.mkdir()
    meta={
        "ho_vii_method":method,
        "input_fits":"/science/n0120.fits",
        "input_sav":"/science/n0120_LGS_trs.sav",
        "p3_pupil_rotation_hypothesis":pupil,
        "p3_pupil_resampling":"physical_binary",
        "p3_pupil_parity":"native",
        "focal_anisoplanatism_algorithm":"symmetry",
        "focal_cache_configured":False,
    }
    profile={
        "cprofile_enabled":False,
        "focal_geometric_kernel":{"cache_status":cache},
        "ho_vii_kernel":{"method":method},
        "psfr_stage_timings_ms":{
            "ho_sf":11500. if method=="legacy" else 7700.,
            "psfr_init_total":28500. if method=="legacy" else 24500.,
        }
    }
    sf=np.zeros((16,16),dtype=float)
    sf[8,7]=.5
    sf[10,8]+=sf_delta
    psf=np.zeros((8,8,3),dtype=float)
    psf[4,4,:]=[.2,.3,.4]
    psf[3,4,1]+=psf_delta
    (folder/"metadata.json").write_text(json.dumps(meta))
    (folder/"psfr_profile_breakdown.json").write_text(json.dumps(profile))
    np.save(folder/"ho_dphi_ao.npy",sf)
    np.save(folder/"telemetry_only_psf.npy",psf)


def test_real_comparator_accepts_both_independent_products_identically(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    _write(old,"legacy")
    _write(new,"factorized")
    result=compare(old,new)
    assert result["parity_passed"]
    assert result["ho_structure_function"]["max_absolute_difference"]==0
    assert result["science_psf"]["max_absolute_difference"]==0
    assert result["ho_sf_speedup"]==pytest.approx(11.5/7.7)


def test_unchanged_psf_cannot_hide_failed_ho_phase_sf(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    _write(old,"legacy")
    _write(new,"factorized",sf_delta=.001)
    result=compare(old,new)
    assert result["science_psf"]["parity_passed"]
    assert not result["ho_structure_function"]["parity_passed"]
    assert not result["parity_passed"]


def test_changed_pupil_geometry_is_not_a_valid_ab_comparison(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    _write(old,"legacy")
    _write(new,"factorized",pupil="zero")
    with pytest.raises(ValueError,match="pupil"):
        compare(old,new)


def test_focal_cache_cold_warm_difference_is_rejected(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    _write(old,"legacy",cache="miss")
    _write(new,"factorized",cache="hit")
    with pytest.raises(ValueError,match="cache"):
        compare(old,new)


def test_incorrect_baseline_method_tag_is_rejected(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    _write(old,"factorized")
    _write(new,"factorized")
    with pytest.raises(ValueError,match="Expected"):
        compare(old,new)
