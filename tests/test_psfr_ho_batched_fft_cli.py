"""A/B Keck HO FFT comparator rejects misleading timings and science changes."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.compare_keck_ho_fft_runs import compare


def _case(where,mode,*,pupil="header",cache="disabled",delta=0.):
    where=Path(where)
    where.mkdir()
    meta={
        "ho_fft_mode":mode,"input_fits":"/example/n0120.fits",
        "input_sav":"/example/n0120_LGS_trs.sav",
        "focal_anisoplanatism_algorithm":"symmetry",
        "p3_pupil_rotation_hypothesis":pupil,
        "p3_pupil_resampling":"physical_binary",
        "p3_pupil_parity":"native",
        "focal_cache_configured":False,
    }
    profiler={
        "focal_geometric_kernel":{"cache_status":cache},
        "cprofile_enabled":False,
        "ho_fft":{"mode":mode,"batch_count":1},
        "psfr_stage_timings_ms":{
            "ho_sf":12000 if mode=="legacy" else 3500,
            "psfr_init_total":28000 if mode=="legacy" else 19000,
        },
    }
    image=np.zeros((8,8,3),dtype=float)
    image[4,4]=[.2,.4,.6]
    image[4,3,0]=delta
    dphi=np.zeros((16,16),dtype=float)
    dphi[6,7]=.8+delta
    (where/"metadata.json").write_text(json.dumps(meta))
    (where/"psfr_profile_breakdown.json").write_text(json.dumps(profiler))
    np.save(where/"telemetry_only_psf.npy",image)
    np.save(where/"ho_dphi_ao.npy",dphi)


def test_matching_science_psf_and_ho_sf_with_batch_speedup(tmp_path):
    old,new=tmp_path/"legacy",tmp_path/"batched"
    _case(old,"legacy")
    _case(new,"batched")
    result=compare(old,new)
    assert result["parity_passed"]
    assert result["ho_sf_speedup"]==pytest.approx(12000/3500)
    assert result["psfr_init_speedup"]==pytest.approx(28000/19000)
    assert result["ho_structure_function"]["max_abs"]==0.
    assert result["science_psf"]["max_abs"]==0.


def test_changed_pupil_rotation_prohibits_attribution(tmp_path):
    old,new=tmp_path/"legacy",tmp_path/"batched"
    _case(old,"legacy")
    _case(new,"batched",pupil="zero")
    with pytest.raises(ValueError,match="pupil"):
        compare(old,new)


def test_mismatched_cache_state_cannot_be_claimed_as_ho_speedup(tmp_path):
    old,new=tmp_path/"legacy",tmp_path/"batched"
    _case(old,"legacy",cache="miss")
    _case(new,"batched",cache="hit")
    with pytest.raises(ValueError,match="cache"):
        compare(old,new)


def test_ho_sf_science_mismatch_is_detected_even_when_psf_near_same(tmp_path):
    old,new=tmp_path/"legacy",tmp_path/"batched"
    _case(old,"legacy")
    _case(new,"batched")
    dphi=np.load(new/"ho_dphi_ao.npy")
    dphi[6,7]+=.001
    np.save(new/"ho_dphi_ao.npy",dphi)
    result=compare(old,new)
    assert not result["parity_passed"]
    assert result["science_psf"]["parity_passed"]
    assert not result["ho_structure_function"]["parity_passed"]


def test_wrong_historical_baseline_tag_is_rejected(tmp_path):
    old,new=tmp_path/"legacy",tmp_path/"batched"
    _case(old,"batched")
    _case(new,"batched")
    with pytest.raises(ValueError,match="Expected"):
        compare(old,new)
