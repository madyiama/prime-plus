"""Cold/warm Keck comparison must never conceal rotated-pupil changes."""
import json
import numpy as np
import pytest
from scripts.compare_keck_cache_runs import compare


def _case(path, *, cache_status, rotation="header", alter=0.):
    path.mkdir()
    meta={
        "input_fits":"/example/n0120.fits",
        "input_sav":"/example/n0120_LGS_trs.sav",
        "focal_anisoplanatism_algorithm":"symmetry",
        "p3_pupil_rotation_hypothesis":rotation,
        "p3_pupil_parity":"native",
        "p3_pupil_resampling":"physical_binary",
        "psfr_total_init_elapsed_s":30. if cache_status=="miss" else 12.,
        "frequency_anisoplanatism_elapsed_s":16. if cache_status=="miss" else .1,
    }
    report={
        "focal_geometric_kernel":{
            "cache_status":cache_status,
            "cache_key":"synthetic_identical_parameters",
        },
        "cprofile_enabled":False,
    }
    image=np.zeros((8,8,3),dtype=float)
    image[4,4,:]=[.2,.3,.5]
    image[4,5,0]=alter
    (path/"metadata.json").write_text(json.dumps(meta),encoding="utf-8")
    (path/"psfr_profile_breakdown.json").write_text(
        json.dumps(report),encoding="utf-8"
    )
    np.save(path/"telemetry_only_psf.npy",image)


def test_exact_cold_warm_keck_psf_and_runtime(tmp_path):
    a,b=tmp_path/"cold",tmp_path/"warm"
    _case(a,cache_status="miss")
    _case(b,cache_status="hit")
    metrics=compare(a,b)
    assert metrics["parity_passed"]
    assert metrics["max_absolute_error"]==0.
    assert metrics["initialization_speedup"]==pytest.approx(2.5)
    assert metrics["nested_anisoplanatism_speedup"]==pytest.approx(160.)


def test_rotated_pupil_hypotheses_must_not_be_compared_as_identical(tmp_path):
    a,b=tmp_path/"cold",tmp_path/"warm"
    _case(a,cache_status="miss")
    _case(b,cache_status="hit",rotation="zero")
    with pytest.raises(ValueError,match="pupil hypothesis"):
        compare(a,b)


def test_no_false_speedup_without_verified_cache_hit(tmp_path):
    a,b=tmp_path/"cold",tmp_path/"warm"
    _case(a,cache_status="miss")
    _case(b,cache_status="miss")
    with pytest.raises(ValueError,match="cold=miss"):
        compare(a,b)


def test_changed_psf_fails_strict_parity(tmp_path):
    a,b=tmp_path/"cold",tmp_path/"warm"
    _case(a,cache_status="miss")
    _case(b,cache_status="hit",alter=.001)
    metrics=compare(a,b)
    assert not metrics["parity_passed"]
    assert metrics["max_absolute_error"]==pytest.approx(.001)
