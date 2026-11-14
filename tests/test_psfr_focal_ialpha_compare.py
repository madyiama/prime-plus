"""Synthetic integrity checks for the Keck Ialpha worker A/B comparator."""
import json

import numpy as np
import pytest

from scripts.compare_keck_focal_ialpha_workers import compare


def write_case(folder, *, workers, psf=None, input_sav="n0120_LGS_trs.sav",
               cache="disabled", parallel_calls=None):
    folder.mkdir()
    meta = {
        "input_fits":"n0120.fits",
        "input_sav":input_sav,
        "p3_pupil_rotation_hypothesis":"header",
        "p3_pupil_resampling":"physical_binary",
        "p3_pupil_parity":"native",
        "ho_vii_method":"factorized_real",
        "focal_anisoplanatism_algorithm":"quadrant",
        "focal_cache_configured":False,
        "focal_workers":workers,
    }
    diagnostics = {
        "cache_status":cache,
        "quadrant_direct":True,
        "ialpha_workers_requested":workers,
        "ialpha_parallel_calls":(int(workers > 1)
                                 if parallel_calls is None
                                 else parallel_calls),
        "n_computed_layers":7,
        "covariance_ms":10.,
        "grouping_ms":0.,
    }
    profile = {
        "cprofile_enabled":False,
        "focal_geometric_kernel":diagnostics,
        "psfr_stage_timings_ms":{"psfr_init_total":100.},
    }
    image = np.ones((10,10,3),dtype=float) if psf is None else psf
    (folder/"metadata.json").write_text(json.dumps(meta),encoding="utf-8")
    (folder/"psfr_profile_breakdown.json").write_text(
        json.dumps(profile),encoding="utf-8")
    np.save(folder/"telemetry_only_psf.npy",image)


def test_accepts_identical_science_cube_and_reports_moments(tmp_path):
    a,b=tmp_path/"serial",tmp_path/"parallel"
    write_case(a,workers=1)
    write_case(b,workers=4)
    report=compare(a,b)
    assert report["science_psf_accepted"]
    assert report["workers"]==4
    assert report["max_absolute_difference"]==0.
    assert report["observable_changes"]["integrated_flux"]==[0.,0.,0.]


def test_rejects_changed_science_data_even_with_same_focal_mode(tmp_path):
    a,b=tmp_path/"serial",tmp_path/"parallel"
    write_case(a,workers=1)
    write_case(b,workers=2,input_sav="other_acquisition.sav")
    with pytest.raises(ValueError,match="provenance"):
        compare(a,b)


def test_rejects_disabled_parallelism_that_falsely_claims_workers(tmp_path):
    a,b=tmp_path/"serial",tmp_path/"parallel"
    write_case(a,workers=1)
    write_case(b,workers=4,parallel_calls=0)
    with pytest.raises(ValueError,match="did not actually run"):
        compare(a,b)


def test_science_psf_mismatch_fails_gate(tmp_path):
    a,b=tmp_path/"serial",tmp_path/"parallel"
    write_case(a,workers=1)
    diff=np.ones((10,10,3),dtype=float)
    diff[2,3,1]+=1e-6
    write_case(b,workers=2,psf=diff)
    report=compare(a,b)
    assert not report["science_psf_accepted"]
    assert report["max_absolute_difference"]>1e-7


def test_rejects_cache_or_profile_bias(tmp_path):
    a,b=tmp_path/"serial",tmp_path/"parallel"
    write_case(a,workers=1)
    write_case(b,workers=2,cache="hit")
    with pytest.raises(ValueError,match="cold direct quadrant"):
        compare(a,b)
