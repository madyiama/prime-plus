"""Keck real-FFT comparator tests: avoid misattributed or invalid speedups."""
import json
import numpy as np
import pytest

from scripts.compare_keck_vii_real_fft import compare


def write_case(folder,method,*,pupil="header",cache="disabled",
               ho_delta=0.,psf_delta=0.):
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
            "ho_sf":7000 if method=="factorized" else 5200,
            "psfr_init_total":23500 if method=="factorized" else 21500,
        },
    }
    psf=np.zeros((12,12,3))
    psf[5,6,:]=[.1,.2,.3]
    psf[7,8,1]+=psf_delta
    ho=np.zeros((24,24))
    ho[9,10]=.5
    ho[9,12]+=ho_delta
    (folder/"metadata.json").write_text(json.dumps(meta))
    (folder/"psfr_profile_breakdown.json").write_text(json.dumps(profile))
    np.save(folder/"telemetry_only_psf.npy",psf)
    np.save(folder/"ho_dphi_ao.npy",ho)


def test_real_vii_comparator_accepts_identical_science_and_ho(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    write_case(old,"factorized")
    write_case(new,"factorized_real")
    report=compare(old,new)
    assert report["science_psf_accepted"]
    assert report["ho_sf_strict_parity_passed"]
    assert report["ho_sf_numerical_warning"] is None
    assert report["science_psf"]["max_absolute_difference"]==0
    assert report["ho_sf_speedup"]==pytest.approx(7/5.2)
    for values in report["science_psf_observable_differences"].values():
        np.testing.assert_array_equal(values,np.zeros(3))


def test_intermediate_ho_sf_warning_does_not_hide_science_psf_acceptance(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    write_case(old,"factorized")
    write_case(new,"factorized_real",ho_delta=1e-8)
    report=compare(old,new)
    assert report["science_psf_accepted"]
    assert not report["ho_sf_strict_parity_passed"]
    assert "1e-12" in report["ho_sf_numerical_warning"]


def test_changed_science_psf_is_a_hard_failure(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    write_case(old,"factorized")
    write_case(new,"factorized_real",psf_delta=.001)
    report=compare(old,new)
    assert not report["science_psf_accepted"]
    assert report["ho_sf_strict_parity_passed"]


def test_changed_pupil_hypothesis_forbids_comparison(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    write_case(old,"factorized")
    write_case(new,"factorized_real",pupil="zero")
    with pytest.raises(ValueError,match="pupil"):
        compare(old,new)


def test_mismatched_geometric_cache_is_rejected(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    write_case(old,"factorized",cache="miss")
    write_case(new,"factorized_real",cache="hit")
    with pytest.raises(ValueError,match="cache"):
        compare(old,new)


def test_wrong_reference_method_is_rejected(tmp_path):
    old,new=tmp_path/"old",tmp_path/"new"
    write_case(old,"factorized_real")
    write_case(new,"factorized_real")
    with pytest.raises(ValueError,match="Expected"):
        compare(old,new)
