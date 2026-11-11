"""Keck PSF-R V1 pinned nominal policy and fail-closed acceptance tests."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.keck_psfr_v1_e2e import DEFAULT_CONFIG,load_nominal,validate_run


def test_v1_nominal_config_is_fixed_and_no_twice_anisokinetism():
    c=load_nominal(DEFAULT_CONFIG)
    assert c["psfr_v1"]["tt_anisokinetism"]=="p3_dani_tt_only"
    assert c["psfr_v1"]["n_layer"]==7
    assert c["psfr_v1"]["pupil_resampling"]=="physical_binary"
    assert c["tip_tilt_noise"]["method"]=="van_dam"
    assert c["tip_tilt_jitter"]["method"]=="legacy_psfr"
    assert c["tip_tilt_jitter"]["mirror_model"]=="unity"


def test_v1_config_rejects_unreviewed_science_changes(tmp_path):
    raw=DEFAULT_CONFIG.read_text()
    changed=raw.replace("ho_covariance_policy = \"strict\"",
                        "ho_covariance_policy = \"small_negative\"")
    p=tmp_path/"changed.toml"
    p.write_text(changed)
    with pytest.raises(ValueError,match="nominal recipe has changed"):
        load_nominal(p)
    changed2=raw.replace('method = "van_dam"','method = "gendron"')
    p.write_text(changed2)
    with pytest.raises(ValueError,match="TT V1 profile/TOML mismatch"):
        load_nominal(p)


def _fixture(directory):
    p=Path(directory)
    p.mkdir(parents=True)
    psf=np.ones((64,64),dtype=float)/(64*64)
    np.save(p/"telemetry_only_psf.npy",psf)
    np.save(p/"p4_otf_tt.npy",np.ones((64,64)))
    metadata={
        "ao_mode":"LGS",
        "p3_pupil_resampling":"physical_binary",
        "p3_pupil_parity":"native",
        "p3_pupil_rotation_hypothesis":"header",
        "tt_kernel_convention":"noll_angular",
        "ho_covariance_policy":"strict",
        "ho_covariance_projection_applied":False,
        "tt_kernel_is_physical_noll":True,
    }
    def write(n,v): (p/n).write_text(json.dumps(v))
    write("metadata.json",metadata)
    write("tt_selection.json",{
        "profile":"reduced_jitter","noise_method":"van_dam",
        "covariance_method":"legacy_psfr","mirror_model":"unity",
    })
    write("p4_tt_angular_chain.json",{
        "tt_kernel_convention":"noll_angular",
        "status":"numerical_chain_verified_instrument_uncalibrated",
        "tt_source_angle_replay":{"tt_centroid_to_otf_units_verified":True},
        "sensor_gain_independently_calibrated":False,
    })
    write("tiptilt_diagnostics.json",{
        "jitter_covariance_method":"legacy_psfr",
    })
    write("anisoplanatism_diagnostics.json",{})
    write("ho_covariance_diagnostics.json",{})
    (p/"comparison_nirc2_psfr.png").write_bytes(b"synthetic")
    return p


def test_v1_artifact_numerical_gates(tmp_path):
    d=_fixture(tmp_path/"ok")
    report=validate_run(d)
    assert report["status"]=="passed_numerical_artifact_gates"
    assert not report["independent_instrument_calibration"]
    assert report["anisokinetism"]=="existing_P3_dani_tt_only"


def test_v1_artifacts_reject_nonphysical_otf_and_ho_repair(tmp_path):
    d=_fixture(tmp_path/"bad")
    o=np.ones((64,64))
    o[32,32]=.9
    np.save(d/"p4_otf_tt.npy",o)
    with pytest.raises(ValueError,match="Nonphysical TT OTF"):
        validate_run(d)
    np.save(d/"p4_otf_tt.npy",np.ones((64,64)))
    meta=json.loads((d/"metadata.json").read_text())
    meta["ho_covariance_projection_applied"]=True
    (d/"metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="HO covariance was repaired"):
        validate_run(d)


def test_v1_cli_has_multi_acquisition_and_check_only():
    script=Path(__file__).resolve().parents[1]/"scripts/keck_psfr_v1_e2e.py"
    cmd=subprocess.run([sys.executable,str(script),"--help"],
                       capture_output=True,text=True,check=False,timeout=20)
    assert cmd.returncode==0,cmd.stderr
    for arg in ("--fits","--sav","--calib","--config","--check-only"):
        assert arg in cmd.stdout
