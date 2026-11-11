"""P4-C conservative TT production choice and switchable noise physics.

The tests do not require archive data or NIRC2 source fitting. They
verify the existing P4-A default is maintained and model alternatives
are passed independently, not mixed into silent empirical corrections.
"""
import tomllib
from pathlib import Path

import numpy as np
import pytest

from p3.telemetry.ttSelection import (
    DEFAULT_REDUCED_JITTER_PROFILE, PROFILES, resolve_tt_selection,
)
from p3.telemetry.ttNoise import estimate_tt_noise
from p3.aoSystem.ttAngular import direct_angular_tt_covariance
from p3.aoSystem.tipTiltGeometry import zernike_tilt_opd_per_arcsec


CONFIG = Path(__file__).resolve().parents[1] / "config/keck_tt_noise.toml"


def load_config():
    with CONFIG.open("rb") as fp:
        return tomllib.load(fp)


def test_provisional_default_is_existing_physical_A1_not_kernel_ablation():
    cfg=load_config()
    resolved=resolve_tt_selection(cfg)
    named=resolve_tt_selection(cfg,profile=DEFAULT_REDUCED_JITTER_PROFILE)
    assert resolved.noise_method == named.noise_method == "van_dam"
    assert resolved.covariance_method == named.covariance_method == "legacy_psfr"
    assert resolved.mirror_model == named.mirror_model == "unity"
    assert resolved.noise_band == named.noise_band == (0.65, 0.95)
    assert resolved.noise_welch == named.noise_welch == 512
    assert resolved.report()["recommended_for_framework_default"]
    assert not resolved.report()["instrument_calibrated"]
    assert resolved.profile is None  # TOML is authoritative until profile requested
    assert "spectral_comparison" in PROFILES


def test_user_changed_TOML_remains_authoritative_unless_explicit_profile():
    cfg=load_config()
    cfg["tip_tilt_noise"]["method"]="gendron"
    cfg["tip_tilt_jitter"]["method"]="spectral_closed_loop"
    c=resolve_tt_selection(cfg)
    assert c.noise_method=="gendron"
    assert c.covariance_method=="spectral_closed_loop"
    assert c.source=="TOML_or_legacy_defaults"
    override=resolve_tt_selection(cfg,profile="reduced_jitter")
    assert override.noise_method=="van_dam"
    assert override.covariance_method=="legacy_psfr"
    assert override.profile=="reduced_jitter"


def test_named_profiles_are_independent_noise_and_atmospheric_switches():
    cfg=load_config()
    spectral=resolve_tt_selection(cfg,profile="spectral_comparison")
    gendron=resolve_tt_selection(cfg,profile="gendron_comparison")
    assert spectral.noise_method=="van_dam"
    assert spectral.covariance_method=="spectral_closed_loop"
    assert gendron.noise_method=="gendron"
    assert gendron.covariance_method=="legacy_psfr"
    for model in (spectral,gendron):
        meta=model.report()
        assert not meta["science_psf_used_for_method_selection"]
        assert not meta["instrument_calibrated"]
        assert not meta["noise_method_physically_validated"]
        assert not meta["apd_noise_model_enabled"]


def test_individual_cli_overrides_are_honored_and_reported():
    cfg=load_config()
    p=resolve_tt_selection(cfg,profile="reduced_jitter",overrides={
        "noise_method":"gendron",
        "covariance_method":"spectral_closed_loop",
        "mirror_model":"ragland_2016_keck_ii",
        "jitter_welch":256,
    })
    assert p.noise_method=="gendron"
    assert p.covariance_method=="spectral_closed_loop"
    assert p.mirror_model=="ragland_2016_keck_ii"
    assert p.jitter_welch==256
    assert "explicit_CLI" in p.source


def test_reject_silent_ignored_mirror_and_invalid_timing_config():
    cfg=load_config()
    with pytest.raises(ValueError,match="does not use the mirror transfer"):
        resolve_tt_selection(cfg,overrides={"mirror_model":"ragland_2016_keck_ii"})
    with pytest.raises(ValueError,match="Unknown TT profile"):
        resolve_tt_selection(cfg,profile="magic_image_fit")
    with pytest.raises(ValueError,match="Unsupported STRAP noise estimator"):
        resolve_tt_selection(cfg,overrides={"noise_method":"not_a_method"})
    with pytest.raises(ValueError,match="Unsupported TT covariance"):
        resolve_tt_selection(cfg,overrides={"covariance_method":"empirical_x4"})
    with pytest.raises(ValueError,match="band"):
        resolve_tt_selection(cfg,overrides={"noise_band":(0.98,0.70)})
    with pytest.raises(ValueError,match="integer >= 128"):
        resolve_tt_selection(cfg,overrides={"noise_welch":27})


def test_changing_noise_only_changes_legacy_residual_not_angular_otf_formula():
    # A physically admissible synthetic test: slowly varying atmospheric
    # image motion plus independent detector noise. The previous pure-white
    # fixture made C_y-(1+pn)C_n non-PSD for BOTH methods and required
    # clipping, so it could not verify the actual residual-turbulence model.
    rng = np.random.default_rng(443)
    count = 4096
    frames = np.arange(count)
    raw = np.column_stack([
        0.16*np.sin(2*np.pi*3*frames/count)
        + 0.05*np.cos(2*np.pi*7*frames/count),
        0.12*np.sin(2*np.pi*5*frames/count + 0.7)
        + 0.04*np.cos(2*np.pi*9*frames/count),
    ]) + rng.normal(0, 0.012, size=(count, 2))

    D = 10.5
    gain = zernike_tilt_opd_per_arcsec(D)
    c1, _ = estimate_tt_noise(
        raw, method="van_dam", sample_rate_hz=125.0,
        rejection_frequency_hz=np.linspace(0, 62.5, 1025),
        rejection_transfer=np.ones(1025),
        high_frequency_band=(0.65, 0.95), nperseg=512,
    )
    c2, _ = estimate_tt_noise(raw, method="gendron")
    # Even small estimator changes must be compared in arcsec², not
    # accidentally swallowed by an OPD/rad²-sized absolute tolerance.
    assert not np.allclose(c1, c2, rtol=1e-4, atol=1e-10)
    pn = 0.12
    observed = np.cov(raw, rowvar=False, bias=True)
    assert np.linalg.eigvalsh(observed - (1+pn)*c1).min() > 0
    assert np.linalg.eigvalsh(observed - (1+pn)*c2).min() > 0
    # A clipping warning is an outright test failure: we are testing the
    # physical covariance algebra, NOT its automatic PSD repair policy.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        physical1, _ = direct_angular_tt_covariance(
            raw, c1*gain**2, D, method="legacy_psfr", pn=pn,
        )
        physical2, _ = direct_angular_tt_covariance(
            raw, c2*gain**2, D, method="legacy_psfr", pn=pn,
        )
    # This validates C_y-C_n without a covariance projection.
    expected1 = (observed-c1) * (np.pi/(180*3600))**2
    expected2 = (observed-c2) * (np.pi/(180*3600))**2
    np.testing.assert_allclose(physical1, expected1, rtol=1e-12, atol=1e-28)
    np.testing.assert_allclose(physical2, expected2, rtol=1e-12, atol=1e-28)
    assert not np.allclose(physical1, physical2, rtol=1e-6, atol=1e-27)
    assert np.linalg.eigvalsh(physical1).min() > 0
    assert np.linalg.eigvalsh(physical2).min() > 0


def test_framework_cli_exposes_profiles_and_independent_switches():
    import subprocess
    import sys

    script=Path(__file__).resolve().parents[1]/"scripts"/"keck_n0013_smoke.py"
    completed=subprocess.run(
        [sys.executable,str(script),"--help"],
        capture_output=True,text=True,check=False,timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--tt-profile" in completed.stdout
    assert "reduced_jitter" in completed.stdout
    assert "spectral_comparison" in completed.stdout
    assert "gendron_comparison" in completed.stdout
    assert "--tt-noise-method" in completed.stdout
    assert "--tt-jitter-model" in completed.stdout


def test_selection_json_can_be_serialized_for_reproduction():
    import json

    choice=resolve_tt_selection(
        load_config(),profile="spectral_comparison",
        overrides={"noise_method":"gendron"},
    )
    copied=json.loads(json.dumps(choice.report(),allow_nan=False))
    assert copied["profile"] == "spectral_comparison"
    assert copied["covariance_method"] == "spectral_closed_loop"
    assert copied["noise_method"] == "gendron"
    assert copied["selection_source"].startswith("named_profile")
    assert copied["instrument_calibrated"] is False
