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
    rng=np.random.default_rng(443)
    raw=rng.normal(0,0.06,size=(2048,2))
    D=10.5
    gain=zernike_tilt_opd_per_arcsec(D)
    c1,_=estimate_tt_noise(
        raw,method="van_dam",sample_rate_hz=125.,
        rejection_frequency_hz=np.linspace(0,62.5,1025),
        rejection_transfer=np.ones(1025),
        high_frequency_band=(0.65,0.95),nperseg=512,
    )
    c2,_=estimate_tt_noise(raw,method="gendron")
    assert not np.allclose(c1,c2,rtol=1e-4,atol=1e-8)
    physical1,_=direct_angular_tt_covariance(
        raw,c1*gain**2,D,method="legacy_psfr",pn=0.12,
    )
    physical2,_=direct_angular_tt_covariance(
        raw,c2*gain**2,D,method="legacy_psfr",pn=0.12,
    )
    assert not np.allclose(physical1,physical2,rtol=1e-4,atol=1e-10)
    # Both routes retain rad² image-motion covariance and physical
    # Noll-normalized angular Gaussian OTF; only noise inference changes.
    assert physical1.shape==physical2.shape==(2,2)
    assert np.all(np.isfinite(physical1))
    assert np.all(np.isfinite(physical2))
