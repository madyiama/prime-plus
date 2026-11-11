"""P1-A regression tests for archived Keck atmosphere / r0 / DIMM evidence.

Fixtures match the actual keys exported by keck_n0013_smoke.py, not
the synthetic geometry dict used by the P2 experiments.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.keck_p1_r0_baseline_audit import analyse_case


def _case(tmp_path, *, dimm=0.74, sign=False, l0=31.0, airmass=1.358):
    folder = tmp_path/"n0120"
    folder.mkdir(exist_ok=True)
    r0, wl = 0.11474041, 0.5e-6
    rad2arc = 180*3600/np.pi
    seeing = 0.976*rad2arc*wl/r0
    atm = {
        "r0_tel_m_at_atmosphere_wavelength": r0,
        "atmosphere_wavelength_m": wl,
        "seeing_tel_arcsec": seeing,
        "seeing_dimm_arcsec": dimm,
        "airmass_nirc2": airmass,
        "dimm_seeing_reference_assumption": "UNKNOWN; conditional zenith only",
        "fitted_L0_m": l0,
        "r0_fit_uncertainty_m": 0.005,
        "fitted_L0_uncertainty_m": 8.0,
        "r0_fit_success": True,
        "r0_fit_cost": 0.05,
        "r0_fit_nfev": 42,
        "r0_fit_active_parameter_bounds": [0, 0],
        "r0_fit_L0_at_upper_bound": l0 >= 100,
        "pseudo_open_loop": {
            "delay_frames": 1.6,
            "frames_used": 999,
            "frames_dropped": 2,
            "method": "future_residual_linear_no_wrap",
            "sign_independently_verified": sign,
        },
    }
    wl_ref = 1.53e-6
    r0_expected = r0*(wl_ref/wl)**(6/5)
    freq = {
        "rescaled_r0_m": r0_expected,
        "r0_expected_wfs_los_m": r0_expected,
        "seeing_reference": "line_of_sight",
    }
    (folder/"atmosphere_reconstruction_audit.json").write_text(
        json.dumps(atm), encoding="utf-8"
    )
    (folder/"frequency_grid_diagnostics.json").write_text(
        json.dumps(freq), encoding="utf-8"
    )
    return folder, atm, freq


def test_archive_contract_preserves_line_of_sight_and_sees_unverified_rtc_sign(tmp_path):
    case, _, _ = _case(tmp_path)
    d = analyse_case(case)
    assert d["r0_wfs_los_m"] == pytest.approx(0.11474041)
    assert d["L0_wfs_fit_m"] == pytest.approx(31.0)
    assert d["r0_forward_ratio_to_expected"] == pytest.approx(1.0)
    assert d["seeing_relative_error"] == pytest.approx(0.0, abs=1e-15)
    assert d["r0_wfs_over_dimm_direct"] != pytest.approx(
        d["r0_wfs_over_dimm_if_zenith_los"], rel=1e-3
    )
    assert d["dimm_reference_independently_verified"] is False
    assert "PSEUDO_OL_RTC_SIGN_UNVERIFIED" in d["quality_flags"]
    assert "FORWARD_R0_LINE_OF_SIGHT_RESCALE_MISMATCH" not in d["quality_flags"]


def test_fit_bounds_are_reported_without_reinterpreting_dimm(tmp_path):
    case, a, f = _case(tmp_path, sign=True, l0=100)
    a["r0_fit_active_parameter_bounds"] = [0, 1]
    f["rescaled_r0_m"] *= 0.832
    (case/"atmosphere_reconstruction_audit.json").write_text(
        json.dumps(a), encoding="utf-8"
    )
    (case/"frequency_grid_diagnostics.json").write_text(
        json.dumps(f), encoding="utf-8"
    )
    flags = analyse_case(case)["quality_flags"]
    assert "PSEUDO_OL_RTC_SIGN_UNVERIFIED" not in flags
    assert "L0_AT_FIT_UPPER_BOUND" in flags
    assert "ATMOSPHERE_FIT_ACTIVE_BOUND" in flags
    assert "FORWARD_R0_LINE_OF_SIGHT_RESCALE_MISMATCH" in flags


def test_missing_dimm_keeps_undetermined_site_reference(tmp_path):
    case, _, _ = _case(tmp_path, dimm=None)
    d = analyse_case(case)
    assert "DIMM_SEEING_UNAVAILABLE" in d["quality_flags"]
    assert d["r0_dimm_direct_m"] is None
    assert d["r0_dimm_if_zenith_los_m"] is None
    assert d["r0_wfs_over_dimm_direct"] is None
    assert d["r0_wfs_over_dimm_if_zenith_los"] is None


def test_invalid_physical_fits_fail_explicitly(tmp_path):
    case, a, _ = _case(tmp_path)
    a["r0_tel_m_at_atmosphere_wavelength"] = -1.0
    (case/"atmosphere_reconstruction_audit.json").write_text(
        json.dumps(a), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="WFS LOS r0"):
        analyse_case(case)


def test_p1_archive_requires_real_named_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="atmosphere_reconstruction"):
        analyse_case(tmp_path)
