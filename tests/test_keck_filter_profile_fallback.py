"""NIRC2 filter-profile regression tests, including n0120 CH4_short headers."""

import numpy as np
import pytest

from p3.telemetry import keckUtils


def _n0120_header():
    # Values read from the actual 2020-12-02 n0120_LGS_trs.sav
    # embedded FITS-header cards. No science FITS is bundled in tests.
    return {
        "CURRINST": "NIRC2",
        "fwiname": "CH4_short",
        "fwoname": "PK50_1.5",
        "CENWAVE": 1.59250,
        "MINWAVE": 1.53000,
        "MAXWAVE": 1.65500,
    }


def test_n0120_unsupported_filter_uses_labelled_header_passband(tmp_path):
    hdr = _n0120_header()
    with pytest.warns(RuntimeWarning, match="UNCALIBRATED flat passband"):
        wavelengths, bw, transmission, dispersion, meta = (
            keckUtils.samplingFilterProfile(
                str(tmp_path), hdr, nWvl=3, return_metadata=True,
            )
        )
    np.testing.assert_allclose(
        wavelengths * 1e6, [1.53, 1.5925, 1.655], rtol=0, atol=1e-12
    )
    assert bw == pytest.approx(0.125e-6)
    np.testing.assert_array_equal(transmission, np.ones(3))
    assert len(dispersion) == 2 and all(len(axis) == 3 for axis in dispersion)
    assert meta["filter_name"] == "CH4_short"
    assert meta["source"] == "fits_header_flat_passband"
    assert meta["flat_transmission_approximation"] is True
    assert not meta["tabulated_transmission_curve_available"]


def test_missing_filter_without_header_bounds_fails_with_useful_error(tmp_path):
    hdr = _n0120_header()
    del hdr["MINWAVE"]
    with pytest.raises(ValueError, match="MINWAVE, CENWAVE or MAXWAVE"):
        keckUtils.samplingFilterProfile(str(tmp_path), hdr)


def test_nonfinite_or_invalid_header_bandpass_is_rejected(tmp_path):
    hdr = _n0120_header()
    hdr["MINWAVE"] = 1.68
    with pytest.raises(ValueError, match="invalid wavelength interval"):
        keckUtils.samplingFilterProfile(str(tmp_path), hdr)
    hdr["MINWAVE"] = float("nan")
    with pytest.raises(ValueError, match="invalid wavelength interval"):
        keckUtils.samplingFilterProfile(str(tmp_path), hdr)


def test_local_table_is_preferred_over_header_proxy(tmp_path):
    hdr = _n0120_header()
    wave = np.linspace(1.50, 1.70, 21)
    trans_pct = 80.0 * np.exp(-0.5 * ((wave-1.60) / 0.04)**2)
    profile_path = tmp_path / "CH4short.dat"
    with profile_path.open("w") as stream:
        stream.write("#Wavelength Transmission\n")
        for w, t in zip(wave, trans_pct):
            stream.write(f"{w:.8f} {t:.8f}\n")
    wavelengths, bw, transmission, dispersion, meta = (
        keckUtils.samplingFilterProfile(
            str(tmp_path), hdr, nWvl=3, bwMin=10e-9, return_metadata=True
        )
    )
    assert meta["source"] == "local_transmission_table"
    assert not meta["flat_transmission_approximation"]
    assert meta["tabulated_transmission_curve_available"]
    assert profile_path.name in meta["profile_file"]
    assert wavelengths.size == 3 and bw > 0
    assert np.all(transmission > 0)


def test_legacy_four_return_values_still_supported(tmp_path):
    hdr = _n0120_header()
    with pytest.warns(RuntimeWarning, match="UNCALIBRATED"):
        result = keckUtils.samplingFilterProfile(str(tmp_path), hdr)
    assert len(result) == 4


def test_primary_filter_selection_ignores_blocking_glass_and_clear():
    hdr = _n0120_header()
    assert keckUtils.getFilterName(hdr) == "CH4_short"
    hdr["fwiname"] = "PK50_1.5"
    hdr["fwoname"] = "Kp"
    assert keckUtils.getFilterName(hdr) == "Kp"
    hdr["fwiname"] = "clear"
    assert keckUtils.getFilterName(hdr) == "Kp"


def test_existing_nirc2_h_filter_uses_tabulated_profile():
    from pathlib import Path
    folder = (
        Path(__file__).resolve().parents[1]
        / "p3/aoSystem/data/KECK_CALIBRATION/NIRC2_FILTERS"
    )
    hdr = {
        "CURRINST": "NIRC2",
        "fwiname": "H",
        "fwoname": "PK50_1.5",
        "CENWAVE": 1.633,
        "MINWAVE": 1.485,
        "MAXWAVE": 1.781,
    }
    wavelengths, bw, throughput, _, info = keckUtils.samplingFilterProfile(
        str(folder), hdr, return_metadata=True
    )
    assert info["source"] == "local_transmission_table"
    assert not info["flat_transmission_approximation"]
    assert wavelengths.size == 3
    assert np.all(np.isfinite(throughput))
    assert np.all(throughput > 0)
    assert bw > 0


def test_missing_entire_filter_directory_is_not_silent_header_fallback(tmp_path):
    directory = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="calibration directory"):
        keckUtils.samplingFilterProfile(str(directory), _n0120_header())


def test_inactive_filter_wheels_cannot_fabricate_passband(tmp_path):
    hdr = _n0120_header()
    hdr["fwiname"] = "clear"
    hdr["fwoname"] = "PK50_1.5"
    with pytest.raises(ValueError, match="No active NIRC2 science passband"):
        keckUtils.samplingFilterProfile(str(tmp_path), hdr)


def test_official_ch4s_profile_alias_is_accepted(tmp_path):
    # Official NIRC2 documentation says CH4S, whereas FITS has CH4_short.
    curve = tmp_path / "CH4S.dat"
    curve.write_text(
        "#Wavelength Transmission\n"
        "1.530 50\n"
        "1.570 78\n"
        "1.592 80\n"
        "1.620 78\n"
        "1.655 50\n"
    )
    name = keckUtils._find_filter_profile_file(
        str(tmp_path), "CH4_short"
    )
    assert name == curve
