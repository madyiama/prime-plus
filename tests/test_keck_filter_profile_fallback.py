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
    assert not meta["calibrated_transmission_curve_available"]


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
        stream.write("#Wavelength Transmission\\n")
        for w, t in zip(wave, trans_pct):
            stream.write(f"{w:.8f} {t:.8f}\\n")
    wavelengths, bw, transmission, dispersion, meta = (
        keckUtils.samplingFilterProfile(
            str(tmp_path), hdr, nWvl=3, return_metadata=True
        )
    )
    assert meta["source"] == "local_transmission_table"
    assert not meta["flat_transmission_approximation"]
    assert meta["calibrated_transmission_curve_available"]
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
