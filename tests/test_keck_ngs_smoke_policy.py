"""Keck NGS smoke CLI policy; no FITS, SAV, or optical fitting required."""

import pytest

from scripts.keck_n0013_smoke import select_tt_jitter_mode


@pytest.mark.parametrize(
    ("ao_mode", "requested", "expected"),
    [
        ("LGS", "spectral_closed_loop", "spectral_closed_loop"),
        ("LGS", "legacy_psfr", "legacy_psfr"),
        ("NGS", "legacy_psfr", "legacy_psfr"),
    ],
)
def test_supported_explicit_tt_modes_remain_unchanged(ao_mode, requested, expected):
    actual, note = select_tt_jitter_mode(
        ao_mode, requested, explicitly_requested=True,
    )
    assert actual == expected
    assert note is None


def test_ngs_spectral_from_configuration_falls_back_with_visible_warning():
    actual, note = select_tt_jitter_mode(
        "NGS", "spectral_closed_loop", explicitly_requested=False,
    )
    assert actual == "legacy_psfr"
    assert "uncalibrated" in note
    assert "NGS" in note


def test_explicit_ngs_spectral_rejected_with_actionable_message():
    with pytest.raises(ValueError, match="requires Keck LGS/STRAP"):
        select_tt_jitter_mode(
            "NGS", "spectral_closed_loop", explicitly_requested=True,
        )
