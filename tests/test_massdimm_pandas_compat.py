"""Regression tests for Keck MASS/DIMM data parsing with modern pandas."""

from pathlib import Path

import numpy as np
import pytest

from p3.telemetry.massdimm import DIMM, MASS, MASSPROF


DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "p3" / "aoSystem" / "data" / "KECK_CALIBRATION"
    / "MASSDIMM" / "20201202"
)


@pytest.mark.parametrize(
    ("reader", "suffix", "field"),
    [
        (DIMM, "dimm", "seeing"),
        (MASS, "mass", "free_seeing"),
        (MASSPROF, "masspro", "profs"),
    ],
)
def test_keck_massdimm_whitespace_files_parse(reader, suffix, field):
    """Read the actual space-delimited Keck calibration files from Dec 2020."""
    item = reader(str(DATA_DIR / f"20201202.{suffix}.dat"))
    assert len(item.year) > 0
    assert len(item.timeInHours) == len(item.year)
    assert np.all(np.isfinite(item.timeInHours))
    values = np.asarray(getattr(item, field))
    assert values.shape[0] == len(item.year)
    assert np.all(np.isfinite(values))
    if suffix == "masspro":
        assert values.shape[1] == 6

