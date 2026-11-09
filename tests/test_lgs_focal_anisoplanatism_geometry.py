"""Regression coverage for LGS focal anisoplanatism actuator geometry."""

from types import SimpleNamespace

import numpy as np
import pytest

from p3.aoSystem.anisoplanatismModel import focal_angular_function


@pytest.mark.parametrize("actuators", [1, [1], np.array([1]), np.array([1.0])])
def test_focal_anisoplanatism_accepts_one_dm_actuator_count(actuators):
    """A one-element nActu1D array must work as well as a scalar."""
    telescope = SimpleNamespace(D=10.5)
    atmosphere = SimpleNamespace(L0=30.0, nL=1, heights=np.array([0.0]))
    source = SimpleNamespace(nSrc=1, direction=np.zeros((2, 1)))
    lgs = SimpleNamespace(height=90000.0, direction=np.zeros((2, 1)))

    result = focal_angular_function(
        telescope, atmosphere, source, lgs,
        nActu=actuators, nOtf=8, samp=2.0,
    )
    assert result.shape == (1, 1, 8, 8)
    assert np.all(np.isfinite(result))


def test_focal_anisoplanatism_rejects_ambiguous_multi_dm_geometry():
    telescope = SimpleNamespace(D=10.5)
    atmosphere = SimpleNamespace(L0=30.0, nL=1, heights=np.array([0.0]))
    source = SimpleNamespace(nSrc=1, direction=np.zeros((2, 1)))
    lgs = SimpleNamespace(height=90000.0, direction=np.zeros((2, 1)))

    with pytest.raises(ValueError, match="single DM"):
        focal_angular_function(
            telescope, atmosphere, source, lgs,
            nActu=np.array([21, 21]), nOtf=8, samp=2.0,
        )
