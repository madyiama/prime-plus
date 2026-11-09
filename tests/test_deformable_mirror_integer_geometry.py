"""Regression tests for integer-valued DM geometry."""

import numpy as np
import pytest

from p3.aoSystem.deformableMirror import deformableMirror


@pytest.mark.parametrize("precision", ["single", "double"])
def test_actuator_count_is_integer_independent_of_precision(precision):
    dm = deformableMirror(
        nActu1D=[21],
        pitch=[0.5625],
        heights=[0.0],
        mechCoupling=[0.11],
        precision=precision,
    )
    assert np.issubdtype(dm.nActu1D.dtype, np.integer)
    assert dm.nActu1D.tolist() == [21]
    # The integer count must be suitable for np.linspace's num parameter.
    assert np.linspace(-1, 1, dm.nActu1D[0]).size == 21
