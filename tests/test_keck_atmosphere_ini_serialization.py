"""Regression: MASS-DIMM 1/3/7-layer vectors must survive INI IO as floats."""

import ast
from configparser import ConfigParser

import numpy as np
import pytest

from p3.telemetry.configFile import _format_numeric_vector
from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.FourierUtils import eqLayers


MASS_WEIGHTS = np.array([
    0.4837829375341058, 2.8954981850976237e-11,
    7.28398762188621e-11, 4.0265521636513835e-06,
    0.044337315959307365, 0.4071794322793533,
    0.06469628757327503,
], dtype=np.float64)
HEIGHTS = np.array([0, 500, 1000, 2000, 4000, 8000, 16000], dtype=float)
WIND = np.array([6.8, 6.9, 7.1, 7.5, 10.0, 26.9, 18.5], dtype=float)


@pytest.mark.parametrize("n_layers", [1, 3, 7])
def test_mass_dimm_compressed_arrays_roundtrip_through_configparser(n_layers):
    if n_layers < 7:
        weights, heights = eqLayers(
            MASS_WEIGHTS, HEIGHTS, n_layers, dtype=np.float64
        )
        _, wind = eqLayers(
            MASS_WEIGHTS, WIND, n_layers, dtype=np.float64
        )
    else:
        weights, heights, wind = MASS_WEIGHTS, HEIGHTS, WIND

    parser = ConfigParser()
    parser.add_section("atmosphere")
    for label, values in {
        "Cn2Weights": weights,
        "Cn2Heights": heights,
        "WindSpeed": wind,
        "WindDirection": np.zeros(n_layers, dtype=np.float32),
    }.items():
        encoded = _format_numeric_vector(values, label)
        assert "np.float" not in encoded
        parser.set("atmosphere", label, encoded)

    recovered = {}
    for label in ("Cn2Weights", "Cn2Heights", "WindSpeed", "WindDirection"):
        raw = parser.get("atmosphere", label)
        # Exact same reader used by the Keck aoSystem; unlike the original
        # str(np.ndarray), this must be a literal list with commas.
        parsed = aoSystem._parse_config_value(None, raw)
        assert isinstance(parsed, list)
        assert all(isinstance(v, float) for v in parsed)
        assert parsed == ast.literal_eval(raw)
        recovered[label] = np.array(parsed)

    assert len(recovered["Cn2Weights"]) == n_layers
    assert np.sum(recovered["Cn2Weights"]) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(recovered["Cn2Weights"], weights, rtol=0, atol=0)
    np.testing.assert_allclose(recovered["Cn2Heights"], heights, rtol=0, atol=0)
    np.testing.assert_allclose(recovered["WindSpeed"], wind, rtol=0, atol=0)


def test_three_layer_numpy_str_was_not_valid_ini_literal():
    old = str(np.array([0.48, 0.05, 0.47], dtype=float))
    assert "," not in old
    assert isinstance(aoSystem._parse_config_value(None, old), str)
    corrected = _format_numeric_vector(
        np.array([0.48, 0.05, 0.47], dtype=float), "Cn2Weights"
    )
    assert aoSystem._parse_config_value(None, corrected) == [0.48, 0.05, 0.47]


@pytest.mark.parametrize("values", [
    np.array([0.5, np.nan, 0.5]),
    np.array([0.5, np.inf, 0.5]),
    np.array([[0.5, 0.5]]),
    [],
    ["0.5", "bad"],
])
def test_atmosphere_ini_writer_rejects_non_finite_or_malformed_vectors(values):
    with pytest.raises(ValueError):
        _format_numeric_vector(values, "Cn2Weights")
