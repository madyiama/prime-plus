"""Normalization must not alter relative spectral fluxes or hide errors."""
import numpy as np
import pytest

from p3.psfr.products import normalize_psf


def test_single_wavelength_unit_flux():
    array = np.zeros((5, 5))
    array[2, 2] = 3
    result = normalize_psf(array)
    assert result.cube.shape == (5, 5, 1)
    assert result.image.sum() == pytest.approx(1.0)
    assert result.image[2, 2] == pytest.approx(1.0)
    np.testing.assert_allclose(result.spectral_fractions, [1.0])


def test_spectral_weights_not_applied_twice():
    array = np.zeros((5, 5, 3))
    array[2, 2, :] = [2, 3, 5]
    original = array.copy()
    result = normalize_psf(array)
    assert result.image.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(result.spectral_fractions, [0.2, 0.3, 0.5])
    np.testing.assert_array_equal(array, original)


def test_single_source_from_4d_and_explicit_multi_source():
    four_dimensional = np.ones((4, 4, 1, 2))
    result = normalize_psf(four_dimensional)
    assert result.cube.shape == (4, 4, 2)
    multiple = np.ones((4, 4, 2, 2))
    with pytest.raises(ValueError, match="source_index"):
        normalize_psf(multiple)
    assert normalize_psf(multiple, source_index=1).image.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="source_index"):
        normalize_psf(multiple, source_index=7)


@pytest.mark.parametrize("invalid", [
    np.zeros((4, 4)),
    np.array([[1, -1], [1, 1]], dtype=float),
    np.array([[1, np.nan], [1, 1]], dtype=float),
    np.array([[1, np.inf], [1, 1]], dtype=float),
])
def test_invalid_physics_fails_loudly(invalid):
    with pytest.raises(ValueError):
        normalize_psf(invalid)


def test_invalid_shape_or_source_index_rejected():
    with pytest.raises(ValueError, match="2-D, 3-D, or 4-D"):
        normalize_psf(np.ones((2,)))
    with pytest.raises(ValueError, match="only valid for 4-D"):
        normalize_psf(np.ones((4, 4, 2)), source_index=0)
