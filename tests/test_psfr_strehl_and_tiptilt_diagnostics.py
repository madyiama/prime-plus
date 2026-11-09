"""Regression tests for tip-tilt diagnostics and polychromatic Strehl inputs."""

import numpy as np
import pytest

from p3.psfr.psfR import _integrate_spectral_psf, _sqrt_checked_variance


def test_integrate_spectral_cube_preserves_flux():
    psf = np.zeros((16, 16, 3), dtype=float)
    psf[8, 8, :] = [1.0, 2.0, 3.0]
    integrated = _integrate_spectral_psf(psf)
    assert integrated.shape == (16, 16)
    assert np.isclose(integrated[8, 8], 6.0)
    assert np.isclose(integrated.sum(), psf.sum())


def test_integrate_monochromatic_psf_returns_2d_image():
    psf = np.ones((16, 16), dtype=float)
    assert _integrate_spectral_psf(psf).shape == (16, 16)


def test_negative_noise_subtracted_variance_remains_unknown():
    with pytest.warns(RuntimeWarning, match="invalid noise-subtracted variance"):
        assert np.isnan(_sqrt_checked_variance(-1e-15, "TT test"))
    assert _sqrt_checked_variance(4.0, "TT test") == 2.0
