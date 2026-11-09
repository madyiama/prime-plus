"""Test-first PSF-R spatial-physics invariants (no Keck data, no image fitting).

The tests are deliberately limited to mathematical/numerical contracts.
They do not certify the instrument-specific DM fitting/anisoplanatism model.
"""
from types import SimpleNamespace as NS

import numpy as np

from p3.psfr.psfR import psfR
from p3.psfr.psfrUtils import modes2Otf


def test_zero_modal_covariance_returns_diffraction_otf_and_zero_sf():
    """An exactly noise-free modal residual cannot create phase blur."""
    n_pupil = 8
    yy, xx = np.mgrid[-1:1:complex(0, n_pupil),
                       -1:1:complex(0, n_pupil)]
    pupil = (xx**2 + yy**2 <= 0.8).astype(float)
    modes = np.column_stack(
        (xx.ravel(), yy.ravel(), (xx * yy).ravel())
    )
    covariance = np.zeros((3, 3))

    otf, dphi = modes2Otf(
        covariance, modes, pupil, n_otf=16, samp=2, basis="Vii"
    )
    assert otf.shape == dphi.shape == (16, 16)
    assert np.all(np.isfinite(otf))
    assert np.all(np.isfinite(dphi))
    np.testing.assert_allclose(dphi, 0.0, atol=1e-14)
    np.testing.assert_allclose(np.max(otf), 1.0, atol=1e-12)


def test_dm_fitting_sf_scales_as_r0_to_minus_five_thirds():
    """At fixed geometry and L0, the normalized fitting SF has known scaling."""
    n = 32
    k = np.fft.fftshift(np.fft.fftfreq(n))
    kx, ky = np.meshgrid(k, k)
    # Positive even PSD. Its normalization is arbitrary because this test
    # checks the theoretically exact r0 exponent, not an instrument coefficient.
    psd = (1.0 + 30.0 * (kx**2 + ky**2))**(-11.0 / 6.0)
    psd[n // 2, n // 2] = 0.0

    model = psfR.__new__(psfR)
    model.ao = NS(tel=NS(D=10.5))
    model.freq = NS(psdKolmo_=psd, sampRef=2.0)

    r0_lo, r0_hi = 0.20, 0.40
    dphi_lo = model.fitting_phase_structure_function(r0_lo)
    dphi_hi = model.fitting_phase_structure_function(r0_hi)

    assert dphi_lo.shape == (n, n)
    assert np.all(np.isfinite(dphi_lo))
    np.testing.assert_allclose(
        dphi_lo, dphi_hi * (r0_hi / r0_lo) ** (5.0 / 3.0),
        rtol=1e-12, atol=1e-11,
    )
    # Structure-function zero lag is the middle cell after fftshift.
    assert abs(dphi_lo[n // 2, n // 2]) < 1e-10
    assert np.max(dphi_lo) > 0
