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


def test_dm_fitting_sf_matches_independent_psd_quadrature_at_one_lag():
    """Independent cosine-integral check, not a repeat of the FFT algorithm."""
    n = 32
    diameter_m = 10.5
    sampling = 2.0
    dk = 1.0 / (diameter_m * sampling)
    k = (np.arange(n) - n // 2) * dk
    kx, ky = np.meshgrid(k, k)
    outer_scale_m = 25.0
    cutoff_inv_m = 0.35
    base_psd = 0.0229 * (kx**2 + ky**2 + outer_scale_m**-2)**(-11.0/6.0)
    base_psd *= (kx**2 + ky**2 >= cutoff_inv_m**2)
    # Discrete grid, with piston contribution omitted.
    base_psd[n // 2, n // 2] = 0.0

    model = psfR.__new__(psfR)
    model.ao = NS(tel=NS(D=diameter_m))
    model.freq = NS(psdKolmo_=base_psd, sampRef=sampling)
    r0_m = 0.20
    dphi = model.fitting_phase_structure_function(r0_m)

    rho_x_m = 1.0 / (n * dk)
    expected = 2 * dk**2 * r0_m**(-5.0/3.0) * np.sum(
        base_psd * (1.0 - np.cos(2 * np.pi * kx * rho_x_m))
    )
    assert expected > 0
    np.testing.assert_allclose(
        dphi[n // 2, n // 2 + 1], expected,
        rtol=1e-10, atol=1e-10,
    )


def test_modal_otf_invariant_under_orthogonal_basis_change():
    """Rotating modal coordinates must not change the physical covariance SF."""
    rng = np.random.default_rng(100)
    n_pupil = 8
    y, x = np.mgrid[-1:1:complex(0, n_pupil),
                      -1:1:complex(0, n_pupil)]
    pupil = ((x*x + y*y) <= 0.8).astype(float)
    modes = np.column_stack([x.ravel(), y.ravel(), (x*y).ravel()])
    # Positive-definite covariance with nonzero correlations.
    matrix = np.array([[0.7, 0.18, -0.06],
                       [0.18, 0.4, 0.11],
                       [-0.06, 0.11, 0.3]])
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated_modes = modes @ rotation
    rotated_covariance = rotation.T @ matrix @ rotation

    otf, dphi = modes2Otf(matrix, modes, pupil, n_otf=16, samp=2)
    otf_rot, dphi_rot = modes2Otf(
        rotated_covariance, rotated_modes, pupil, n_otf=16, samp=2
    )
    np.testing.assert_allclose(dphi, dphi_rot, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(otf, otf_rot, atol=1e-10, rtol=1e-10)


def test_modal_otf_rejects_substantially_negative_covariance_eigenvalue():
    import pytest

    x, y = np.meshgrid(np.linspace(-1, 1, 8),
                       np.linspace(-1, 1, 8))
    modes = np.column_stack([x.ravel(), y.ravel()])
    matrix = np.diag([1.0, -0.1])
    with pytest.raises(ValueError, match="positive-semidefinite"):
        modes2Otf(matrix, modes, np.ones((8, 8)), n_otf=16, samp=2)


def test_lgs_anisokinetism_cross_covariance_uses_xy_not_transposes(monkeypatch):
    """Off-diagonal tilt covariance must give a true XY mixed term."""
    from types import SimpleNamespace as NS
    import p3.aoSystem.anisoplanatismModel as anisomod

    y, x = np.meshgrid([-1., 0., 1.], [-1., 0., 1.])
    covariance = np.array([[2.0, 0.5], [0.5, 1.3]])

    class FakeZernike:
        def __init__(self, indices, n):
            assert indices == [2, 3] and n == 3
            self.modes = np.array([2*x, 2*y])

        def anisokinetism(self, *args):
            return covariance

    monkeypatch.setattr(anisomod, "zernike", FakeZernike)
    atm = NS(nL=1, heights=np.array([1000.]),
             slab=lambda layer: NS(r0=1.0))
    src = NS(direction=[np.array([1e-6]), np.array([0.])])
    gs = NS(direction=[np.array([0.]), np.array([0.])])
    result = anisomod.anisokinetism_function(
        NS(), atm, src, gs, nOtf=3, samp=2
    )
    expected = (covariance[0, 0] * x**2
                + 2 * covariance[0, 1] * x*y
                + covariance[1, 1] * y**2)
    np.testing.assert_allclose(result[0, 0], expected, atol=1e-12)


def test_atmosphere_reference_wavelength_change_preserves_all_layer_r0_scales():
    """The live PSF-R setter cannot leave individual layers at 500 nm."""
    from p3.aoSystem.atmosphere import atmosphere

    lambda_atm = 500e-9
    lambda_sci = 2.055501e-6
    original_r0 = 0.16
    atm = atmosphere(
        lambda_atm, original_r0,
        weights=[0.35, 0.65], heights=[0.0, 6000.0],
        wSpeed=[10., 20.], wDir=[0., 45.], L0=25.,
    )
    original_layers = np.array([atm.slab(i).r0 for i in range(atm.nL)])
    factor = (lambda_sci / lambda_atm)**(6/5)

    atm.wvl = lambda_sci
    np.testing.assert_allclose(atm.r0, original_r0 * factor, rtol=1e-14)
    np.testing.assert_allclose(
        [atm.slab(i).r0 for i in range(atm.nL)],
        original_layers * factor, rtol=1e-14,
    )
    np.testing.assert_allclose(
        sum(atm.slab(i).r0**(-5/3) for i in range(atm.nL)),
        atm.r0**(-5/3), rtol=1e-14,
    )

    atm.wvl = lambda_atm
    np.testing.assert_allclose(atm.r0, original_r0, rtol=1e-14)
    np.testing.assert_allclose(
        [atm.slab(i).r0 for i in range(atm.nL)],
        original_layers, rtol=1e-14,
    )
