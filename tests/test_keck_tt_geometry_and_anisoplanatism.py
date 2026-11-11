"""Tests for Keck STRAP geometry, physical jitter OTF, and LGS anisoplanatism."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.tipTiltGeometry import (
    ARCSEC_TO_RAD,
    angular_jitter_structure_function,
    opd_covariance_to_angular,
    project_covariance_to_psd,
    zernike_tilt_opd_per_arcsec,
)
from p3.psfr.psfR import psfR


def test_rms_zernike_tilt_scale_matches_keck_geometry():
    # Z2 = 2 rho cos(phi), coefficient a = D * theta / 4.
    expected = 10.5 * ARCSEC_TO_RAD / 4
    assert zernike_tilt_opd_per_arcsec(10.5) == pytest.approx(expected)
    assert expected == pytest.approx(1.2726359129e-5, rel=1e-9)


def test_angular_otf_matches_analytic_gaussian_convolution():
    diameter = 10.5
    wavelength = 2.1e-6
    sigma_x_rad = 40e-3 * ARCSEC_TO_RAD
    sigma_y_rad = 60e-3 * ARCSEC_TO_RAD
    angular_cov = np.diag([sigma_x_rad**2, sigma_y_rad**2])
    opd_cov = angular_cov * (diameter / 4) ** 2
    np.testing.assert_allclose(
        opd_covariance_to_angular(opd_cov, diameter),
        angular_cov,
        rtol=1e-13,
    )
    u, v = np.meshgrid(np.linspace(-0.8, 0.8, 9), np.linspace(-0.8, 0.8, 9))
    dphi = angular_jitter_structure_function(
        angular_cov, diameter, wavelength, u, v
    )
    # f_ang = D/lambda * (u,v). Gaussian OTF = exp(-2pi^2 f C f).
    expected_otf = np.exp(-2 * np.pi**2 * (diameter/wavelength)**2 *
                          (sigma_x_rad**2*u**2 + sigma_y_rad**2*v**2))
    np.testing.assert_allclose(np.exp(-0.5*dphi), expected_otf, rtol=1e-13)
    assert np.all(dphi >= -1e-12)


def test_non_psd_covariance_warns_and_projection_is_positive():
    raw = np.diag([-1e-14, 2e-14])
    with pytest.warns(RuntimeWarning, match="not positive-semidefinite"):
        projected, eigenvalues = project_covariance_to_psd(raw, "test TT")
    assert eigenvalues[0] < 0
    np.testing.assert_allclose(projected, np.diag([0.0, 2e-14]), atol=1e-27)
    assert np.linalg.eigvalsh(projected).min() >= -1e-28


def test_psfr_with_excessive_noise_has_physical_jitter_otf_and_diagnostics():
    # Raw per-axis 0.5e-12 m²; noise with pn=0.1 over-subtracts it.
    samples = np.array([
        [1e-6, 0.0], [-1e-6, 0.0], [0.0, 1e-6], [0.0, -1e-6],
    ])
    model = psfR.__new__(psfR)
    # __new__ bypasses __init__: explicitly select the historical covariance
    # path that this non-PSD projection regression is meant to exercise.
    model.tt_covariance_mode = 'legacy_psfr'
    model.trs = NS(
        tipTilt=NS(
            slopes=samples, nExp=len(samples),
            Cn_tt=np.eye(2) * 5e-13,
            command_units_verified=False,
        ),
        ttloop=NS(tf=NS(pn=0.1)),
        tel=NS(pupilAngle=0.0),
    )
    u, v = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    model.freq = NS(wvlRef=2.1e-6, sampRef=2.0, U_=u, V_=v)
    model.ao = NS(tel=NS(D=10.5))

    with pytest.warns(RuntimeWarning, match="not positive-semidefinite"):
        sf = model.jitter_phase_structure_function()
    assert sf.shape == u.shape
    assert np.all(np.isfinite(sf))
    assert np.min(sf) >= -1e-12
    assert np.max(np.exp(-0.5*sf)) <= 1.0 + 1e-12
    assert model.tiptilt_diagnostics["corrected_trace_m2"] < 0
    assert model.tiptilt_diagnostics["negative_variance_projected"]
    assert not model.tiptilt_diagnostics["noise_model_validated"]
    assert np.linalg.eigvalsh(model.tt_model_cov).min() >= -1e-25


def _psfr_stub(aniso, layers, layer_kernels):
    size = 4
    model = psfR.__new__(psfR)
    model.freq = NS(isAniso=aniso, wvlRef=2e-6)
    model.ao = NS(
        atm=NS(
            nL=layers, wvl=500e-9,
            weights=np.ones(layers) / layers,
        ),
        src=NS(nSrc=1),
    )
    model.dphi_ao = np.zeros((size, size))
    model.dphi_tt = np.zeros((size, size))
    model.dphi_fit = np.zeros((size, size))
    model.dphi_alias = np.zeros((size, size))
    model.dphi_ani = np.asarray(layer_kernels).reshape(1, layers, size, size)
    return model


def test_one_layer_lgs_anisoplanatism_is_included_in_forward_model():
    model = _psfr_stub(True, 1, np.ones((1, 4, 4)) * 10)
    r0 = 0.2
    sf = model.phase_structure_function(r0, [1, 1], Cn2=None)
    expected = 10 * r0**(-5/3) * (500e-9/2e-6)**2
    assert sf.shape == (4, 4, 1)
    np.testing.assert_allclose(sf, expected)
    assert np.all(sf > 0)


def test_multilayer_anisoplanatism_in_science_wavelength_units():
    raw = np.array([
        np.full((4, 4), 2.0),
        np.full((4, 4), 6.0),
    ])
    model = _psfr_stub(True, 2, raw)
    sf = model.phase_structure_function(0.2, [1, 1], Cn2=[2.0, 5.0])
    np.testing.assert_allclose(sf, (2*2 + 6*5) * (500e-9/2e-6)**2)


def test_no_anisoplanatism_returns_base_sf_only():
    model = _psfr_stub(False, 1, np.ones((1, 4, 4)) * 10)
    sf = model.phase_structure_function(0.2, [1, 1], Cn2=None)
    np.testing.assert_array_equal(sf, np.zeros((4, 4, 1)))


def test_jitter_otf_frequency_grid_has_correct_camera_pixel_scale_at_any_wavelength():
    """Independent angular-FFT reference checks spatial grid and lambda cancellation.

    For a fixed on-sky Gaussian jitter, its OTF sampled at a fixed camera
    plate scale cannot change merely because the reference wavelength changes.
    Assumes psdExpansion=False (standard Keck PSF-R), kRef integer.
    """
    from p3.aoSystem.FourierUtils import instantiateAngularFrequencies

    diameter = 10.5
    pixel_scale_mas = 9.942
    mas_to_rad = ARCSEC_TO_RAD / 1000.0
    plate_scale_rad = pixel_scale_mas * mas_to_rad
    sigma_x_rad = 24.0 * mas_to_rad
    sigma_y_rad = 39.0 * mas_to_rad
    angular_cov = np.diag([sigma_x_rad**2, sigma_y_rad**2])

    n_pix = 128
    for k_ref in (1, 2):
        n_otf = n_pix * k_ref
        u, v, *_ = instantiateAngularFrequencies(n_otf, fact=2)
        ix, iy = np.mgrid[0:n_otf, 0:n_otf]
        qx = ix - n_otf // 2
        qy = iy - n_otf // 2

        # Direct DFT characteristic function on the camera's pixel grid:
        # its exponent depends only on jitter measured in camera pixels.
        expected = np.exp(-2*np.pi**2 * (
            (sigma_x_rad/plate_scale_rad)**2 * (qx/n_pix)**2
            + (sigma_y_rad/plate_scale_rad)**2 * (qy/n_pix)**2
        ))

        for wavelength in (1.65e-6, 2.055501e-6, 2.35e-6):
            # frequencyDomain standard (psdExpansion=False) definition:
            sampling = k_ref * wavelength / (diameter * plate_scale_rad)
            dphi = angular_jitter_structure_function(
                angular_cov, diameter, wavelength,
                sampling/2 * u, sampling/2 * v,
            )
            np.testing.assert_allclose(
                np.exp(-0.5*dphi), expected, atol=1e-14, rtol=1e-12,
            )
            assert dphi[n_otf//2, n_otf//2] == pytest.approx(0.0)
