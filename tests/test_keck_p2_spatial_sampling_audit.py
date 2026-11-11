"""P2: physical OTF lags, native Flicker support, and Kolmogorov controls.

These are synthetic deterministic tests, no Keck science data necessary.
For the *production* method, check the raw native map followed by
interpolateSupport remains precisely the legacy focal_angular_function
output. The new physically sampled method is audit-only.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.FourierUtils import interpolateSupport
from p3.aoSystem.anisoplanatismModel import focal_angular_function
from scripts.keck_p2_spatial_sampling_audit import (
    native_lag_axis,
    otf_lag_axis,
    resample_native_to_otf,
    relative_error_on_otf_support,
    synthetic_spatial_calibration,
)


def _circular_diffraction_otf(n=128, diameter=10.5, samp=3.0231069111135773):
    """Continuous ideal circular pupil autocorrelation, OTF(0)=1."""
    lag = otf_lag_axis(n, diameter, samp)
    yy, xx = np.meshgrid(lag, lag, indexing="ij")
    z = np.minimum(np.hypot(xx, yy)/diameter, 1.0)
    otf = (2/np.pi) * (
        np.arccos(z) - z*np.sqrt(np.maximum(1-z*z, 0))
    )
    return np.maximum(otf, 0)


def test_n0120_science_otf_lag_m_matches_camera_pixel_scale():
    D = 10.5
    N = 128
    samp = 3.0231069111135773
    pixel_rad = 9.942e-3/206264.806247
    wavelength = 1.53e-6
    axis = otf_lag_axis(N, D, samp)
    assert axis[N//2] == 0
    assert axis[N//2+1] == pytest.approx(
        wavelength/(N*pixel_rad), rel=5e-8
    )


def test_legacy_grid_has_half_the_reconstruction_lag_span():
    D = 10.5
    samp = 3.0231069111135773
    n_actu = 21
    n_otf = 128
    legacy = native_lag_axis(n_actu, D, samp, "legacy")
    physical = native_lag_axis(n_actu, D, samp, "physical_diameter")
    target = otf_lag_axis(n_otf, D, samp)
    assert len(legacy) == 85
    assert legacy[42] == 0
    assert physical[42] == 0
    assert physical.max() == pytest.approx(D)
    assert legacy.max()/abs(target.min()) == pytest.approx(0.48837209302)
    assert physical.max() > legacy.max()


def test_stationary_powerlaw_exposes_legacy_index_zoom_error():
    g = {
        "n_actu": 21,
        "diameter_m": 10.5,
        "samp_ref": 3.0231069111135773,
        "otf_size": 128,
        "otf": _circular_diffraction_otf(),
    }
    q = synthetic_spatial_calibration(g)
    assert q["legacy"]["legacy_index_zoom_relative_rms_error"] > 0.2
    assert (
        q["physical_diameter"]["physical_interpolation_relative_rms_error"]
        < 0.02
    )
    assert q["physical_diameter"][
        "otf_weight_outside_native_support"
    ] < 1e-8


def test_physical_axis_resampling_of_stationary_powerlaw():
    D, samp, n_actu, n_otf = 10.5, 3.0, 21, 128
    source_axis = native_lag_axis(n_actu, D, samp, "physical_diameter")
    target_axis = otf_lag_axis(n_otf, D, samp)
    y, x = np.meshgrid(source_axis, source_axis, indexing="ij")
    src = np.hypot(x, y)**(5/3)
    dst, outside = resample_native_to_otf(
        src, source_axis, target_axis
    )
    yy, xx = np.meshgrid(target_axis, target_axis, indexing="ij")
    exact = np.hypot(xx, yy)**(5/3)
    otf = _circular_diffraction_otf(samp=samp)
    assert relative_error_on_otf_support(dst, exact, otf) < 0.02
    assert np.sum(np.maximum(otf, 0)[outside]) == pytest.approx(0, abs=1e-10)
    assert dst[n_otf//2, n_otf//2] == pytest.approx(0)


def test_native_flicker_mode_preserves_original_legacy_result():
    tel = NS(D=10.5)
    atm = NS(L0=31.045692, heights=np.array([8500.0]), nL=1)
    src = NS(nSrc=1, direction=[np.zeros(1), np.zeros(1)])
    lgs = NS(height=np.array([120_000.0]), direction=[
        np.zeros(1), np.zeros(1)
    ])
    n_otf, samp, n_actu = 24, 2.0, 1
    old = focal_angular_function(
        tel, atm, src, lgs, [n_actu], n_otf, samp
    )
    native = focal_angular_function(
        tel, atm, src, lgs, [n_actu], n_otf, samp,
        output_grid="native", pupil_grid_mode="legacy",
    )
    assert old.shape == (1, 1, n_otf, n_otf)
    assert native.shape == (1, 1, 5, 5)
    np.testing.assert_allclose(
        old[0, 0], interpolateSupport(native[0, 0], n_otf),
        rtol=3e-12, atol=1e-12,
    )


def test_legacy_and_physical_pupil_modes_change_native_geometry():
    tel = NS(D=10.5)
    atm = NS(L0=np.inf, heights=np.array([8500.0]), nL=1)
    src = NS(nSrc=1, direction=[np.zeros(1), np.zeros(1)])
    lgs = NS(height=np.array([120_000.0]), direction=[
        np.zeros(1), np.zeros(1)
    ])
    inputs = (tel, atm, src, lgs, [1], 24, 2.0)
    legacy = focal_angular_function(
        *inputs, output_grid="native", pupil_grid_mode="legacy",
    )
    physical = focal_angular_function(
        *inputs, output_grid="native", pupil_grid_mode="physical_diameter",
    )
    assert legacy.shape == physical.shape
    assert np.all(np.isfinite(legacy))
    assert np.all(np.isfinite(physical))
    assert np.linalg.norm(physical-legacy) > 0
    with pytest.raises(ValueError, match="pupil_grid_mode"):
        focal_angular_function(*inputs, pupil_grid_mode="bogus")
    with pytest.raises(ValueError, match="output_grid"):
        focal_angular_function(*inputs, output_grid="bogus")
