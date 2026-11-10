"""Unified r0 API: legacy invariance and IDL-inspired Keck DM method.

Synthetic tests do not use a science PSF, DIMM, archived IDL SAV files,
or a telemetry file. On-sky correctness remains an explicit gate.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.telemetry.seeingEstimator import (
    _IDL_LGS_BAD,
    _fit_amplitude,
    _model_modal_variance,
    _phase_sf,
    estimate_r0,
    modified_dm_projection,
    spectral_command_noise_variance,
)


def test_legacy_api_reports_exact_existing_fit_and_does_not_mutate():
    trs = NS(
        atm=NS(r0_tel=0.1147404127, L0_tel=31.045692, wvl=5e-7),
        wfs=NS(jIndex=list(range(4, 265))),
    )
    sd = NS(trs=trs, noiseMethod="autocorrelation")
    result = estimate_r0(sd, method="zernike")
    assert result["r0_m"] == pytest.approx(0.1147404127)
    assert result["L0_m"] == pytest.approx(31.045692)
    assert result["noll_first"] == 4
    assert result["noll_last"] == 264
    assert trs.atm.r0_tel == pytest.approx(0.1147404127)
    with pytest.raises(ValueError, match="Unknown"):
        estimate_r0(sd, method="unknown")


def test_idl_2018_lgs_noll_exclusions_have_seventy_five_surviving_modes():
    modes = set(range(7, 121))
    good = modes - _IDL_LGS_BAD
    assert len(good) == 75
    assert {1, 2, 3}.isdisjoint(good)
    assert 7 not in good
    assert 120 not in good


def test_kolmogorov_von_karman_structure_function_limits():
    rho = np.linspace(0.0, 9.0, 19)
    infinite = _phase_sf(rho, 9.0, np.inf)
    assert infinite[0] == 0
    np.testing.assert_allclose(
        infinite, 6.88*(rho/9.0)**(5.0/3.0), rtol=1e-12,
    )
    finite = _phase_sf(rho, 9.0, 31)
    assert finite[0] == pytest.approx(0.0, abs=1e-13)
    assert np.all(finite[1:] < infinite[1:])
    assert np.all(finite >= 0)


def test_exact_phase_covariance_projection_is_positive_for_piston_free_modes():
    n = 11
    yy, xx = np.indices((n, n), dtype=float)
    x = xx-(n-1)/2
    y = yy-(n-1)/2
    q = np.stack([x, y, x*y], axis=0)
    q /= np.max(np.abs(q), axis=(1, 2))[:, None, None]
    q -= np.mean(q, axis=(1, 2), keepdims=True)
    ref = _model_modal_variance(q, 9.0, np.inf, 9.0)
    finite = _model_modal_variance(q, 9.0, 31.0, 9.0)
    assert np.all(ref > 0)
    assert np.all(finite > 0)
    assert np.all(finite < ref)


def test_model_amplitude_and_L0_grid_recover_synthetic_parameters():
    model = [
        np.array([1.0, 0.6, 0.2]),
        np.array([1.4, 0.8, 0.3]),
        np.array([2.0, 1.0, 0.4]),
    ]
    r0 = 0.15
    dia = 9.0
    amplitude = (dia/r0)**(5/3)
    y = amplitude*model[1]
    best, profile, diagnostics = _fit_amplitude(
        y, model, np.array([7, 8, 9]), np.array([3,3,3]),
        dia, [10., 30., 90.],
    )
    assert best["L0_m"] == pytest.approx(30)
    assert best["r0_m"] == pytest.approx(0.15, rel=1e-12)
    assert len(profile) == 3


def test_white_modal_command_noise_with_unity_transfer_is_recovered():
    fs = 1000.0
    rng = np.random.default_rng(44)
    sigma = np.array([0.02, 0.05, 0.08])
    commands = rng.standard_normal((2**15, 3))*sigma
    freq = np.linspace(0.0, fs/2, 200)
    transfer = np.ones_like(freq)
    recovered, diagnostics = spectral_command_noise_variance(
        commands, fs, freq, transfer
    )
    np.testing.assert_allclose(
        recovered, sigma**2, rtol=0.18
    )
    assert diagnostics["selected_frequency_bins"] > 10


def test_dm_annular_modified_projection_shapes_and_no_tt_or_piston():
    # Synthetic Gaussian influences, only geometrical API check.
    n = 25
    yy, xx = np.indices((n, n), dtype=float)
    coords = np.linspace(3, n-4, 9)
    influences = []
    for cy in coords:
        for cx in coords:
            influences.append(np.exp(
                -((xx-cx)**2+(yy-cy)**2)/3.0
            ))
    IF = np.stack(influences, axis=-1).reshape(n*n, len(influences))
    out = modified_dm_projection(
        IF, 10.5, n, j_start=7, j_stop=14,
        pinv_rcond=1e-3
    )
    assert out["phase_probes"].shape == (8, n, n)
    assert out["dm_to_modal"].shape[0] == 8
    assert out["actuator_indices"].size > 11
    assert out["annulus_pixel_count"] > 80
    np.testing.assert_allclose(
        out["phase_probes"].sum(axis=(1, 2)), 0, atol=1e-10
    )
    assert out["noll"][0] == 7
    assert out["noll"][-1] == 14


def test_LGS_method_rejects_missing_instrument_without_implicit_fallback():
    sd = NS(trs=NS(aoMode="NGS"))
    with pytest.raises(ValueError, match="requires Keck LGS"):
        estimate_r0(sd, method="jolissaint")



def test_keck_2018_nominal_valid_actuator_mask_is_exactly_320_with_correct_if_scale():
    # This fixture reproduces the geometry of Keck's 21x21 DM, where
    # 349 nominally valid actuators become 320 after excluding 5
    # central + 24 edge ones, independently of influence peak positions.
    n = 25
    n_dm = 21
    pitch = 0.5625
    xy = np.linspace(-10*pitch, 10*pitch, n)
    yy, xx = np.meshgrid(xy, xy, indexing="ij")
    dmxy = (np.arange(n_dm)-10)*pitch
    ay, ax = np.meshgrid(dmxy, dmxy, indexing="ij")
    actual_radius = np.hypot(ax, ay)
    valid = actual_radius <= 10.54*pitch
    assert np.sum(valid) == 349
    # Smooth nonzero influences deliberately overlap strongly.
    IF = np.exp(
        -(
            (xx.ravel()[:, None]-ax.ravel()[None, :])**2
            +(yy.ravel()[:, None]-ay.ravel()[None, :])**2
        )/(2*0.38**2)
    )
    out = modified_dm_projection(
        IF, 10.5, n, j_start=7, j_stop=14,
        dm_pitch_m=pitch, valid_actuators=valid,
    )
    assert len(out["actuator_indices"]) == 320
    assert out["valid_actuators_provided"] is True
    assert out["dm_influence_extent_m"] == pytest.approx(11.25)
    assert out["pupil_sample_pitch_m"] == pytest.approx(11.25/(n-1))
    assert out["pixels_per_dm_pitch"] == pytest.approx((n-1)/20)
    assert out["analysis_outer_diameter_m"] == pytest.approx(9.0)

    bad_valid = valid.copy()
    bad_valid[10, 13] = False
    with pytest.raises(ValueError, match="invalid DM actuators"):
        modified_dm_projection(
            IF, 10.5, n, j_start=7, j_stop=14,
            dm_pitch_m=pitch, valid_actuators=bad_valid,
        )


def test_physical_phase_structure_sampling_changes_reference_model():
    n = 11
    y, x = np.indices((n, n), dtype=float)
    probe = (x-(n-1)/2)[None, :, :]
    probe -= probe.mean(axis=(1, 2), keepdims=True)
    wrong = _model_modal_variance(
        probe, 9.0, 31.0, 10.5
    )
    correct = _model_modal_variance(
        probe, 9.0, 31.0, 10.5,
        pixel_pitch_m=11.25/(n-1)
    )
    assert correct[0] > wrong[0] > 0
    with pytest.raises(ValueError, match="grid pitch"):
        _model_modal_variance(
            probe, 9.0, 31.0, 10.5, pixel_pitch_m=0
        )



def test_log_variance_least_squares_uses_mean_not_median_log_ratio():
    # With a skewed log ratio distribution, the median is NOT the
    # minimizer of the mean squared logarithmic residual.
    scale = np.array([1., 1., 100.])
    base = np.array([1., 2., 3.])
    models = [base, base*1.2, base*1.5]
    best, profile, _ = _fit_amplitude(
        base*scale, models,
        np.array([9, 10, 12]), np.array([3,3,4]),
        9.0, [10., 20., 30.],
    )
    expected = np.exp(np.mean(np.log(scale)))
    assert profile[0]["amplitude_D_over_r0_to_5_3"] == pytest.approx(
        expected, rel=1e-12
    )


def test_dm_theory_uses_angular_physical_pitch_in_comparing_models():
    n = 11
    y, x = np.indices((n, n), dtype=float)
    probe = (x-(n-1)/2)[None, :, :]
    probe = probe-probe.mean()
    a = _model_modal_variance(
        probe, 9.0, np.inf, 10.5, pixel_pitch_m=0.10
    )
    b = _model_modal_variance(
        probe, 9.0, np.inf, 10.5, pixel_pitch_m=0.20
    )
    np.testing.assert_allclose(
        b/a, (0.20/0.10)**(5/3), rtol=1e-10
    )
