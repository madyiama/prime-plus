"""Keck quad-cell STRAP detector-noise regression tests (no FITS required)."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.tipTiltGeometry import zernike_tilt_opd_per_arcsec
from p3.telemetry.ttApdNoise import (
    QUADRANT_PATTERNS,
    apd_imat_matrix,
    apd_delta_method_covariance,
    apd_poisson_monte_carlo,
    normalized_quad_cell,
    reproduce_keck_centroids,
)
from p3.telemetry.systemDiagnosis import systemDiagnosis
from p3.telemetry.ttClosedLoop import from_p3_tiptilt_loop
from p3.telemetry.ttNoise import estimate_tt_noise


SKY = [74.04, 34.68, 41.504, 72.664]
IMAT = [0.1785, 0.842, 0.95, -0.2]


def _counts(n=384):
    rng = np.random.default_rng(55)
    raw = rng.poisson([130, 89, 98, 128], size=(n, 4)).astype(float)
    return raw


def test_2020_matrix_layout_and_centroid_reproduction():
    raw = _counts()
    quad, flux = normalized_quad_cell(raw, SKY)
    matrix = apd_imat_matrix(IMAT)
    np.testing.assert_allclose(
        matrix, [[-0.842, 0.1785], [0.2, 0.95]]
    )
    expected = quad @ matrix.T
    audit = reproduce_keck_centroids(raw, SKY, IMAT, expected)
    assert audit["centroid_reproduction_valid"]
    np.testing.assert_allclose(audit["centroid_rms_error_arcsec"], [0, 0])
    assert audit["mean_net_counts_per_frame"] > 0
    assert np.min(flux) > 0
    with pytest.raises(ValueError, match="does not reproduce"):
        reproduce_keck_centroids(
            raw, SKY, IMAT, expected + [0.01, 0.0],
        )


def test_apd_normalized_ratio_jacobian_matches_finite_differences():
    counts = np.array([[120., 89., 98., 130.]])
    sky = np.array([30., 20., 20., 30.])
    quad, flux = normalized_quad_cell(counts, sky)
    matrix = apd_imat_matrix(IMAT)
    jac = matrix @ (
        QUADRANT_PATTERNS-quad[0,:,None]
    )/flux[0]
    eps = 1e-4
    for i in range(4):
        positive = counts.copy()
        negative = counts.copy()
        positive[0,i] += eps
        negative[0,i] -= eps
        c_plus, _ = normalized_quad_cell(positive, sky)
        c_minus, _ = normalized_quad_cell(negative, sky)
        numerical = ((c_plus-c_minus) @ matrix.T)[0]/(2*eps)
        np.testing.assert_allclose(numerical, jac[:,i], rtol=1e-6, atol=1e-10)


def test_apd_delta_propagation_has_correct_units_and_psd():
    covariance, diag = apd_delta_method_covariance(_counts(), SKY, IMAT)
    assert covariance.shape == (2,2)
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-13)
    assert np.linalg.eigvalsh(covariance).min() >= 0
    np.testing.assert_allclose(
        covariance,
        np.asarray(diag["centroid_covariance_arcsec2"]) *
        zernike_tilt_opd_per_arcsec(10.5)**2,
    )
    assert not diag["apd_gain_independently_calibrated"]
    assert not diag["noise_model_validated"]


def test_unknown_apd_excess_gain_and_read_noise_are_explicit_parameters():
    raw = _counts()
    p, _ = apd_delta_method_covariance(
        raw, SKY, IMAT, excess_noise_factor=1.0,
    )
    twice, _ = apd_delta_method_covariance(
        raw, SKY, IMAT, excess_noise_factor=2.0,
    )
    np.testing.assert_allclose(twice, 2*p, rtol=1e-13)
    half, _ = apd_delta_method_covariance(
        raw, SKY, IMAT, effective_photoelectrons_per_count=2,
    )
    np.testing.assert_allclose(half, p/2, rtol=1e-13)
    more, _ = apd_delta_method_covariance(
        raw, SKY, IMAT, read_noise_variance_counts2=12.0,
    )
    assert np.trace(more) > np.trace(p)
    with pytest.raises(ValueError, match=">= 1"):
        apd_delta_method_covariance(raw, SKY, IMAT, excess_noise_factor=0.4)


def test_apd_poisson_monte_carlo_matches_first_order_at_high_flux():
    # Conditional pure-Poisson baseline; high rates suppress ratio nonlinearity.
    counts = np.full((128,4), [2200., 1800., 2100., 2400.])
    sky = np.array([15., 10., 10., 15.])
    approximate, _ = apd_delta_method_covariance(counts, sky, IMAT)
    sampled, audit = apd_poisson_monte_carlo(
        counts, sky, IMAT, trials_per_frame=256, seed=77,
    )
    np.testing.assert_allclose(
        np.diag(sampled), np.diag(approximate), rtol=0.08,
    )
    assert audit["nonpositive_denominator_draws"] == 0
    assert not audit["noise_model_validated"]


def test_monte_carlo_includes_stochastic_normalization_denominator():
    counts = np.full((48,4), [30., 30., 30., 30.])
    delta, _ = apd_delta_method_covariance(
        counts, np.zeros(4), IMAT,
    )
    monte, _ = apd_poisson_monte_carlo(
        counts, np.zeros(4), IMAT,
        trials_per_frame=512, seed=2020,
    )
    # At low counts the estimator is no longer identical to the first
    # order approximation. Monte Carlo should remain PSD and close.
    assert np.linalg.eigvalsh(monte).min() >= 0
    assert not np.allclose(
        monte, delta, rtol=0.001, atol=0.0,
    )
    np.testing.assert_allclose(
        np.diag(monte), np.diag(delta), rtol=0.12,
    )


def test_invalid_flux_background_and_units_fail_closed():
    raw = _counts()
    with pytest.raises(ValueError, match="positive"):
        normalized_quad_cell(np.full((4,4), 10.), [20]*4)
    with pytest.raises(ValueError, match="four"):
        apd_imat_matrix([1,2,3])
    with pytest.raises(ValueError, match="positive"):
        apd_delta_method_covariance(
            raw, SKY, IMAT, effective_photoelectrons_per_count=0,
        )
    with pytest.raises(ValueError, match="length-four"):
        apd_delta_method_covariance(
            raw, SKY, IMAT, read_noise_variance_counts2=[1,2,3],
        )


def test_van_dam_rejection_matches_ragland_plant_not_unity():
    fs, n = 125., 8192
    rng = np.random.default_rng(31)
    f = np.linspace(0, fs/2, 2049)
    W = np.sinc(f/fs)*np.exp(-1j*np.pi*f/fs)
    K = np.full_like(f, .4, dtype=complex)
    P = np.exp(-2j*np.pi*f/fs)
    tf = NS(
        freq=f, wfs=W, servo=K, lag=P,
        rtf=1/(1+W*K*P),
    )
    trs = NS(
        aoMode="LGS",
        dm=NS(com=rng.normal(0, 1e-7, (n, 4))),
        holoop=NS(tf=NS(ctf=np.ones_like(f), wfs=np.ones_like(f))),
        tipTilt=NS(slopes=rng.normal(0, 2e-7, (n,2))),
        ttloop=NS(freq=fs, tf=tf),
    )
    diag = systemDiagnosis.__new__(systemDiagnosis)
    diag.trs = trs
    _, actual = diag.get_noise_variance(
        tt_noise_method="van_dam",
        tt_mirror_model="ragland_2016_keck_ii",
    )
    model = from_p3_tiptilt_loop(
        trs.ttloop, mirror_model="ragland_2016_keck_ii"
    )
    direct, _ = estimate_tt_noise(
        trs.tipTilt.slopes, sample_rate_hz=fs,
        rejection_frequency_hz=f,
        rejection_transfer=model.sensitivity,
    )
    np.testing.assert_allclose(actual, direct, rtol=1e-13, atol=1e-30)
    assert trs.tipTilt.noise_rejection_mirror_model == "ragland_2016_keck_ii"
    assert trs.tipTilt.noise_estimation_diagnostics["rejection_mirror_model"] == (
        "ragland_2016_keck_ii"
    )
    assert not trs.tipTilt.noise_estimation_diagnostics[
        "instrument_loop_gain_independently_validated"
    ]
