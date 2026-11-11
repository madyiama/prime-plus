"""P4-B research-only STRAP audits: synthetic, no Keck FITS required."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.telemetry.ttApdNoise import (
    apd_imat_matrix, normalized_quad_cell,
)
from p3.telemetry.ttClosedLoop import (
    from_p3_tiptilt_loop,
)
from p3.telemetry.ttP4BAudit import (
    BANDS, WELCH_SIZES, MIRRORS,
    loop_identity_report, sensitivity_audit, flux_segment_audit,
)


def loop_fixture(fs=125.0):
    f = np.linspace(0,fs/2,2049)
    W = np.sinc(f/fs)*np.exp(-1j*np.pi*f/fs)
    K = np.full(f.shape,0.26,dtype=complex)
    lag = np.exp(-2j*np.pi*f/fs)
    S = 1/(1+W*K*lag)
    return NS(freq=fs,tf=NS(freq=f,wfs=W,servo=K,lag=lag,rtf=S))


def mock_centroids(n=4096):
    rng=np.random.default_rng(717)
    white=rng.normal(size=(n,2))*0.025
    white[:,1] += 0.13*white[:,0]
    white[:,0] += 0.003*np.sin(np.arange(n)*0.25)
    return white.astype(np.float64)


def test_loop_feedback_identities_and_ragland_hypothesis_are_explicit():
    loop=loop_fixture()
    unity=from_p3_tiptilt_loop(loop,"unity")
    ragland=from_p3_tiptilt_loop(loop,"ragland_2016_keck_ii")
    for model in (unity,ragland):
        report=loop_identity_report(model,loop.freq)
        assert report["algebra_verified"]
        assert max(report["transfer_algebra_errors"].values()) < 1e-12
        assert report["sensor_noise_to_measured_gain"] > 0
        assert not report["keck_2020_plant_independently_calibrated"]
    assert not np.allclose(unity.sensitivity,ragland.sensitivity,rtol=1e-4,atol=1e-6)


def test_prespecified_van_dam_grid_and_baseline_no_science_image_ranking():
    loop=loop_fixture()
    result=sensitivity_audit(
        mock_centroids(),loop,
        bands=((0.65,0.95),(0.75,0.95)),
        welch_sizes=(256,512),
    )
    assert len(result["rows"]) == 2*2*2
    assert result["not_ranked_by_science_image"]
    assert not result["physical_model_validated"]
    assert result["reference"]["welch_nperseg"] == 512
    assert result["reference"]["band_fraction_nyquist"] == [0.65,0.95]
    assert set(result["fixed_unity_noise_plant_sensitivity"]) == set(MIRRORS)
    for r in result["rows"]:
        assert r["sensor_noise"]["trace_arcsec2"] > 0
        assert r["spectral_welch"]["fit_bin_count"] >= 8
        assert 0 <= r["spectral_welch"][
            "negative_signed_residual_bin_fraction"][0] <= 1
        assert not r["sensor_noise_is_independently_calibrated"]


def synthetic_apds(n=4096, *, varying=True):
    rng=np.random.default_rng(777)
    profile=np.array([1.0,1.3,1.6,2.0,2.6,3.0,3.5,4.0]) if varying else np.ones(8)
    counts=np.concatenate([
        rng.poisson(80.0*f, size=(512,4))
        for f in profile
    ]).astype(float)
    sky=np.zeros(4)
    imat=np.array([.1785,.842,.95,-.2])
    quad,_=normalized_quad_cell(counts,sky)
    centroids=quad@apd_imat_matrix(imat).T
    return counts,sky,imat,centroids


def test_segmented_apd_flux_and_van_dam_preserve_time_order_and_units():
    raw,sky,imat,centroids=synthetic_apds()
    out=flux_segment_audit(
        raw,sky,imat,centroids,loop_fixture(),
        segment_length=512,nperseg=256,
    )
    assert out["segment_count"] == 8
    assert out["discarded_trailing_frames"] == 0
    assert out["sufficient_flux_leverage"]
    assert out["flux_vs_van_dam_spearman_rho"] is not None
    assert out["flux_stratification_selection_bias_avoided"]
    assert out["all_APD_instrument_gains_unverified"]
    rows=out["blocks"]
    assert rows[0]["start_frame"] == 0
    assert rows[0]["end_frame_exclusive"] == rows[1]["start_frame"]
    assert rows[0]["predicted_ideal_apd_noise"]["trace_arcsec2"] > (
        rows[-1]["predicted_ideal_apd_noise"]["trace_arcsec2"]
    )
    assert all(x["van_dam_noise"]["psd_with_roundoff_only"] for x in rows)


def test_flux_insufficiency_is_reported_not_overinterpreted():
    raw,sky,imat,centroids=synthetic_apds(varying=False)
    out=flux_segment_audit(
        raw,sky,imat,centroids,loop_fixture(),
        segment_length=512,nperseg=256,
    )
    assert out["segment_count"] == 8
    assert not out["sufficient_flux_leverage"]
    assert out["flux_vs_van_dam_spearman_rho"] is None
    assert "undetermined" in out["flux_trend_inference"]


def test_invalid_alignment_and_segment_sizes_fail_closed():
    raw,sky,imat,centroids=synthetic_apds()
    with pytest.raises(ValueError,match="frame-aligned"):
        flux_segment_audit(raw,sky,imat,centroids[:-1],loop_fixture())
    with pytest.raises(ValueError,match="two Welch windows"):
        flux_segment_audit(
            raw,sky,imat,centroids,loop_fixture(),
            segment_length=256,nperseg=256
        )
    with pytest.raises(ValueError,match="centroid"):
        flux_segment_audit(
            raw,sky,imat,centroids+0.01,loop_fixture(),
            segment_length=512,nperseg=256
        )


def test_prespecified_grid_cannot_silently_be_replaced_with_one_model():
    assert BANDS == ((0.55,0.85),(0.65,0.95),(0.75,0.95))
    assert WELCH_SIZES == (256,512,1024)
    assert MIRRORS == ("unity","ragland_2016_keck_ii")
