"""P2: anisoplanatic SF must vanish at zero baseline, OTF(0)=1.

An additive SF offset introduced by focal-angular map interpolation leaves
the flux-normalized image unchanged but biases direct OTF-based Strehl and
the derived Maréchal-equivalent WFE. The n0120 reported offset was 0.30434
rad^2 at one equivalent layer, versus 0.17625 rad^2 at seven layers.
"""

import numpy as np
import pytest

from p3.psfr.psfR import _anisoplanatism_zero_lag


@pytest.mark.parametrize("raw_dc", [0.0, 0.176253846978, 0.304339917734])
def test_anisoplanatism_zero_lag_correction_preserves_normalized_psf(raw_dc):
    n = 64
    yy, xx = np.mgrid[:n, :n]
    r2 = (yy-n//2)**2 + (xx-n//2)**2
    dphi_raw = raw_dc + 0.0005*r2
    dphi = _anisoplanatism_zero_lag(dphi_raw)
    assert dphi[n//2, n//2] == pytest.approx(0.0, abs=1e-14)

    # A normalized, diffraction-like Gaussian OTF with DC exactly 1.
    dl = np.exp(-0.002*r2)
    raw_otf = dl * np.exp(-0.5*dphi_raw)
    normalized_otf = dl * np.exp(-0.5*dphi)
    assert normalized_otf[n//2, n//2] == pytest.approx(1.0, abs=1e-14)
    assert raw_otf[n//2, n//2] == pytest.approx(np.exp(-raw_dc/2))
    assert np.sum(normalized_otf)/np.sum(raw_otf) == pytest.approx(
        np.exp(raw_dc/2), rel=1e-12
    )

    raw_psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(raw_otf))).real
    normalized_psf = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(normalized_otf))
    ).real
    np.testing.assert_allclose(
        raw_psf/raw_psf.sum(),
        normalized_psf/normalized_psf.sum(),
        rtol=1e-12, atol=1e-14,
    )


def test_reported_n0120_peak_ratio_is_exactly_dc_normalized_strehl_ratio():
    raw_sr_ratio = 15.684207879879512/13.658213317279253
    dc_one = 0.304339917734454
    dc_seven = 0.1762538469780601
    corrected_sr_ratio = raw_sr_ratio * np.exp((dc_seven-dc_one)/2)
    expected_peak_ratio = 0.0108295148391602/0.010054346159479945
    assert corrected_sr_ratio == pytest.approx(expected_peak_ratio, rel=5e-8)


def test_one_source_and_multi_source_zero_lag_normalization():
    n = 16
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    spatial = ((x-n//2)**2 + (y-n//2)**2).astype(float)
    cube = np.stack([spatial + 0.2, 0.3*spatial + 0.9], axis=2)
    corrected = _anisoplanatism_zero_lag(cube)
    np.testing.assert_allclose(corrected[n//2, n//2], [0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(corrected[:, :, 0], spatial)
    np.testing.assert_allclose(corrected[:, :, 1], 0.3*spatial)


def test_nonfinite_and_wrong_shape_rejected():
    with pytest.raises(ValueError, match="must be"):
        _anisoplanatism_zero_lag(np.zeros(4))
    with pytest.raises(ValueError, match="non-finite"):
        _anisoplanatism_zero_lag(np.array([[0.0, np.nan], [0.0, 0.0]]))
