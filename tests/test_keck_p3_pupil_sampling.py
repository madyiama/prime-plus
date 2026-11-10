"""P3 optical-mask regression: bounded transmission, orientation and telescope integration.

No Keck science image or empirical PSF fit is used here.
"""
import numpy as np
import pytest

from p3.aoSystem.pupilTransforms import (
    resample_binary_pupil, pupil_sampling_diagnostics,
)


def test_zero_angle_same_grid_is_exact():
    source = np.zeros((64, 64))
    source[5:22, 9:31] = 1
    np.testing.assert_array_equal(
        resample_binary_pupil(source, 64, 0, supersample=4),
        source,
    )


def test_90_degree_rotation_has_known_parity_and_area():
    source = np.zeros((64, 64))
    source[3:9, 40:51] = 1
    rotated = resample_binary_pupil(source, 64, 90, supersample=4)
    np.testing.assert_array_equal(rotated, np.rot90(source, 1))
    assert rotated.sum() == pytest.approx(source.sum())
    assert not np.array_equal(source, rotated)


def test_oblique_rotation_remains_bounded_and_preserves_area():
    y, x = np.indices((128, 128))
    mask = ((x-63.5)**2+(y-63.5)**2 < 45**2).astype(float)
    mask[((np.abs(x-63.5) < 1.5) | (np.abs(y-63.5) < 1.5))] = 0
    result = resample_binary_pupil(mask, 128, 46.76723, supersample=4)
    assert result.min() >= 0 and result.max() <= 1
    assert np.count_nonzero((result > 0) & (result < 1)) > 0
    assert abs(result.mean() - mask.mean()) < 0.007
    d = pupil_sampling_diagnostics(mask, result, 46.76723, 4)
    assert d["zero_angle_parity_calibrated"] is False
    assert d["output_mean_intensity_transmission"] <= d["output_mean_amplitude"]


def test_invalid_binary_mask_is_not_silently_clipped():
    source = np.ones((32, 32))
    source[0, 0] = 0.5
    with pytest.raises(ValueError, match="0/1"):
        resample_binary_pupil(source, 32)
    with pytest.raises(ValueError, match="resolution"):
        resample_binary_pupil(np.ones((32, 32)), 8)


def test_keck_telescope_uses_identical_bounded_pupil_in_otf(tmp_path):
    from astropy.io import fits
    from p3.aoSystem.telescope import telescope

    source = np.zeros((96, 96))
    y, x = np.indices(source.shape)
    source[((x-47.5)**2+(y-47.5)**2 < 38**2)
           & ((x-47.5)**2+(y-47.5)**2 > 11**2)] = 1
    source[(x > 45) & (x < 48)] = 0
    path = tmp_path / "test_binary_pupil.fits"
    fits.writeto(path, source)

    tel = telescope(
        D=10.5, resolution=100,
        pupilAngle=46.76723, path_pupil=str(path),
        pupil_resampling="physical_binary", verbose=False,
    )
    expected = resample_binary_pupil(source, 100, 46.76723)
    np.testing.assert_allclose(np.asarray(tel.pupil), expected, rtol=0, atol=0)
    assert np.min(tel.pupil) >= 0 and np.max(tel.pupil) <= 1
    assert tel.pupil_resampling_diagnostics["method"] == (
        "physical_binary_subpixel_area_average"
    )
    assert tel.pupil_resampling_diagnostics["zero_angle_parity_calibrated"] is False


def test_legacy_sampling_is_default_for_generic_telescopes():
    from p3.aoSystem.telescope import telescope
    tel = telescope(D=8, resolution=64, obsRatio=0.2, verbose=False)
    assert tel.pupil_resampling == "legacy"
    assert tel.pupil.shape == (64, 64)
