"""P3 pupil-orientation audit contracts; no on-sky calibration claimed."""
import numpy as np
import pytest

from p3.telemetry.keckPupilAudit import (
    pupil_orientation_report, binary_pupil_on_grid, diffraction_intensity,
    read_mask,
)


def _n0120():
    return {
        "EL": 47.40723, "ROTPPOSN": 0.64,
        "ROTPOSN": 0.70, "INSTANGL": 0.70,
        "PMSNAME": "open", "PMSANGL": 349.20361, "PMRANGL": 150.9975,
    }


def test_kaon_nirc2_and_wfs_angles_do_not_reuse_sky_pa():
    report = pupil_orientation_report(_n0120())
    assert report["sky_pa_deg"] == pytest.approx(0.0)
    assert report["nirc2_pupil_angle_deg"] == pytest.approx(46.76723)
    assert report["wfs_pupil_angle_deg"] == pytest.approx(-46.76723)
    assert report["pmsangl_header_deg"] == pytest.approx(349.20361)
    assert report["pupil_mask_mechanical_angle_deg"] is None
    assert report["mask_fits_zero_angle_calibrated"] is False
    assert report["camera_array_rotation_sign_calibrated"] is False
    assert report["candidate_scipy_array_rotation_deg"]["kaon_nirc2_plus_90"] == pytest.approx(136.76723)
    assert report["candidate_scipy_array_rotation_deg"]["as_archived_no_rotation"] == 0


def test_unknown_and_unavailable_incirc_masks_cannot_fallback_to_open(tmp_path):
    with pytest.raises(ValueError, match="PMSNAME"):
        pupil_orientation_report({**_n0120(), "PMSNAME": "UNEXPECTED"})
    incircle = pupil_orientation_report({**_n0120(), "PMSNAME": "incircle"})
    with pytest.raises(FileNotFoundError, match="No calibrated local FITS"):
        read_mask(tmp_path, incircle)


def test_mask_rotation_is_bounded_area_average_and_changes_feature():
    mask = np.zeros((64, 64))
    mask[8:13, 43:51] = 1
    zero = binary_pupil_on_grid(mask, 64, 0)
    turned = binary_pupil_on_grid(mask, 64, 90)
    assert turned.min() >= 0 and turned.max() <= 1
    assert turned.sum() == pytest.approx(zero.sum())
    assert not np.array_equal(zero, turned)


def test_oblique_rotation_anti_aliases_and_never_overshoots():
    yy, xx = np.indices((96, 96))
    open_aperture = ((xx-47.5)**2+(yy-47.5)**2 < 37**2)
    pupil = binary_pupil_on_grid(open_aperture.astype(float), 96, 46.76723)
    assert np.any((pupil > 0) & (pupil < 1))
    assert pupil.min() >= 0 and pupil.max() <= 1
    assert np.mean(pupil) == pytest.approx(np.mean(open_aperture), abs=0.006)


def test_diffraction_energy_and_symmetry_without_atmospheric_fit():
    yy, xx = np.indices((64, 64))
    ann = (np.hypot(xx-31.5, yy-31.5) < 24) & (
        np.hypot(xx-31.5, yy-31.5) > 9
    )
    psf = diffraction_intensity(ann, oversample=2)
    assert psf.shape == (128, 128)
    assert np.isfinite(psf).all()
    assert np.min(psf) >= 0
    assert psf.sum() == pytest.approx(1.0, rel=1e-12)


def test_missing_mask_fits_fails_explicitly(tmp_path):
    report = pupil_orientation_report(_n0120())
    with pytest.raises(FileNotFoundError, match="not found"):
        read_mask(tmp_path, report)
