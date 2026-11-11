"""Keck AO pure physical-contract tests: temporal alignment, Fried r0 and pupil."""

import numpy as np
import pytest

from p3.telemetry.systemDiagnosis import (
    interpolate_forward_residual_no_wrap,
    fried_seeing_arcsec,
)
from p3.telemetry.keckUtils import getPupilAngles, getPA


def test_pseudo_open_loop_integer_delay_moves_frames_not_actuators():
    n = 9
    # Every actuator is given a unique, noninterchangeable signature.
    frames = np.column_stack([
        1000 + 3 * np.arange(n),
        -20 + 7 * np.arange(n),
        np.arange(n)**2,
    ]).astype(float)
    aligned, valid = interpolate_forward_residual_no_wrap(frames, 2.0)
    np.testing.assert_array_equal(valid, [True] * 7 + [False] * 2)
    np.testing.assert_allclose(aligned[:7], frames[2:])
    np.testing.assert_allclose(aligned[7:], 0.0)
    # Critically: cannot reproduce by flattened np.roll.
    assert not np.array_equal(aligned[:7], np.roll(frames, -2)[:7])


def test_pseudo_open_loop_fractional_delay_and_no_circular_boundary():
    frames = np.column_stack([
        10.0 * np.arange(7), 100.0 + np.arange(7)**2
    ])
    aligned, valid = interpolate_forward_residual_no_wrap(frames, 0.25)
    np.testing.assert_array_equal(valid, [True] * 6 + [False])
    np.testing.assert_allclose(
        aligned[:6], 0.75 * frames[:6] + 0.25 * frames[1:]
    )
    np.testing.assert_allclose(aligned[-1], 0.0)


def test_pseudo_open_loop_zero_delay_and_invalid_input():
    signal = np.arange(12.0).reshape(6, 2)
    actual, mask = interpolate_forward_residual_no_wrap(signal, 0)
    np.testing.assert_array_equal(actual, signal)
    assert mask.all()
    with pytest.raises(ValueError, match="non-negative"):
        interpolate_forward_residual_no_wrap(signal, -1)
    with pytest.raises(ValueError, match="remain"):
        interpolate_forward_residual_no_wrap(signal, 5.5)


def test_fried_seeing_r0_inverse_at_500nm_and_chromatic_scaling():
    r0_500 = 0.16
    wavelength_500 = 500e-9
    angle_500 = fried_seeing_arcsec(r0_500, wavelength_500)
    assert angle_500 == pytest.approx(0.976 * 206264.806247 * 500e-9 / 0.16)
    r0_k = r0_500 * (2.2e-6 / wavelength_500) ** 1.2
    # epsilon scales as lambda^(-1/5) when r0 follows lambda^(6/5).
    assert fried_seeing_arcsec(r0_k, 2.2e-6) == pytest.approx(
        angle_500 * (2.2e-6 / wavelength_500) ** (-0.2)
    )


def test_n0120_nirc2_pupil_orientation_kaon1160_separate_from_sky_pa():
    hdr = dict(
        EL=47.40723, ROTPPOSN=0.64000,
        ROTPOSN=0.70000, INSTANGL=0.70000,
    )
    info = getPupilAngles(hdr)
    assert info["sky_position_angle_deg"] == pytest.approx(0)
    assert info["nirc2_pupil_angle_deg"] == pytest.approx(46.76723)
    assert info["wfs_pupil_angle_deg"] == pytest.approx(-46.76723)
    assert getPA(hdr) == pytest.approx(46.76723)
    assert info["mask_zero_offset_calibrated"] is False


def test_missing_pupil_rotation_header_has_no_sky_pa_fallback():
    hdr = dict(EL=47.4, ROTPOSN=0.7, INSTANGL=0.7)
    with pytest.raises(ValueError, match="ROTPPOSN"):
        getPupilAngles(hdr)
