"""Analytical STRAP integration and latency transfer-function checks."""
import numpy as np
import pytest

from p3.telemetry.systemDiagnosis import (
    _tt_sensor_integration_response,
    _tt_latency_response,
)


def test_strap_sensor_integration_is_causal_boxcar():
    fs = 125.0
    freq = np.array([0.0, fs/4, fs/2])
    response = _tt_sensor_integration_response(freq, fs)
    expected = np.sinc(freq/fs) * np.exp(-1j*np.pi*freq/fs)
    np.testing.assert_allclose(response, expected, atol=1e-14)
    assert response[0] == pytest.approx(1.0 + 0j)
    assert abs(response[-1]) == pytest.approx(2/np.pi, rel=1e-13)
    assert np.angle(response[-1]) == pytest.approx(-np.pi/2, abs=1e-13)


def test_strap_delay_has_2pi_frequency_phase_and_unit_amplitude():
    fs = 125.0
    response = _tt_latency_response(np.array([0.0, fs/4]), fs, 1.0)
    np.testing.assert_allclose(response, np.array([1.0, -1j]), atol=1e-14)
    np.testing.assert_allclose(np.abs(response), np.ones(2), atol=1e-14)


def test_tt_delay_from_seconds_and_frames_are_equivalent():
    fs = 124.99406154
    delay_seconds = 0.001
    frequencies = np.linspace(0, fs/2, 103)
    actual = _tt_latency_response(frequencies, fs, delay_seconds*fs)
    expected = np.exp(-2j * np.pi * frequencies * delay_seconds)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
