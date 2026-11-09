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


def test_loop_positive_fft_grid_uses_sample_rate_once():
    from p3.telemetry.systemDiagnosis import _positive_fft_frequency_grid

    fs, n = 125.0, 4278
    f = _positive_fft_frequency_grid(fs, n)
    np.testing.assert_allclose(f, np.fft.rfftfreq(n, d=1 / fs)[1:])
    assert f[0] == pytest.approx(fs / n)
    assert f[-1] == pytest.approx(fs / 2)
    assert len(f) == n // 2


def test_high_order_loop_boxcar_and_lag_use_same_physical_time_units_as_strap():
    """HO integration/delay must use 2*pi*f*t, not just f*t."""
    from types import SimpleNamespace as NS
    from p3.telemetry.systemDiagnosis import systemDiagnosis

    def loop(fs, seconds):
        return NS(
            freq=fs, lat=seconds,
            tf=NS(num=np.array([0.3, 0., 0., 0.]),
                  den=np.array([0., 0., 0.])),
        )

    sd = systemDiagnosis.__new__(systemDiagnosis)
    sd.trs = NS(
        holoop=loop(500., 0.002),
        ttloop=loop(125., 0.008),
        wfs=NS(nExp=512),
        tipTilt=NS(nExp=512),
    )
    sd.temporal_transfer_function()
    for lo in (sd.trs.holoop, sd.trs.ttloop):
        f = lo.tf.freq
        expected_sensor = np.sinc(f / lo.freq) * np.exp(-1j*np.pi*f/lo.freq)
        expected_lag = np.exp(-2j*np.pi*f*lo.lat)
        np.testing.assert_allclose(lo.tf.wfs, expected_sensor, atol=1e-14)
        np.testing.assert_allclose(lo.tf.lag, expected_lag, atol=1e-14)
        assert f[0] == pytest.approx(lo.freq / 512)
