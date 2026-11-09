"""Test AO-loop transfer-function integration with NumPy 2 and SciPy."""

from types import SimpleNamespace

import numpy as np

from p3.telemetry.systemDiagnosis import systemDiagnosis


def _loop(frame_rate):
    return SimpleNamespace(
        freq=frame_rate,
        lat=0.002,
        tf=SimpleNamespace(
            num=np.array([0.2, 0.0, 0.0, 0.0]),
            den=np.array([-0.4, 0.0, 0.0]),
        ),
    )


def test_transfer_functions_integrate_high_order_and_tip_tilt_noise():
    """Both loops must compute finite noise propagation factors without np.trapz."""
    trs = SimpleNamespace(
        holoop=_loop(1000.0),
        ttloop=_loop(125.0),
        wfs=SimpleNamespace(nExp=1024),
        tipTilt=SimpleNamespace(nExp=128),
    )
    diagnosis = systemDiagnosis.__new__(systemDiagnosis)
    diagnosis.trs = trs
    diagnosis.temporal_transfer_function()
    for loop in (trs.holoop, trs.ttloop):
        assert np.isfinite(loop.tf.pn)
        assert loop.tf.pn > 0
        assert loop.tf.freq.size == loop.tf.ntf.size
