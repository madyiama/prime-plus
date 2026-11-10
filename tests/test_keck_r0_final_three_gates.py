"""Synthetic scientific checks for final r0/L0 audit; no Keck observations."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.telemetry.r0FinalAudit import (
    sample_shifted_residual,
    fixed_l0_modal_fit,
    sign_lag_diagnostic,
    gain_hypothesis_diagnostic,
    analyse_final_three_gates,
)


def test_fractional_and_negative_rtc_lags_do_not_wrap_or_mix_actuators():
    y = np.column_stack([10.0*np.arange(20), 500+np.arange(20)**2])
    copy = y.copy()
    idx = np.array([3, 5, 11])
    np.testing.assert_allclose(
        sample_shifted_residual(y, idx, 0.5),
        0.5*y[idx] + 0.5*y[idx+1],
    )
    np.testing.assert_allclose(
        sample_shifted_residual(y, idx, -0.5),
        0.5*y[idx-1] + 0.5*y[idx],
    )
    with pytest.raises(ValueError, match="outside"):
        sample_shifted_residual(y, np.array([0, 1]), -0.5)
    np.testing.assert_array_equal(y, copy)


def test_absolute_and_relative_variance_fits_recover_amplitude():
    radial = np.repeat(np.arange(3, 15), 2)
    model = 2.0/(radial.astype(float)**2)
    r0 = 0.11
    scaling = 2.5
    fit = fixed_l0_modal_fit(scaling*model, model, radial, r0, 3, 14)
    correct = r0*scaling**(-3/5)
    assert fit["r0_absolute_loss_m"] == pytest.approx(correct)
    assert fit["r0_log_loss_m"] == pytest.approx(correct)
    assert fit["shape_rms_log_residual"] == pytest.approx(0, abs=1e-13)
    negative = model.copy()
    negative[radial == 4] = -1
    assert fixed_l0_modal_fit(
        negative, model, radial, r0, 3, 14
    )["status"] == "nonpositive_modal_variance"


def test_gain_bias_exponent_is_minus_six_fifths_and_not_autoapplied():
    out = gain_hypothesis_diagnostic(0.1147404127, 0.4095e-6)
    expected = 0.1147404127 * (0.6/0.4095)**(-6/5)
    assert out["alternatives"][0][
        "r0_m_if_all_OPD_scales_were_changed"
    ] == pytest.approx(expected)
    assert out["conversion_interpretation_verified"] is False


def test_all_sign_lag_hypotheses_share_frames_and_never_choose_winner():
    rng = np.random.default_rng(7)
    cmd = rng.normal(size=(200, 20))
    residual = rng.normal(size=(200, 20))
    original = cmd.copy()
    ref = np.linspace(0.5, 1.5, 12)
    radial = np.arange(3, 15)
    U = np.zeros((12, 20))
    U[:, :12] = np.eye(12)
    res = sign_lag_diagnostic(
        cmd, residual, U, np.arange(20), 2.3, ref, radial,
        r0_reference_m=0.11, wavelength_m=500e-9,
        frame_limit=128,
    )
    assert len(res["results"]) == 6
    assert {x["residual_sign"] for x in res["results"]} == {-1, 1}
    assert len({x["n_time_samples"] for x in res["results"]}) == 1
    assert res["rtc_sign_independently_validated"] is False
    np.testing.assert_array_equal(cmd, original)


def test_three_gate_wrapper_keeps_current_atmosphere_and_projections():
    rng = np.random.default_rng(9)
    nact = 25
    nmodes = 21
    valid = np.arange(nmodes)
    commands = rng.normal(scale=1e-8, size=(200, nact))
    residual = rng.normal(scale=1e-8, size=(200, nact))
    U = np.zeros((nmodes, nmodes))
    U[:] = np.eye(nmodes)
    model = 1.0/np.arange(1, nmodes+1)
    atm = NS(wvl=500e-9, r0_tel=0.12, L0_tel=31.0)
    dm = NS(com=commands, nActuators=[5], pitch=[0.5625],
            volt2meter=0.4095e-6)
    trs = NS(
        atm=atm, dm=dm, rec=NS(res=residual),
        wfs=NS(coeffs_mod=model, var_meas=np.full(nmodes, 1e-15),
               var_noise_zer=np.zeros(nmodes)),
        mat=NS(u2z=U, dmIF=np.ones((25, 25))),
        holoop=NS(lat=0.001, freq=1000),
        tel=NS(D=10.5), aoMode="LGS",
    )
    sd = NS(trs=trs, z=NS(n=np.arange(2, 23)),
            actuInPupil=valid, mZZ=np.eye(nmodes))
    cmd_before = commands.copy()
    report = analyse_final_three_gates(sd, frame_limit=128)
    assert len(report["radial_mode_sensitivity"]) == 5
    assert len(report["rtc_sign_lag_sensitivity"]["results"]) == 6
    assert report["projection"]["n_projection_modes"] == nmodes
    assert report["noise"]["noise_WFS_independently_calibrated"] is False
    assert atm.r0_tel == 0.12 and atm.L0_tel == 31.0
    np.testing.assert_array_equal(commands, cmd_before)
