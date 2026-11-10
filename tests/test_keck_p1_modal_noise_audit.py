"""P1-B: synthetic regression for Zernike/noise diagnostic, no Keck SAV.

The tests demonstrate that command differences mix temporal signal
and noise, ensure the old periodic LGS command proxy can be reproduced,
and show that no diagnostic mutates the input telemetry.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.keck_p1_modal_noise_audit import (
    analyse_diagnosis,
    fixed_l0_noise_sensitivity,
    lag_difference_variances,
    project_lag_variance,
    write_audit,
)


def _fake_system_diagnosis():
    nframe = 128
    t = np.arange(nframe, dtype=float)
    # Include smooth deterministic turbulence as well as a fast actuator
    # fluctuation. The lag-difference 'noise' proxy is not a true
    # independent measurement of the WFS detector noise.
    commands = np.column_stack([
        1e-7 * np.sin(t/11.0+0.7*k)
        + 1.5e-8*np.sin((k+1)*t/2)
        for k in range(4)
    ])
    residual = 0.2*commands + np.roll(commands, 1, axis=0)*0.03
    reconstructor = np.eye(4)
    noise = lag_difference_variances(commands, 1, periodic=True)
    raw_var = noise*8+np.array([1,2,3,4])*1e-15
    wl = 500e-9
    theory = (2*np.pi/wl)**2 * (raw_var-noise)
    dm = NS(com=commands.copy(), pseudo_ol_frames_used=126,
            pseudo_ol_sign_independently_verified=False)
    wfs = NS(
        var_meas=raw_var, var_noise_zer=noise,
        coeffs_mod=theory,
        jIndex=[4,5,6,7],
        slopes=np.column_stack([commands, commands]),
    )
    atm = NS(wvl=wl, r0_tel=0.1, L0_tel=31.0)
    mat = NS(u2z=reconstructor)
    trs = NS(dm=dm, rec=NS(res=residual.copy()), wfs=wfs,
             atm=atm, mat=mat, aoMode="LGS")
    sd = NS(trs=trs, actuInPupil=np.ones(4, dtype=bool),
            z=NS(n=np.array([2,2,2,3])),
            mZZ=np.eye(4),
            res_r0=NS(success=True, cost=0.0),
            noiseMethod="autocorrelation")
    return sd


def test_lag_difference_proxy_detects_atmosphere_even_with_no_detector_noise():
    t = np.arange(256)
    # Pure smooth physical signal, detector noise identically zero.
    wavefront = np.sin(t/9.0)[:, None]
    estimate = lag_difference_variances(wavefront, 1, periodic=False)
    assert estimate[0] > 0.0


def test_lag_proxy_is_finite_record_by_default_and_never_modifies_input():
    a = np.array([[0.0, 4.0], [1.0, 4.0], [2.0, 4.0],
                  [3.0, 4.0], [10.0, 4.0]])
    before = a.copy()
    finite = lag_difference_variances(a, 1)
    periodic = lag_difference_variances(a, 1, periodic=True)
    np.testing.assert_array_equal(a, before)
    assert periodic[0] > finite[0]
    assert finite[1] == periodic[1] == 0.0


def test_modal_projection_lag_proxy_preserves_units_and_linear_mapping():
    a = np.arange(48, dtype=float).reshape(12, 4)*1e-8
    R = np.array([[1.0, 0.5], [0.1, -2.0]])
    expected = lag_difference_variances(
        a[:, [1, 3]] @ R.T, 1, periodic=False
    )
    measured = project_lag_variance(a, np.array([1, 3]), R)
    np.testing.assert_allclose(measured, expected, atol=0, rtol=1e-13)
    np.testing.assert_array_equal(a, np.arange(48).reshape(12, 4)*1e-8)


def test_noise_subtraction_sensitivity_at_fixed_L0_increases_r0_with_noise():
    model = np.array([4.,4.,4.,2.])
    raw = np.array([5.,5.,5.,3.])*1e-14
    noise = np.full(4, 1e-14)
    rows = fixed_l0_noise_sensitivity(
        raw, noise, model, [2,2,2,3], 5e-7, 0.12,
    )
    finite_r0 = [v["fixed_L0_r0_m"] for v in rows]
    assert all(np.isfinite(x) for x in finite_r0)
    assert finite_r0 == sorted(finite_r0)
    assert rows[2]["noise_subtraction_multiplier"] == 1.0


def test_real_diagnosis_contract_including_write_and_no_input_mutation(tmp_path):
    sd = _fake_system_diagnosis()
    before_commands = sd.trs.dm.com.copy()
    before_residual = sd.trs.rec.res.copy()
    d = analyse_diagnosis(sd)
    assert d["fit"]["n_modes"] == 4
    assert d["fit"]["r0_wfs_los_m"] == 0.1
    assert d["projection"]["nominal_tip_tilt_excluded"] is True
    assert d["noise_proxies"][
        "legacy_noise_mode_variance_reproduction_relative_error"
    ] < 1e-12
    assert "PSEUDO_OL_RTC_SIGN_UNVERIFIED" in d["quality_flags"]
    assert len(d["modes"]) == 4
    assert len(d["noise_scale_sensitivity"]) == 4
    np.testing.assert_array_equal(sd.trs.dm.com, before_commands)
    np.testing.assert_array_equal(sd.trs.rec.res, before_residual)
    write_audit(d, tmp_path)
    for path in [
        "p1_modal_noise_audit.json",
        "p1_modal_noise_modes.csv",
        "p1_modal_noise_radial.csv",
        "p1_modal_noise_sensitivity.csv",
        "p1_modal_noise_radial.png",
    ]:
        assert (tmp_path/path).is_file()


def test_never_include_noll_piston_or_tiptilt_in_HO_fit():
    sd = _fake_system_diagnosis()
    sd.trs.wfs.jIndex = [2,3,4,5]
    with pytest.raises(ValueError, match="exclude Noll piston and tilt"):
        analyse_diagnosis(sd)


def test_noise_over_subtraction_is_flagged_not_silently_clipped():
    sd = _fake_system_diagnosis()
    sd.trs.wfs.var_meas[0] = sd.trs.wfs.var_noise_zer[0]*0.5
    d = analyse_diagnosis(sd)
    assert "NEGATIVE_OR_ZERO_NOISE_SUBTRACTED_MODE" in d["quality_flags"]
    assert d["modes"][0]["variance_corrected_OPD_m2"] < 0
