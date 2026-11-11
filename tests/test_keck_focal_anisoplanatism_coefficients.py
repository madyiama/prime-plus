"""Independent Hardy (1998) cone-effect coefficient and unit tests.

Reference: Beltramo-Martin et al., MNRAS 487 (2019), eq. 12,
https://academic.oup.com/mnras/article/487/4/5450/5519866
The 0.452 coefficient in that equation is NOT the 0.425 found in an
older duplicate P3 helper. 0.423 is required when writing the expression
in Fried-r0 units rather than integrated Cn2 dh units.
"""

from types import SimpleNamespace as NS
import numpy as np
import pytest

from p3.aoSystem.anisoplanatismModel import (
    focal_anisoplanatism_phase_variance,
    focal_anisoplanatism_wfe,
    focal_anisoplanatism_variance,
)


def _case(heights, weights, *, wavelength=500e-9, r0=0.11474041274562248):
    return (
        NS(D=10.5),
        NS(
            heights=np.asarray(heights, dtype=float),
            weights=np.asarray(weights, dtype=float),
            r0=float(r0),
            wvl=float(wavelength),
        ),
        NS(height=np.array([90_000.0])),
    )


def _reference_phase_variance(tel, atm, lgs):
    """Independent direct Hardy integral over atmospheric layer strengths."""
    x = atm.heights / lgs.height[0]
    reference_kernel = (0.5 * x**(5/3) - 0.452*x*x)/0.423
    reference_kernel = np.where(
        x >= 1, 1.0299, np.clip(reference_kernel, 0, 1.0299)
    )
    return (tel.D/atm.r0)**(5/3) * np.dot(atm.weights, reference_kernel)


def test_analytical_cone_coefficients_match_independent_hardy_expression():
    tel, atm, lgs = _case(
        heights=[0., 4000., 8000., 16000.],
        weights=[0.48, 0.04, 0.41, 0.07],
    )
    phase_variance = focal_anisoplanatism_phase_variance(tel, atm, lgs)
    np.testing.assert_allclose(
        phase_variance, _reference_phase_variance(tel, atm, lgs),
        rtol=1e-13, atol=1e-14,
    )
    expected_nm = np.sqrt(phase_variance) * atm.wvl*1e9/(2*np.pi)
    assert focal_anisoplanatism_wfe(tel, atm, lgs) == pytest.approx(expected_nm)
    # Historic API name was misleading: it also returns WFE in nm.
    assert focal_anisoplanatism_variance(tel, atm, lgs) == pytest.approx(expected_nm)


def test_n0120_analytical_7_layers_have_lower_cone_wfe_than_equivalent_layer():
    # Physical n0120 MASS/DIMM profile captured in the atmospheric audit.
    weights = np.array([
        0.4837829375341058, 2.8954981850976237e-11,
        7.28398762188621e-11, 4.0265521636513835e-06,
        0.044337315959307365, 0.4071794322793533,
        0.06469628757327503,
    ])
    heights = np.array([0., 500., 1000., 2000., 4000., 8000., 16000.])
    heq = np.sum(weights*heights**(5/3))**(3/5)
    seven = _case(heights, weights)
    one = _case([heq], [1.0])
    wfe_one = focal_anisoplanatism_wfe(*one)
    wfe_seven = focal_anisoplanatism_wfe(*seven)
    assert heq == pytest.approx(6043.11093372411, rel=1e-7)
    assert wfe_one == pytest.approx(312.4173095, rel=3e-6)
    assert wfe_seven == pytest.approx(295.0074943, rel=3e-6)
    assert wfe_seven < wfe_one
    # The reference formula and the OTF-based error budget describe
    # different approximations; a different trend is a reason to audit,
    # not to tune the analytical coefficient.


def test_ground_layer_zero_and_above_beacon_full_unsensed_variance():
    tel, atm, lgs = _case([0], [1.0])
    assert focal_anisoplanatism_phase_variance(tel, atm, lgs) == pytest.approx(0)
    tel, atm, lgs = _case([95_000], [1.0])
    assert focal_anisoplanatism_phase_variance(
        tel, atm, lgs
    ) == pytest.approx(1.0299 * (tel.D/atm.r0)**(5/3))


@pytest.mark.parametrize("weights", [
    [0.7, 0.7],
    [-0.1, 1.1],
    [np.nan, 1.0],
])
def test_cone_analytical_coefficients_reject_invalid_weights(weights):
    tel, atm, lgs = _case([0, 8000], weights)
    with pytest.raises(ValueError, match="normalized layer weights"):
        focal_anisoplanatism_phase_variance(tel, atm, lgs)


def test_opd_wfe_is_invariant_under_consistent_wavelength_scaling():
    tel, atm, lgs = _case([0, 8000], [0.5, 0.5])
    wfe_500 = focal_anisoplanatism_wfe(tel, atm, lgs)
    atm_k = NS(
        heights=atm.heights, weights=atm.weights,
        r0=atm.r0*(2.2e-6/atm.wvl)**(6/5), wvl=2.2e-6,
    )
    assert focal_anisoplanatism_wfe(
        tel, atm_k, lgs
    ) == pytest.approx(wfe_500, rel=1e-12)
