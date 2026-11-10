"""P2: validate the exact Kolmogorov limit of Flicker's Keck LGS cone kernel.

The finite-von-Karman implementation combines six McDonald covariance
terms, each tending to 3/5 as L0->infinity. Multiplying their differences
by L0**(5/3) produces catastrophic cancellation for huge finite L0.

The exact limit follows from
Ialpha(z) = 3/5 - a*z**(5/3) + o(z**(5/3))
where a=-Gamma(-5/6)/(2**(8/3)*Gamma(11/6)).
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.special import gamma

from p3.aoSystem.anisoplanatismModel import focal_angular_function


def _cone_kernel(l0, *, n_actu=1, n_otf=24, samp=2.5):
    tel = NS(D=10.5)
    atm = NS(L0=l0, heights=np.array([8208.7]), nL=1)
    src = NS(nSrc=1, direction=[np.zeros(1), np.zeros(1)])
    lgs = NS(
        height=np.array([122252.3]),
        direction=[np.zeros(1), np.zeros(1)],
    )
    return focal_angular_function(
        tel, atm, src, lgs, np.array([n_actu]), n_otf, samp
    )


def test_mcdonald_small_argument_asymptotic_coefficient():
    from p3.aoSystem.FourierUtils import mcDonald

    nu = 5/6
    a = -gamma(-nu)/(2**(2*nu+1)*gamma(1+nu))
    assert a > 0
    z = np.array([1e-3, 1e-4, 1e-5])
    measured = (0.6 - mcDonald(z))/z**(5/3)
    # The O(z^2) correction decays only like z^(1/3) relatively.
    assert np.all(np.isfinite(measured))
    assert measured[-1] == pytest.approx(a, rel=0.06)
    assert abs(measured[-1]-a) < abs(measured[0]-a)


def test_exact_infinite_outer_scale_is_finite_nontrivial_and_bounded():
    infinite = _cone_kernel(np.inf)
    assert infinite.shape == (1, 1, 24, 24)
    assert np.all(np.isfinite(infinite))
    assert np.max(np.abs(infinite)) > 0
    assert np.max(np.abs(infinite)) < 1e3


def test_finite_outer_scale_converges_to_analytical_infinite_limit():
    at_300 = _cone_kernel(300)
    at_100000 = _cone_kernel(100_000)
    infinity = _cone_kernel(np.inf)
    err_300 = np.linalg.norm(at_300 - infinity)
    err_100000 = np.linalg.norm(at_100000 - infinity)
    assert np.isfinite(err_100000)
    assert err_100000 < err_300
    assert err_100000/np.linalg.norm(infinity) < 0.08


def test_invalid_outer_scale_is_rejected():
    for value in (0, -5, -np.inf, np.nan):
        with pytest.raises(ValueError, match="Outer scale"):
            _cone_kernel(value)


def test_diffraction_otf_does_not_repair_unmasked_nonstationary_covariance_average():
    # A synthetic one-dimensional pair example.  For a baseline of one
    # pixel, three pairs exist on the 4-point computation grid; the real
    # pupil illuminates only the first two.
    pair_dphi = np.array([0.4, 1.4, 2.4])
    pupil = np.array([1.0, 1.0, 1.0, 0.0])
    weights = pupil[:-1]*pupil[1:]
    assert list(weights) == [1.0, 1.0, 0.0]
    exact_normalized_otf_transfer = np.sum(
        weights*np.exp(-0.5*pair_dphi)
    )/weights.sum()
    # matrix_to_map acts like a uniform pair average at this baseline,
    # then multiplying by the diffraction OTF supplies only the
    # aggregate pair count, not each pair's lost covariance information.
    unmasked_average = float(pair_dphi.mean())
    stationary_approx_transfer = np.exp(-0.5*unmasked_average)
    assert exact_normalized_otf_transfer != pytest.approx(
        stationary_approx_transfer, rel=0.01
    )
    # Even with the correct mask, <exp(-D/2)> and exp(-<D>/2)
    # are distinct for a nonstationary phase structure function.
    masked_average = np.sum(weights*pair_dphi)/weights.sum()
    assert exact_normalized_otf_transfer != pytest.approx(
        np.exp(-masked_average/2), rel=0.01
    )


def test_stationary_phase_structure_function_factors_exactly():
    pupil = np.array([1.0, 1.0, 1.0, 0.0])
    weights = pupil[:-1]*pupil[1:]
    dphi = np.array([0.7, 0.7, 0.7])
    exact = np.sum(weights*np.exp(-dphi/2))/weights.sum()
    stationary = np.exp(-dphi[0]/2)
    assert exact == pytest.approx(stationary, rel=1e-14)
