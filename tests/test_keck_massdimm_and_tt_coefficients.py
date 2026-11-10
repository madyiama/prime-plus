"""MASS/DIMM input quality and explicit TT OTF kernel-convention tests."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.telemetry.massdimm import DIMM, MASS, MASSPROF, CombineMASSandDIMM
from p3.psfr.psfR import psfR


@pytest.mark.parametrize("kind", ["dimm", "mass", "masspro"])
def test_site_monitor_rejects_stale_samples_without_verbose(kind, tmp_path):
    # Original telemetry classes convert 00:00 HST to 10:00 UTC.
    date = "2020 12 02 00 00 00"
    if kind == "masspro":
        line = date + " " + " ".join(["1e-14"] * 6) + " 0.50\n"
        cls = MASSPROF
    elif kind == "mass":
        line = date + " 0.50\n"
        cls = MASS
    else:
        line = date + " 0.75\n"
        cls = DIMM
    target = tmp_path / f"data.{kind}"
    target.write_text(line)
    monitor = cls(str(target), verbose=False)

    good, delta = monitor.indexTime([10.5])
    assert float(np.asarray(delta).reshape(-1)[0]) == pytest.approx(0.5)
    assert np.all(np.asarray(good, dtype=float) >= 0)

    stale, delta = monitor.indexTime([12.0])
    assert float(np.asarray(delta).reshape(-1)[0]) == pytest.approx(2.0)
    assert np.all(np.asarray(stale, dtype=float) == -1)


def test_mass_dimm_ground_cn2_integral_requires_total_seeing_above_mass():
    upper = np.full(6, 1.0e-14)
    combined = CombineMASSandDIMM(0.8, 0.45, upper, wvl=500e-9)
    assert combined.shape == (7,)
    assert np.all(np.isfinite(combined))
    assert np.all(combined >= 0)
    np.testing.assert_array_equal(combined[1:], upper)
    with pytest.raises(ValueError, match="exceeds"):
        CombineMASSandDIMM(0.4, 0.7, upper)
    with pytest.raises(ValueError, match="nonnegative"):
        CombineMASSandDIMM(0.8, 0.5, np.array([1e-14, -1e-14]*3))


def _minimal_tt_psfr_model():
    model = psfR.__new__(psfR)
    model.ao = NS(tel=NS(D=10.5))
    model.trs = NS(tel=NS(pupilAngle=0.0))
    u, v = np.meshgrid(
        np.arange(16, dtype=float)/16 - 0.5,
        np.arange(16, dtype=float)/16 - 0.5
    )
    model.freq = NS(wvlRef=2.2e-6, sampRef=2.0, U_=u, V_=v)
    return model


def test_tt_kernel_noll_and_legacy_difference_is_traceable_factor_16():
    model = _minimal_tt_psfr_model()
    # Coefficients in RMS-normalized Noll-Z2/Z3 OPD units, not angle.
    cov_opd = np.diag([1e-14, 2e-14])
    model.tt_kernel_dphi_multiplier_vs_noll = 1.0
    dphi_noll = model.jitter_phase_structure_function(Ctt=cov_opd)
    model.tt_kernel_dphi_multiplier_vs_noll = 1.0 / 16.0
    dphi_legacy = model.jitter_phase_structure_function(Ctt=cov_opd)
    np.testing.assert_allclose(dphi_noll, 16.0*dphi_legacy, atol=1e-14, rtol=1e-13)
    assert np.max(dphi_noll) > 0
    # An independent angular covariance must give the physical formula:
    cov_theta = cov_opd * (4.0/10.5)**2
    expected = (2*np.pi*10.5/2.2e-6)**2 * (
        cov_theta[0,0]*model.freq.U_**2
        + cov_theta[1,1]*model.freq.V_**2
    )
    np.testing.assert_allclose(dphi_noll, expected, atol=1e-13, rtol=1e-13)
