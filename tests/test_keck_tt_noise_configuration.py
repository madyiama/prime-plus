"""Verify configuration and integration of Keck TT-noise method selection."""
from pathlib import Path
from types import SimpleNamespace as NS
import tomllib

import numpy as np

from p3.telemetry.systemDiagnosis import systemDiagnosis


def _fake_diagnosis():
    n = 8192
    rng = np.random.default_rng(4)
    s = rng.normal(0, [1e-7, 2e-7], (n, 2))
    frequencies = np.linspace(0, 62.5, 4096)
    trs = NS(
        aoMode="LGS",
        dm=NS(com=rng.normal(0, 1e-7, (n, 4))),
        tipTilt=NS(slopes=s),
        ttloop=NS(freq=125.0, tf=NS(
            freq=frequencies, rtf=np.ones_like(frequencies),
        )),
    )
    diag = systemDiagnosis.__new__(systemDiagnosis)
    diag.trs = trs
    return diag


def test_keck_default_toml_uses_van_dam():
    p = Path(__file__).parents[1] / "config" / "keck_tt_noise.toml"
    with p.open("rb") as file:
        cfg = tomllib.load(file)["tip_tilt_noise"]
    assert cfg["method"] == "van_dam"
    assert len(cfg["high_frequency_band"]) == 2


def test_system_diagnosis_uses_van_dam_for_lgs_and_preserves_ho_method():
    diag = _fake_diagnosis()
    cn_ho, cn_tt = diag.get_noise_variance()
    assert cn_ho.shape == (4, 4)
    assert cn_tt.shape == (2, 2)
    assert np.linalg.eigvalsh(cn_tt).min() >= -1e-24
    assert diag.trs.tipTilt.noise_estimation_method == "van_dam"
    assert diag.trs.tipTilt.noise_estimation_diagnostics["fit_bin_count"] >= 8


def test_system_diagnosis_can_select_gendron():
    diag = _fake_diagnosis()
    _, cn_tt = diag.get_noise_variance(tt_noise_method="gendron")
    assert cn_tt.shape == (2, 2)
    assert diag.trs.tipTilt.noise_estimation_method == "gendron"


def test_system_diagnosis_can_select_legacy_autocorrelation():
    diag = _fake_diagnosis()
    _, cn_tt = diag.get_noise_variance(tt_noise_method="legacy_autocorrelation")
    assert cn_tt.shape == (2, 2)
    assert diag.trs.tipTilt.noise_estimation_method == "legacy_autocorrelation"
