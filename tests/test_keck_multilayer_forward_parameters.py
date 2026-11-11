"""Keck telemetry-only PSF-R forward vector: 1/3/7 layer contract.

No science image, Keck archive, Fourier propagation, or PSF fitting needed.
The test explicitly exercises FourierUtils' native parameter parser, whose
atmospheric interpretation changes when nL>1 and nRecLayers>1.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.FourierUtils import sort_params_from_labels
from p3.psfr.psfR import psfR
from scripts.keck_n0013_smoke import _telemetry_forward_parameters


def _model(n_layers, n_reconstructed=7, r0=0.3654040960224877):
    weights = np.arange(1, n_layers + 1, dtype=float)
    weights /= weights.sum()
    ao = NS(
        atm=NS(
            nL=n_layers, r0=r0, weights=weights, wvl=1.53e-6
        ),
        dms=NS(nRecLayers=n_reconstructed),
        src=NS(nSrc=1, wvl=np.array([1.53e-6])),
        cam=NS(
            transmittance=np.array([1.0, 0.8, 0.6]),
            dispersion=[np.zeros(3), np.zeros(3)],
        ),
    )
    return NS(
        ao=ao,
        nwvl=3,
        n_param_atm=min(n_layers, n_reconstructed),
        n_param_dphi=2,
        param_labels=["Cn2", "gho", "gtt", "F", "dx", "dy", "bkg", "stat"],
    )


@pytest.mark.parametrize("n_layers", [1, 3, 7])
def test_forward_vector_roundtrip_and_gain_position(n_layers):
    model = _model(n_layers)
    params = _telemetry_forward_parameters(model)
    cn2, parsed_r0, gains, jitter, star, stat = (
        sort_params_from_labels(model, params)
    )
    assert len(params) == (n_layers if n_layers > 1 else 1) + 2
    np.testing.assert_allclose(gains, [1.0, 1.0])
    assert parsed_r0 == pytest.approx(model.ao.atm.r0, rel=1e-12)
    assert jitter is None
    assert stat is None
    np.testing.assert_allclose(star[0], [[1.0, 0.8, 0.6]])
    if n_layers == 1:
        assert cn2 is None
        assert params[0] == pytest.approx(model.ao.atm.r0)
    else:
        assert len(cn2) == n_layers
        np.testing.assert_allclose(
            cn2, model.ao.atm.weights*model.ao.atm.r0**(-5/3)
        )
        assert cn2.sum() == pytest.approx(model.ao.atm.r0**(-5/3))


def test_seven_layer_vector_prevents_three_parameter_parse_failure():
    model = _model(7)
    _, _, bad_gains, _, _, _ = sort_params_from_labels(
        model, [model.ao.atm.r0, 1.0, 1.0]
    )
    assert bad_gains == []  # The exact error in the original user traceback.
    params = _telemetry_forward_parameters(model)
    _, _, gains, _, _, _ = sort_params_from_labels(model, params)
    assert gains == [1.0, 1.0]


def test_multilayer_forward_uses_layer_integrated_turbulence_for_anisoplanatism():
    model = _model(7)
    params = _telemetry_forward_parameters(model)
    cn2, parsed_r0, gains, _, _, _ = sort_params_from_labels(model, params)
    ny = nx = 4
    kernel = np.arange(1, 8, dtype=float)
    yy, xx = np.indices((ny, nx))
    separations = (yy - ny//2)**2 + (xx - nx//2)**2
    physics = psfR.__new__(psfR)
    physics.ao = model.ao
    physics.freq = NS(isAniso=True, wvlRef=1.53e-6)
    physics.dphi_ao = np.zeros((ny, nx))
    physics.dphi_tt = np.zeros((ny, nx))
    physics.dphi_fit = np.zeros((ny, nx))
    physics.dphi_alias = np.zeros((ny, nx))
    physics.dphi_ani = kernel[np.newaxis, :, np.newaxis, np.newaxis] * (
        0.1 + separations[np.newaxis, np.newaxis, :, :]
    )
    sf = physics.phase_structure_function(parsed_r0, gains, Cn2=cn2)
    assert sf.shape == (ny, nx, 1)
    expected_scale = np.dot(
        model.ao.atm.weights, kernel
    ) * parsed_r0**(-5.0/3.0)
    np.testing.assert_allclose(
        sf[:, :, 0], expected_scale*separations, atol=1e-12, rtol=1e-12
    )
    assert sf[ny//2, nx//2, 0] == pytest.approx(0, abs=1e-12)
    assert physics.dani_raw_zero_lag_rad2[0] == pytest.approx(
        0.1*expected_scale
    )


def test_multilayer_atmosphere_with_single_reconstructor_expects_r0_not_strengths():
    model = _model(7, n_reconstructed=1)
    params = _telemetry_forward_parameters(model)
    assert len(params) == 3
    cn2, r0, gains, *_ = sort_params_from_labels(model, params)
    assert cn2 is None
    assert r0 == pytest.approx(model.ao.atm.r0)
    assert gains == [1.0, 1.0]


@pytest.mark.parametrize("weights", [
    np.array([0.2, 0.3, 0.4]),
    np.array([0.5, -0.2, 0.7]),
    np.array([0.5, np.nan, 0.5]),
])
def test_bad_layer_weights_are_rejected_before_forward(weights):
    model = _model(3)
    model.ao.atm.weights = weights
    with pytest.raises(ValueError, match="normalized to one"):
        _telemetry_forward_parameters(model)


def test_invalid_reference_r0_rejected_before_forward():
    model = _model(7, r0=0.0)
    with pytest.raises(ValueError, match="positive r0"):
        _telemetry_forward_parameters(model)


def test_missing_multilayer_ho_tt_gains_raise_clear_error():
    model = psfR.__new__(psfR)
    with pytest.raises(ValueError, match="two finite HO/TT gains"):
        model.phase_structure_function(0.3654, [], Cn2=np.ones(7))


def test_smoke_rejects_local_noncanonical_tt_conversion_before_long_reconstruction():
    from unittest.mock import patch

    from scripts.keck_n0013_smoke import _validate_canonical_tt_geometry

    _validate_canonical_tt_geometry(10.5)
    invalid = np.eye(2) * (1.0/10.5)**2
    with patch(
        "p3.aoSystem.tipTiltGeometry.opd_covariance_to_angular",
        return_value=invalid,
    ):
        with pytest.raises(ValueError, match="Noncanonical TT"):
            _validate_canonical_tt_geometry(10.5)
