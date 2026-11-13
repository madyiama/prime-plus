"""Synthetic regression of the stateless HO boundary and legacy adapter."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.psfr.ho import estimate_ho_statistics, reconstruct_ho_otf
from p3.psfr.psfR import psfR
from p3.psfr.psfrUtils import modes2Otf


def fixture():
    rng = np.random.default_rng(123456)
    commands = rng.normal(scale=1e-8, size=(101, 3))
    noise = np.diag([1e-18, 2e-18, 1.5e-18])
    yy, xx = np.mgrid[-1:1:8j, -1:1:8j]
    pupil = ((xx*xx+yy*yy) < 0.9).astype(float)
    modes = np.column_stack((xx.ravel(), yy.ravel(), (xx*yy).ravel()))
    return commands, noise, pupil, modes


@pytest.mark.parametrize("method", ["dm-based", "slopes-based"])
def test_estimator_matches_legacy_algebra_without_input_mutation(method):
    commands, noise, _, _ = fixture()
    commands_original, noise_original = commands.copy(), noise.copy()
    gain, wavelength = 0.8, 2.055e-6
    residual = (np.diff(commands, axis=0)/gain
                if method == "dm-based" else commands)
    empirical = np.dot(residual.T, residual)/residual.shape[0]
    corrected = (2*np.pi/wavelength)**2*(empirical-noise)
    stats = estimate_ho_statistics(
        commands, noise, reference_wavelength_m=wavelength,
        method=method, loop_gain=gain if method == "dm-based" else None,
    )
    np.testing.assert_array_equal(stats.empirical_covariance_m2, empirical)
    np.testing.assert_array_equal(stats.modal_covariance_rad2, corrected)
    assert stats.n_samples == residual.shape[0]
    assert stats.method == method
    np.testing.assert_array_equal(commands, commands_original)
    np.testing.assert_array_equal(noise, noise_original)


def test_functional_otf_matches_legacy_vii_bitwise():
    commands, noise, pupil, modes = fixture()
    stats = estimate_ho_statistics(
        commands, noise, reference_wavelength_m=2.055e-6, loop_gain=0.8,
    )
    diagnostics = {}
    new_otf, new_sf = reconstruct_ho_otf(
        stats, modes, pupil, 16, sampling=2,
        covariance_diagnostics=diagnostics,
    )
    old_otf, old_sf = modes2Otf(
        stats.modal_covariance_rad2, modes, pupil, 16, samp=2,
        covariance_psd_policy="strict",
    )
    np.testing.assert_array_equal(new_otf, old_otf)
    np.testing.assert_array_equal(new_sf, old_sf)
    assert diagnostics["policy"] == "strict"
    assert diagnostics["projection_applied"] is False
    assert new_otf.max() == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["dm-based", "slopes-based"])
def test_legacy_adapter_preserves_ho_science_and_error_budget(method):
    commands, noise, pupil, modes = fixture()
    gain, wavelength = 0.8, 2.055e-6
    model = psfR.__new__(psfR)
    model.ao = NS(rtc=NS(holoop={"gain": gain}), tel=NS(pupil=pupil))
    model.freq = NS(wvlRef=wavelength, nOtf=16, sampRef=2)
    model.trs = NS(
        dm=NS(com=commands), rec=NS(res=commands),
        mat=NS(dmIF=modes), wfs=NS(Cn_ao=noise),
    )
    model.ho_covariance_policy = "strict"
    model.ho_covariance_diagnostics = {}
    residual = (np.diff(commands, axis=0)/gain
                if method == "dm-based" else commands)
    raw = np.dot(residual.T, residual)/residual.shape[0]
    _, expected = modes2Otf(
        (2*np.pi/wavelength)**2*(raw-noise), modes, pupil, 16,
        samp=2, covariance_psd_policy="strict",
    )
    sf = model.residual_phase_structure_function(method=method)
    np.testing.assert_array_equal(sf, expected)
    np.testing.assert_array_equal(model.Cao, raw)
    assert model.ho_covariance_diagnostics["policy"] == "strict"


@pytest.mark.parametrize("gain", [None, 0, np.nan, np.inf])
def test_dm_requires_finite_nonzero_explicit_loop_gain(gain):
    commands, noise, _, _ = fixture()
    with pytest.raises(ValueError, match="loop_gain"):
        estimate_ho_statistics(
            commands, noise, reference_wavelength_m=2e-6, loop_gain=gain
        )


def test_reject_invalid_method_samples_noise_and_wavelength():
    commands, noise, _, _ = fixture()
    base = dict(reference_wavelength_m=2e-6, loop_gain=1.0)
    for measurements in (commands[:1], commands[:0], commands.ravel()):
        with pytest.raises(ValueError, match="HO measurements|frames"):
            estimate_ho_statistics(measurements, noise, **base)
    with pytest.raises(ValueError, match="noise covariance shape"):
        estimate_ho_statistics(commands, np.eye(2), **base)
    corrupt = noise.copy()
    corrupt[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        estimate_ho_statistics(commands, corrupt, **base)
    with pytest.raises(ValueError, match="non-finite"):
        damaged = commands.copy()
        damaged[1, 0] = np.nan
        estimate_ho_statistics(damaged, noise, **base)
    with pytest.raises(ValueError, match="HO method"):
        estimate_ho_statistics(
            commands, noise, method="invalid", **base
        )
    for wavelength in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="reference wavelength"):
            estimate_ho_statistics(
                commands, noise,
                reference_wavelength_m=wavelength, loop_gain=1.0,
            )


def test_strict_policy_rejects_negative_noise_subtracted_covariance():
    commands, _, pupil, modes = fixture()
    differences = np.diff(commands, axis=0)
    raw = np.dot(differences.T, differences)/len(differences)
    noise = raw + np.eye(raw.shape[0])*1e-15
    stats = estimate_ho_statistics(
        commands, noise, reference_wavelength_m=2e-6, loop_gain=1.0,
    )
    assert np.linalg.eigvalsh(stats.modal_covariance_rad2).min() < 0
    with pytest.raises(ValueError, match="positive-semidefinite"):
        reconstruct_ho_otf(stats, modes, pupil, 16)


def _full_cube_vii_reference(covariance, modes, pupil, n_otf, *, samp=2):
    """Pre-optimization full padded cube, retained ONLY in small science tests."""
    from p3.aoSystem import FourierUtils

    n_pup = pupil.shape[0]
    padded = FourierUtils.enlargeSupport(pupil, samp)
    fft_pup = np.fft.fft2(padded)
    conj_pup = np.conjugate(fft_pup)
    G = np.fft.fftshift(np.real(np.fft.ifft2(abs(fft_pup)**2)))
    den = np.zeros_like(G)
    mask = G / G.max() > 1e-7
    den[mask] = 1 / G[mask]

    values, eigenvectors = np.linalg.eigh(
        0.5 * (covariance + covariance.T)
    )
    assert np.all(values >= 0)
    full = np.dot(modes, eigenvectors)
    full = full.reshape(n_pup, n_pup, len(values))
    full = FourierUtils.enlargeSupport(full, samp) * padded[:, :, None]
    buf = np.zeros_like(G)
    for mode_index in range(len(values)):
        Mk = full[:, :, mode_index]
        Vk = (
            np.real(np.fft.fft2(Mk**2) * conj_pup)
            - abs(np.fft.fft2(Mk))**2
        )
        buf = buf + values[mode_index] * Vk
    sf = den * np.fft.fftshift(np.real(np.fft.ifft2(2 * buf)))
    otf = G * np.exp(-0.5 * sf)
    otf = otf * (G > 1e-5)
    otf = FourierUtils.interpolateSupport(otf, n_otf)
    sf = FourierUtils.interpolateSupport(sf, n_otf)
    otf = otf / otf.max()
    return otf, sf


@pytest.mark.parametrize("n_modes", [3, 8, 16])
def test_vii_streamed_eigenmodes_match_independent_full_cube_reference(
    n_modes, monkeypatch,
):
    """Preserve complex Fourier Vii physics while avoiding 3-D padding."""
    from p3.aoSystem import FourierUtils

    rng = np.random.default_rng(156 + n_modes)
    n_pup = 8
    y, x = np.mgrid[-1:1:8j, -1:1:8j]
    pupil = ((x*x+y*y) < 0.9).astype(float)
    modes = rng.normal(size=(n_pup*n_pup, n_modes))
    basis = rng.normal(size=(n_modes, n_modes))
    covariance = basis @ basis.T / n_modes + np.eye(n_modes) * 0.1

    reference_otf, reference_sf = _full_cube_vii_reference(
        covariance, modes, pupil, 16, samp=2,
    )
    original = FourierUtils.enlargeSupport
    seen = []

    def observe_support(cube, sampling):
        seen.append(tuple(np.asarray(cube).shape))
        return original(cube, sampling)

    monkeypatch.setattr(FourierUtils, "enlargeSupport", observe_support)
    otf, sf = modes2Otf(
        covariance, modes, pupil, n_otf=16, samp=2,
        covariance_psd_policy="strict",
    )

    # The former implementation enlarged the full
    # (8,8,n_modes) cube to (16,16,n_modes).
    assert seen
    assert all(len(shape) == 2 for shape in seen)
    np.testing.assert_allclose(otf, reference_otf, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(sf, reference_sf, rtol=1e-11, atol=1e-11)
