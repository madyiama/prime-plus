"""Pure high-order (HO) telemetry statistics and OTF reconstruction.

This module reproduces the historical P3 estimator, including its *uncentred*
second moment and the lack of a Bessel correction. Measurements must be OPD
in metres in the DM actuator basis. No covariance PSD projection is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

import numpy as np


@dataclass(frozen=True)
class HOStatistics:
    """A pupil-independent HO product, before Fourier propagation.

    empirical_covariance_m2 is the actuator OPD second moment [m**2].
    modal_covariance_rad2 is the noise-subtracted phase covariance [rad**2]
    at the reference science wavelength. It may be indefinite.
    """

    empirical_covariance_m2: np.ndarray
    modal_covariance_rad2: np.ndarray
    n_samples: int
    method: str
    reference_wavelength_m: float


def estimate_ho_statistics(
    measurements: np.ndarray,
    noise_covariance_m2: np.ndarray,
    *,
    reference_wavelength_m: float,
    method: str = "dm-based",
    loop_gain: float | None = None,
) -> HOStatistics:
    """Estimate HO statistics without an aoSystem or telemetry object.

    Measurements are [frame, actuator] OPD metres: consecutive raw DM
    commands for dm-based, already reconstructed residual OPD for
    slopes-based (NOT WFS centroid slopes). DM mode differences successive
    commands and divides by the loop gain; slopes mode does neither.

    The result uses Cemp = residual.T @ residual / N (no mean removal) and
    Cphase = (2*pi/wavelength)**2 * (Cemp - Cnoise).
    The noise covariance must already be calibrated in actuator OPD m**2.
    """
    if method not in ("dm-based", "slopes-based"):
        raise ValueError("HO method must be 'dm-based' or 'slopes-based'")
    data = np.asarray(measurements, dtype=float)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("HO measurements must have shape [frame, actuator]")
    min_frames = 2 if method == "dm-based" else 1
    if data.shape[0] < min_frames:
        raise ValueError("HO method requires more measurement frames")
    if not np.isfinite(data).all():
        raise ValueError("HO measurements contain non-finite samples")
    n_act = data.shape[1]
    noise = np.asarray(noise_covariance_m2, dtype=float)
    if noise.shape != (n_act, n_act):
        raise ValueError("HO noise covariance shape differs from actuator count")
    if not np.isfinite(noise).all():
        raise ValueError("HO noise covariance contains non-finite values")
    wavelength = float(reference_wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("HO reference wavelength must be finite and positive")

    if method == "dm-based":
        if loop_gain is None:
            raise ValueError("HO dm-based method requires explicit loop_gain")
        gain = float(loop_gain)
        if not np.isfinite(gain) or gain == 0:
            raise ValueError("HO loop_gain must be finite and nonzero")
        residual = np.diff(data, axis=0) / gain
    else:
        residual = data

    empirical = np.dot(residual.T, residual) / residual.shape[0]
    corrected = (2 * np.pi / wavelength)**2 * (empirical - noise)
    if not np.isfinite(corrected).all():
        raise ValueError("HO phase covariance became non-finite after scaling")
    return HOStatistics(
        empirical_covariance_m2=empirical,
        modal_covariance_rad2=corrected,
        n_samples=residual.shape[0],
        method=method,
        reference_wavelength_m=wavelength,
    )


def reconstruct_ho_otf(
    statistics: HOStatistics,
    influence_modes: np.ndarray,
    pupil: np.ndarray,
    n_otf: int,
    *,
    sampling: float = 2,
    basis: str = "Vii",
    covariance_psd_policy: str = "strict",
    covariance_diagnostics: MutableMapping | None = None,
    fft_mode: str = "legacy",
    fft_batch_size: int = 16,
    fft_workers: int = 1,
    fft_diagnostics: MutableMapping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (OTF, phase SF rad**2) using the UNCHANGED legacy Vii/Uij kernel.

    This boundary is stateless except for an explicitly passed optional
    diagnostics dictionary. The default covariance PSD policy remains strict.
    """
    from p3.psfr.psfrUtils import modes2Otf

    return modes2Otf(
        statistics.modal_covariance_rad2, influence_modes, pupil, n_otf,
        samp=sampling, basis=basis,
        covariance_psd_policy=covariance_psd_policy,
        covariance_diagnostics=covariance_diagnostics,
        fft_mode=fft_mode,fft_batch_size=fft_batch_size,
        fft_workers=fft_workers,fft_diagnostics=fft_diagnostics,
    )
