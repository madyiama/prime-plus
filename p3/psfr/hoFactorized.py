"""Experimental algebraic factorization of the exact Vii modal formula.

For pupil-masked spatial eigenmodes M_k and eigenvalue weights lambda_k:

sum_k lambda_k * (Re[FFT(M_k**2) * conj(FFT(P))] - |FFT(M_k)|**2)
= Re[FFT(sum_k lambda_k * M_k**2) * conj(FFT(P))]
  - sum_k lambda_k * |FFT(M_k)|**2.

This uses one rather than n_modes FFTs of squared modes. The
PHYSICAL expression is identical; floating-point summation/FFT
reordering MAY NOT reproduce the original result within 1e-12 for
every OTF pixel. Never enable in science runs without checking both
HO phase SF and the final PSF on the exact Keck acquisition.

No cached pupil/OTF/eigenmodes, parallel workers or GPU.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np


def accumulate_vii_factorized(
    modes_padded: np.ndarray,
    eigenvalues: np.ndarray,
    pupil_fft_conjugate: np.ndarray,
    *,
    diagnostics: dict | None = None,
) -> np.ndarray:
    """Compute the Vii numerator in original float64/complex128 precision.

    The modal sums are sequential in exactly the same eigenvalue order,
    but their grouping changes relative to the historical sum of two FFT
    terms; this is an EXPERIMENTAL approximation at roundoff level.
    """
    t0=perf_counter()
    modes=np.asarray(modes_padded)
    weights=np.asarray(eigenvalues)
    conj=np.asarray(pupil_fft_conjugate)
    if (modes.ndim!=3 or conj.shape!=modes.shape[:2] or weights.ndim!=1
            or weights.size!=modes.shape[2]):
        raise ValueError("Vii factorized modal/pupil dimensions are inconsistent")
    if not (np.isrealobj(modes) and np.isfinite(modes).all()
            and np.isfinite(weights).all() and (weights>=0).all()
            and np.isfinite(conj).all()):
        raise ValueError("Vii factorized inputs must be finite real modes / PSD weights")
    # Use the original np.fft backend. No modal FFT vectorization or
    # multithreading here; the experimental variable is factorization alone.
    sum_squares=np.zeros(modes.shape[:2],dtype=np.float64)
    sum_power=np.zeros(modes.shape[:2],dtype=np.float64)
    for k in range(modes.shape[2]):
        mode=modes[:,:,k]
        weighted_sq=weights[k]*(mode**2)
        sum_squares=sum_squares+weighted_sq
        spectrum=np.fft.fft2(mode)
        sum_power=sum_power+weights[k]*np.abs(spectrum)**2
    out=np.real(np.fft.fft2(sum_squares)*conj)-sum_power
    if diagnostics is not None:
        diagnostics.update({
            "method":"factorized",
            "mode_count":int(weights.size),
            "fft2_count_nominal":int(weights.size)+1,
            "square_mode_fft2_eliminated":int(weights.size)-1,
            "sequential_mode_summation":True,
            "fft_backend":"numpy",
            "workers":1,
            "pupil_or_ho_otf_cached":False,
            "kernel_elapsed_ms":(perf_counter()-t0)*1000,
            "numerical_reordering":True,
            "requires_per_case_ho_sf_and_psf_parity":True,
        })
    return out
