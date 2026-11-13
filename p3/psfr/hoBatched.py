"""Memory-bounded batched real-mode Fourier transforms for Vii HO statistics.

Exactly the original Gendron Vii per-mode expression:
Vk = real(FFT2(Mk**2)*conj(FFT2(P))) - abs(FFT2(Mk))**2,
buf += eigenvalue[k]*Vk.

Only FFT evaluation is grouped (SciPy pocketfft with axes=(-2,-1)
on contiguous [mode,y,x] batches);
modal weighting and summation remain strictly in ascending mode order.
No FFT, HO mode, or pupil result is cached across observations.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy import fft as scipy_fft


MAX_BATCH_WORK_BYTES = 256 * 1024 * 1024


def validate_fft_options(mode, batch_size, workers):
    if mode not in ("legacy", "batched"):
        raise ValueError("HO FFT mode must be 'legacy' or 'batched'")
    if (not isinstance(batch_size, (int, np.integer))
            or isinstance(batch_size, (bool, np.bool_)) or batch_size<1):
        raise ValueError("HO FFT batch_size must be a positive integer")
    if (not isinstance(workers, (int, np.integer))
            or isinstance(workers, (bool, np.bool_)) or workers<1):
        raise ValueError("HO FFT workers must be a positive integer")
    if mode=="legacy" and (int(batch_size)!=16 or int(workers)!=1):
        raise ValueError(
            "Non-default FFT batch_size/workers require HO FFT mode batched"
        )


def accumulate_vii_batched(
    modes_padded, eigenvalues, pupil_fft_conjugate, *,
    batch_size=16, workers=1, diagnostics=None,
):
    """Return the legacy Vii frequency-domain sum, before IFFT.

    Input modes already incorporate CURRENT pupil mask, rotation and
    sampling. No result is reused between distinct calls.
    Both fft2 operations transform spatial axes (0,1), never the modal
    axis (2). Memory guard avoids large temporary NumPy/SciPy arrays.
    """
    validate_fft_options("batched",batch_size,workers)
    modes=np.asarray(modes_padded)
    weights=np.asarray(eigenvalues)
    conj=np.asarray(pupil_fft_conjugate)
    if (modes.ndim!=3 or conj.shape!=modes.shape[:2]
            or weights.ndim!=1 or weights.size!=modes.shape[2]):
        raise ValueError("Invalid Vii mode/weight/pupil FFT dimensions")
    if (not np.isfinite(modes).all() or not np.isfinite(weights).all()
            or not np.isfinite(conj).all()):
        raise ValueError("Vii batch inputs must be finite")
    if not np.isrealobj(modes):
        raise ValueError("Vii pupil-weighted modes must be real")
    if not np.all(weights>=0):
        raise ValueError("Vii eigenvalue weights must be nonnegative")
    # Two double-precision real arrays and two complex FFT arrays,
    # with conservative overhead for broadcasts, squares and FFT work.
    single=8*modes.shape[0]*modes.shape[1]
    upper=int(batch_size)*single*12
    if upper>MAX_BATCH_WORK_BYTES:
        raise ValueError(
            "HO FFT batch would exceed 256 MiB scratch budget; "
            "decrease --ho-fft-batch-size"
        )
    buf=np.zeros(modes.shape[:2],dtype=np.float64)
    t0=perf_counter()
    n_batches=0
    for begin in range(0,modes.shape[2],int(batch_size)):
        end=min(modes.shape[2],begin+int(batch_size))
        # Contiguous mode-first layout gives pocketfft a contiguous 2-D
        # transform plane per mode. Merely passing [y,x,mode] with axes
        # (0,1) is strided and can be SLOWER than the original loop.
        # This is a temporary local copy of the CURRENT pupil modes.
        slab=np.moveaxis(modes[:,:,begin:end],-1,0).copy(order="C")
        fft_modes=scipy_fft.fft2(slab,axes=(-2,-1),workers=int(workers))
        fft_squares=scipy_fft.fft2(slab**2,axes=(-2,-1),workers=int(workers))
        for i,k in enumerate(range(begin,end)):
            # Preserve original per-mode expression and accumulation ORDER:
            # no einsum, BLAS sum reduction or double counting.
            vk=(np.real(fft_squares[i]*conj)
                - np.abs(fft_modes[i])**2)
            buf=buf+weights[k]*vk
        n_batches+=1
    if diagnostics is not None:
        diagnostics.update({
            "mode":"batched",
            "batch_size":int(batch_size),
            "workers":int(workers),
            "mode_count":int(weights.size),
            "batch_count":int(n_batches),
            "fft_storage_layout":"contiguous_mode_first_y_x",
            "vii_fft_and_accumulation_ms":1000*(perf_counter()-t0),
            "batch_scratch_budget_bytes":MAX_BATCH_WORK_BYTES,
            "pupil_modes_or_ffts_cached":False,
        })
    return buf
