"""Mathematically equivalent accelerated focal-angular covariance helpers.

The exact six Flicker terms admit symmetries under exchange of pupil
point-pair indices. The expensive McDonald Bessel function Ialpha is
radial/even, so I6 == I3.T; I4 and I5 have just n_ph**2 distinct
arguments, not n_ph**4. I1 and I2 use the signed integer baseline grid.

This changes ONLY evaluation order/rounding, not physical equations.
Scientific production keeps its legacy path unless explicitly opted in
and validated against full Keck NIRC2 PSF results.
"""
from __future__ import annotations

import numpy as np

from p3.aoSystem import FourierUtils



def parallel_ialpha(x, y, *, executor=None, workers=1,
                    min_chunk_elements=4096, diagnostics=None):
    """Evaluate the *unchanged* SciPy Ialpha kernel in deterministic row chunks.

    Only the independent elementwise radial McDonald evaluations are
    parallelized; all subsequent 6-term arithmetic and bincount reductions
    remain single-threaded and preserve the original order. Input arrays are
    broadcast as zero-copy views, not materialized pair-cubes. Each worker
    writes to a private result and the caller assembles rows in order.

    The executor is owned by the caller and reused for all focal layers.
    No interpolation, approximations or float32. Small arrays remain serial
    to avoid overhead; L0=infinity never calls this function.
    """
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("Ialpha workers must be exactly 1, 2 or 4")
    a, b = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                               np.asarray(y, dtype=np.float64))
    if (executor is None or workers == 1 or a.ndim != 2 or
            a.size < workers*min_chunk_elements or a.shape[0] < workers):
        if diagnostics is not None:
            diagnostics["ialpha_serial_calls"] = diagnostics.get(
                "ialpha_serial_calls", 0) + 1
        return FourierUtils.Ialpha(a, b)
    if diagnostics is not None:
        diagnostics["ialpha_parallel_calls"] = diagnostics.get(
            "ialpha_parallel_calls", 0) + 1
    edges = np.linspace(0, a.shape[0], workers+1, dtype=np.intp)
    tasks = [
        executor.submit(FourierUtils.Ialpha, a[edges[i]:edges[i+1]],
                        b[edges[i]:edges[i+1]])
        for i in range(workers)
    ]
    out = np.empty(a.shape, dtype=np.float64)
    for i, task in enumerate(tasks):
        out[edges[i]:edges[i+1]] = task.result()
    return out


def grouped_matrix_to_map(matrix, sep_x, sep_y):
    """Same negative-quadrant mean + reflection as legacy matrix_to_map.

    Legacy constructs a boolean mask of size ~n_ph**6 for each layer.
    Here we count all pupil pairs by their integer baseline once
    (O(n_ph**4) time and memory). No pupil-mask or output-symmetry
    convention is changed.
    """
    mat=np.asarray(matrix)
    if mat.ndim!=2 or mat.shape[0]!=mat.shape[1]:
        raise ValueError("Focal covariance matrix must be square 2D")
    n=int(np.sqrt(mat.shape[0]))
    if n*n!=mat.shape[0]:
        raise ValueError("Focal covariance dimension must be n_ph squared")
    sx=np.asarray(sep_x)
    sy=np.asarray(sep_y)
    if (sx.shape!=mat.shape or sy.shape!=mat.shape or
            not np.isfinite(sx).all() or not np.isfinite(sy).all() or
            not np.array_equal(sx,np.rint(sx)) or
            not np.array_equal(sy,np.rint(sy))):
        raise ValueError("Pupil baselines must be finite integer-coordinate maps")
    included=(sx<=0)&(sy<=0)&(sx>=1-n)&(sy>=1-n)
    flat=((sy[included].astype(np.intp)+n-1)*n+
          (sx[included].astype(np.intp)+n-1))
    counts=np.bincount(flat,minlength=n*n)
    if np.any(counts==0):
        raise ValueError("Pupil baseline grid is incomplete")
    vals=mat[included]
    if np.iscomplexobj(mat):
        totals=(np.bincount(flat,weights=np.real(vals),minlength=n*n)
                +1j*np.bincount(flat,weights=np.imag(vals),minlength=n*n))
    else:
        totals=np.bincount(flat,weights=vals,minlength=n*n)
    quadrant=(totals/counts).reshape(n,n)
    out=np.empty((2*n-1,2*n-1),dtype=quadrant.dtype)
    out[:n,:n]=quadrant
    out[:n,n:]=np.fliplr(quadrant[:,:-1])
    out[n:,:]=np.flipud(out[:n-1,:])
    return out



def _single_real_scalar(value, *, name):
    """Accept a scalar or NumPy singleton; never silently select a beacon.

    Keck's LGS height is often stored as a shape-(1,) NumPy array.
    float(array) fails with NumPy >= 2 even though the physical value
    is unambiguous. A multi-element array is not an admissible scalar
    for this single-beacon focal covariance kernel.
    """
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar") from exc
    if values.size != 1:
        raise ValueError(
            f"{name} must contain exactly one value; got shape {values.shape}"
        )
    return float(values.reshape(-1)[0])


def symmetric_focal_layer(
    pupil_axis_m, X1, Y1, rho_x, rho_y, sep_x, sep_y,
    *, cone_fraction, theta_shift_x_m, theta_shift_y_m,
    finite_outer_scale_m,
):
    """Return Flicker's original unfiltered point-pair covariance matrix.

    finite_outer_scale_m is either a finite positive L0 or +inf,
    corresponding to the exact Kolmogorov limit. The 6 covariance
    terms and their SIGNS are those used in the historical code.
    """
    u=np.asarray(pupil_axis_m,dtype=float)
    n=u.size
    if n<2 or any(np.shape(a)!=(n*n,n*n) for a in
                   (X1,Y1,rho_x,rho_y,sep_x,sep_y)):
        raise ValueError("Expected n_ph point-pair covariance geometry")
    g=_single_real_scalar(cone_fraction,name="cone_fraction")
    thx=_single_real_scalar(theta_shift_x_m,name="theta_shift_x_m")
    thy=_single_real_scalar(theta_shift_y_m,name="theta_shift_y_m")
    l0=_single_real_scalar(finite_outer_scale_m,name="finite_outer_scale_m")
    if (not np.isfinite(g) or not np.isfinite(thx) or
            not np.isfinite(thy) or
            not (np.isfinite(l0) and l0>0 or np.isposinf(l0))):
        raise ValueError("Invalid cone and outer-scale parameters")
    # For both the legacy uniform and physical-diameter pupil axes,
    # baselines are integer multiples of the 1-D pixel pitch.
    step=u[1]-u[0]
    if not np.isfinite(step) or step<=0 or not np.allclose(
        np.diff(u),step,rtol=1e-12,atol=1e-14
    ):
        raise ValueError("Focal symmetric evaluator requires a uniform pupil grid")
    lag=(np.arange(-n+1,n)*step)
    lag_x,lag_y=np.meshgrid(lag,lag)
    iy=(np.asarray(sep_y,dtype=np.intp)+n-1)
    ix=(np.asarray(sep_x,dtype=np.intp)+n-1)
    if (iy.min()<0 or iy.max()>=2*n-1 or ix.min()<0 or ix.max()>=2*n-1):
        raise ValueError("Pupil lag indices out of bounds")
    idx=np.arange(n*n,dtype=np.intp)

    if np.isposinf(l0):
        def covariance(x,y):
            return np.hypot(x,y)**(5.0/3.0)
        # Kolmogorov sign: +I1+I2-I3+I4+I5-I6
        coeffs=(1.,1.,-1.,1.,1.,-1.)
        constant=0.
    else:
        f0=2*np.pi/l0
        def covariance(x,y):
            return FourierUtils.Ialpha(f0*x,f0*y)
        # Finite L0 sign: 2 I0-I1-I2+I3-I4-I5+I6
        coeffs=(-1.,-1.,1.,-1.,-1.,1.)
        constant=6./5.

    I1=covariance(lag_x,lag_y)[iy,ix]
    I2=covariance((1-g)*lag_x,(1-g)*lag_y)[iy,ix]
    I3=covariance(rho_x-g*X1+thx,rho_y-g*Y1+thy)

    # Original geometry intentionally uses opposite pupil-pair index
    # directions in x and y. These indexing patterns retain EXACTLY
    # that convention (not an implicit telescope parity correction).
    I4_small=covariance(g*u[None,:]-thx,g*u[:,None]-thy)
    I5_small=covariance(-g*u[:,None]-thx,-g*u[None,:]-thy)
    I4=I4_small[(idx%n)[:,None],(idx//n)[None,:]]
    I5=I5_small[(idx//n)[:,None],(idx%n)[None,:]]
    # Radial covariance: I6(i,j) = I3(j,i), including angular shift.
    I6=I3.T
    a,b,c,d,e,f=coeffs
    return constant+a*I1+b*I2+c*I3+d*I4+e*I5+f*I6


def direct_focal_quadrant_map(
    pupil_axis_m, *, cone_fraction, theta_shift_x_m, theta_shift_y_m,
    finite_outer_scale_m, workers=1, executor=None, diagnostics=None,
):
    """Flicker six-term covariance, directly averaged over the legacy quadrant.

    For vector_to_separation's exact indexing, a pair (i,j) belongs to
    the negative-lag quadrant iff ix >= jx AND iy <= jy, where flattened
    pupil indices are i=ix*n+iy and j=jx*n+jy. Only these ordered pairs
    are evaluated. This does NOT change the signs, pupil grid, radial
    McDonald kernel, outer-scale limit or final reflected map.

    No full (n*n, n*n) pupil-pair matrix is allocated. This helper is valid
    only for the scalar/identity Hfilter path; the caller must fall back
    to the original full matrix before a non-scalar spatial filter.
    """
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("Ialpha workers must be exactly 1, 2 or 4")
    if workers > 1 and executor is None:
        raise ValueError("Parallel Ialpha requires a managed thread executor")
    u = np.asarray(pupil_axis_m, dtype=np.float64)
    n = u.size
    if u.ndim != 1 or n < 2 or not np.isfinite(u).all():
        raise ValueError("Expected a finite one-dimensional pupil axis")
    step = u[1] - u[0]
    if (not np.isfinite(step) or step <= 0 or
            not np.allclose(np.diff(u), step, rtol=1e-12, atol=1e-14)):
        raise ValueError("Quadrant evaluation requires a uniform pupil grid")
    g = _single_real_scalar(cone_fraction, name="cone_fraction")
    thx = _single_real_scalar(theta_shift_x_m, name="theta_shift_x_m")
    thy = _single_real_scalar(theta_shift_y_m, name="theta_shift_y_m")
    l0 = _single_real_scalar(finite_outer_scale_m,
                              name="finite_outer_scale_m")
    if (not np.isfinite(g) or not np.isfinite(thx) or
            not np.isfinite(thy) or not (
                np.isfinite(l0) and l0 > 0 or np.isposinf(l0))):
        raise ValueError("Invalid cone and outer-scale parameters")

    if np.isposinf(l0):
        def covariance(x, y):
            return np.hypot(x, y)**(5.0/3.0)
    else:
        f0 = 2*np.pi/l0
        def covariance(x, y):
            return parallel_ialpha(f0*x, f0*y, executor=executor,
                                   workers=workers, diagnostics=diagnostics)

    # Exactly the legacy negative-lag selection. Its two axes run in
    # opposite index directions because of vector_to_separation's
    # historical X/Y flattening conventions: rho_x=u[jx]-u[ix],
    # rho_y=u[iy]-u[jy].
    ix, jx = np.tril_indices(n)
    iy, jy = np.triu_indices(n)
    rho_x = (u[jx]-u[ix])[:, None]
    rho_y = (u[iy]-u[jy])[None, :]
    ui_x, uj_x = u[ix][:, None], u[jx][:, None]
    ui_y, uj_y = u[iy][None, :], u[jy][None, :]

    # I3(i,j), I4(i,j), I5(i,j), I6(i,j)=I3(j,i).
    # I4 and I5 need n*n distinct Bessel evaluations, not n**4.
    i4_small = covariance(g*u[None, :]-thx, g*u[:, None]-thy)
    i5_small = covariance(-g*u[:, None]-thx, -g*u[None, :]-thy)
    pair = covariance(rho_x-g*uj_x+thx, rho_y-g*ui_y+thy)
    i4 = i4_small[iy[None, :], jx[:, None]]
    i5 = i5_small[ix[:, None], jy[None, :]]
    i6 = covariance(-rho_x-g*ui_x+thx, -rho_y-g*uj_y+thy)
    if np.isposinf(l0):
        # Kolmogorov: I1+I2-I3+I4+I5-I6.
        pair = -pair+i4+i5-i6
    else:
        # Finite von Karman: 6/5-I1-I2+I3-I4-I5+I6.
        pair = pair-i4-i5+i6

    # A scalar baseline term may be averaged AFTER pair aggregation;
    # its value is identical for all pairs in a given baseline bin.
    # np.bincount is the same operation used by grouped_matrix_to_map.
    lag_bin = (((iy-jy)[None, :]+n-1)*n
               + ((jx-ix)[:, None]+n-1))
    totals = np.bincount(lag_bin.ravel(), weights=pair.ravel(),
                         minlength=n*n)
    counts = np.bincount(lag_bin.ravel(), minlength=n*n)
    if np.any(counts == 0):
        raise ValueError("Incomplete focal quadrant baseline grid")
    quadrant = (totals/counts).reshape(n, n)
    lag = np.arange(1-n, 1)*step
    lag_x, lag_y = np.meshgrid(lag, lag)
    i1 = covariance(lag_x, lag_y)
    i2 = covariance((1-g)*lag_x, (1-g)*lag_y)
    if np.isposinf(l0):
        quadrant += i1+i2
    else:
        quadrant += 6.0/5.0-i1-i2

    out = np.empty((2*n-1, 2*n-1), dtype=np.float64)
    out[:n, :n] = quadrant
    out[:n, n:] = np.fliplr(quadrant[:, :-1])
    out[n:, :] = np.flipud(out[:n-1, :])
    return out
