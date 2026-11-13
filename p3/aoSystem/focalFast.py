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
    g=float(cone_fraction)
    thx=float(theta_shift_x_m)
    thy=float(theta_shift_y_m)
    l0=float(finite_outer_scale_m)
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
