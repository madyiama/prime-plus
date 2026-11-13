"""Stellar photometry/astrometry fitting with a frozen telemetry-only PSF.

No HO/TT/atmospheric parameter can change here. A PSF template is loaded
once; fit variables are only source fluxes, detector x/y coordinates and
a shared constant detector background.

Positions use Numpy image conventions: x = column, y = row. Positive dx
moves right, positive dy moves down. Band-limited shifts use zero padding
to avoid periodic wrap; any interpolation ringing is reported, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import fourier_shift
from scipy.optimize import least_squares

from .template import PSFTemplate


@dataclass(frozen=True)
class StellarFitResult:
    """Arrays of fitted x/y positions and fluxes, with full model and residual."""
    positions_xy_px: np.ndarray
    fluxes: np.ndarray
    background: float
    model: np.ndarray
    residual: np.ndarray
    valid_pixels: np.ndarray
    converged: bool
    status: int
    message: str
    nfev: int
    rms: float
    weighted_chi2: float | None
    reduced_chi2: float | None
    interpolation_negative_to_peak: float
    initial_positions_xy_px: np.ndarray


class _FourierTemplateShift:
    """Cache padded template FFT; shift without re-FFTing the pupil per call."""

    def __init__(self, image):
        kernel = np.asarray(image, dtype=np.float64)
        if kernel.ndim != 2 or min(kernel.shape) < 4:
            raise ValueError("Stellar PSF template must be a 2-D array")
        ny, nx = kernel.shape
        self.shape = (ny,nx)
        self.ys = ny // 2
        self.xs = nx // 2
        padded = np.zeros((ny*2,nx*2),dtype=float)
        padded[self.ys:self.ys+ny,self.xs:self.xs+nx] = kernel
        self.fft = np.fft.fftn(padded)

    def shift(self, dx, dy):
        """Pixel translation without re-normalizing truncated finite FoV."""
        if not np.isfinite(dx) or not np.isfinite(dy):
            raise ValueError("Non-finite star displacement")
        shifted=np.fft.ifftn(
            fourier_shift(self.fft,shift=(float(dy),float(dx)))
        ).real
        ny,nx=self.shape
        return shifted[self.ys:self.ys+ny,self.xs:self.xs+nx]


def render_stellar_scene(template, *, positions_xy_px, fluxes, background=0.):
    """Build an additive star scene with fixed PSF and constant background.

    Position refers to the location of the PSF's brightest pixel. The
    detector grid is [y,x] and no subpixel resampling changes PSF physics.
    Flux is the coefficient of the unit-flux unshifted template, NOT
    the finite-support integral of a shifted/cropped source.
    """
    if not isinstance(template,PSFTemplate):
        raise TypeError("A validated PSFTemplate is required")
    positions=np.asarray(positions_xy_px,dtype=float)
    f=np.asarray(fluxes,dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or
            positions.shape[0] < 1 or f.shape != (positions.shape[0],) or
            not np.isfinite(positions).all() or not np.isfinite(f).all()
            or np.any(f < 0) or not np.isfinite(background)):
        raise ValueError("Positions [star,xy], nonnegative flux and background required")
    ny,nx=template.image.shape
    if (np.any(positions[:,0]<0) or np.any(positions[:,0]>=nx) or
            np.any(positions[:,1]<0) or np.any(positions[:,1]>=ny)):
        raise ValueError("All stars must have centres inside the detector image")
    shifter=_FourierTemplateShift(template.image)
    peak_y,peak_x=np.unravel_index(np.argmax(template.image),template.image.shape)
    scene=np.full((ny,nx),float(background),dtype=float)
    for (x,y),flux in zip(positions,f):
        scene+=flux*shifter.shift(x-peak_x,y-peak_y)
    return scene


def fit_stellar_scene(
    observed: np.ndarray,
    template: PSFTemplate,
    *,
    initial_positions_xy_px: np.ndarray | None = None,
    initial_fluxes: np.ndarray | None = None,
    initial_background: float | None = None,
    sigma: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    radius_px: float | None = None,
    max_position_offset_px: float = 5.,
    robust_loss: str = "linear",
    f_scale: float = 1.,
    max_nfev: int = 300,
) -> StellarFitResult:
    """Fit flux, x, y per star and one constant background.

    Default objective is unweighted least squares (no invented detector
    noise law). A provided strictly positive sigma map turns it into
    inverse-variance-weighted least squares. Valid_mask=True means INCLUDE.
    Optional radius restricts the fitting ROI around initial star positions.

    Only fluxes are constrained positive. x/y are bounded within
    max_position_offset_px of supplied seeds and the science image.
    AO, TT, r0 and pupil are NEVER fitted here.

    This first-stage fitter does not quantify stellar/AO degeneracies,
    independent sky-angle calibration or parameter uncertainties.
    """
    if not isinstance(template,PSFTemplate):
        raise TypeError("A validated PSFTemplate is required")
    data=np.asarray(observed,dtype=float)
    ny,nx=template.image.shape
    if data.shape != (ny,nx):
        raise ValueError("Science image and PSF template must have identical [y,x] shapes")
    if not np.isfinite(max_position_offset_px) or max_position_offset_px<=0:
        raise ValueError("max_position_offset_px must be finite and positive")
    if not isinstance(max_nfev,int) or max_nfev<1:
        raise ValueError("max_nfev must be positive")
    if robust_loss not in ("linear","soft_l1","huber"):
        raise ValueError("Unsupported robust loss")
    if not np.isfinite(f_scale) or f_scale<=0:
        raise ValueError("f_scale must be finite positive")

    valid=np.isfinite(data)
    if valid_mask is not None:
        mask=np.asarray(valid_mask)
        if mask.shape!=data.shape or mask.dtype.kind!="b":
            raise ValueError("valid_mask must be a boolean image with True=include")
        valid&=mask
    noise=None
    if sigma is not None:
        noise=np.asarray(sigma,dtype=float)
        if noise.shape!=data.shape:
            raise ValueError("sigma must match science image dimensions")
        valid&=np.isfinite(noise)&(noise>0)
    if initial_positions_xy_px is None:
        if not np.any(valid):
            raise ValueError("No valid image pixels for initial star detection")
        y,x=np.unravel_index(
            np.argmax(np.where(valid,data,-np.inf)),data.shape
        )
        starts=np.asarray([[float(x),float(y)]],dtype=float)
    else:
        starts=np.asarray(initial_positions_xy_px,dtype=float)
        if (starts.ndim != 2 or starts.shape[1] != 2 or
                starts.shape[0] < 1 or not np.isfinite(starts).all() or
                np.any(starts[:,0]<0) or np.any(starts[:,0]>=nx) or
                np.any(starts[:,1]<0) or np.any(starts[:,1]>=ny)):
            raise ValueError("Initial centres must be [star,xy] inside image")
    if radius_px is not None:
        if not np.isfinite(radius_px) or radius_px <= max_position_offset_px:
            raise ValueError("radius_px must exceed max_position_offset_px")
        yy,xx=np.indices(data.shape)
        region=np.zeros(data.shape,dtype=bool)
        for x,y in starts:
            region |= (xx-x)**2+(yy-y)**2<=radius_px**2
        valid &= region
    n=starts.shape[0]
    n_parameters=3*n+1
    if valid.sum() <= n_parameters:
        raise ValueError("Not enough valid science pixels for stellar parameters")
    if initial_background is None:
        bg=float(np.nanmedian(data[valid]))
    else:
        bg=float(initial_background)
        if not np.isfinite(bg):
            raise ValueError("Initial background must be finite")
    if initial_fluxes is None:
        peak=float(template.image.max())
        flux_initial=[]
        for x,y in starts:
            pixel=float(data[int(np.rint(y)),int(np.rint(x))])
            flux_initial.append(max(1e-12,(pixel-bg)/peak))
        flux_initial=np.asarray(flux_initial,dtype=float)
    else:
        flux_initial=np.asarray(initial_fluxes,dtype=float)
        if (flux_initial.shape!=(n,) or not np.isfinite(flux_initial).all()
                or np.any(flux_initial<0)):
            raise ValueError("Initial stellar fluxes must be nonnegative [star]")
    shifter=_FourierTemplateShift(template.image)
    peak_y,peak_x=np.unravel_index(np.argmax(template.image),template.image.shape)

    start=np.empty(n_parameters)
    lo=np.full(n_parameters,-np.inf)
    hi=np.full(n_parameters,np.inf)
    for i,((x,y),flux) in enumerate(zip(starts,flux_initial)):
        k=3*i
        start[k:k+3]=(flux,x,y)
        lo[k]=0.
        lo[k+1]=max(0.,x-max_position_offset_px)
        lo[k+2]=max(0.,y-max_position_offset_px)
        hi[k+1]=min(nx-1.,x+max_position_offset_px)
        hi[k+2]=min(ny-1.,y+max_position_offset_px)
        # For an unshifted star exactly on a boundary, scipy least_squares
        # accepts the initial point and automatically moves into bounds.
    start[-1]=bg

    def model_from(x):
        out=np.full(data.shape,x[-1],dtype=float)
        for i in range(n):
            flux,px,py=x[3*i:3*i+3]
            out+=flux*shifter.shift(px-peak_x,py-peak_y)
        return out

    if noise is None:
        def residual(x):
            return (model_from(x)-data)[valid]
    else:
        def residual(x):
            return ((model_from(x)-data)/noise)[valid]

    fit=least_squares(
        residual,start,bounds=(lo,hi),method="trf",
        loss=robust_loss,f_scale=f_scale,max_nfev=max_nfev,
        ftol=1e-10,xtol=1e-10,gtol=1e-10,
    )
    model=model_from(fit.x)
    residual_image=data-model
    subset=residual_image[valid]
    rms=float(np.sqrt(np.mean(subset**2)))
    chi2=(float(np.sum((subset/noise[valid])**2))
          if noise is not None else None)
    dof=int(valid.sum())-n_parameters
    reduced=(chi2/dof if chi2 is not None and dof>0 else None)
    # Report signed interpolation ringing rather than clipping the FFT
    # shifted template or changing the model during optimization.
    negatives=0.
    for i in range(n):
        shifted=shifter.shift(
            fit.x[3*i+1]-peak_x,fit.x[3*i+2]-peak_y
        )
        negatives=max(negatives,max(0.,-float(np.min(shifted)))/
                      max(float(np.max(shifted)),1e-30))
    return StellarFitResult(
        positions_xy_px=fit.x[:-1].reshape(n,3)[:,1:3].copy(),
        fluxes=fit.x[:-1].reshape(n,3)[:,0].copy(),
        background=float(fit.x[-1]),
        model=model,
        residual=residual_image,
        valid_pixels=valid.copy(),
        converged=bool(fit.success),
        status=int(fit.status),
        message=str(fit.message),
        nfev=int(fit.nfev),
        rms=rms,
        weighted_chi2=chi2,
        reduced_chi2=reduced,
        interpolation_negative_to_peak=float(negatives),
        initial_positions_xy_px=starts.copy(),
    )
