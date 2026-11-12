"""Standalone wavelength-dependent OTF-to-detector PSF optical kernel.

This does NOT replace P3's validated Keck V1 forward model.  It provides
an independently testable monochromatic building block for the future
polychromatic renderer. A phase SF from P3 must first be mapped to THIS
physical pupil-baseline (metres) grid; the old nOtf SF array cannot simply
be reused on an unrelated sampling grid.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class PupilLagGrid:
    """Pupil amplitude and normalized zero-OPD OTF at physical lag samples.

    Pupil grid pixel spacing is D / n_pupil in metres. The OTF grid
    has shape [2*n_pupil-1,2*n_pupil-1] and is centred at zero lag.
    """
    amplitude: np.ndarray
    diameter_m: float
    lag_step_m: float
    otf_lag: np.ndarray


@dataclass(frozen=True)
class MonochromaticResult:
    """Unit-integral detector PSF and sampled OTF at its wavelength."""
    psf: np.ndarray
    otf: np.ndarray
    wavelength_m: float
    negative_pixel_fraction_before_cleanup: float
    negative_flux_fraction_before_cleanup: float


def _autocorrelation_otf(field):
    """Exact discrete field autocorrelation, normalized at zero lag."""
    ac = fftconvolve(field, np.conj(field[::-1, ::-1]), mode="full")
    middle = ac.shape[0]//2
    dc = float(np.real(ac[middle, middle]))
    if not np.isfinite(dc) or dc <= 0:
        raise ValueError("Pupil autocorrelation has nonpositive zero lag")
    return ac / dc


def prepare_pupil_lag_grid(amplitude, *, diameter_m):
    """Build a reusable pupil autocorrelation once for zero static OPD."""
    p = np.asarray(amplitude,dtype=float)
    d = float(diameter_m)
    if (p.ndim!=2 or p.shape[0]!=p.shape[1] or p.shape[0]<4 or
            not np.isfinite(p).all() or np.any(p<0) or
            not np.isfinite(d) or d<=0 or not np.any(p>0)):
        raise ValueError("Physical pupil must be finite, nonnegative, square")
    return PupilLagGrid(
        amplitude=p.copy(),diameter_m=d,lag_step_m=d/p.shape[0],
        otf_lag=_autocorrelation_otf(p),
    )


def _sample_lag_grid(array, geometry, lambda_m, fy, fx):
    """Interpolate physical lag at rho = wavelength * angular frequency.

    Real and imaginary parts are handled separately to support static OPD
    phases. Bilinear interpolation of a finite lag grid is used; no
    extrapolation outside the pupil autocorrelation support is allowed.
    """
    n=geometry.amplitude.shape[0]
    coordinates=np.asarray([
        (lambda_m*fy/geometry.lag_step_m + n-1).ravel(),
        (lambda_m*fx/geometry.lag_step_m + n-1).ravel(),
    ])
    real=map_coordinates(
        np.real(array),coordinates,order=1,mode="constant",cval=0.
    ).reshape(fx.shape)
    if np.iscomplexobj(array):
        imag=map_coordinates(
            np.imag(array),coordinates,order=1,mode="constant",cval=0.
        ).reshape(fx.shape)
        return real+1j*imag
    return real


def detector_monochromatic_psf(
    geometry: PupilLagGrid, *, wavelength_m, pixel_scale_rad, n_pixels,
    static_opd_m=None, phase_sf_reference_rad2=None,
    reference_wavelength_m=None,
    allow_numerical_ringing_clipping=False,
    max_ringing_peak_fraction=1e-4,
    max_ringing_flux_fraction=1e-4,
):
    """Render a detector-sampled monochromatic PSF on a FIXED angular grid.

    Pupil autocorrelation evaluated at rho = lambda * spatial_frequency;
    the Gaussian residual OTF factor is exp[-Dphi_ref(rho)/2 *
    (lambda_ref/lambda)**2]. Static OPD phase is exp[i*2*pi*OPD/lambda].
    The ideal box pixel MTF is sinc(pixel_scale*fx)*sinc(pixel_scale*fy).
    Final PSF is a unit-flux detector image with no stellar photometry.

    For phase_sf_reference_rad2, the input has to sample the exact same
    physical pupil-lag grid as geometry.otf_lag. No regridding from the
    historical P3 nOtf grid is performed here.

    Sampling the finite pupil-lag OTF can cause small negative FFT ringing
    at insufficient pupil resolution. It is REJECTED by default. An
    explicitly opted-in, audited clipping can be used for diagnostics if
    both peak and total negative flux are below the requested thresholds.
    """
    lam=float(wavelength_m)
    pix=float(pixel_scale_rad)
    if (not np.isfinite(lam) or lam<=0 or not np.isfinite(pix) or pix<=0
            or not isinstance(n_pixels,(int,np.integer)) or n_pixels<8):
        raise ValueError("Wavelength, pixel scale and n_pixels must be positive")

    lag_otf=geometry.otf_lag
    if static_opd_m is not None:
        opd=np.asarray(static_opd_m,dtype=float)
        if opd.shape!=geometry.amplitude.shape or not np.isfinite(opd).all():
            raise ValueError("Static OPD must match the pupil [y,x] in metres")
        field=geometry.amplitude*np.exp((2j*np.pi/lam)*opd)
        lag_otf=_autocorrelation_otf(field)

    freqs=np.fft.fftshift(np.fft.fftfreq(n_pixels,d=pix))
    fy,fx=np.meshgrid(freqs,freqs,indexing="ij")
    otf=_sample_lag_grid(lag_otf,geometry,lam,fy,fx)

    if phase_sf_reference_rad2 is not None:
        sf=np.asarray(phase_sf_reference_rad2,dtype=float)
        if (sf.shape!=geometry.otf_lag.shape or
                not np.isfinite(sf).all() or np.min(sf)<-1e-10):
            raise ValueError("Reference phase SF must be finite and nonnegative on pupil lags")
        c=sf.shape[0]//2
        if abs(sf[c,c])>1e-10:
            raise ValueError("Reference phase SF must vanish at zero lag")
        if reference_wavelength_m is None:
            raise ValueError("Reference wavelength required for phase SF")
        ref=float(reference_wavelength_m)
        if not np.isfinite(ref) or ref<=0:
            raise ValueError("Invalid reference phase-SF wavelength")
        sampled=_sample_lag_grid(sf,geometry,lam,fy,fx)
        otf=otf*np.exp(-.5*(ref/lam)**2*sampled)

    otf=otf*np.sinc(pix*fx)*np.sinc(pix*fy)
    c=n_pixels//2
    otf[c,c]=1.0  # by definition: normalized pupil, Dphi(0)=0, pixel MTF(0)=1
    psf=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(otf))).real
    peak=float(psf.max())
    if not np.isfinite(psf).all() or peak<=0:
        raise ValueError("Invalid monochromatic PSF or pupil sampling")
    neg=psf[psf<0]
    min_rel=float(-min(0,float(psf.min()))/peak)
    negative_flux=float(-neg.sum()/max(float(psf.sum()),1e-30))
    if neg.size:
        if not allow_numerical_ringing_clipping:
            raise ValueError(
                "Negative PSF pixels from lag-OTF sampling; refine pupil "
                "sampling or explicitly opt into audited numerical clipping "
                f"(min/peak={min_rel:.3g}, negative_flux={negative_flux:.3g})"
            )
        if (min_rel>max_ringing_peak_fraction or
                negative_flux>max_ringing_flux_fraction):
            raise ValueError("Negative PSF ringing exceeds declared clipping budget")
        warnings.warn(
            "Small negative monochromatic PSF ringing clipped with explicit "
            f"audit (min/peak={min_rel:.3g}, negative_flux={negative_flux:.3g}); "
            "this is a numerical approximation, not a physical correction.",
            RuntimeWarning,stacklevel=2,
        )
        psf=np.maximum(psf,0)
    psf=psf/psf.sum()
    return MonochromaticResult(
        psf=psf,otf=otf,wavelength_m=lam,
        negative_pixel_fraction_before_cleanup=min_rel,
        negative_flux_fraction_before_cleanup=negative_flux,
    )
