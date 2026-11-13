"""PSF-prior source detection with explicit isolated-pixel artifact rejection.

Matched-filter scores use the supplied detector sigma map or an empirical
robust high-frequency noise scale, not an assumed Poisson/read-noise model.
Single-exposure morphology cannot prove that a pixel is physically hot;
rejected detections are labelled "hot_pixel_like", never certified defects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .stellar import _FourierTemplateShift, fit_stellar_scene
from .template import PSFTemplate


@dataclass(frozen=True)
class SourceCandidate:
    seed_xy_px: tuple[int, int]
    fitted_xy_px: tuple[float, float] | None
    flux: float
    matched_snr: float
    halo_snr: float
    psf_minus_spike_delta_chi2: float
    coverage: float
    classification: str


@dataclass(frozen=True)
class SourceDetection:
    positions_xy_px: np.ndarray
    initial_fluxes: np.ndarray
    accepted: tuple[SourceCandidate, ...]
    rejected: tuple[SourceCandidate, ...]
    background: float
    noise_sigma: float | None
    noise_source: str
    min_snr: float
    min_halo_snr: float
    max_sources: int
    examined_candidates: int
    input_valid_pixels: int
    final_detection_valid_pixels: int
    fit_valid_pixels: np.ndarray
    stopped_reason: str


def _robust_sigma_from_differences(image, valid):
    """Empirical MAD of adjacent differences divided by sqrt(2)."""
    parts = []
    for axis in (0, 1):
        diffs = np.diff(image, axis=axis)
        left, right = ((valid[:-1, :], valid[1:, :]) if axis == 0
                       else (valid[:, :-1], valid[:, 1:]))
        use = left & right
        if np.any(use):
            parts.append(diffs[use])
    if not parts:
        raise ValueError("Not enough neighboring valid pixels to estimate noise")
    sample = np.concatenate(parts)
    med = np.median(sample)
    sigma = float(1.4826*np.median(np.abs(sample-med))/np.sqrt(2.))
    if sigma <= 0 or not np.isfinite(sigma):
        # A mathematical fallback for exactly noise-free synthetic images,
        # NOT a claim about actual detector noise.
        sigma = max(1.,float(np.max(np.abs(image[valid]))))*1e-6
    return sigma


def _psf_core(template, radius):
    """Odd-size kernel centred on the PSF's brightest pixel."""
    p = np.asarray(template.image, dtype=float)
    peak_y, peak_x = np.unravel_index(np.argmax(p),p.shape)
    y,x = np.indices((2*radius+1,2*radius+1))
    sy = y+peak_y-radius
    sx = x+peak_x-radius
    inside = (sy>=0)&(sy<p.shape[0])&(sx>=0)&(sx<p.shape[1])
    core = np.zeros((2*radius+1,2*radius+1),dtype=float)
    core[inside] = p[sy[inside],sx[inside]]
    if core[radius,radius] <= 0:
        raise ValueError("Invalid PSF core for source detection")
    flank = float(np.sum(core**2)-core[radius,radius]**2)
    if flank < 1e-8*float(np.sum(core**2)):
        raise ValueError(
            "PSF prior is unresolved: cannot distinguish stars from hot pixels"
        )
    return core


def detect_stellar_sources(
    observed: np.ndarray,
    template: PSFTemplate,
    *,
    sigma: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    max_sources: int = 8,
    min_snr: float = 6.,
    min_halo_snr: float = 4.,
    min_separation_px: float = 4.,
    core_radius_px: int = 9,
    min_coverage: float = .75,
    max_examined: int = 128,
    refine_positions: bool = True,
) -> SourceDetection:
    """Find stars using the PSF shape, not a list of brightest pixels.

    For each matched-filter candidate, a star must beat a single-pixel
    impulse in chi-square improvement AND its surrounding PSF wings must
    have positive signal-to-noise even after excluding the central pixel.
    Rejected candidate centres are masked for subsequent detection and
    the final stellar fit. Once a star is accepted, an optional robust
    joint fit of accepted stars refines its coordinates on a frozen PSF;
    the search continues on residuals. It never recomputes AO quantities.

    This conservative search has a minimum separation and cannot resolve
    arbitrarily close binaries or certify the status of detector defects.
    """
    if not isinstance(template,PSFTemplate):
        raise TypeError("A validated PSFTemplate is required")
    data=np.asarray(observed,dtype=float)
    if data.shape!=template.image.shape:
        raise ValueError("Observed image and template must have identical shapes")
    if not isinstance(max_sources,(int,np.integer)) or max_sources<1:
        raise ValueError("max_sources must be >= 1")
    if not isinstance(max_examined,(int,np.integer)) or max_examined<max_sources:
        raise ValueError("max_examined must be >= max_sources")
    if not isinstance(core_radius_px,(int,np.integer)) or core_radius_px<2:
        raise ValueError("core_radius_px must be >= 2")
    if (not np.isfinite(min_snr) or min_snr<=0 or
        not np.isfinite(min_halo_snr) or min_halo_snr<=0 or
        not np.isfinite(min_separation_px) or min_separation_px<1 or
        not np.isfinite(min_coverage) or not 0<min_coverage<=1):
        raise ValueError("Invalid detection SNR, separation, or coverage")

    usable=np.isfinite(data)
    if valid_mask is not None:
        mask=np.asarray(valid_mask)
        if mask.shape!=data.shape or mask.dtype.kind!="b":
            raise ValueError("valid_mask must be boolean and match image shape")
        usable &= mask
    if sigma is not None:
        noise=np.asarray(sigma,dtype=float)
        if noise.shape!=data.shape:
            raise ValueError("sigma must have the same shape as image")
        usable &= np.isfinite(noise)&(noise>0)
    if np.count_nonzero(usable)<25:
        raise ValueError("Too few valid detector pixels for detection")
    initial_background=float(np.median(data[usable]))
    if sigma is None:
        empirical_sigma=_robust_sigma_from_differences(data,usable)
        noise=np.full(data.shape,empirical_sigma,dtype=float)
        noise_type="empirical_adjacent_difference_MAD_not_calibrated"
    else:
        empirical_sigma=None
        noise_type="user_supplied_sigma"

    radius=int(core_radius_px)
    kernel=_psf_core(template,radius)
    k2=(kernel*kernel)[::-1,::-1]
    reversed_kernel=kernel[::-1,::-1]
    full_norm=float(np.sum(kernel**2))
    centre=float(kernel[radius,radius])
    detection_mask=usable.copy()
    accepted=[]
    rejected=[]
    fitted_positions=[]
    fitted_fluxes=[]
    background=initial_background
    scene=np.full(data.shape,background,dtype=float)
    yy,xx=np.indices(data.shape)
    median_noise=float(np.median(noise[usable]))
    attempted=0
    stop="max_sources"

    def render_from_seeds():
        shifter=_FourierTemplateShift(template.image)
        py,px=np.unravel_index(np.argmax(template.image),template.image.shape)
        model=np.full(data.shape,background,dtype=float)
        for (sx,sy),sf in zip(fitted_positions,fitted_fluxes):
            model+=sf*shifter.shift(sx-px,sy-py)
        return model

    while len(accepted)<max_sources and attempted<max_examined:
        weights=np.zeros(data.shape,dtype=float)
        weights[detection_mask]=1./noise[detection_mask]**2
        residual=np.where(detection_mask,data-scene,0.)
        dot=fftconvolve(residual*weights,reversed_kernel,mode="same")
        norm=np.maximum(0.,fftconvolve(weights,k2,mode="same"))
        reference_norm=full_norm/median_noise**2
        good=(norm>=min_coverage*reference_norm)&detection_mask
        matched=np.full(data.shape,-np.inf,dtype=float)
        matched[good]=dot[good]/np.sqrt(norm[good])
        for xp,yp in fitted_positions:
            matched[(xx-xp)**2+(yy-yp)**2<min_separation_px**2]=-np.inf
        loc=np.unravel_index(np.argmax(matched),matched.shape)
        matched_snr=float(matched[loc])
        if not np.isfinite(matched_snr) or matched_snr<min_snr:
            stop="below_matched_snr"
            break
        y,x=map(int,loc)
        attempted+=1
        spike_snr=float(residual[y,x]*np.sqrt(weights[y,x]))
        halo_dot=float(dot[y,x]-weights[y,x]*centre*residual[y,x])
        halo_norm=float(norm[y,x]-weights[y,x]*centre**2)
        halo_snr=halo_dot/np.sqrt(halo_norm) if halo_norm>0 else -np.inf
        # Both quantities are improvements over a flat residual. A star
        # must explain the local patch better than a single-pixel impulse.
        delta_chi2=matched_snr**2-max(0.,spike_snr)**2
        flux=max(0.,float(dot[y,x]/norm[y,x]))
        coverage=float(norm[y,x]/reference_norm)
        if halo_snr<min_halo_snr or delta_chi2<=0:
            classification=(
                "hot_pixel_like" if delta_chi2<=0 else
                "insufficient_psf_wings"
            )
            rejected.append(SourceCandidate(
                (x,y),None,flux,matched_snr,float(halo_snr),
                float(delta_chi2),coverage,classification,
            ))
            detection_mask[y,x]=False
            continue

        fitted_positions.append((float(x),float(y)))
        fitted_fluxes.append(flux)
        if refine_positions:
            try:
                solution=fit_stellar_scene(
                    data,template,
                    initial_positions_xy_px=np.asarray(fitted_positions),
                    initial_fluxes=np.asarray(fitted_fluxes),
                    initial_background=background,
                    valid_mask=detection_mask,sigma=noise,
                    max_position_offset_px=min(1.5,min_separation_px/2),
                    robust_loss="soft_l1",f_scale=1.,max_nfev=100,
                )
                fitted_positions=[tuple(p) for p in solution.positions_xy_px]
                fitted_fluxes=list(solution.fluxes)
                background=float(solution.background)
                scene=solution.model
            except (ValueError,RuntimeError,FloatingPointError):
                # Refinement is a convenience, never a mandatory physical
                # correction. Fall back to the explicitly accepted seeds.
                scene=render_from_seeds()
        else:
            scene=render_from_seeds()
        accepted.append(SourceCandidate(
            (x,y),tuple(map(float,fitted_positions[-1])),
            float(fitted_fluxes[-1]),matched_snr,float(halo_snr),
            float(delta_chi2),coverage,"star_candidate",
        ))

    if attempted>=max_examined and len(accepted)<max_sources:
        stop="max_examined"
    return SourceDetection(
        positions_xy_px=np.asarray(fitted_positions,dtype=float).reshape(-1,2),
        initial_fluxes=np.asarray(fitted_fluxes,dtype=float).reshape(-1),
        accepted=tuple(accepted),rejected=tuple(rejected),
        background=float(background),noise_sigma=empirical_sigma,
        noise_source=noise_type,min_snr=float(min_snr),
        min_halo_snr=float(min_halo_snr),max_sources=int(max_sources),
        examined_candidates=attempted,input_valid_pixels=int(usable.sum()),
        final_detection_valid_pixels=int(detection_mask.sum()),
        fit_valid_pixels=detection_mask.copy(),stopped_reason=stop,
    )
