"""Measured NIRC2 passbands and photon-weighted spectral integration.

No optical propagation here: callers must independently compute each
monochromatic PSF on a common detector/angular grid. The historic P3
renderer must not be mistaken for this chromatic forward model.
"""
from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np


@dataclass(frozen=True)
class FilterProfile:
    wavelength_um: np.ndarray
    transmission: np.ndarray  # dimensionless, not full telescope throughput
    source: str
    audit: dict


@dataclass(frozen=True)
class PhotonQuadrature:
    wavelength_m: np.ndarray
    weights: np.ndarray
    total_photon_integral: float
    effective_wavelength_m: float


def read_kai_filter(path, *, clip_negative=True, merge_duplicates=True):
    """Read raw KAI NIRC2 .dat: wavelength [um], measured transmission [%].

    Negative laboratory baselines are explicitly clipped, with warning and
    audit; duplicate wavelengths are averaged; NaN or >100% are errors.
    No 1%-transmission threshold or rectangular-filter approximation.
    """
    data=np.loadtxt(path,comments="#",dtype=float)
    if data.ndim!=2 or data.shape[1]!=2 or data.shape[0]<2:
        raise ValueError("KAI file must contain two columns and at least two rows")
    if not np.isfinite(data).all():
        raise ValueError("KAI filter contains non-finite samples")
    if np.any(data[:,0]<=0) or np.any(data[:,1]>100):
        raise ValueError("Invalid wavelength or >100% transmission")
    order=np.argsort(data[:,0],kind="stable")
    wave=data[order,0]
    trans=data[order,1]/100.
    uniq, inverse, counts=np.unique(
        wave,return_inverse=True,return_counts=True
    )
    n_duplicates=len(wave)-len(uniq)
    if n_duplicates and not merge_duplicates:
        raise ValueError("KAI file contains duplicate wavelengths")
    if n_duplicates:
        trans=np.bincount(inverse,weights=trans)/counts
        wave=uniq
    negatives=int(np.count_nonzero(trans<0))
    if negatives and not clip_negative:
        raise ValueError("KAI file contains negative transmission")
    if negatives:
        warnings.warn(
            f"{Path(path).name}: clipped {negatives} negative measured "
            "transmission samples to zero",RuntimeWarning,stacklevel=2
        )
        trans=np.maximum(trans,0)
    if wave.size<2 or not np.any(trans>0):
        raise ValueError("KAI filter has no valid passband")
    return FilterProfile(
        wavelength_um=wave.copy(),transmission=trans.copy(),source=str(path),
        audit={"raw_samples":int(data.shape[0]),
               "duplicate_wavelength_samples":int(n_duplicates),
               "clipped_negative_samples":negatives,
               "source_units":"um, percent",
               "complete_system_throughput":False},
    )


def photon_quadrature(profile, *, n_bins, photon_sed=None, atmosphere=None):
    """Return photon-centroid wavelengths [m] and filter-integrated weights.

    photon_sed must be sampled on profile wavelengths in photons/um (arbitrary
    normalization). Convert an energy F_lambda first by multiplying by lambda.
    atmosphere is independently measured transmission in [0,1]. Weights
    include delta-lambda exactly once; no flux is applied to the PSF here.
    """
    if not isinstance(n_bins,(int,np.integer)) or n_bins<1:
        raise ValueError("n_bins must be positive integer")
    w=np.asarray(profile.wavelength_um,dtype=float)
    t=np.asarray(profile.transmission,dtype=float)
    if (w.ndim!=1 or w.size<2 or t.shape!=w.shape or
        not np.isfinite(w).all() or not np.isfinite(t).all() or
        np.any(w<=0) or np.any(np.diff(w)<=0) or
        np.any(t<0) or np.any(t>1)):
        raise ValueError("Invalid filter wavelength/transmission arrays")
    def array_factor(name,values,throughput=False):
        if values is None: return np.ones_like(w)
        a=np.asarray(values,dtype=float)
        if (a.shape!=w.shape or not np.isfinite(a).all() or
            np.any(a<0) or (throughput and np.any(a>1))):
            raise ValueError(name+" must be finite, nonnegative, same grid")
        return a
    density=t*array_factor("photon_sed",photon_sed)*array_factor(
        "atmosphere",atmosphere,True
    )
    edges=np.linspace(w[0],w[-1],n_bins+1)
    areas=np.zeros(n_bins)
    centres=np.zeros(n_bins)
    for i in range(n_bins):
        lo,hi=edges[i:i+2]
        points=np.r_[lo,w[(w>lo)&(w<hi)],hi]
        rates=np.interp(points,w,density)
        area=np.trapezoid(rates,points)
        areas[i]=area
        centres[i]=np.trapezoid(points*rates,points)/area if area>0 else .5*(lo+hi)
    total=float(areas.sum())
    if not np.isfinite(total) or total<=0:
        raise ValueError("Integrated photon flux is zero")
    weights=areas/total
    return PhotonQuadrature(
        wavelength_m=centres*1e-6,weights=weights,
        total_photon_integral=total,
        effective_wavelength_m=float(np.dot(weights,centres)*1e-6),
    )


def integrate_monochromatic_psfs(cube,weights):
    """Integrate precomputed unit-flux PSFs, without reapplying throughput.

    Shape [y,x,wavelength] or [y,x,source,wavelength]. Output leaves the
    source axis if present; source flux and positions are applied elsewhere.
    """
    p=np.asarray(cube,dtype=float)
    v=np.asarray(weights,dtype=float)
    if p.ndim not in (3,4) or min(p.shape[:2])<2:
        raise ValueError("PSF cube must be [y,x,(source),wavelength]")
    if v.ndim!=1 or v.size!=p.shape[-1]:
        raise ValueError("Spectral weights must match wavelength axis")
    if not np.isfinite(p).all() or np.any(p<0):
        raise ValueError("PSF slices must be finite and nonnegative")
    if not np.isfinite(v).all() or np.any(v<0) or not np.isclose(
        v.sum(),1,rtol=1e-12,atol=1e-12
    ):
        raise ValueError("Spectral weights must be nonnegative and sum to one")
    if not np.allclose(p.sum(axis=(0,1)),1,rtol=1e-9,atol=1e-11):
        raise ValueError("Each monochromatic PSF must already have unit flux")
    return np.tensordot(p,v,axes=([-1],[0]))
