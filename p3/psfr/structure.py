"""Pure PSF-R phase structure function assembly; AO terms are precomputed."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhaseStructureResult:
    """Phase SF cube [y,x,source] rad**2 and anisoplanatism diagnostics."""
    dphi_rad2: np.ndarray
    anisoplanatism_rad2: np.ndarray | None
    raw_zero_lag_rad2: np.ndarray | None
    zero_lag_reference: str | None


def normalize_anisoplanatism_zero_lag(dphi):
    """Subtract central OTF sample, preserving historical normalization."""
    dphi = np.asarray(dphi, dtype=float)
    if dphi.ndim not in (2, 3) or min(dphi.shape[:2]) < 2:
        raise ValueError("Anisoplanatism dphi must be [y,x] or [y,x,source]")
    if not np.all(np.isfinite(dphi)):
        raise ValueError("Anisoplanatism dphi contains non-finite samples")
    centre = (dphi.shape[0]//2, dphi.shape[1]//2)
    return dphi - dphi[centre]


def assemble_phase_structure_function(
    *, r0_m, gains, dphi_ao, dphi_tt, dphi_fit, dphi_alias,
    n_sources, anisoplanatism_kernel=None, layer_strengths=None,
    layer_weights=None, atmosphere_wavelength_m=None,
    science_wavelength_m=None,
):
    """Build HO + TT + r0**(-5/3)*(fit+alias) + anisoplanatism.

    HO, TT, fit, alias: [y,x] rad**2 at science reference wavelength.
    Anisoplanatism: [source,layer,y,x] at atmosphere wavelength.
    Strengths: one value per layer, in historical r0**(-5/3) units.
    This is the existing P3 algebra, not a new physical model.
    """
    if gains is None:
        raise ValueError("PSF-R requires two finite HO/TT gains [gho, gtt]")
    g = np.asarray(gains, dtype=float)
    if g.shape != (2,) or not np.isfinite(g).all():
        raise ValueError("PSF-R requires two finite HO/TT gains [gho, gtt]")
    r0 = float(r0_m)
    if not np.isfinite(r0) or r0 <= 0:
        raise ValueError("PSF-R requires finite positive r0")
    if not isinstance(n_sources, (int, np.integer)) or n_sources < 1:
        raise ValueError("PSF-R n_sources must be positive integer")
    terms = [np.asarray(v, dtype=float) for v in (
        dphi_ao, dphi_tt, dphi_fit, dphi_alias
    )]
    shape = terms[0].shape
    if len(shape) != 2 or min(shape) < 2 or any(x.shape != shape for x in terms):
        raise ValueError("PSF-R kernels must share [y,x] shape")
    if not all(np.isfinite(x).all() for x in terms):
        raise ValueError("PSF-R kernels contain non-finite values")

    # Keep exact legacy grouping, including its unmodified TT gain.
    on_axis = g[0]*terms[0] + g[1]*terms[1] + r0**(-5/3)*(
        terms[2] + terms[3]
    )
    if anisoplanatism_kernel is None:
        return PhaseStructureResult(
            np.repeat(on_axis[:, :, np.newaxis], n_sources, axis=2),
            None, None, None
        )

    kernel = np.asarray(anisoplanatism_kernel, dtype=float)
    if (kernel.ndim != 4 or kernel.shape[0] != n_sources or
            kernel.shape[2:] != shape or kernel.shape[1] < 1):
        raise ValueError("Anisoplanatism kernel must be [source,layer,y,x]")
    if not np.isfinite(kernel).all():
        raise ValueError("Anisoplanatism kernel contains non-finite values")
    if layer_strengths is None:
        if layer_weights is None:
            raise ValueError("Anisoplanatism needs layer strengths or weights")
        strength = np.asarray(layer_weights, dtype=float).reshape(-1)*r0**(-5/3)
    else:
        strength = np.asarray(layer_strengths, dtype=float).reshape(-1)
    if strength.size != kernel.shape[1]:
        raise ValueError("Anisoplanatism: Cn2 must have one value per layer")
    if not np.isfinite(strength).all():
        raise ValueError("Anisoplanatism Cn2 contains non-finite values")
    if atmosphere_wavelength_m is None or science_wavelength_m is None:
        raise ValueError("Anisoplanatism requires both wavelengths")
    atm_wvl = float(atmosphere_wavelength_m)
    sci_wvl = float(science_wavelength_m)
    if (not np.isfinite(atm_wvl) or not np.isfinite(sci_wvl) or
            atm_wvl <= 0 or sci_wvl <= 0):
        raise ValueError("Anisoplanatism wavelengths must be positive and finite")
    raw = np.einsum(
        'slxy,l->xys', kernel, strength*(atm_wvl/sci_wvl)**2
    )
    centre = (raw.shape[0]//2, raw.shape[1]//2)
    raw_zero_lag = np.asarray(raw[centre], dtype=float).copy()
    dani = normalize_anisoplanatism_zero_lag(raw)
    return PhaseStructureResult(
        on_axis[:, :, np.newaxis] + dani,
        dani, raw_zero_lag, "central_otf_sample",
    )
