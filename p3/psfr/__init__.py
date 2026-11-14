"""Telemetry-based PSF reconstruction.

The legacy psfR class is retained as an internal compatibility engine while
the public interface is progressively moved to pure functions.
"""
from .products import PSFProduct, normalize_psf
from .ho import HOStatistics, estimate_ho_statistics, reconstruct_ho_otf
from .structure import PhaseStructureResult, assemble_phase_structure_function
from .spectral import (FilterProfile, PhotonQuadrature, read_kai_filter,
                       photon_quadrature, integrate_monochromatic_psfs)
from .chromatic import (PupilLagGrid, MonochromaticResult,
                        prepare_pupil_lag_grid, detector_monochromatic_psf)
from .template import (PSFTemplate, template_from_psfr,
                       save_psfr_template, load_psfr_template)
from .stellar import (StellarFitResult, render_stellar_scene, fit_stellar_scene)

__all__ = [
    "PSFProduct", "normalize_psf",
    "HOStatistics", "estimate_ho_statistics", "reconstruct_ho_otf",
    "PhaseStructureResult", "assemble_phase_structure_function",
    "FilterProfile", "PhotonQuadrature", "read_kai_filter",
    "photon_quadrature", "integrate_monochromatic_psfs",
    "PupilLagGrid", "MonochromaticResult",
    "prepare_pupil_lag_grid", "detector_monochromatic_psf",
    "PSFTemplate", "template_from_psfr",
    "save_psfr_template", "load_psfr_template",
    "StellarFitResult", "render_stellar_scene", "fit_stellar_scene",
]
