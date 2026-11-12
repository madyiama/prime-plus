"""Telemetry-based PSF reconstruction.

The legacy psfR class is retained as an internal compatibility engine while
the public interface is progressively moved to pure functions.
"""
from .products import PSFProduct, normalize_psf
from .ho import HOStatistics, estimate_ho_statistics, reconstruct_ho_otf
from .structure import PhaseStructureResult, assemble_phase_structure_function
from .spectral import (FilterProfile, PhotonQuadrature, read_kai_filter,
                       photon_quadrature, integrate_monochromatic_psfs)

__all__ = [
    "PSFProduct", "normalize_psf",
    "HOStatistics", "estimate_ho_statistics", "reconstruct_ho_otf",
    "PhaseStructureResult", "assemble_phase_structure_function",
    "FilterProfile", "PhotonQuadrature", "read_kai_filter",
    "photon_quadrature", "integrate_monochromatic_psfs",
]
