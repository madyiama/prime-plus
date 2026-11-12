"""Telemetry-based PSF reconstruction.

The legacy psfR class is retained as an internal compatibility engine while
the public interface is progressively moved to pure functions.
"""
from .products import PSFProduct, normalize_psf

__all__ = ["PSFProduct", "normalize_psf"]
