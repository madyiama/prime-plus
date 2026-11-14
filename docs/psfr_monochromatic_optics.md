# Stage 4C: wavelength-dependent pupil-lag PSF renderer (experimental)

Base branch: refactor/psfr-core
Working branch: refactor/psfr-monochromatic-optics
THIS CHANGE DOES NOT MODIFY THE VALIDATED KECK V1 PATH.

This module provides a physically explicit, standalone monochromatic
optical kernel to make the next polychromatic PR testable:

1. Physical pupil amplitude [y,x] sampled at D / n_pupil metres.
2. Pupil complex field at wavelength lambda:
   P(x) * exp(2*pi*i*static_OPD(x)/lambda).
3. Complex pupil autocorrelation OTF in physical lag rho [m].
   This is computed once if static OPD = 0 and reused for all wavelengths.
4. Evaluate pupil OTF at rho = lambda*nu where nu is angular
   spatial frequency in cycles/radian of the FIXED detector pixel grid.
5. If an AO residual SF is supplied on the SAME physical pupil-lag
   grid, multiply by exp[-0.5*Dphi_ref(rho)*(lambda_ref/lambda)^2].
6. Account for integration across square detector pixels by the
   box pixel MTF sinc(pixel_scale*nu_x)*sinc(pixel_scale*nu_y).
7. Inverse FFT to obtain the detector-sampled monochromatic PSF.
   Normalize to unit integral; stellar photometry is kept separate.

The physical pupil size, camera angular pixel scale and wavelength
are explicit inputs. The AO SF grid cannot be guessed.

IMPORTANT SCIENTIFIC LIMITATION
The legacy P3 phase SF arrays use freq.nOtf/sampRef support, which is
not guaranteed to be the same sampling as the physical autocorrelation
lag grid of a telescope pupil. This module DOES NOT blindly reuse or
interpolate those arrays: a validated rho-space bridge needs its own
tests and comparisons before Keck telemetry can be propagated here.
It must also establish whether the existing anisoplanatic Dphi uses
the correct pupil-lag orientation and reference wavelength.

The finite lag-grid bilinear interpolation may produce slight
negative values after FFT on an inadequately sampled pupil.
They are REJECTED by default. Optional numerical clipping must be
explicitly requested with peak and total-flux upper budgets, and it
raises RuntimeWarning with measured affected fractions. It cannot
be mistaken for a physical correction.

The current passband integration from stage 4A works ONLY on PSF
slices already rendered at distinct wavelengths on the same detector
angular/pixel grid. No repeated AO telemetry estimation is needed
for these independent lambda evaluations.

Test:

    uv run --with pytest python -m pytest -q tests/test_psfr_monochromatic_optics.py
    uv run --with pytest python -m pytest -q

Planned before Keck integration:
- Verify chromatic diffraction scaling on ideal circular aperture;
- Verify finite pupil truncation and pixel MTF convergence;
- Implement explicit lag-coordinate bridge from P3 to physical pupil;
- Validate wavelength dependence of all HO/TT/fit/alias/aniso terms;
- Verify energy normalization, Strehl and pupil rotation/parity;
- Cross-validate mono limit with current Keck PSF-R before replacing it;
- Validate Kp/H bands with KAI measured photon quadrature.

Performance: pupil autocorrelation is cached across wavelengths
when static OPD is absent. FFTs on a fixed detector grid are small
relative to potentially >100s focal-anisoplanatism preprocessing,
which is being profiled separately in PR #34.
