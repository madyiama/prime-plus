# Keck PSF-R V1 frozen reference and fitting handoff

Working branch refactor/psfr-fitting-interface
Base branch refactor/psfr-core; main stays unchanged.

## Release decision

The operational reference is the existing Keck LGS STRAP V1 telemetry
pipeline. Baseline science: Keck NIRC2 pupil physical_binary/header/native;
HO strict PSD; seven atmospheric layers; Van Dam noise + legacy_psfr
TT covariance + unity mirror + noll_angular TT kernel.

The following aspects are NOT yet calibrated:
- absolute STRAP sky-to-camera gain/APD noise,
- camera/TT parity in all acquisitions,
- absolute science-image Strehl,
- proper chromatic diffraction in Keck V1.

The independent pupil-lag chromatic demo in PR #35 tested the optics on
an ideal circular pupil, NOT the true segmented Keck NIRC2 optics. Do
not silently substitute it for the reference forward model yet.

## Fit-ready product

The template is derived from the validated telemetry-only PSF forward
product, with no new AO initialization. It includes:
- image: [row, col], normalized unit flux over current image support;
- cube: [row, col, spectral bin], already weighted by the V1 forward;
- spectral_fractions, inherited without double-weighting;
- positive detector pixel scale in milliarcseconds per pixel;
- source provenance / code revision / pupil and TT policies;
- flags identifying uncalibrated STRAP and legacy mono+weights model.

To export an already reconstructed nominal case without repeating
the 153.9 s model construction:

    uv run python scripts/export_psfr_template.py \
       --case results/psfr_v1_acceptance/n0120 \
       --pixel-scale-mas 9.942

IMPORTANT: 9.942 mas/pixel is only an example for NIRC2 narrow camera.
Verify the actual observation's detector mode/calibration first.
If the metadata already contains the pixel scale, a conflicting CLI
value is rejected.

This writes:
- psfr_fit_template.npz (no pickles: image/cube/fractions/metadata);
- psfr_fit_template.json (human-readable manifest).

The exporter refuses incomplete or nonnominal V1 diagnostic cases
using the existing acceptance validator. It never fits the science
image, never changes HO/TT parameters, never clips negative PSF
pixels, and never adjusts the pupil or transmission.

## Next sprint: PSF fitting

1. Load the frozen template once.
2. On a well-defined, unsaturated, background-calibrated stellar cutout,
   fit only flux, subpixel dx/dy and local constant background.
3. Derive noise/weight/masking model and residual diagnostics. Do not
   compare an unregistered star field to a centred point-source PSF.
4. Introduce simultaneous multi-star fitting and field-dependent PSFs.
5. Only after degeneracy/identifiability checks, optionally permit
   selected AO parameters r0 / g_HO / g_TT, with explicit priors and
   telemetry-only baseline comparisons. Jitter calibration must not
   be retrofitted merely to match diffraction ring contrast.

The current P3 psfrRun.py uses obsolete imports from a historic fitting
package and must not be treated as the new public API.

## Verification

    uv run --with pytest python -m pytest -q tests/test_psfr_template.py
    uv run --with pytest python -m pytest -q

Then export the user's existing n0120 result and inspect the metadata.
No additional Keck FITS/SAV acquisition is needed for this export step.
