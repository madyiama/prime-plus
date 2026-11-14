# Keck/NIRC2 PSF-R → Stellar fitting: repeatable workflow

Working feature branch: refactor/psfr-stellar-fitting
Based on refactor/psfr-fitting-interface (PR #36), and ultimately
refactor/psfr-core. Do not merge to main.

## 1. Reconstruct the Keck V1 PSF-R after deleting old results

From project root, on this working branch:

    git fetch origin
    git switch --track origin/refactor/psfr-stellar-fitting
    uv sync

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --output results/psfr_v1_fit

The FITS/SAV paths are example paths from the prior local workflow.
Verify that the files really exist. The V1 runner checks Keck LGS
STRAP nominal physics: 7 layers, pupil physical_binary/header/native,
Van Dam / legacy_psfr / unity / noll_angular TT, and strict HO PSD.

Expected:
- results/psfr_v1_fit/n0120/telemetry_only_psf.npy
- results/psfr_v1_fit/n0120/observed_nirc2_cutout.npy
- results/psfr_v1_fit/n0120/metadata.json
- results/psfr_v1_fit/n0120/psfr_v1_run.log
- results/psfr_v1_fit/psfr_v1_acceptance.json

The saved observed cutout is exactly telemetryKeck.trs.cam.image,
with the same orientation and cropping as the PSF-R's figure. The
pixel scale is now recorded from model.ao.cam.psInMas. Old runs
cannot reliably be manually recropped without a new equivalence check.

## 2. Export the fixed PSF-R fitting template

    uv run python scripts/export_psfr_template.py \
      --case results/psfr_v1_fit/n0120

This uses the V1 acceptance validator and metadata. The camera scale
is inferred from the newly recorded camera_pixel_scale_mas, with an
optional explicit --pixel-scale-mas cross-check.

Outputs:
- psfr_fit_template.npz
- psfr_fit_template.json

No telescope or AO recalculation occurs here.

## 3. Fit a single star

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120

Default initialization finds the brightest valid image pixel. The
fitting parameters are a single nonnegative flux, x column coordinate,
y row coordinate, and constant background.

For two stars, give two approximate positions (not blind catalog
detection):

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120 \
      --xy 62 64 --xy 80 71 \
      --max-offset 4

Those coordinates are EXAMPLES; use actual star positions in the
telemetryKeck 128x128 cutout. For a crowded field, optionally pass
--radius 18, which masks everything outside the union of circles of
radius 18 pixels centred on the initial guesses.

To exclude saturated pixels explicitly:

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120 \
      --saturation YOUR_DETECTOR_THRESHOLD

The science cutout's units must match the detector saturation
threshold; do not use an arbitrary ADU value. Other options include
--mask boolean.npy (True=include), and --sigma noise_map.npy with
per-pixel standard deviations. The default objective is UNWEIGHTED
least-squares; no invented NIRC2 photon/read-noise model is assumed.

Outputs in results/psfr_v1_fit/n0120/stellar_fit:
- fit_report.json: fluxes, pixel offsets, background, residual RMS,
  optimizer status, active mask, fixed AO and calibration limitations
- model.npy
- residual.npy
- valid_pixels.npy
- fit_comparison.png

The x axis is detector column, y axis detector row (increasing down
array index). Reported mas offsets are relative to the template peak
and are NOT absolute calibrated sky position angles.

## Scientific scope

This is FIRST the fitting of parameters STELLAR ONLY:
per-source flux, x/y detector position, one background.

The PSF-R is frozen; no HO, TT, r0, Cn2, pupil, throughput or spectral
model is changed. Fitting does not calibrate STRAP independently.
The current Keck V1 "broadband" product remains the historic
single-reference-wavelength PSF with varying spectral weights.
The standalone Kp chromatic demo does not yet use Keck AO functions
of structure and is NOT substituted into the V1 model.

The Fourier subpixel shift is padded to avoid periodic wrap, but
truncated pixel support can change the measured total flux; it is
not re-normalized to conceal that truncation. The interpolation may
introduce small signed ringing: report its negative-to-peak ratio
rather than silently clipping. Uncertainty estimation and
Poisson-likelihood optimisation are future steps when the detector
variance/gain model is available.

## Test and merge gates

    uv run --with pytest python -m pytest -q tests/test_psfr_stellar.py
    uv run --with pytest python -m pytest -q

Before integrating into psfr-core:
1. synthetic flux/astrometry/background/multi-star recovery tests pass;
2. fresh LGS E2E gate passes and observed cutout is shape-aligned;
3. template export is identical to original model-only PSF;
4. fitting leaves all AO diagnostics and reference PSF unchanged;
5. residuals are inspected for saturation, companions, centering and
   model mismatch before considering changes to TT or HO.
