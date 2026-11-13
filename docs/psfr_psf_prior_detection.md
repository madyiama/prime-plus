# PSF-R prior source detection before stellar fitting

Feature branch feat/psfr-psf-prior-detection, based on
refactor/psfr-stellar-fitting (PR #37), stacked on #36.
Neither the reference Keck V1 PSF-R nor main is changed.

## Purpose

A brightest-pixel detector can confuse a hot detector pixel, a cosmic
ray or other single-pixel excess with a star. For a telemetry-only
reconstructed PSF prior, candidate detection should require a spatial
signal with the correct PSF footprint, not simply the brightest pixel.

The implementation is conservative and diagnostic:

1. Estimate a constant background from the valid science cutout.
2. Obtain a noise scale from the supplied positive sigma map, or from
   the robust MAD of neighbor differences, without claiming to model
   NIRC2 read/Poisson noise. The latter is NOT a calibrated detection
   false-positive rate.
3. Cross-correlate the residual with a PSF-R core centred on the
   reconstructed PSF's actual maximum. Respect user masks/saturation.
4. Test the candidate against the alternative of a single hot pixel:
   compare the weighted chi-square gain of the PSF versus a one-pixel
   impulse, and require an independent positive halo SNR with the
   candidate's central pixel explicitly excluded.
5. Rejected point-like outliers are masked from subsequent source
   searches AND from the final stellar fit. They remain available in
   the source catalog and are called "hot_pixel_like", never treated
   as confirmed physical detector defects.
6. Joint-refit the currently accepted stars with the FROZEN PSF and
   a robust loss for detection only, then search the residual. A
   minimum source separation prevents counting the same star again.
7. Stop at configurable SNR limits, number of stars, or examined
   candidates. The returned count is a detection hypothesis, not
   a ground-truth source catalogue.

A second exposure/dark frame, bad-pixel map and injection/recovery
benchmarks are needed to quantify reliability. An unresolved true
stellar PSF may be impossible to distinguish from a single-pixel
artifact. Close companions below the configured separation cannot be
reliably counted by this greedy algorithm. Imperfect residual HO/TT,
anisoplanatism and Keck pupil model could cause spurious candidates.

## Restart with existing reconstructed PSF-R results

No new telescope model is needed for detection/fitting after PSF
reconstruction. The previously exported fitting template and exact
telemetryKeck cutout are reused:

    git fetch origin
    git switch --track origin/feat/psfr-psf-prior-detection
    uv sync

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120 --detect-only

Expected files in results/psfr_v1_fit/n0120/stellar_fit:

- source_detection.json: accepted and rejected coordinates, PSF and
  halo SNRs, PSF-vs-single-pixel statistic, thresholds and stop reason;
- source_detection.png: candidates overlaid on the NIRC2 cutout;
- source_detection_valid_pixels.npy: valid pixels AFTER rejection.

To run detection then simultaneous flux/x/y/background fitting:

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120

Extra stellar fitting results are fit_report.json, fit_comparison.png,
model.npy and residual.npy. Model parameters AO/TT/r0/pupil never move.
No Keck FITS/SAV reprocessing takes place for these operations.

For fainter targets tune the detection thresholds explicitly:

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120 \
      --max-sources 6 --detection-snr 6 --halo-snr 4 \
      --min-separation 4 --core-radius 9

For a crowded field or unreliable auto-detection, retain the manual
mode (no auto rejection or catalogue):

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120 \
      --xy 62 64 --xy 83 75

The positions above are illustrations, NOT determined NIRC2 sources.
Optional --sigma, --mask, --saturation are honored by both detector
and stellar optimizer. The mask convention is True=valid.
If default star search finds zero sources, it writes the detection
catalogue but does not invoke a bogus one-star fit.

## Full rebuild if all previous results have been deleted

Use the original FITS/SAV pair and Keck nominal configuration:

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --output results/psfr_v1_fit

    uv run python scripts/export_psfr_template.py \
      --case results/psfr_v1_fit/n0120

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120

This reproduces the unchanged Keck V1 model; the new detector uses
only the saved .npy images and .npz PSF template. The chromatic PSF-R
model is not yet integrated or scientifically validated on Keck.

## Acceptance gates

- Inject multiple subpixel synthetic PSF stars plus hotter isolated
  pixels; reject pixels and recover the correct source count.
- Empty field with a hot pixel must not count a star.
- Handle sigma and explicit valid masks and keep original inputs
  unchanged.
- Save a complete detection audit and mask; never silently delete a
  detector pixel from the science image.
- Test with real Keck n0120 and visually inspect accepted/rejected
  catalogue and residuals before merging. Adjust threshold/ROI for
  anisoplanatic mismatch and other real NIRC2 behaviors.

    uv run --with pytest python -m pytest -q tests/test_psfr_detection.py
    uv run --with pytest python -m pytest -q
