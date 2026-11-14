# Keck/NIRC2 PSF-R: constrained AO parameter fitting

Working branch: feat/psfr-ao-parameter-fit
Base: feat/psfr-psf-prior-detection (PR #38), itself on PR #37/#36.
Do not merge to main. Keep refactor/psfr-core untouched until validation.

## Science definition

We use P3 Keck V1's already measured and computed phase SFs:

    D_phi(r0,gho,gtt) = gho*D_HO + gtt*D_TT
                        + r0**(-5/3)*(D_fit + D_alias)
                        + D_aniso(r0, Cn2 weights)

The TT, HO, aliasing, fitting and LGS anisoplanatism kernels are
computed ONCE, not per fit iteration. The saved OTF static+pixel
transmission and P3 reference-wavelength detector renderer are also
reused. Neither the historical 3-bin spectral weighting nor the
experimentally demonstrated circular-pupil Kp chromatic model is
silently substituted for the Keck V1 model.

Keck V1 is a numerical reconstruction, not a calibrated telescope
PSF or a calibrated STRAP absolute optical gain. gho and gtt are
multipliers of PRECOMPUTED phase SFs, not direct sensor gains.

Reference values:
- r0 = the Keck line-of-sight Fried length at the saved atmosphere
  reference wavelength, from WFS systemDiagnosis;
- gho = 1, gtt = 1.
- gtt is FIXED by default because STRAP calibration is uncertain.
  Fitting gtt requires --allow-exploratory-gtt and gets a tight prior.

The default optimizer only fits r0: it is deliberately safer than
fitting strongly correlated r0+HO+TT simultaneously. The user may
enable r0 gho, and can inspect a Jacobian condition diagnostic.
Bounds relative to telemetry: r0 +/-25%, gho 0.70..1.30,
gtt 0.85..1.15 (exploratory only). Priors: log(r0/r0_tel) sigma=.15,
gho-1 sigma=.15, gtt-1 sigma=.07. These are *working priors*
not Keck-derived confidence intervals.

The fit alternates:
1. Fit stellar flux, x/y positions, and shared background against
   the frozen telemetry-only PSF, using current masks/noise maps.
2. Fit selected AO parameters with those stars temporarily fixed,
   using P3 reference-wavelength Fourier propagation and Gaussian
   telemetry-centred priors.
3. Refit stellar parameters to the updated AO PSF; repeat twice.

Results include initial/final images and residual RMS, fitted
parameter values, bounds/priors and degeneracy warnings. Do not
interpret lower residual RMS as an independent sensor calibration.

## Fresh reconstruction: capture AO basis once

Old results without an AO basis do NOT contain enough information
to vary r0/gho/gtt correctly. You must reconstruct Keck V1 once
with the NEW explicit export flag (using the same measured data):

    git fetch origin
    git switch --track origin/feat/psfr-ao-parameter-fit
    uv sync
    uv run --with pytest python -m pytest -q tests/test_psfr_ao_fit.py

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --output results/psfr_v1_fit \
      --export-ao-basis

Before writing ao_fit_basis.npz the exporter checks the nominal
phase-SF/Fourier reconstruction against the original nominal PSF
using rtol=1e-12, atol=1e-12. If it fails, no AO fitting is allowed.
Inspect the log; do not loosen tolerances or silently rescale PSFs.
The full ~154-second AO initialization occurs at most ONCE per
reconstructed telemetry run, not in fitting iterations.

Expected new files:
- n0120/ao_fit_basis.npz (frozen HO/TT/fit/alias/aniso and optical OTF)
- n0120/ao_fit_basis_parity.json (numerical baseline equivalence)
- n0120/telemetry_only_psf.npy and observed_nirc2_cutout.npy
- n0120/metadata.json
- regular Keck V1 acceptance files

## Prepare stellar input

    uv run python scripts/export_psfr_template.py \
      --case results/psfr_v1_fit/n0120

    uv run python scripts/fit_psfr_stars.py \
      --case results/psfr_v1_fit/n0120

The stellar fit defaults to PSF-prior candidate detection (PR #38).
Inspect source_detection.png and fit_comparison.png first.
The detector can still confuse AO speckles with close or faint stars.

To avoid fitting 8 spurious detections when ~2 are expected, pass
--n-stars 2 to the AO fit; this selects the TWO BRIGHTEST sources
from the previous stellar-fit report, not the ground truth. It does
not replace visually verifying coordinates, and it is not an
automatic SIMBAD crossmatch. You may instead provide exactly
two --xy detector column/row seeds as manual initial positions.

## Fit AO without recalculating telemetry

Default recommended preliminary run (one AO variable):

    uv run python scripts/fit_psfr_ao.py \
      --case results/psfr_v1_fit/n0120 \
      --n-stars 2 \
      --ao-params r0

Next, only if the first fit and residuals are well behaved:

    uv run python scripts/fit_psfr_ao.py \
      --case results/psfr_v1_fit/n0120 \
      --n-stars 2 \
      --ao-params r0 gho

Explicit TT sensitivity experiment, NOT nominal science calibration:

    uv run python scripts/fit_psfr_ao.py \
      --case results/psfr_v1_fit/n0120 \
      --n-stars 2 \
      --ao-params gtt \
      --allow-exploratory-gtt

Optional --sigma path.npy provides calibrated positive per-pixel
noise estimates if available. If missing, the default empirical
MAD of the baseline stellar fit residual is used for numerical
optimization only. Residual speckles are spatially correlated,
so this is NOT a calibrated likelihood, chi-square, detection
false-positive rate or posterior interval.

Outputs under n0120/hybrid_ao_fit:
- ao_fit_report.json: telemetry baseline, fitted AO, prior values,
  stellar nuisance parameters, RMS, forward parity and warnings;
- baseline_model.npy and hybrid_model.npy;
- hybrid_residual.npy;
- ao_fit_comparison.png.

The number and identities of stars are held fixed during each AO
fit. A later SIMBAD/astroquery crossmatch should constrain source
count and sky positions via real FITS WCS, proper motion and epoch
matching. It must NOT be replaced by unverified pixel coordinates
or a hardcoded two-star limit.

## Validation gates

- Source shapes match the recorded detector geometry.
- Baseline parity at r0_tel,gho=gtt=1 is <= 1e-12 tolerances.
- AO data archive stores arrays without pickled objects.
- Synthetic AO perturbations change PSF and are recoverable with
  explicit priors and known source positions; parameter bounds and
  TT opt-in are enforced.
- Run on n0120 and compare both the nominal baseline and hybrid
  residuals; assess degeneracy and inspect source multiplicity.
- Only then merge this stacked PR into the reference branch.
