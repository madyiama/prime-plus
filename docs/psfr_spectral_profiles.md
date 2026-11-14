# PSF-R stage 4A — measured NIRC2 spectral profiles

Working branch: refactor/psfr-spectral-profiles
Target branch: refactor/psfr-core (NEVER main).

## Sources

Official Keck NIRC2 filter list (central wavelength, band limits):
https://www2.keck.hawaii.edu/inst/nirc2/filters.html

Source spreadsheet linked on that page:
https://www2.keck.hawaii.edu/inst/nirc2/filter_passbands.xls

Keck AO Imaging (KAI) ASCII measured profiles, at pinned upstream commit:
https://github.com/Keck-DataReductionPipelines/KAI/tree/main/kai/filters

    uv run python scripts/fetch_nirc2_filters.py --filters Kp Ks K J H
    uv run --with pytest python -m pytest -q tests/test_psfr_spectral.py

Download is explicit and SHA-1 verified against upstream Git blob. No
automatic HTTP connection occurs during PSF-R. Raw data stay unchanged.
KAI files have wavelength [micrometres] and transmission [percent].
Some include duplicate wavelengths and slightly negative baselines.
The reader averages duplicate wavelengths, clips negative baseline to
zero with a RuntimeWarning, and returns a cleaning audit. Non-finite
source values are an error and require separate scientific inspection.

## Units, SED and throughput

Photon weights use the integral across each measured passband segment:

    W_j = integral_bin_j T_filter(lambda) S_ph(lambda)
          T_atmosphere(lambda) d_lambda

    w_j = W_j / sum(W_j)

Here S_ph is PHOTON flux density per micrometre (arbitrary normalization);
if the input is energy F_lambda, convert to photons first by multiplying
by lambda/(h*c). Individual bin wavelengths are photon centroids,
not arbitrary uniformly spaced wavelengths. No implicit rectangular
top-hat approximation, or SED application twice.

The filter transmission alone is not total telescope+AO+camera
throughput. Add atmospheric and other measured factors explicitly.
Different stars can have different SED weights.

## Status: not yet a full chromatic PSF forward model

This stage only prepares authentic passbands, photon-integrated quadrature,
and a strict unit-flux PSF-cube integration primitive.

The existing sf_3D_to_psf_3D path reuses a reference-wavelength PSF;
sf_3D_to_psf_4D has different phase-unit conventions. Neither is
silently swapped into the validated Keck V1 runner.

Next stage: physically propagate at each wavelength, ensuring:

- diffraction-limited OTF with lambda-dependent pupil-lag geometry;
- phase SF converted as (lambda_ref/lambda)^2 at matching pupil baseline;
- static OPD phase with 1/lambda scaling;
- pixel MTF on a COMMON detector angular grid and correct binning;
- per-slice unit flux, then photon-weighted spectral integration;
- flux/astrometry/background rendered separately;
- explicit analytical Airy/Maréchal/flux/spectral-convergence tests.

## Speedup plan and how to measure

- Cache immutable AO telemetry estimates, pupil FFT and autocorrelation,
  DM modes and grid/OTF geometry; do not recompute at each fitter step.
- Vectorize Vii FFTs over a LIMITED batch of modes. Preserve strict PSD
  checks and validate against original Vii kernel before switching.
- Avoid repeated spline support interpolation on the same grid. Profile
  scipy.fft backends and BLAS threading with controlled env vars.
- GPU CUDA/CuPy is useful only on NVIDIA hardware. On Apple Silicon,
  MLX/Metal could be tested separately; small jobs can lose on transfers.
- Benchmark actual Keck FITS+SAV by stage and peak RSS on the same machine.
  The existing 64-mode Vii ~80ms is NOT a full PSF-R benchmark.
