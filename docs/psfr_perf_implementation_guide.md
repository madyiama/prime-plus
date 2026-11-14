# PSF-R: performance implementation guide and closure (2026-11-14)

**Scope:** experimental branch `perf/psfr-focal-cache-profiling` / draft PR
[#42](https://github.com/madyiama/prime-plus/pull/42).
The optimized paths are **opt-in**. The production reference remains the
historical `legacy` calculation, including strict HO covariance PSD
validation. **Do not promote any of these methods to default or merge #42
directly into `refactor/psfr-core`**: #42's base is the unvalidated fitting
branch `feat/psfr-ao-parameter-fit` / #39.

This guide is an architectural index for the implementation. Detailed
math and local protocols remain in the per-method documents:
- [Focal symmetry](psfr_focal_symmetry_perf.md) (#40)
- [Focal cache/profiling](psfr_focal_cache_profiling.md) (#42)
- [Vii factorization](psfr_vii_factorization_otf4d_audit.md) (#44)
- [Vii real FFT](psfr_vii_real_fft.md) (#45)
- [Direct focal quadrant](psfr_focal_quadrant.md) (#46)
- [Exact Ialpha CPU workers](psfr_focal_ialpha_workers.md) (#47)

## Implementation map

| Problem / why it was slow | Code and algorithm | Mathematical contract |
| --- | --- | --- |
| Dense Flicker covariance repeats Bessel work on identical separations | `p3/aoSystem/focalFast.py:symmetric_focal_layer` evaluates unique I1/I2 lag grids and I4/I5 point grids; I6 follows I3 transposition | The six terms and finite-L0 McDonald `FourierUtils.Ialpha` stay unchanged |
| Full pair matrix computed even though only one quadrant is averaged | `direct_focal_quadrant_map` enumerates exactly the historical negative-lag pupil-pair subset and aggregates with `np.bincount` | The historical opposite signed X/Y flattening is intentional; the other quadrants use the existing reflection |
| Full focal matrix invalid to bypass when a spatial filter is requested | `anisoplanatismModel.focal_angular_function` falls back to the full symmetry-plus-filter path when `h_filt` is not scalar | Never discard `Hfilter` or cache its output as geometry-only |
| Expensive independent Bessel evaluations on CPU | `focalFast.parallel_ialpha`: fixed row blocks, same SciPy `Ialpha`, float64, managed `ThreadPoolExecutor`, ordered assembly | No spline/table/interpolation; 1 worker remains default; only `quadrant` may use 2 or 4 |
| Repeated geometric focal SF across identical observation geometry | `p3/aoSystem/focalCache.py` SHA-256 key and validated atomic `.npz` | Only unweighted, unfiltered **focal geometry**; never optical pupil/OTF, TT, HO, NCPA or finished science PSF |
| Vii kernel computes per-mode squared-mode FFT redundantly | `p3/psfr/hoFactorized.py:accumulate_vii_factorized`: factor `FFT(sum λM²)` | Algebraically same covariance; floating-point summation order changes |
| Full complex FFTs duplicate the Hermitian half spectrum | `accumulate_vii_factorized_real` with `np.fft.rfft2/irfft2` | Same double-precision physical kernel; numerical parity must be evaluated rather than assumed |
| Total constructor time obscures the actual bottleneck | `p3/psfr/psfR.py:timing_breakdown_ms`, `psfr_profile_breakdown.json`, optional cProfile | Nested timers are subsets; cProfile overhead invalidates absolute wall-time comparisons |

## Command-line switches and safeguards

These switches are available in the experimental Keck smoke/E2E scripts
`scripts/keck_n0013_smoke.py` and `scripts/keck_psfr_v1_e2e.py`:

```text
--ho-vii-method {legacy,factorized,factorized_real}   # default legacy
--focal-algorithm {legacy,symmetry,quadrant}           # default legacy
--focal-workers {1,2,4}                              # default 1
--focal-cache-dir <directory>                        # optional; default off
--profile-psfr                                       # off unless requested
```

The CPU workers setting >1 is valid only with `quadrant`. Finite
`L0>0` uses the exact McDonald kernel; `L0=+inf` uses the analytic
Kolmogorov limit and opens no Bessel thread pool. For a matrix
`Hfilter`, direct quadrant and cache are bypassed. No gpu backend,
PSD correction, altered TT propagation or model fitting is enabled by
these performance switches.

The covariance `strict` HO policy rejects negative modes beyond the
numerical threshold; do not silently clip them to make the model run.
The pupil rotation, mask/parity/resampling and per-exposure HO/NCPA/TT
remain exposure-specific. The geometry-only cache is not an optical
or detector PSF cache.

## Exact numerical acceptance vs physical calibration

1. Synthetic tests compare the SF/OTF against `legacy` or `symmetry`
   and exercise multiple `L0`, sampling and field-direction cases.
2. The final Keck A/B comparisons hold FITS/SAV, pupil orientation,
   HO/TT recipe and all settings fixed while changing one performance
   option. Compare full `telemetry_only_psf.npy` cubes, wavelengths,
   flux, peak, centroids and timing. A strict comparison normally
   uses `rtol=atol=1e-12`.
3. A pass is **numerical equivalence for those inputs**, not an
   absolute Keck/STRAP calibration, AO-covariance physical proof or
   validation of a PSF science fit. Broader tests on another pupil
   orientation, NGS/LGS and multiple atmospheric conditions remain open.

### Empirical Mac measurements supplied by the maintainer

| Case | Observation |
| --- | --- |
| Synthetic #46, nActu=7, seven layers | Symmetry 255.8 ms → quadrant 125.0 ms (**2.05×**); relative L2 ≈3.1e-15; `tracemalloc` 9.91 → 1.88 MB (NOT native RSS) |
| Synthetic #47, nActu=10 | 1 / 2 / 4 worker median 393.4 / 225.4 / 146.1 ms; **2.69×** at 4, bitwise parity |
| Synthetic #47, nActu=21 | 1 / 2 / 4 worker median 6.56 / 3.75 / 2.50 s; **2.62×** at 4, bitwise parity |
| Real Keck n0120, quadrant, 4 vs 1 worker | focal kernel 5.800 → 2.575 s (**2.25×**); full PSF (128, 128, 3) bitwise equal; identical flux, peak and centroid |
| Same n0120 pair | init PSF-R 13.388 → 11.796 s (single pair of runs, NOT a repeated benchmark) |
| Later optimized n0120 run | init PSF-R **8.287 s**: HO SF **5.135 s**, frequency domain **2.327 s**, other ≈0.825 s (different execution; not directly comparable) |
| Complete E2E shell timing | **29.257 s** elapsed: telemetry, computations, acceptance checks, figures and output I/O; distinct from PSF-R constructor |

The benchmark numbers are supplied Mac measurements, not measurements
recomputed by CI. The last PR #47 CI succeeded (856 passed, 97 skipped,
88 subtests). Direct Keck 1 vs 4-worker parity **passed for n0120**.
No claim is made for a second independent pupil rotation.

## Reproduce end-to-end with explicit opt-in methods

```bash
git fetch origin
git switch perf/psfr-focal-cache-profiling
git pull --ff-only
uv sync

export P3_DISABLE_GPU=TRUE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

time uv run python scripts/keck_psfr_v1_e2e.py \
  --fits "/path/to/n0120.fits" \
  --sav "/path/to/n0120_LGS_trs.sav" \
  --calib p3/aoSystem/data/KECK_CALIBRATION \
  --config config/psfr_keck_v1.toml \
  --ho-vii-method factorized_real \
  --focal-algorithm quadrant \
  --focal-workers 4 \
  --output results/psfr_v1_perf_closeout
```

Read `results/psfr_v1_perf_closeout/psfr_v1_acceptance.json` for
numerical artifact gates and
`results/psfr_v1_perf_closeout/n0120/psfr_profile_breakdown.json`
for initialization stages. For strict comparison:
`scripts/compare_keck_focal_ialpha_workers.py`; for synthetic:
`scripts/bench_psfr_focal_ialpha_workers.py`.

**Do not enable cProfile when timing end-to-end speed.** Do not sum
`frequency_domain` and `focal_kernel_seconds`; the latter is nested
inside the former. `psfr_init_total` is part of the shell `time` total.

## End-of-performance-work decision

- Freeze the experimental numerical optimizations as opt-in;
  do not chase GPU or further threading solely for a faster
  constructor.
- Keep #42 a **draft** because its base #39 carries fitting features
  that must not be imported into the refactor core implicitly.
- When ready to integrate validated performance improvements, open a
  new **isolated** PR targeting `refactor/psfr-core` with only the
  HO/focal code, tests, docs and Keck acceptance evidence—no fitting.
- Prioritize multi-acquisition scientific reliability: HO covariance
  PSD, STRAP calibration, correct NIRC2 pupil/PSF morphology and
  chromatic physics, before the fitting stage.
