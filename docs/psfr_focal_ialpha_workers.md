# PR47 — exact CPU multithreaded Ialpha in direct focal quadrant (Keck PSF-R)

Branch: perf/psfr-focal-ialpha-workers
Target: perf/psfr-focal-cache-profiling (PR46 already merged, PR45 inherited).
DO NOT merge into main/refactor or promote defaults before real Keck science tests.

## Mathematical contract

The finite-L0 Flicker six-term covariance retains exactly the same
McDonald radial function:
    Ialpha(x,y)=Ialpha(hypot(x,y))
    Ialpha(r)=r**(5/6)*scipy.special.kv(5/6,r)/
              (2**(5/6)*scipy.special.gamma(11/6))
and Ialpha(0)=3/5. No spline, lookup, approximation, changed coefficients,
float32, GPU, fitting/HO/TT modification, or optical covariance trick.

Only the *independent elementwise evaluations* of Ialpha in PR46's
direct negative-lag quadrant are partitioned into disjoint row blocks.
Workers call the same scipy.special.kv-backed FourierUtils.Ialpha with
the same double-precision inputs. The original row order is restored
before adding/subtracting Flicker terms; np.bincount grouping and the
quadrant reflection remain serial. The thread pool is created once
per uncached focal reconstruction and reused across all layers/sources.
There are no simultaneous writes to the same output arrays.

To avoid task setup overhead, arrays smaller than 4096 elements/worker
use the serial kernel. Single-thread remains the default (workers=1).
L0=+inf uses the exact Kolmogorov power law and opens no thread pool.
A non-scalar Hfilter forces the exact full symmetry/filter path and
bypasses geometric caching; workers do not modify the filtered physics.

## CLI

    --focal-algorithm quadrant --focal-workers {1,2,4}

A workers count above 1 is rejected for legacy or symmetry. The default
for all APIs/CLIs is 1, preserving existing science behavior.

The optional geometry-only disk cache key is deliberately independent
of worker count because parallel scheduling does not change the
elementwise numerical kernel, and the exact output is required to
agree with the 1-worker result. With an explicit cache hit the
effective workers in diagnostics are 0 (no Ialpha evaluated).
NEVER cache Keck pupil/OPD/NCPA/optical OTF/HO/TT/science PSF.

## Tests

    git fetch origin
    git switch --track origin/perf/psfr-focal-ialpha-workers
    uv sync
    export P3_DISABLE_GPU=TRUE
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1

    uv run --with pytest python -m pytest -q \
        tests/test_psfr_focal_ialpha_workers.py \
        tests/test_psfr_focal_quadrant.py \
        tests/test_psfr_focal_fast.py \
        tests/test_psfr_focal_cache.py
    uv run --with pytest python -m pytest -q

Thread scheduling does not change the input values to SciPy; strict
elementwise equal tests and final focal SF 1e-12 comparisons run for
2/4 workers, native non-symmetric source coordinates, finite-L0,
L0=infinity, matrix filter fallback and independent cache copies.

## Repeated Mac synthetic benchmark

    uv run python scripts/bench_psfr_focal_ialpha_workers.py \
      --n-actu 10 --layers 7 --n-otf 128 --trials 5 \
      --output results/bench_psfr_focal_ialpha_workers.json

Repeat with --n-actu 7 and --n-actu 14. The reported medians include
thread management in the focal kernel; run with cProfile OFF. Metrics:
strict SF parity 1e-12, exact equality, actual parallel call counts,
speedup relative to 1 worker. Small input sizes may get slower with
threads — no speedup is assumed, and 4 workers is not necessarily best.

## Real Keck n0120 A/B and full PSF gate

Run three separate cases with exactly the same FITS/SAV, NIRC2 pupil
rotation/parity, config, HO/TT, no geometric cache and no cProfile:

    for workers in 1 2 4; do
      uv run python scripts/keck_psfr_v1_e2e.py \
        --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
        --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
        --calib p3/aoSystem/data/KECK_CALIBRATION \
        --config config/psfr_keck_v1.toml \
        --ho-vii-method factorized_real \
        --focal-algorithm quadrant \
        --focal-workers "$workers" \
        --output "results/ialpha_workers_$workers"
    done

    for workers in 2 4; do
      uv run python scripts/compare_keck_focal_ialpha_workers.py \
        --serial results/ialpha_workers_1/n0120 \
        --parallel "results/ialpha_workers_$workers/n0120" \
        --output "results/compare_keck_ialpha_workers_$workers.json"
    done

The comparator refuses different FITS/SAV/HO/pupil, a cache hit,
cProfile, a non-direct quadrant computation, or an unused thread pool.
The full spectral science PSF cube requires rtol=atol=1e-12.
Report measured HO-independent focal kernel time (covariance+grouping),
full PSF-R initialization, flux, peak and centroids by wavelength.
Repeat timing rounds and another acquisition with a DIFFERENT pupil
rotation before any default change. Faster synthetic kernel is not
proof of a scientifically acceptable or faster Keck E2E result.
