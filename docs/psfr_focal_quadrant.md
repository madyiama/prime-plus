# PR46 — direct negative-lag focal covariance quadrant (Keck PSF-R)

Branch: perf/psfr-focal-quadrant
Target: perf/psfr-focal-cache-profiling (PR #42, after merge of #45).
Do not merge this stack into main or refactor/psfr-core until real-data validation.

## Physics and exact geometry

The opt-in --focal-algorithm quadrant evaluates the existing six-term
Flicker focal-angular phase covariance, with the same finite-outer-scale
McDonald Ialpha = scipy.special.kv(5/6, x) and the exact Kolmogorov
L0=+inf limit. This is NOT a radial approximation, interpolation change,
GPU port, altered HO/TT correction or a change in production default.

The existing symmetric covariance path computes a full n_ph**2 x n_ph**2
matrix and THEN selects the negative-lag quadrant to construct its
averaged map. The new path computes only those contributing ordered
pupil-point pairs:
  i = ix*n_ph + iy, j = jx*n_ph + jy
  ix >= jx, iy <= jy
  rho_x = u[jx] - u[ix], rho_y = u[iy] - u[jy].
The unusual opposite sign conventions of rho_x and rho_y are part
of the historical vector_to_separation code; do not "correct" them.

The direct method evaluates I3 and its transposed I6 on this subset,
I4/I5 on n_ph x n_ph grids, I1/I2 on a per-lag grid, then uses the
same np.bincount means and quadrant reflection as the existing
grouped_matrix_to_map. It does not allocate a full n_ph**4 pair matrix.
All values use float64; changing summation order can change the last
few bits. The exact six terms and coefficients remain unchanged.

A non-scalar spatial Hfilter MUST use the complete historical
symmetry matrix and multiplication BEFORE averaging. It is never
silently discarded. The direct path is only used for scalar filters.
The old --focal-algorithm legacy and symmetry remain unchanged.
The default remains legacy in all constructors and Keck entry points.

## Cache

Geometry-only cache is optional. Keys include the covariance algorithm
so that symmetry and quadrant do not share arrays with slightly
different floating summation order. Existing symmetry keys are unchanged.
No optical PSF, rotated Keck pupil, NCPA/OPD, HO, TT or final OTF is
cached. A non-scalar Hfilter bypasses caching entirely.

The diagnostic "covariance_ms" includes lag grouping for direct
quadrant, so compare covariance_ms + grouping_ms between modes, not
either timer independently. "quadrant_direct" and
"quadrant_filtered_fallback" show which path was used.

## Regression and synthetic benchmark

    git fetch origin
    git switch --track origin/perf/psfr-focal-quadrant
    uv sync

    uv run --with pytest python -m pytest -q tests/test_psfr_focal_quadrant.py
    uv run --with pytest python -m pytest -q

    export P3_DISABLE_GPU=TRUE
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1

    uv run python scripts/bench_psfr_focal_quadrant.py \
      --n-actu 7 --layers 7 --n-otf 128 --trials 5 \
      --output results/bench_psfr_focal_quadrant.json

The benchmark records medians, strict full-layer SF parity
(rtol=1e-12, atol=2e-11), and tracemalloc tracked peak allocations.
It is NOT whole-process RSS or real Keck speedup. Do not run with
cProfile to claim absolute performance.

## Keck n0120 A/B (no cache, same HO/TT and science inputs)

    for method in symmetry quadrant; do
      uv run python scripts/keck_psfr_v1_e2e.py \
        --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
        --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
        --calib p3/aoSystem/data/KECK_CALIBRATION \
        --config config/psfr_keck_v1.toml \
        --ho-vii-method factorized_real \
        --focal-algorithm "$method" \
        --output "results/quadrant_$method"
    done

    uv run python scripts/compare_keck_focal_quadrant.py \
      --symmetry results/quadrant_symmetry/n0120 \
      --quadrant results/quadrant_quadrant/n0120 \
      --output results/compare_keck_focal_quadrant.json

Both runs must use the SAME FITS/SAV, pupil orientation, resampling,
parity and HO method. Both runs MUST have no geometric cache and no
cProfile. The comparator rejects missing provenance, different
settings, nonfinite or incompatible cubes. Full Keck spectral PSF
acceptance uses rtol=atol=1e-12; it also reports per-wavelength
flux, peak and centroid shifts, focal-kernel and constructor times.
The Python tests do not substitute for these Keck measurements.

Repeat on a second real Keck acquisition with a different pupil
rotation. A reproducible improvement in both focal and full PSF-R
time is needed before promoting quadrant, and main remains unchanged.
