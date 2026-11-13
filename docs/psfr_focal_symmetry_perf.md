# PSF-R: Focal anisoplanatism symmetry optimization

Working branch: perf/psfr-focal-aniso-symmetry
Target: feat/psfr-ao-parameter-fit (PR #39). Do not merge into main.

## The bottleneck hypothesis

Measured Keck reconstruction: 153.9 s in psfR initialization,
~168 s for the full E2E case. Focal-angular Flicker covariance
is a likely contributor, NOT yet a proven share of that runtime.
The per-case metadata now records:
- psfr_total_init_elapsed_s (full constructor)
- frequency_anisoplanatism_elapsed_s (nested part of constructor)
- focal_anisoplanatism_algorithm (legacy or symmetry)

## Mathematical optimization, with historical fallback

New optional focal_angular_function covariance_algorithm="symmetry":
- McDonald covariance Ialpha is radial/even: I6(i,j)=I3(j,i),
  eliminating one full n_ph**4 Bessel evaluation per layer.
- I4 and I5 need only n_ph**2 unique argument pairs; evaluate
  small 2D grids instead of repeating each across pupil pairs.
- I1 and I2 depend only on pupil baseline; evaluate unique
  (2*n_ph-1)**2 lags instead of n_ph**4 pupil pairs.
- Baseline averaging uses np.bincount grouping O(n_ph**4),
  instead of n_ph**6 boolean masks in the historical matrix_to_map.
- The finite-L0 SciPy special.kv function and exact infinite-L0
  Kolmogorov limit are both preserved. No radial approximation,
  lower precision, astrophysical model changes, or GPU conversion.
- The legacy covariance and matrix average remain DEFAULT.

Isolated Python CPU checks of the six-term kernel at n_ph=15
showed roughly 4.9-5.9x shorter covariance evaluation and ~1e-15
pointwise differences for finite outer scale. This is NOT a Keck
end-to-end acceleration claim; benchmark the real acquisition.

## Quick synthetic benchmark

    git fetch origin
    git switch --track origin/perf/psfr-focal-aniso-symmetry
    uv sync

    uv run --with pytest python -m pytest -q tests/test_psfr_focal_fast.py
    uv run --with pytest python -m pytest -q

    uv run python scripts/bench_psfr_focal_aniso.py \
      --n-actu 7 --layers 7 --trials 3

Writes results/bench_psfr_focal_aniso.json with both durations,
max absolute error, relative L2 error and focal-only speedup.

## Real Keck A/B acceptance

Both commands must use the identical FITS, SAV, calibration and Keck V1
configuration; this keeps all HO/TT and pupil science untouched.

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --output results/focal_legacy \
      --focal-algorithm legacy

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --output results/focal_symmetry \
      --focal-algorithm symmetry

    uv run python scripts/compare_keck_focal_runs.py \
      --legacy results/focal_legacy/n0120 \
      --symmetry results/focal_symmetry/n0120

The comparator requires all PSF elements to agree to rtol=atol=1e-12,
and reports both full PSF-R construction and nested anisoplanatism
duration. Abort the candidate on failure; do NOT silently relax
numerical tolerances or accept altered HO/TT physics.

Any measured speedup should be checked over repeated runs to avoid
conflating CPU thermal throttling, thread settings, disk caching,
and unrelated processes. The 80ms synthetic Vii FFT is not the
priority while an entire Keck model spends ~154s initializing.

## Next if this is not the dominant contributor

Use the existing profiling branch/PR #34 to collect per-stage timings
and cProfile output, then target the measured bottleneck rather than
preemptively moving small FFTs to a GPU. CUDA/CuPy is NVIDIA-only;
a Metal/MLX backend on Apple Silicon needs additional FFT/special-kv
precision and transfer-cost studies.
