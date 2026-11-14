# PSF-R CPU performance: Vii factorization and 4D OTF loop audit

Branch: perf/psfr-vii-factorized-otf4d-audit
Base: perf/psfr-focal-cache-profiling (PR #42).
PR #43 was closed without merge. Neither main nor refactor/psfr-core
is changed.

## Contract

- CPU only, float64 and complex128, NumPy FFT, no multiple workers.
- Reconstruct the Keck rotated pupil and all HO/TT/optical OTFs on
  every observation; do not cache them.
- Unmodified covariance PSD rejection, eigenvalues/eigenvectors,
  pupil sampling, modal weights, FITS/SAV and wavelength physics.
- Scientific acceptance: rtol=atol=1e-12 on BOTH the 2D HO phase SF
  and the entire 128x128x3 science PSF from real Keck telemetry.
- Mathematical identities can change floating-point rounding. All
  experiments are opt-in; the original scientific path stays default.

## 1. Experimental Vii factorization

Historical:
    sum[k] lambda[k] * (Re[FFT2(Mk**2)*conj(FFT2(P))] - abs(FFT2(Mk))**2)

Algebraic factorization:
    A = sum[k] lambda[k]*Mk**2
    B = sum[k] lambda[k]*abs(FFT2(Mk))**2
    result = Re[FFT2(A)*conj(FFT2(P))] - B

For 441 modes, transform count falls from 882 to 442,
without approximating the Vii physical equation. The summation/FFT
operation order changes, so the numerical difference may exceed
the 1e-12 gate, especially after dividing by small pupil OTF
autocorrelation. This is an experimental option, not a new default.

### Synthetic tests

    git fetch origin
    git switch --track origin/perf/psfr-vii-factorized-otf4d-audit
    uv sync

    # All candidate/baseline timings: one CPU worker, GPU disabled.
    export P3_DISABLE_GPU=TRUE
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1

    uv run --with pytest python -m pytest -q \
      tests/test_psfr_vii_factorized.py tests/test_psfr_otf4d_loops.py
    uv run --with pytest python -m pytest -q

    uv run python scripts/bench_psfr_vii_factorized.py \
      --n-pupil 48 --n-modes 120 --trials 3

Output: results/bench_psfr_vii_factorized.json.
On strict parity failure the script writes JSON and exits nonzero.
This is intentional: no scientific promotion is allowed.

### Real n0120 A/B

Run same physical input with legacy/factorized. No focal disk cache in
either run, for unbiased benchmarking.

    for method in legacy factorized; do
      uv run python scripts/keck_psfr_v1_e2e.py \
        --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
        --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
        --calib p3/aoSystem/data/KECK_CALIBRATION \
        --config config/psfr_keck_v1.toml \
        --output "results/ho_vii_$method" \
        --focal-algorithm symmetry \
        --ho-vii-method "$method"
    done

    uv run python scripts/compare_keck_vii_factorized.py \
      --legacy results/ho_vii_legacy/n0120 \
      --factorized results/ho_vii_factorized/n0120

Output: results/compare_keck_vii_factorized.json.
The comparator rejects changed FITS/SAV, pupil orientation,
geometric cache status or cProfile usage. Both the HO phase SF
and science PSF must pass strict per-pixel equivalence before merge.
No change to the user-facing default HO Vii algorithm.

## 2. Investigation of 4D OTF/PSF for loops

Current p3/aoSystem/FourierUtils.py sf_3D_to_psf_4D:
- recomputes wavelength-specific static optical OTF with getStaticOTF;
- applies wavelength-dependent jitter and angular phase scaling;
- performs the spatial FFT inversion per wavelength/source;
- uses potentially different k_[wavelength] bin/crop geometry;
- normalizes each separate monochromatic PSF.

Only some portions can be vectorized. The provided microbenchmark
tests precomputed static OTF and phasor across all wavelengths,
comparing the historical wavelength loop to one vectorized 4D
OTF product and NumPy FFT along ONLY two spatial axes.

It reports time, numerical parity and approximate RAM via
tracemalloc, but EXCLUDES getStaticOTF, binning, crop and spectral
flux normalization. It does not modify science code, and cannot
establish whole-Keck speedups by itself.

    uv run python scripts/bench_psfr_otf4d_loops.py \
      --n-pupil 128 --sources 1 --wavelengths 3 --trials 15

    uv run python scripts/bench_psfr_otf4d_loops.py \
      --n-pupil 128 --sources 4 --wavelengths 12 --trials 10 \
      --output results/bench_psfr_otf4d_large.json

The n0120 science path has 1 source and 3 bins: full 4D
vectorization may not be worthwhile due to allocation overhead.
The prior profile measured total error-budget/forward initialization
at ~0.120 s, whereas HO phase SF took ~11.25 s; focus on HO first.

## 3. Decision process

- Reject candidate on any 1e-12 HO SF or science PSF mismatch,
  even when the final image alone is nearly identical.
- If passing, require reproducible speedup at single-thread and
  realistic peak RAM, on at least two pupil orientations.
- If factorization fails strict numerics, retain the research report,
  keep legacy production behavior and examine per-mode FFT buffer
  reuse and memory layout without reordering FFTs or sums.
- No automatic promotion or merging to stable refactor branches.
