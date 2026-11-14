# PSF-R: real one-sided FFT for factorized HO Vii (PR #45)

Branch: perf/psfr-vii-real-fft
Base: perf/psfr-focal-cache-profiling (PR #42). PR #44 merged;
PR #43 closed without merge. No changes to main or refactor/psfr-core.

## Mathematical and instrument contract

Keck n0120 accepted the existing optional complex factorization:
HO SF computation 12.426 to 6.934 s; full science PSF max absolute
difference 4.337e-19 against historical Vii. Its internal HO SF
max difference 1.377e-11 is explicitly documented and never hidden.

The factorized Vii expression uses 441 complex FFT2s of 441 real
pupil-masked modal maps Mk, plus one complex FFT2 of the sum
of weighted mode squares. Each real input has Hermitian Fourier
symmetry. The new opt-in one-sided algorithm is:

    A = sum_k lambda[k] * Mk**2
    H = sum_k lambda[k] * abs(RFFT2(Mk))**2
    half = real(RFFT2(A) * conj(FFT2(P))[:,:nx//2+1]) - H
    spatial = IRFFT2(2*half, s=(ny,nx))
    dphi = den*fftshift(spatial)

The usual pupil autocorrelation G, denominator den, pupil
rotation, normalizing masks, modal eigensystem, PSD checks, OTF,
PSF and science wavelength model remain unchanged. The original
FULL NumPy FFT of the CURRENT pupil is still used for G and
its conjugate half spectrum within this SAME call. NO pupil,
mode, OTF or HO FFT cache. No multiworker or GPU.

An explicit irfft2 output shape preserves odd and even pupil
support sizes. The one-sided modal power sum reduces FFT scratch
but does not eliminate the full M_all modal cube. The reported
tracemalloc memory is an approximation, not total process RSS.

The existing factorized-complex mode remains default for
comparisons; the application default stays legacy.

## Local environment and unit tests

    git fetch origin
    git switch --track origin/perf/psfr-vii-real-fft
    uv sync

    export P3_DISABLE_GPU=TRUE
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1

    uv run --with pytest python -m pytest -q \
      tests/test_psfr_vii_real_fft.py \
      tests/test_psfr_vii_real_fft_cli.py \
      tests/test_psfr_vii_real_fft_bench.py
    uv run --with pytest python -m pytest -q

## 1. Synthetic benchmark

    uv run python scripts/bench_psfr_vii_real_fft.py \
      --n-pupil 48 --n-modes 120 --trials 5 \
      --output results/bench_psfr_vii_real_fft.json

Compares complex factorized and real factorized with HO OTF/SF
numerical differences, median runtime and tracemalloc peaks.
A strict 1e-12 synthetic internal parity failure exits nonzero
AFTER recording the result for investigation.

## 2. Real Keck n0120 A/B

Both cases must use the SAME FITS/SAV, pupil, rotation, config
and cache status. No geometric cache in either run to isolate
FFT acceleration. No cProfile for timing.

    for method in factorized factorized_real; do
      uv run python scripts/keck_psfr_v1_e2e.py \
        --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
        --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
        --calib p3/aoSystem/data/KECK_CALIBRATION \
        --config config/psfr_keck_v1.toml \
        --output "results/vii_real_fft_$method" \
        --focal-algorithm symmetry \
        --ho-vii-method "$method"
    done

    uv run python scripts/compare_keck_vii_real_fft.py \
      --complex results/vii_real_fft_factorized/n0120 \
      --real results/vii_real_fft_factorized_real/n0120 \
      --output results/compare_keck_vii_real_fft.json

The science acceptance gate is rtol=atol=1e-12 on the FULL
128x128x3 reconstructed science PSF. The internal HO SF strict
parity is preserved as a separate diagnostic, never silently
relaxed. The Keck comparator also reports centroid, flux and
peak changes per wavelength, plus HO and whole constructor times.

The real-FFT method must show a reproducible speedup to justify
integration. Additional pupil orientations and stellar fits
should be audited before changing the production default.

## Follow-ons

If real FFT passes science and runtime validation, merge into
stacked PR #42, not main. If the PSF gate fails, keep the existing
validated complex-factorized path. Further CPU-only improvements
may target allocation reuse and FFT data layout; for cold geometry
the next significant target is the exact Bessel/Ialpha kernel,
not the already-cheap 4D wavelength-loop FFT.
