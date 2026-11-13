# Keck/NIRC2 — memory-bounded HO Vii FFT batching

Working branch: perf/psfr-ho-batched-fft. Base: perf/psfr-focal-cache-profiling
(PR #42), stacked on the AO fitting refactor. Do not modify main or
refactor/psfr-core before real instrument acceptance.

## Evidence and scope

Profiling of Keck n0120 with a warm focal geometry cache:
- PSF-R init 11.89 seconds;
- HO residual phase SF ~11.25 seconds;
- Vii modes2Otf ~10.97 seconds; 890 FFT2 calls ~9.49 seconds.

Experimental implementation groups spatial FFTs into modal batches.
For each mode k it computes exactly the historical Vii expression:

    Vk = real(FFT2(Mk**2)*conj(FFT2(P))) - abs(FFT2(Mk))**2
    buf = buf + lambda[k]*Vk

The only altered operation is FFT evaluation: scipy.fft.fft2 over
array [y,x,n_modes_in_batch], axes=(0,1), with configurable workers.
The summation is still done in original increasing eigenmode order.
Do NOT reduce modes with einsum, change PSD policy, or interpret
minor roundoff differences as a change in physical covariance.

The current Keck pupil, telescope rotation, DM influence modes and
optical OTFs are ALWAYS rebuilt for each new observation. No modal
FFT, HO structure function, pupil or OTF is cached. The batching
scratch budget is capped at 256 MiB. The Uij estimator is not
accelerated. The old NumPy implementation is still DEFAULT.

## Synthetic test

    git fetch origin
    git switch --track origin/perf/psfr-ho-batched-fft
    uv sync
    uv run --with pytest python -m pytest -q tests/test_psfr_ho_batched_fft.py
    uv run --with pytest python -m pytest -q

    uv run python scripts/bench_psfr_ho_fft.py \
      --pupil-size 64 --modes 200 \
      --batch-sizes 8 16 32 --workers 1 --trials 3

Output: results/bench_psfr_ho_fft.json. The synthetic median
speedup is indicative only, NOT a verified Keck speedup.
Try workers=1 first and measure 2 or 4 only after parity.

## Strict real Keck A/B: no focal cache, same pupil orientation

Use identical input, calibration, physical pupil hypotheses, and
anisoplanatism algorithm for both runs. Disable focal cache for BOTH
runs so a cold-cache run is never compared with a warm-cache run.

    for mode in legacy batched; do
      uv run python scripts/keck_psfr_v1_e2e.py \
        --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
        --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
        --calib p3/aoSystem/data/KECK_CALIBRATION \
        --config config/psfr_keck_v1.toml \
        --output "results/ho_fft_$mode" \
        --focal-algorithm symmetry \
        --ho-fft-mode "$mode" \
        --ho-fft-batch-size 16 --ho-fft-workers 1
    done

    uv run python scripts/compare_keck_ho_fft_runs.py \
      --legacy results/ho_fft_legacy/n0120 \
      --batched results/ho_fft_batched/n0120

Output: results/compare_keck_ho_fft.json. The comparator checks:
- same FITS/SAV, physical Keck pupil, focal model and cache state;
- ho_dphi_ao.npy parity at rtol=atol=1e-12;
- the full 128x128x3 telemetry_only_psf.npy cube at same tolerances;
- HO-specific and overall PSF-R initialization speedups.

A parity failure means STOP, not a reason to increase tolerance.
For a cache-hot comparison, prewarm the geometric cache once, then
repeat BOTH modes with the SAME --focal-cache-dir path; verify
cache_status=hit in both psfr_profile_breakdown.json files.

## Acceptance

Tests cover uneven batches, worker count, rotated asymmetric pupils,
strict covariance PSD checks, purity of HO inputs, explicit Uij
non-support and scratch memory budgeting. At least one real Keck
A/B must pass both HO SF and science PSF parity before any merge.
Repeat measured timings to isolate CPU thermal/frequency variability.
