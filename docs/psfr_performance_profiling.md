# PSF-R initialization timing and Keck profiler

Working branch: perf/psfr-initialization-profiling
Target branch: refactor/psfr-core ONLY, not main.

Observed Keck run timings, seconds:
telemetryKeck 6.0, systemDiagnosis 6.1, STRAP figure 1.2,
psfR init 153.9, psfR forward 0.0, comparison figure 0.9.

Hypothesis: focal-angular anisoplanatism in Keck SLAO is expensive,
because several modified-Bessel covariance terms are evaluated
over every pair of pupil points and every atmosphere layer.
We do not assume this is proven before profiling.

The normal Keck E2E now writes a per-case file named
psfr_timing_breakdown.json containing top-level wall times:
ao_system, frequency_domain, fourier_model, fitting_sf, aliasing_sf,
tiptilt_sf, ho_sf, error_budget_including_forward, psfr_init_total.

The nested frequency anisoplanatism timer (freq.tani) is part of
the frequency-domain stage; DO NOT ADD it to the total again.
Fourier model's own internal timers are nested too.

To collect cumulative function costs with Python's built-in cProfile:

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits /path/n0120.fits \
      --sav /path/n0120_LGS_trs.sav \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --output results/psfr_profile \
      --profile-psfr

Inspect:

    cat results/psfr_profile/n0120/psfr_timing_breakdown.json
    head -n 65 results/psfr_profile/n0120/psfr_init_profile_top.txt

Also generates psfr_init.pstats for detailed sorting.
cProfile has overhead: do not compare its total against the
unprofiled 153.9-second run without accounting for profiler impact.

Candidate optimizations after profiling:
- cache reusable pupil/lag/atmospheric geometry across model instances;
- if proven to dominate, accelerate radially symmetric covariance
  Bessel function evaluations using accuracy-controlled interpolation;
- batch Vii FFTs only after profiling (64-mode core ~80 milliseconds);
- CPU FFT workers first; GPU only with large repeated workloads.

No model physics, numerical tolerances or Keck V1 config are altered.

Tests:
    uv run --with pytest python -m pytest -q tests/test_psfr_timing.py
    uv run --with pytest python -m pytest -q
