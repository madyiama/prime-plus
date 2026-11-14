# PSF-R profiling and safe focal geometry disk cache (Keck)

Branch: perf/psfr-focal-cache-profiling
PR target: feat/psfr-ao-parameter-fit (stacked refactor).
Scientific reference: refactor/psfr-core; main MUST remain untouched.

## Critical Keck pupil-rotation boundary

For every exposure, Keck/NIRC2 pupil angle, native FITS mask,
parity, resampling, OPD/NCPA, pixel/static/diffraction OTF and
HO residual modal support MUST be rebuilt from the current telemetry.
They are **never stored nor retrieved** from the new cache.

Only the UNFILTERED, unweighted LGS focal-angular structure function
produced by Flicker's symmetric six-term geometric covariance is
cached. That function is independent of the actual rotated pupil
mask in the current P3 implementation. The key includes:
- exact telescope diameter, actuator count, nOtf, sampling and
  pupil-axis grid convention;
- atmospheric layer heights and finite/infinite outer scale;
- LGS height/direction and all science source directions;
- algorithm and versioned schema, native/OTF output mode.

The physical pupil rotation itself is intentionally NOT in the key
because this unfiltered geometric formula does not read it. If field
directions change with instrument rotation, they ARE key inputs and
invalidate the cache. A non-scalar Hfilter bypasses caching to avoid
reusing a pupil-dependent filtered product. Cache never holds a
model's final OTF, HO or TT.

Atmospheric r0 and layer Cn2 weights are not key inputs: the output
is intentionally a per-layer, r0-free kernel. Each model reweights
all layers with its own atmosphere values AFTER cache retrieval.
This is tested by modifying both values on warm cache.

No persistent cache is used unless explicitly requested.
Each key is SHA-256 of serialized exact float64 geometry and model
version. Each .npz stores a version/schema, expected key, shape,
finite array and SHA-256 of raw float64 bytes. Corrupt or stale
files are disregarded and recomputed; writes are atomic. No pickle
and no global cache state. Caller receives a private array copy.
Only this geometric cache's own entries are retained, with a
maximum of 64 entries or 256 MiB: the least recently used entries
are evicted safely. The cache can always be regenerated and never
deletes observation, pupil, OTF or other user result files.

## 1. Basic profiling without cProfile overhead

    git fetch origin
    git switch --track origin/perf/psfr-focal-cache-profiling
    uv sync

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --focal-algorithm symmetry \
      --output results/focal_profile_no_cache

Per case:
- n0120/psfr_profile_breakdown.json: AO system, frequencyDomain,
  fourierModel, fit SF, aliasing SF, TT SF, HO SF, error budget,
  nested frequency-domain anisoplanatism time, and cached focal
  substeps (covariance, grouping, interpolation).
- n0120/metadata.json: model and cache provenance.

Nested times are subsets, NOT additive. A PSF-R initialization may
include an initial science PSF in the error-budget method.

## 2. Measure a cold and a warm disk cache on identical Keck inputs

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --focal-algorithm symmetry \
      --focal-cache-dir results/keck_focal_geometry_cache \
      --output results/focal_cache_cold

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --focal-algorithm symmetry \
      --focal-cache-dir results/keck_focal_geometry_cache \
      --output results/focal_cache_warm

Verify cold/warm numerical equivalence automatically:

    uv run python scripts/compare_keck_cache_runs.py \
      --cold results/focal_cache_cold/n0120 \
      --warm results/focal_cache_warm/n0120

This produces results/compare_keck_focal_cache.json and rejects a
different FITS/SAV, pupil rotation hypothesis, mismatched cache key,
missing cache hit, or differing PSF (rtol=atol=1e-12).

Expect cold cache_status=miss and warm=hit. The output
PSF must remain identical to 1e-12. The geometric focal part
should drop substantially on warm cache; full Keck run still
needs per-exposure pupil rotation and OTF/HO/TT calculations.
One run alone on a cold cache does not imply a speedup.

## 3. Profile remaining C/NumPy routines in detail

    uv run python scripts/keck_psfr_v1_e2e.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
      --calib p3/aoSystem/data/KECK_CALIBRATION \
      --config config/psfr_keck_v1.toml \
      --focal-algorithm symmetry --profile-psfr \
      --output results/focal_cprofile

Read:
- n0120/psfr_init_profile_top.txt (top 60 functions by cumulative time)
- n0120/psfr_init.pstats (Python cProfile; scipy/numpy calls may be
  collapsed into underlying C routines)
- n0120/psfr_profile_breakdown.json

cProfile introduces overhead; benchmark absolute timing with
--profile-psfr DISABLED.

## 4. Acceptance gates

    uv run --with pytest python -m pytest -q tests/test_psfr_focal_cache.py
    uv run --with pytest python -m pytest -q

Tests include:
- same geometric kernel reused under a ROTATED, asymmetric pupil
  while separately computed diffraction OTFs differ;
- changed diameter, atmospheric layer altitude, L0, science/LGS
  direction or laser height invalidates cached geometry;
- r0/Cn2 updated after a hit;
- non-scalar filtered focal path never caches;
- legacy/cold/warm symmetry physics parity;
- corrupt ZIP contents are never reused; no mutable aliases.

For actual Keck parity, compare the cold/warm
telemetry_only_psf.npy cubes and confirm the same FITS/SAV and
pupil-orientation metadata. Do NOT compare different rotations
as if their full science PSFs should match: they SHOULD differ.

## Caveat

A disk cache is most valuable when two reconstructions have
**exactly identical** focal geometry. In many science observations
changing pointing, field direction or altitude prevents a hit.
The no-cache cold-run improvement is profiling/instrumentation
only; the main PR #40 symmetric mathematical acceleration already
provides the proven ~6.78x n0120 improvement.
