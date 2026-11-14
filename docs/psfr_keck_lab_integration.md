# Keck PSF Lab — consolidated PSF-R and fitting integration

**Target branch:** `refactor/psfr-core`, assembled via PR #50 after
PRs #36–#38 and the opt-in performance/fitting experiments #39/#42.

This is a **numerically optimized telemetry PSF-R + explicit stellar/AO
fitting stack**, not an independently calibrated Keck instrument model.
`main` is unchanged until separate science acceptance.

## Architecture and API boundaries

1. `p3.telemetry` and `p3.aoSystem`: Keck FITS/SAV, AO configuration,
   NIRC2 physical pupil, atmosphere, pupil-lag and Fourier geometry.
2. `p3.psfr.psfR`: telemetry-only reconstruction, error budget, full
   science PSF and structured diagnostics. **Default algorithms** are
   `ho_vii_method='factorized_real'` and
   `focal_algorithm='quadrant'`, with 4 CPU `Ialpha` workers when the
   quadrant path applies. For a matrix `Hfilter`, the exact full
   filtered covariance path is used. The geometric disk cache remains OFF.
3. `p3.psfr.template`: immutable flux-normalized science template and
   provenance / detector sampling, exported from an accepted Keck case.
4. `p3.psfr.stellar` and `p3.psfr.detection`: detector-frame source
   detection, subpixel flux/position/background fitting, masking and
   residual diagnostics. The AO telemetry kernel is **frozen**.
5. `p3.psfr.ao_basis` and `p3.psfr.ao_fit`: optional reuse of
   previously computed HO/TT/fit/alias/aniso structure functions for
   tightly constrained r0 and optional HO gain fitting. No telemetry
   re-initialization is done per stellar optimizer iteration. The TT
   gain remains fixed by default.
6. `scripts/keck_psfr_v1_e2e.py`: reproducible numerical acceptance
   with per-acquisition logs. `scripts/export_psfr_template.py`,
   `scripts/fit_psfr_stars.py` and `scripts/fit_psfr_ao.py` are
   **explicit** fitting operations; they do not run automatically.

The physical V1 recipe `config/psfr_keck_v1.toml` remains locked:
van Dam / legacy_psfr / unity / noll_angular, seven-layer atmosphere,
physical_binary/header/native Keck pupil, strict HO covariance PSD.
The label "legacy_psfr" in the **TT covariance policy** is independent
of the new accelerated HO/focal numerics; do not change TT physics by
confusing these names.

## Optimized defaults and reference switches

| Method | Default in consolidated PSF-R | Historical reference |
| --- | --- | --- |
| HO Vii | `factorized_real` | `--ho-vii-method legacy` |
| Focal covariance | `quadrant` | `--focal-algorithm legacy` |
| Focal CPU workers | 4 (quadrant), 1 (other paths) | `--focal-workers 1` |
| Disk geometry cache | off | `--focal-cache-dir PATH` (explicit only) |
| HO covariance PSD policy | `strict` | `strict` (unchanged) |
| Fitting | off in telemetry-only run | explicit separate CLI |

All defaults preserve double-precision mathematical kernels but the
`factorized_real` and direct `quadrant` transformations reorder some
floating-point operations: **bitwise parity is not guaranteed against
historical HO/Flicker kernels**. In the observed Keck n0120 A/B of
1-vs-4 workers, the optimized PSF cube (128,128,3) was bitwise
identical; this only proves that particular worker comparison.

## Reproducible Mac check on n0120

```bash
git fetch origin
git switch refactor/psfr-core
git pull --ff-only
uv sync

export P3_DISABLE_GPU=TRUE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

time uv run python scripts/keck_psfr_v1_e2e.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
  --calib p3/aoSystem/data/KECK_CALIBRATION \
  --config config/psfr_keck_v1.toml \
  --output results/psfr_keck_lab_n0120 \
  --export-ao-basis
```

**No performance flags are necessary**: the optimized configuration is
now the default for PSF-R. The optional `--export-ao-basis` is required
only if planning the later r0/HO fitting step. Omit it to run a
telemetry-only PSF-R with no AO fitting exports.

Inspect:
- `results/psfr_keck_lab_n0120/psfr_v1_acceptance.json`: numerical gate.
- `.../n0120/psfr_v1_run.log`: complete telemetry / SF diagnostics.
- `.../n0120/psfr_profile_breakdown.json`: HO/focal timings and methods.
- `.../n0120/psfr_timing_breakdown.json`: PR34 structured timing.
- `.../n0120/comparison_nirc2_psfr.png`: registered visual comparison.
- `.../n0120/ao_fit_basis_parity.json`: optional frozen AO basis parity.

### Force the original unoptimized numerical reference

```bash
uv run python scripts/keck_psfr_v1_e2e.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
  --calib p3/aoSystem/data/KECK_CALIBRATION \
  --ho-vii-method legacy \
  --focal-algorithm legacy \
  --output results/psfr_keck_lab_legacy
```

The CLI automatically uses 1 focal worker when `legacy` is selected,
unless an incompatible explicit override is supplied.

### Stellar and AO fitting — AFTER a passed V1 reconstruction

```bash
uv run python scripts/export_psfr_template.py \
  --case results/psfr_keck_lab_n0120/n0120

uv run python scripts/fit_psfr_stars.py \
  --case results/psfr_keck_lab_n0120/n0120

uv run python scripts/fit_psfr_ao.py \
  --case results/psfr_keck_lab_n0120/n0120 \
  --n-stars 2 --ao-params r0
```

Inspect `source_detection.json`, detection overlay,
`stellar_fit/fit_report.json`, AO prior/condition-number diagnostics,
and model/residual images. `--n-stars 2` means the two brightest
candidates, **not** independently confirmed stellar identities.

## Observed performance (Mac, n0120, not a universal guarantee)

- Exact focal `Ialpha` 4-vs-1 worker comparison: **2.25×** faster
  (5.800 to 2.575 seconds), 128×128×3 science PSF bitwise identical.
- One optimized PSF-R constructor: **8.287 seconds**, including
  5.135 s HO SF and 2.327 s frequency-domain initialization.
- End-to-end shell time on a separate n0120 run: **29.257 seconds**.
  The second value is inclusive of the first plus telemetry ingestion,
  science artifacts, image plotting and file I/O; do not add them.
- Repeat on other pupil rotations, atmospheres and NGS/LGS
  configurations before declaring general physical or speed validity.

## Keck PSF Lab integration rules

- Pin the exact `prime-plus` commit in the lab's deployment manifest,
  container mount or dependency version; do not track a mutable tip
  without recording the SHA. Do **not** copy only `psfR.py`: the
  `p3.psfr`, `aoSystem`, `telemetry` and calibration contracts are
  coupled.
- Keep input FITS/SAV, configuration version, pupil angle, spectral
  convention, algorithm selection, effective workers and provenance
  in the job metadata. Preserve the PSF-R outputs and full log.
- Initial telemetry PSF-R must run only once per case; source/stellar
  fitting and constrained AO fit consume the saved immutable templates.
- A failed strict HO PSD check is a **scientific model/estimation
  failure**, not something the lab should silently repair or mask.
- Linux containers may be RAM-constrained: PR #41 SIGKILL/RSS
  investigation is **not independently validated** and streaming
  eigenmodes are not silently enabled in the optimized real-FFT path.
  Inspect cgroup memory events and record RSS peaks in real deployments.
- Avoid automatic fitting to make a failed telemetry-only PSF pass:
  scientific residuals and TT/HO assumptions must remain traceable.

## Tests and next integration

```bash
uv run --with pytest python -m pytest -q
uv run --with pytest python -m pytest -q \
  tests/test_psfr_ao_fit.py tests/test_psfr_ao_fit_cli.py \
  tests/test_psfr_focal_quadrant.py tests/test_psfr_focal_ialpha_workers.py \
  tests/test_psfr_vii_real_fft.py
```

After PR #50 CI and local Keck acceptance, merge into
`refactor/psfr-core`, then connect `keck-psf-lab` by a **separate PR**
in the observatory repository. No change to `main` is implied.
