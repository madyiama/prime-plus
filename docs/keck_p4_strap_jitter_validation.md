# P4 — STRAP/NIRC2: identify physical jitter without image-based tuning

**Status:** opt-in diagnostics on branch `feat/psfr-p4-strap-jitter-validation`.
Parent: `main` with PR8, PR16 and PR17 merged. P1 r0/L0 and
P2/P3 forward components remain unchanged. This is a scientific
**audit**, NOT a certified Keck STRAP calibration.

## Why the image looks too broad

Keck n0013 with TT enabled appears markedly broader than n0012
without TT, but those are **different observations** and the
science scene appears binary. Do not use that comparison to estimate
jitter. A first historical n0013 van Dam+unity run gave approximately
31.6 mas RMS per axis for the legacy covariance and 41.2 mas for
spectral reconstruction, with about 87.7% of observed STRAP centroid
variance attributed to filtered measurement noise. These are
conditional numerical predictions, not independently validated sky
motion; they MUST be reproduced with the exact new model revision.

## Theoretical geometry and provenance

Keck STRAP B.DTTCENTROIDS are *interpreted* as angular residuals in
arcsec and converted to RMS Noll Zernike tilt OPD [m] using
`a = D*theta/4`. The model builds a Gaussian image-motion OTF

`Dphi_tt(u) = (2*pi*D/lambda)^2 [u_sensor]^T C_theta [u_sensor]`,

where `C_theta=(4/D)^2*C_OPD` (rad²), and
`u_sensor=Q(angle,parity)*u_camera`. Therefore a pure coordinate
rotation/reflection does NOT change trace or eigenvalues of the
angular jitter covariance: it changes only the ellipse orientation.

The earlier Keck method equated `tt_axis_angle_deg` with
`tel.pupilAngle`. That may be inaccurate because the STRAP-axis
reference and the NIRC2 pupil-mask rotation have different
instrument definitions. The absolute STRAP→detector angle and optical
gain remain **unknown**.

P4 adds `tt_axis_angle_deg` (optional) and `tt_axis_parity` (native,
flip_x or flip_y) to `psfR`. **When angle is omitted and parity
is native, the historical pupil-angle coupling and PSF formula are
retained.** Explicit angles are marked *uncalibrated hypotheses*.
No 90-degree offset is assumed automatically.

PR#8 also retained `legacy_unscaled_opd`, a diagnostic-only kernel
that multiplies the physically normalized Noll angular Dphi by 1/16.
A lower blur with this kernel does not establish that the original
normalization was physically correct.

## P4-A: the same FITS and same SAV, four ablations

```bash
git fetch origin
git switch --track origin/feat/psfr-p4-strap-jitter-validation
uv run --with pytest python -m pytest -q \
    tests/test_keck_p4_tt_frame.py \
    tests/test_psfr_spectral_tt_kernel.py \
    tests/test_keck_tt_geometry_and_anisoplanatism.py

# Use the actual SAME science+telemetry for all four:
uv run python scripts/keck_p4_tt_benchmark.py \
    --fits "/path/to/n0013.fits" \
    --sav "/path/to/n0013_LGS_trs.sav" \
    --calib p3/aoSystem/data/KECK_CALIBRATION \
    --output results/n0013_P4 \
    --n-layer 7 \
    --pupil-resampling physical_binary \
    --pupil-rotation header \
    --pupil-parity native \
    --noise-band 0.65 0.95 --welch-nperseg 512
```

This invokes `scripts/keck_n0013_smoke.py` four times:

| Variant | TT physical covariance | OTF convention | Use |
|---|---|---|---|
| A0_no_tt | ablated | Noll | model-only reference |
| A1_legacy_physical | `legacy_psfr` | `noll_angular` | historical covariance + physical Noll kernel |
| A2_legacy_no16_diagnostic | `legacy_psfr` | `legacy_unscaled_opd` | exact *1/16 Dphi* diagnostic, never choose by science fit |
| A3_spectral_physical | `spectral_closed_loop` | `noll_angular` | conditional physical loop W/K/P |

All use `--tt-noise-method van_dam` with the same HF band,
`--tt-mirror-model unity`, the same Welch nperseg, identical
`r0/L0/Cn2`, fixed aperture and detector passband. The reporter
verifies these invariants and checks that A1 and A2 share
**identical jitter covariances**, as required for a pure kernel
normalization comparison.

The runner deliberately stops at the first failed numerical experiment
and saves `p4_failed_variant.json` and the variant's
`p4_console.log`. Do not turn non-PSD residual covariance into
positive values just to complete the benchmark. With partial results,
the other variants can be run individually.

## Outputs and interpretation

- `p4_tt_ablation_report.json`: input and code SHA, common
  atmosphere/pupil, sensor-noise covariance, model-only PSF energy
  and peak, physical angular jitter major/minor RMS and FWHM.
- `p4_tt_ablation_model_psfs.png`: four equally scaled model PSF
  core images. No observed image registration or source fitting.
- Each subfolder: `metadata.json`, `tiptilt_diagnostics.json`,
  `tt_noise_diagnostics.json`, `tt_noise_psd.png`,
  `telemetry_only_psf.npy`, `p4_tt_metrics.json`,
  `frequency_grid_diagnostics.json`, `p4_console.log`.
- `tt_noise_spectral_cancellation`: for each STRAP axis,
  signed `P_measured - |S|² P_sensor_noise` integrals and
  frequency-bin fraction with negative residual.
  **Do not clip negative bins**; they can arise from finite
  Welch sampling and uncertain transfer functions.
  This spectrum is NOT the physical image-motion PSD.

If inputs are already available, regenerate the consolidated report:
```bash
uv run python scripts/keck_p4_tt_benchmark.py \
    --fits "/path/to/n0013.fits" \
    --sav "/path/to/n0013_LGS_trs.sav" \
    --calib p3/aoSystem/data/KECK_CALIBRATION \
    --output results/n0013_P4 --summarize-only
```

## P4-B and P4-C: next instrument checks (not replaced by ablation)

1. Determine STRAP quad-cell optical gain as a function of APD flux,
   seeing, loop frequency, and detector noise. The APD_IMAT mapping
   reproduces stored centroids but does not prove arcsec-on-sky scale.
2. Establish 2020 Keck TTM servo K, WFS integration W, plant P,
   latency and feedback sign from independent RTC calibration.
3. Confront the signed, frequency-resolved atmosphere PSD
   `(P_y-|S|² P_n)/|W|²` and the actuator noise PSD
   `|P*K*S|²P_n`; document cancellation, negative bins, band-edge
   leakage, Welch variance accounting and low-frequency vibration.
4. Calibrate STRAP→NIRC2 angle/parity using independent on-sky or
   mechanical orientation, not the diffraction pupil zero angle.
5. Re-run the same four variants on at least two additional matched
   LGS telemetry/PSF observations, preferably an isolated star.
   Binary component positions, flux ratio and background may later
   be fitted with PSF fixed, but never infer a TT scale by simply
   forcing an agreement to n0013's observed double source.

## Acceptance gate

**Software:** synthetic frame and ablation tests pass, full project
pytest passes, no numerical covariance clipping, run reproducible
with version and configuration saved.

**Physical:** TT RMS / covariance / directional ellipse can be
explained through identified sensor noise, calibrated gain,
mirror/control transfer, and independent image motion.
The four-variant plot alone is not a physical calibration.

Until then PR P4 remains draft. Do not update P1 r0/L0 or P3
pupil rotation to compensate TT disagreement.
