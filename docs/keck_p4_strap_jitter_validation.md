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

## P4 n0013 initial on-sky telemetry run — quantitative interpretation (11/11/2026)

**Input:** one n0013 FITS/SAV, same Git SHA `0cb4982`, NIRC2
LGS, `n-layer=7`, `r0_model_m=0.65062548` at science
2.055501 µm, `L0_tel=13.12937 m`, `pupilAngle=84.92398 deg`,
binary pupil native, van Dam + mirror unity, no source fitting.
Report uploaded by the project user; physical gain not calibrated.

| Variant | TT angular RMS/axis (mas) | Kernel effective RMS/axis (mas) | Peak flux per pixel | peak/A0 | EE r3 |
|---|---:|---:|---:|---:|---:|
| A0 disabled | n/a | n/a | 0.0199346 | 1 | 0.333957 |
| A1 legacy, Noll | 31.5789 | 31.5789 | 0.00568719 | 0.28529 | 0.140409 |
| A2 legacy, /16 | 31.5789 | **7.89473** | 0.0172174 | 0.86369 | 0.305711 |
| A3 spectral, Noll | 41.1348 | 41.1348 | 0.00391403 | 0.19634 | 0.102633 |

The A1/A2 identical covariance proves the difference is **only the
OTF Noll-kernel normalization**, not the underlying TT estimator. The
A2 31.6 mas label in the original graph was misleading: its displayed
blur corresponds to *one quarter* this RMS. P4 now writes both and
fixes panel-title line breaks, without changing PSFs. **Do not promote
A2 as a physically calibrated solution just because it resembles an
observed image.**

Legacy STRAP covariance trace:
`C_y=3.22775226685e-12 m²`,
`C_n=2.90472962723e-12 m²` (89.992% of measured),
`C_model=C_y-C_n=3.2302274804e-13 m²` (no PSD projection).
A 5% relative change in the assumed Cn trace corresponds to
`1.45236e-13 m²`, **45% of the inferred legacy physical
variance trace**. This is a local *error-propagation illustration*,
not a justified +/-5% experimental noise error bar. The actual
matrix eigenvalues must be checked before accepting each sensitivity.

The measured WFS PSD integrals have predicted filtered-noise
fractions x=89.2343%, y=87.4997%; negative
`PSD_measured-PSD_predicted_noise` in 40.856% and
38.521% of frequency bins, respectively. Finite Welch variance
can produce negative residual bins; do **not** clip them or
reinterpret the signed difference as a directly measured image
motion PSD.

Spectral physical covariance:
`trace=5.48096547791e-13 m²`, **1.697 x A1**,
comprising `4.02493883378e-13` atmospheric (73.43%)
and `1.45602664412e-13` propagated sensor noise (26.57%).
The A3 noise model propagates W/K/P and is explicitly
uncalibrated for Keck 2020. It being broader than A1
does not establish whether either is physically correct.

**Decision:** benchmark identifies extreme *noise-subtraction
sensitivity* plus a legitimate *16x kernel exponent* contrast.
It does NOT yet measure a real TT displacement independently, nor
quantify the binary PSF's true single-source core width. Do not
merge P4 as an instrument calibration.

### Next discriminating checks

1. Inspect raw `tt_noise_psd.png` and complete
   `tt_noise_diagnostics.json`: high-frequency plateau, band
   0.65–0.95 Nyquist, stationarity, detrending, Welch sampling.
2. Re-estimate van Dam noise with a **predeclared** set of bands
   (0.55–0.85, 0.65–0.95, 0.75–0.95) and Welch
   segment lengths 256/512/1024 **without choosing a value
   based on image agreement**. Report each covariance, negative
   spectrum fraction, physical RMS and any non-PSD failure.
3. Check STRAP optical centroid gain, APD normalization,
   real mirror P, WFS integration W and controller K on the
   n0013 2020 RTC; evaluate the sensitivity to unity vs
   independently calibrated Ragland mirror, using consistent
   rejection transfer in both noise and physical jitter.
4. On at least two more matched acquisitions, check sensor-noise
   stability and/or derive TT from independently calibrated
   TTM commands. Compare isolated-star or
   astrometrically fitted binary *with PSF fixed*; no free
   empirical TT amplitude.

## P4-D: photon-counting APD cross-check and factor-16 clarification

**Re-use existing PR7**, do not independently reinvent its noise
model: `p3/telemetry/ttApdNoise.py` and
`scripts/keck_strap_apd_audit.py` reconstruct the normalized
four-quadrant `APDCOUNTS` to `DTTCENTROIDS` through archived
`APD_IMAT` and `APD_SKY_BACK`.

The 2017 Keck telemetry reference explicitly labels
`DTTCENTROIDS` as **arcsec** and `APDCOUNTS` as detector
**counts**, not as verified photoelectrons. The n0013
centroid-reproduction RMS is ~1.49e-8/1.46e-8 arcsec
under the archived mapping. This proves numerical agreement
with the recorded transformation, **not absolute optical
centroid gain**.

The earlier PR7 n0013 reference comparison (all covariances
traces in 1e-12 m² Zernike RMS OPD):
- van Dam sensor-input: 2.905;
- Gendron temporal: 3.101 (signal-location caution);
- APD first-order ideal counting: 2.494;
- APD Poisson Monte Carlo: 2.547;
- observed centroid (loop-filtered): 3.228.

Mean APD raw counts ~446.5/frame and background-subtracted
net counts ~223.6/frame. Even after subtracting the mean
background, background photon shot noise remains. Ideal APD
noise is ~16.5% **lower** than van Dam, so under the same
loop transfer it predicts **larger**, not smaller, residual
atmospheric TT. This cannot explain away the broad A1 PSF
unless the real APD excess noise/counter gain or instrument
model differs from the ideal Poisson hypothesis.

For a quantitative comparison, propagate per-frame count
variances through
`q_j = sum_i A_{ji} (C_i-B_i)/sum_i (C_i-B_i)` and
`y=Mq`. Its Jacobian is
`dq_j/dC_i=(A_{ji}-q_j)/F`, so
`C_y=J diag(Var(C_i))J.T`. Parameters
`effective_photoelectrons_per_count`,
`excess_noise_factor`, read/dark and sky-estimate noise
must be measured **independently**, not fitted to the
science PSF. Compare APD and van Dam at the same sensor
input location; filter both through `S(f)` before
comparing to measured closed-loop centroid covariance.

The correct TT kernel can be written with **only angular
covariance** in the NIRC2 image frame:
`OTF_TT(f_ang)=exp[-2*pi²*f_ang.T*C_theta*f_ang]`.
The archived centroid is a measured angular offset;
no Zernike conversion is physically required to construct
this OTF. PR2's apparent factor 16 arose only because
the *intermediate* covariance was converted to Noll RMS
OPD via `a=D*theta/4`, so the inverse `theta=4*a/D`
must be applied in the normalized-baseline OTF.
Omitting this inverse creates an effective four-times
smaller angle RMS. It does NOT prove that STRAP on-sky
noise is 4x the recorded centroid motion.
P4 regression test `test_apd_archived_angular_centroids_reproduce_same_gaussian_otf_without_extra_factor`
checks both pathways.

**Sensitivity, not a calibration proposal:** holding
the current legacy `trace(C_y)=3.227752e-12` fixed,
to reduce `trace(C_y-C_n)` by 16 would require
`trace(C_n)` ~3.20756e-12, an increase of ~10.43%
over the van Dam estimate 2.90473e-12. This illustrates
why APD gain/excess-noise uncertainty can dominate the
inferred jitter. One must not tune the noise floor to
achieve the A2 image.

Re-run PR7's independent check with the **same n0013** pair:
```bash
uv run --with pytest python -m pytest -q \
  tests/test_keck_strap_apd_noise.py tests/test_keck_p4_tt_frame.py
uv run python scripts/keck_strap_apd_audit.py \
  --fits "/path/to/n0013.fits" \
  --sav "/path/to/n0013_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --mirror-model unity \
  --output results/n0013_P4_APD
```
Outputs: `strap_apd_noise_audit.json`,
`strap_apd_noise_comparison.png` and
`strap_apd_noise_report.md`. Repeat with the same
acquisition and `--mirror-model ragland_2016_keck_ii`
in a separate folder as a sensitivity test of the RTC
plant model, **not** validation of 2016 coefficients
for 2020.

Acceptance: direct angular and Noll OPD implementations
agree numerically, raw APD count-to-centroid transformation
reproduces archived centroids, photon noise estimates include
explicit detector calibration status, and spectral noise
comparison uses the same W/K/P/S. No absolute physical
jitter is certified without independent gain/transfer
calibrations.

## P4 numerical chain consolidated — source-to-OTF gate

The new source-to-OTF implementation and acceptance criteria are
documented separately in
`docs/keck_p4_tt_angular_otf_numeric_contract.md`.

It independently repeats the sensor-noise estimate directly on
`DTTCENTROIDS` in arcsec², compares to the historic Noll OPD² route,
then independently reconstructs the legacy and spectral physical
covariance in rad². It checks the pixel-scale Fourier grid and the
analytic Fourier transform of Gaussian image motion, and records
`p4_tt_angular_chain.json` and `p4_otf_tt.npy`.
A unit mismatch is an error, not a warning.

**Important:** old n0013 P4 output folders do not contain these fields
and are NOT retroactively validated by `--summarize-only`.
Re-run the four ablations after updating the P4 branch to obtain a
new `tt_angular_replay_verified_for_all_enabled_variants=true`
status. A0 has no TT kernel by design.

This changes neither the identified noise nor the nominal Gaussian
jitter covariance; it is a numerical proof/consistency check.
Do not claim absolute Keck calibration: quad-cell gain, detector
photon statistics, RTC W/K/P, STRAP frame/parity and science
path vibrations remain open.
