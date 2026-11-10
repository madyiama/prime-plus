# P3 — Keck/NIRC2 pupil geometry and orientation audit

## Scope and scientific priority

P3 begins after the P2 physics baseline (PR #8) and the P1 r0
diagnostic branch (PR #16). **Do not replace or retune the historical
r0/L0 estimator in P3.** Final r0 calibration (RTC sign/latency,
DM OPD gain, modal selection, noise and multi-acquisition DIMM
comparison) is intentionally deferred until the end of the
reconstruction project.

P3 targets issue #14: propagate and check the telescope pupil
geometry in the NIRC2 science-camera frame independently of AO
turbulence, jitter, NCPA and focal-plane fitting.

## Source-backed geometry

- KAON 1160, Ragland (2017): NIRC2-pupil angle = `EL - ROTPPOSN`;
  WFS-pupil angle = `ROTPPOSN - EL`. Their rotation senses
  are opposite and their reference shapes differ by 90 degrees.
  `ROTPOSN - INSTANGL` is sky PA, not pupil PA.
- `Keck_Pupil_shape.pdf`: actual M1 maximal diameter 10.949 m;
  central obscuration diameter 2.600 m; inside circle 9.018 m;
  equivalent-area circle diameter 9.822 m; DM controlled aperture
  11.250 m. NIRC2 LargeHex outer diameter 10.656 m; Incircle
  outer diameter 8.720 m. These are *different definitions* and
  cannot silently inherit `tel.D=10.5m` as a calibrated aperture.
- For n0120 header: `EL=47.40723`, `ROTPPOSN=0.64000`,
  `ROTPOSN=INSTANGL=0.70000`. This yields NIRC2 angle
  +46.76723 deg and WFS angle -46.76723 deg while sky PA is zero.

**Unknowns**: absolute zero of the stored FITS mask, array-axis
handedness relative to the camera frame, PMSANGL/PMRANGL
mechanical-to-optical rotation, FITS mask alignment, pupil drift,
and NCPA phase frame. KAON 1160 does not calibrate them. Do not
assign +90 deg as an automatically validated additional correction.

## This initial P3 PR

- Adds a read-only audit `p3/telemetry/keckPupilAudit.py`.
- Separates NIRC2, WFS and sky PA, and records raw mechanical
  `PMSANGL` / `PMRANGL` without inventing a calibrated
  mask angle.
- Distinguishes archived OPEN and LARGEHEX FITS masks.
  INCIRCLE is reported explicitly unavailable instead of replaced
  by OPEN without notice.
- Produces binary nearest-neighbour pupil masks and normalized,
  diffraction-only Fourier intensity maps for five *diagnostic*
  angles: archived 0, KAON nominal, nominal +/-90 and reverse sign.
  This exploration is not wired into the production telescope
  or PSF-R model and does not choose the best angle based on image
  residuals.
- Includes synthetic regression tests for frame convention,
  mask resolution and mathematical diffraction invariants.

## Local verification

```bash
git switch feat/psfr-p3-pupil-orientation-audit
uv run --with pytest python -m pytest -q tests/test_keck_p3_pupil_audit.py
uv run python scripts/keck_p3_pupil_audit.py \
  --fits "/path/to/n0120.fits" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --output "results/n0120_P3_pupil" \
  --resolution 256 --oversample 2
```

Outputs:
- `p3_pupil_orientation_audit.json` containing angle provenance and
  explicit uncalibrated fields;
- `p3_pupil_diffraction_hypotheses.png`;
- five binary pupil arrays and five unit-flux diffraction arrays.

## Next P3 validation gate (not claimed complete)

1. Inspect the archived NIRC2 pupil FITS orientation versus the
   reference zero and the 90-degree WFS/camera distinction; use
   segment/spider asymmetry, not a symmetric hexagon alone.
2. Compare *observed* NIRC2 diffraction directions on at least
   2-3 independent high-SNR, unsaturated and correctly classified
   science images, after accounting for pixel scale and camera
   orientation. Derive/record the sign and reference-angle mapping
   rather than visually selecting a favorite panel.
3. Only then wire a calibrated rotation into the telescope/OTF
   creation path. Preserve a legacy switch and test the full
   diffraction PSF, not merely the header value.
4. Separately inventory the mechanical focus keywords for
   NGS phase-diversity sequences; do not convert offsets into
   nm of Z4 without an instrument sensitivity calibration.
5. Compare the aperture areas and pupil OTF for OPEN,
   LARGEHEX, and INCIRCLE where actual calibration exists.
   Keep NCPA disabled until the phase map is matched in frame,
   reference epoch and units.

Do not optimize a rotation, defocus or turbulence parameter merely
to improve visual agreement with n0120.

## P3 optical-model integration — bounded binary mask (second increment)

The first P3 script was a **read-only orientation/diffraction audit**.
The second increment also wires a strictly opt-in **production-optical
pupil transform** into `telescope` (not into r0, L0 or the PSF fitting):

- `PupilResampling='legacy'`: unchanged generic INI default;
  original scipy spline rotation + linear resizing retained as baseline.
- `PupilResampling='physical_binary'`: archived binary FITS mask is
  rotated and area-integrated on an oversampled output grid
  (4 subpixels per axis); complex-pupil amplitude remains in [0,1],
  without arbitrary cubic overshoot or post-hoc clipping.
  Fractional edge amplitude represents area occupancy within a
  computational pixel. The raw FITS is required to be 0/1, otherwise
  an explicit error is emitted.
- A complete mask inventory from `PMSNAME` is mandatory in this
  corrected mode; unknown masks or missing `INCIRCLE` data fail
  instead of silently substituting the archived OPEN mask.
- Diagnostic pupil script now uses the **same transform** as the
  optional optical model, so the plotted diffraction hypotheses
  correspond to the telescope pupil transform.
- The experimental rotation sensitivity flag is
  `--p3-pupil-rotation {header,as_archived,reverse_sign,plus_90,minus_90}`.
  The default `header` uses the KAON 1160 angle already in the
  telemetry. No rotation is automatically calibrated. The camera
  angle selected for model construction is recorded in the metadata;
  the TT kernel also depends on `trs.tel.pupilAngle`, so isolate
  pupil effects with `--disable-tt-jitter` when comparing alternatives.

### Tomorrow's ordered protocol

Use the actual n0120 science FITS / SAV. Each output must be
in its own directory; never overwrite a baseline:

```bash
git fetch origin
git switch feat/psfr-p3-pupil-orientation-audit
git pull --ff-only
uv run --with pytest python -m pytest -q \
  tests/test_keck_p3_pupil_audit.py \
  tests/test_keck_p3_pupil_sampling.py

# Cheap, observationally independent diffraction hypotheses
uv run python scripts/keck_p3_pupil_audit.py \
  --fits "/path/to/n0120.fits" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --output results/n0120_P3_diffraction

# Historical pupil/rotation baseline with TT ablated
uv run python scripts/keck_n0013_smoke.py \
  --fits "/path/to/n0120.fits" \
  --sav "/path/to/n0120_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --output results/n0120_P3_legacy \
  --n-layer 7 --disable-tt-jitter \
  --p3-pupil-resampling legacy --p3-pupil-rotation header

# Same physical/geometric inputs, bounded pupil interpolation only
uv run python scripts/keck_n0013_smoke.py \
  --fits "/path/to/n0120.fits" \
  --sav "/path/to/n0120_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --output results/n0120_P3_bounded \
  --n-layer 7 --disable-tt-jitter \
  --p3-pupil-resampling physical_binary --p3-pupil-rotation header
```

Compare the files `p3_model_pupil_diagnostics.json`,
`metadata.json`, `comparison_nirc2_psfr.png`,
`diffraction_otf.npy` and `telemetry_only_psf.npy`
when present. Verify that both runs share exactly the same
r0/L0, Cn2 profile, TT ablation, DM, wavelength bins and
calibration. Compare pupil effective area and diffraction
spike directions, not merely the visually sharpest image.
Only after that assess alternate angle hypotheses; no
automatic 'best' is selected. Do not compare diffraction
patterns for different masks as proof of a rotation sign.

### Remaining physical calibration gate

**The mask zero-angle, parity, physical pixels-to-metres scale
and NIRC2 optical path/moving mask angle have not been
independently verified.** KAON 1160 supplies a *rotation
difference convention*, not an absolute FITS pupil registration.
The pupil binary remapping is numerically motivated, but
not an on-sky absolute orientation calibration. Use
multi-acquisition diffraction directions to decide the
proper image-array and telescope physical transform.

P3 branch regression tests for this increment:
`tests/test_keck_p3_pupil_audit.py`,
`tests/test_keck_p3_pupil_sampling.py`.
The full project test suite and the on-sky n0120 runs
have not been executed in this environment.
