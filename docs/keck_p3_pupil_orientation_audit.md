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
