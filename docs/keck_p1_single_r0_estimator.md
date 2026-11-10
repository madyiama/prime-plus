# P1 — Single Keck r0 estimator API, two explicit physical methods

**Branch:** feat/psfr-p1-r0-timing-audit; draft PR #16 only.
**Status:** experimental Jolissaint adaptation added; **no on-sky validation
yet** and no replacement of the production Zernike estimates.

## User-supplied sources studied

- \`estimateSeeing.m\` — a mixed MATLAB/IDL sketch using the 9m DM
  and old external Noll/von Karman lookup tables. It contains a
  literal syntax error (\`L0s(mn])\`) and undefined \`phasemask\`,
  \`zernrec\`, \`vmodes_ph\`, \`var_vk\`, \`L0s\`.
  It is NOT a standalone reproducible Jolissaint implementation.
- \`scriptTestsSeeingMonitor.m\` — simulation of true versus WFS
  reconstructed phase, including a separate WFS aliasing experiment.
- \`testGetr0L0.m\` — test of a slope-based PSF-R \`getr0L0\`
  function for a simulated Shack-Hartmann; not the same input as the
  2018 Keck DM-command seeing estimator.
- \`seeingMonitor.zip\` — **real Jolissaint IDL code** (2019 archive):
  \`seeing.pro\`, \`get_seeing.pro\`, \`dm_noise_covmat.pro\`,
  \`actuator_to_zernike_matrices.pro\`,
  \`appzernike_coefficients_covariance_annular_pupil_montecarlo.pro\`,
  \`loop_transfer_functions.pro\`, \`attos.pro\`, helpers.

The **precomputed IDL SAV calibration matrices are not included**
(\`matZI_7_171_23.sav\`, \`matZZ_7_171.sav\` and the
165x165 Kolmogorov covariance file). Consequently a binary-identical
IDL reproduction is impossible from those uploads alone.

## ONE Python entrypoint

\`p3.telemetry.seeingEstimator.estimate_r0(diagnosis, method=...)\`

- \`method="zernike"\`: returns the original \`systemDiagnosis\` r0/L0
  untouched, including the old pseudo-OL and DM-lag-noise assumptions.
- \`method="jolissaint"\`: new **experimental** reconstruction of the
  published Keck **closed-loop DM-command** method. Does not use the
  science PSF, a DIMM fit or a pseudo-OL wavefront.

Both return \`r0_m\`, \`L0_m\`, reference wavelength and provenance;
Jolissaint also returns the L0 fit profile, individual Noll modal
variances, spectral-noise estimates, mode selection, and warnings.

### What the experimental Jolissaint branch implements

1. A physically sampled **9.00m / 2.65m annular pupil** on the actual
   telescope-coordinate grid; 0.5625m Keck actuator pitch.
2. Keck DM influence-function matrix from the live calibrated
   \`telemetryKeck\` instance; selection via actuator influence peaks
   within the nominal 2018 IDL DM region.
3. Build **165 modified Zernikes** Noll **7–171** by projecting the
   ideal annular polynomials into the Keck DM influence-function
   basis, then reconstruct coefficients with the modified mode Gram.
   Explicit per-mode piston-gauge removal ensures a finite
   Kolmogorov phase covariance. No LGS piston/tip-tilt.
4. Project **closed-loop DMCOMMAND** (in **OPD metres** as provided
   by \`telemetryKeck\`) into these modes, converting to phase radians
   at the WFS reference wavelength only once.
5. PSD of each DM-modified mode via Welch; deconvolve by the existing
   closed-loop transfer \`H_CL/H_WFS\` to estimate a high-frequency
   white-noise plateau and reconvolve \`|H_CL/H_WFS|²\` to obtain
   a **command-side modal noise covariance diagonal**.
   The exact 2018 IDL used full actuator covariance incl. cross terms;
   linear modal projection before cross-spectrum is equivalent for
   the diagonal but Welch/window/median-estimator choices differ.
   The HF band remains **experimentally chosen**, not calibrated on
   n0120. \`noise="none"\` is an explicit diagnostic ablation.
6. The original IDL Monte Carlo theory calibration is replaced with
   a **deterministic physical phase-structure-function quadrature**:
   project \`Dphi(rho;r0=D,L0)\` with the same modified-Zernike
   phase functionals, using FFT linear convolution (no spatial wrap).
   At L0=infinity \`Dphi=6.88(|rho|/D)^(5/3)\`; finite L0 uses
   the analytic McDonald von Karman phase structure.
7. Select the 2018 LGS Noll mask of **75 usable modes from 7–120**
   (\`mode_selection="idl_2018"\`, default). This mask was developed
   for the ORIGINAL Keck telemetry and is NOT automatically certified
   on n0120. \`mode_selection="all"\` uses all positive-variance
   modes in Noll 7..120.
8. Fit \`r0,L0\` from *relative* log modal variances, not the
   low-order-dominated raw absolute residual. L0 is profiled on
   a documented finite + infinity grid; **do not claim a confidence
   interval or an independently identified L0** from this alone.
9. The published IDL **empirical WFS aliasing polynomial** may be
   enabled explicitly via \`alias_correction=True\` but is OFF by
   default. It has not been validated for 2020 NIRC2 telemetry.

**This is a physics-matched adaptation, not a claim that all 2018
Keck calibration factors are already verified on our observation.**
Output sets \`r0_is_validated=False\` and never alters \`sd.trs.atm\`.

## User command — single n0120 run, both estimators

From the same P1 branch and with the same FITS/SAV/calibration path as
the existing successful smoke:

\`\`\`bash
git switch feat/psfr-p1-r0-timing-audit
git pull --ff-only
uv run --with pytest python -m pytest -q tests/test_keck_p1_unified_estimator.py
uv run python scripts/keck_n0013_smoke.py \
  --fits "/path/to/existing/n0120.fits" \
  --sav "/path/to/existing/n0120_LGS_trs.sav" \
  --calib "/path/to/existing/keck_calibration" \
  --output "$PWD/results/n0120_P1_compare_estimators" \
  --n-layer 7 --p1-only --r0-method both
\`\`\`

Output: \`p1_r0_estimator_comparison.json\`, plus the existing P1-B
noise and radial variance audit files. No science-image PSF computed.
Only ONE \`systemDiagnosis\` run; Jolissaint then analyzes the same
loaded Keck commands independently.

For a strict noise control, repeat in a DIFFERENT output directory
with \`--r0-method jolissaint --jolissaint-noise none\`.
For the old published but **unvalidated-on-n0120** WFS alias model,
add \`--jolissaint-alias\` separately.

## n0120 first on-sky result and critical geometry correction

**First execution**: legacy \`r0=0.1147404m, L0=31.0457m\`
at 500nm LOS; Jolissaint prototype \`r0=0.0998327m, L0=15m\`,
75 selected LGS Noll modes, full 165-mode constructed basis,
modified Gram condition 65.77, 205 Welch high-frequency bins.
The selected-mode spectral noise subtracted was **0.07394%**
of their summed measured variance (median per-mode 0.139%,
maximum per-mode ~0.39%). It cannot explain the ~13% r0
difference by itself. The log variance profile cost at L0=15m was
0.01144 (rms log residual ~10.7%), versus 0.01490 at 22m and
0.02123 at 31m. This profile has **no validated confidence interval**.

**DO NOT treat the first prototype estimate as a calibrated result**:
three implementation issues identified after reviewing this output
and the original IDL and corrected within this same PR:

1. **367 DM channels instead of precisely 320**. The old Python
   code estimated actuator selection from the spatial peaks of
   overlapping influence functions, an invalid coordinate proxy.
   Original \`actuator_to_zernike_matrices.pro\` and
   \`get_seeing.pro\` define the nominal 21x21 Keck grid and select
   \`1.2*pitch < r < 10.05*pitch\`, corresponding to **320
   of the 349 physically valid channels**. Corrected selection
   uses nominal actuator coordinates and the loaded
   \`trs.dm.validActuators\` mask; it **rejects** 21x21 cases
   without exactly 320 selected actuators.
2. **Wrong spatial sampling of DM influence functions**.
   \`deformableMirror.setInfluenceFunction(resolution)\` samples
   a physical **11.25m** grid for the 21x21 DM at 0.5625m pitch,
   whereas the prototype mistakenly sampled the phase model
   and annulus on a 10.5m grid. Corrected pupil coordinates
   span +/-5.625m and the covariance kernel uses physical
   pixel separation \`11.25/(resolution-1)\` throughout.
   For resolution=100, the actual sampling is **4.95
   pixels per DM pitch**; this is **below the IDL paper's
   recommended >=13 pixels/pitch**, so convergence to the
   original method remains outstanding.
3. **Log-amplitude fit criterion inconsistent**: the prototype
   previously used the median log ratio but minimized the
   *mean squared* log residual. Corrected to the **mean** log
   ratio, the analytic optimum for the stated loss.

After these corrections the first n0120 Jolissaint r0/L0 values
are **obsolete and must be recomputed**. The updated JSON now
records \`actuator_count\` (expected 320), \`dm_influence_extent_m\`,
\`pixels_per_dm_pitch\`, selected-mode noise fractions, and
per-selected-Noll measured/theoretical variance ratios.

No production estimator change, no new PR. Do not compare to DIMM
as an equality target; seeing from the on-telescope AO path includes
dome seeing and may differ from external monitors.

## Validation gates before production substitution

1. Synthetic phase/noise tests, covariance positivity, reference
   wavelength conversion, flat spectral-noise recovery and modal
   exclusion pass on macOS.
2. On-sky Keck run completes with plausible 9m annulus,
   actuator count, Gram condition, Noll selection and PSD plateau.
   There must be no sign/OPD-unit ambiguity.
3. Compare results with and without modal noise subtraction and the
   old alias polynomial, and inspect L0 profile. Flat/boundary L0
   profiles mean **L0 unconstrained**, not physically measured.
4. Verify the 2018 calibration matrices/PSD normalization against
   original IDL outputs if the missing SAV files become available.
5. Multiple independent observations with contemporaneous
   seeing/dome metadata; DIMM is NOT an equality calibration.
6. Never count LGS atmospheric TT twice; this estimator starts at
   Noll 7 and uses no TT for L0.

DO NOT merge a change to the active PSF-R seeing estimator solely
because the new r0 agrees better with DIMM or a fitted science PSF.
