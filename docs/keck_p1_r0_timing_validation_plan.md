# Keck P1 — pseudo-open-loop timing and atmospheric r0/L0 validation

**Status:** started after freezing P2 golden reference 2026-11-10.
**No production estimator or timing convention changed in this PR.**

Related issue: [#13](https://github.com/madyiama/prime-plus/issues/13).
Dependency: the P2 reference baseline
\`docs/keck_p2_frozen_reference_2026-11-10.md\`.

## Scientific question

Does the WFS reconstructed *pseudo-open-loop* modal covariance correctly
identify \`r0\` and \`L0\` at the science line of sight, independently of
the later NIRC2 PSF, given the RTC latency and sensor noise model?

Avoid using an observed fitted Strehl as the r0 calibration target.

### Already corrected — do not regress

- Multi-actuator \`np.roll\` without axis was removed. Only temporal
  interpolation, no wrapping, with \`interpolate_forward_residual_no_wrap\`.
- Historic sign convention is \`u_ol(t)=u_dm(t)+res(t+delay)\`; no external
  timestamp proof of that sign/offset exists.
- WFS \`r0\` is already line-of-sight; the generated INI declares
  \`SeeingReference='line_of_sight'\`, preventing an additional
  \`airmass^(-3/5)\` conversion.
- Seeing must obey \`epsilon=0.976*lambda/r0\`, and
  \`r0(lambda)=r0(lambda0)*(lambda/lambda0)^(6/5)\`.
- Joint \`r0,L0\` fitting and its uncertainties are exposed in
  \`atmosphere_reconstruction_audit.json\` in each completed run.

### P1-A: archive/provenance audit (this draft PR)

Run \`scripts/keck_p1_r0_baseline_audit.py\` on existing smoke output
folders. For every case, report:

1. WFS r0 at the *atmosphere* wavelength and LOS seeing conversion;
2. forward model LOS r0 and wavelength-scaling identity;
3. pseudo-open-loop declared latency/frame exclusion/sign verification;
4. joint r0/L0 success, active bounds, uncertainties and fit cost;
5. **both** independent DIMM comparison scenarios:
   raw direct monitor seeing vs WFS LOS, and
   DIMM assumed zenith-normalized then adjusted to science airmass.

Site DIMM reference definition is UNKNOWN; do not choose the version
closer to WFS r0 and call it validated. A 1-layer and 7-layer result of
the same Keck image is not a pair of independent measurements.

### P1-B: direct modal/noise subtraction audit (current branch)

The present \`systemDiagnosis\` workflow **does not directly fit the
\`OFFSETCENTROID\` WFS slopes**. It fits DM actuator-space
pseudo-open-loop \`DMCOMMAND + shifted RESIDUALWAVEFRONT\` after
\`u2z = pinv(Z^T Z) Z^T IF_DM\`. The default Noll lower bound is
\`j0=4\`, excluding tip-tilt nominally. That does NOT prove zero
tip-tilt leakage through a finite-resolution, non-orthogonal DM basis.

Major unresolved issue: \`slopes_to_noise(self.trs.dm.com)\`
estimates a lag-one **closed-loop DM command difference covariance**,
with *periodic* end-to-start wrapping for LGS, then
\`var_noise_zer=diag(u2z C_DM,lag1 u2z^T)\` is subtracted from the
measured pseudo-open-loop Zernike variance. The command differences
have true atmospheric and closed-loop temporal contributions and are
not, by themselves, a physically independent WFS noise measurement.
This may bias both r0 and L0. The helper also demeans the input DM
commands IN PLACE. Current r0 fit sets \`noise=1\` by default,
subtracting the entire lag covariance proxy, and uses the **median**
of corrected modal variances for each radial order.

This P1-B opt-in audit does NOT change production noise, reconstructor
or timestamps. From live \`systemDiagnosis\` it saves:
- measured, lag-proxy and corrected Zernike variances, by Noll mode
  and by radial order, plus theoretical fitted variance and residuals;
- fraction of the signal attributed to the lag proxy and any
  nonpositive corrected modal variance;
- equality test between the original LGS periodic actuator lag
  estimator and its reconstruction in modal space;
- *non-calibrated* comparison of periodic DM, finite-window DM and
  finite-window residual lag-1 variance proxies; raw WFS centroid
  differences labelled in pixel² (never added to OPD² without
  reconstructing the sensor gain);
- fixed-fitted-L0 r0 variance-amplitude sensitivity under noise
  multipliers 0, 0.5, 1.0, 1.5 (not a full joint fit);
- Gram matrix conditioning and the fact that the projection currently
  uses an ideal circular/annular Zernike basis, not an instrument-adapted
  DM modal basis.

Run on the SAME \`--sav/--fits/--calib\` paths as an existing smoke:
\`\`\`bash
uv run python scripts/keck_n0013_smoke.py \
  --sav /path/to/n0120_LGS_trs.sav \
  --fits /path/to/n0120_science.fits \
  --calib /path/to/keck_calibration \
  --output results/n0120_P1_modal_noise \
  --n-layer 7 --p1-only
\`\`\`
The paths above are placeholders, **not verified as existing**.
\`--p1-only\` stops immediately after the atmospheric
reconstruction diagnostics and does not compute the PSF.
Outputs \`p1_modal_noise_audit.json\`, three CSVs and a radial
variance plot. Run tests with
\`uv run --with pytest python -m pytest -q
tests/test_keck_p1_modal_noise_audit.py\`.

**Physics-matched reference for next steps:** Jolissaint, Ragland,
Christou & Wizinowich (2018), *Determination of the optical
turbulence parameters from the adaptive optics telemetry: critical
analysis and on-sky validation*, Appl. Opt. 57(27), 7837,
DOI:10.1364/AO.57.007837. This is a Keck-specific DM-command-based
seeing method, not a generic WFS tutorial. It explicitly treats
(a) a **9m/2.56m annular fitting pupil** to exclude DM edge and
central obscuration artifacts, (b) Zernike modes projected into
the DM influence-function basis and their **modified turbulence
covariance**, (c) selecting good radial modes, (d) WFS noise
inferred from **DM temporal PSD and closed-loop transfer function**
with attenuation compensation, and (e) WFS aliasing plus its
correlated terms. Reported DM-derived seeing tracks independent
simultaneous open-loop PSFs within ~10% but is much less correlated
with non-collocated MASS/DIMM. Reimplement/test these corrections
in controlled variants, **not** by tuning the current DIMM ratio.

### Next instrumental/model alternatives

1. **Keck published DM-variance method (preferred baseline)**:
   matched 9m/2.56m annular pupil and DM-modified Zernike covariance,
   select calibrated modes, estimate noise through temporal PSD
   and AO transfer function, correct WFS aliasing.
2. **Pseudo-open-loop slope covariance**: build slopes from
   residual centroids + delayed DM interaction model and a correctly
   calibrated slope-to-wavefront reconstructor; subtract sensor noise
   in slope covariance, then propagate by the same modal projector.
   This is a cross-check, not an immediate replacement, because
   WFS optical gain, lenslet geometry and reconstructed slopes
   require Keck 2020 calibration.
3. **Temporal lagged covariance**: estimate turbulence from nonzero
   frame lags where white noise does not add a zero-lag spike,
   under explicit assumptions on frozen flow and temporal sampling.
4. **Fixed-L0 and profile-likelihood r0/L0 fits**: amplitude-only
   r0 fit at chosen L0, then a joint noise/r0/L0 sensitivity map
   with actual residual variances by radial order.

The 2018 Keck study and an independent 2003 Keck open-loop study
indicate that **WFS calibration and modal projection**, rather than
DIMM agreement, should control claims of absolute r0 accuracy.

### P1-B: *next* — RTC timing/sign sensitivity on raw Keck telemetry

Build a **read-only** replay of \`u_ol(t)\` for both time sign
conventions and delay offsets around recorded \`lat*freq\`, holding
identical valid frame intersections, actuator masks and noise models.
Compare per-mode and per-radial-order covariances, fit residuals,
r0/L0 estimates, uncertainty, and saturation at L0=100m.
Do **not** pick the sign or delay by whichever matches DIMM best:
seek timestamps, WFS integration window, DM command convention and RTC
latency documentation. If unavailable, retain both as explicit
conditional scenarios and leave the sign unvalidated.

Check whether \`get_noise_variance(...)\` modifies the input command
array in place, and whether that change is intentional before
pseudo-OL reconstruction. Test synthetic multi-actuator known-delay
telemetry so a column swap, circular wrap or sign inversion fails.

### P1-C: joint fit robustness and site comparison

- Save measured and noise-subtracted Zernike variances *by radial order*,
  together with the model at the fitted (r0,L0), not just summary cost.
- Profile the objective in r0 and L0 to diagnose degeneracy; flag
  active bounds and fit curvature/credible uncertainty.
- Quantify sensitivity to WFS noise model, actuator selection,
  high-order radial range and pseudo-OL time alignment.
- Compare multiple *independent* LGS/NGS observations with site DIMM
  timestamps and explicit zenith reduction convention. Anomalous
  records remain labelled; do not discard them solely on Strehl.

## Entry command

\`\`\`bash
git switch feat/psfr-p1-r0-timing-audit
git pull --ff-only
uv run --with pytest python -m pytest -q tests/test_keck_p1_r0_baseline_audit.py
uv run python scripts/keck_p1_r0_baseline_audit.py \
  --cases "$PWD/results/n0120_P2_1layers" "$PWD/results/n0120_P2_7layers" \
  --output "$PWD/results/n0120_P1_r0_baseline"
\`\`\`

Outputs: \`p1_r0_baseline_audit.json\` and \`.csv\`.
Flags are **diagnostic**. \`PSEUDO_OL_RTC_SIGN_UNVERIFIED\` is expected
until an independent physical timing reference exists.

## P1 completion gate

P1 can only be declared *instrumentally validated* when RTC
time sign/integration alignment has independent provenance,
pseudo-OL synthetic tests and reproducible raw replay pass,
radial Zernike-variance fit residual/uncertainties are visible,
and multi-observation comparisons do not depend on selecting a
favourable DIMM zenith convention or tuning the science PSF.
