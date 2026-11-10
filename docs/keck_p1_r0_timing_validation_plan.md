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
