# Keck STRAP APD detector-noise calibration: independent physics audit (PR #7)

## Scientific contract

**Do not calibrate AO reconstruction against the NIRC2 science PSF.**
The primary result is a sensor-input noise covariance with:
- explicit signal location (four APD counters / STRAP centroid / science TT);
- units (counter units², arcsec² and RMS-Zernike OPD m²);
- sourced equations or explicit algebraic derivation;
- independently verifiable calibration dependencies;
- synthetic regression tests and honest validation flags.

This PR adds an independent conditional noise model, **not** a replacement
for van Dam (the default STRAP noise estimator). It is stacked on PR #6
(which itself depends on PR #5). The baseline is *not calibrated* for
Keck 2020 merely because an analytic variance is available.

## 1. Quad-cell reconstruction and units

The Keck STRAP detector uses four avalanche photodiodes arranged as a
quad cell (van Dam et al. 2006, PASP 118, §2.4; Looze, van Dam and
Johansson 2006, SPIE 6272, §2).

Let C_i be the observed raw counts and B_i the recorded constant sky mean.
Let z_i = C_i-B_i, F = Σ_i z_i, and the two signed contrasts:

    A = [[+1,+1,-1,-1], [+1,-1,-1,+1]]
    q_j = Σ_i A_ji z_i / F

The 2020 `n0013` SAV stores `APD_IMAT=[a,b,c,d]`. The matrix that
reproduces archived `DTTCENTROIDS` is:

    M = [[-b,a],[-d,c]]
    y_arcsec = M q.

**The layout of M is empirically identified from the archived n0013
telemetry, not a universal Keck instrument specification.** It must be
re-checked for each new acquisition. For this dataset its reproduction
RMS is 1.49e-8 and 1.46e-8 arcsec on X/Y.

The physically justified Keck telescope Zernike-angle conversion is
`a_rms_OPD = D*theta/4`; this is used consistently to transform
centroid noise covariance from arcsec² to m² without a fitted
coefficient (see `tipTiltGeometry.py`).

Crucially, **reproducing the recorded centroid does not calibrate the
optical centroid gain**: this depends on the guide-star spot size,
quad-cell optical throughput, acquisition geometry, etc. See
van Dam et al. (2006) §2.4 on calibration and gain uncertainty.

## 2. Conditional photon-counting variance (delta method)

The Jacobian of the normalized contrast is derived directly:

    ∂q_j / ∂C_i = (A_ji-q_j)/F

With `J = M ∂q/∂C` and conditional detector count covariance `C_C`:

    C_y(frame) ≈ J C_C J^T               [arcsec²]
    C_y,mean = average_over_frames C_y(frame).

For independently recorded detector channels with calibrated
`p` effective photoelectrons/counter-unit, excess-noise
factor `F_excess`, readout and sky-estimator variances:

    Var(C_i) = F_excess * C_i/p
               + sigma_read,i² + sigma_sky_estimate,i²

This is a **specific hypothesis for detector gain and APD variance**,
not an equation quoted directly from van Dam, nor a verified Keck
calibration. `C_i` contains gross counts (source plus background),
because the recorded photon noise does not vanish when subtracting
an estimated constant sky mean.

The independently usable defaults in
`config/keck_tt_noise.toml` -> `[strap_apd_reference]` are
`p=1, F_excess=1, read=0, sky_estimate_variance=0`. They are explicitly
marked `calibration_status="unverified_hypothesis"`. These are ideal
Poisson limits, **not measured Keck 2020 parameters**.

Independently measured photon-counter gain, excess-noise, dark-current,
readout and background-estimator statistics must be provided before
one can treat these absolute APD noise amplitudes as calibrated.

## 3. Monte-Carlo denominator propagation

For every observed time frame, draw independent Poisson raw detector
counts conditional on the observed mean rate, reconstruct the
nonlinear normalized quad-cell centroid, compute its sample covariance
across trials, then average covariances across frames. Simulation
options (seed/trials) are fixed for reproducibility.

This is a *pure Poisson* Monte Carlo. It does NOT model avalanche
multiplication distributions, read noise or covariance of the
sky estimate; such missing effects cannot be recovered from
`APD_IMAT` alone.

In the archived `n0013` sample, the pure Poisson Monte-Carlo trace
was ~2.1% above the first-order Jacobian baseline (128 trials per
frame). This figure is a numerical sensitivity check, not a gain
calibration. For very low flux, the code flags >10% invalid
realizations with non-positive background-subtracted denominators
rather than silently asserting an accurate noise covariance.

The approach is motivated by van Dam et al. (2006), §2.4, who
explicitly note that fluctuations of the normalized centroid
denominator make simplified analytic photon-noise estimates less
accurate for faint TT guide stars.

## 4. Compare to van Dam and Gendron at the correct signal location

Van Dam and the APD models report **sensor-input** noise covariance
`C_n` in RMS Zernike OPD units. With the controller model:

    W = sensor response,  K = controller,  P = plant,
    S = [1+WPK]^(-1)

a sensor-input noise `C_n` produces a measured-centroid contribution

    C_noise,measured = g_S*C_n,
    g_S = (2/fs) ∫_0^(fs/2) |S(f)|² df.

Compare `C_y-g_S C_n` only as a contribution in **measured
centroids**, NOT as physical on-sky image motion.

Gendron's temporal autocorrelation estimate is included as an
*alternative empirical diagnostic*, but its signal location may not
be identical to white sensor-input `C_n` when applied naively in
closed loop. It must not be treated as interchangeable simply because
its matrix has the same dimensions.

**New critical coherence fix**: `systemDiagnosis` accepts
`tt_mirror_model` so van Dam uses the correct rejection `S(f)`
for either the pure-delay or the optional Ragland 2016 actuator.
`scripts/keck_n0013_smoke.py` passes the same model to van Dam and
to the spectral PSF-R jitter propagation. The chosen model is
recorded in `noise_estimation_diagnostics` and
`tiptilt_diagnostics`. This removes an internal *model-choice*
inconsistency; it does **not** validate Ragland's mirror coefficients
for 2020 or the optical centroid gain.

The legacy PSF mode keeps the pure-delay model by default.

## 5. Running the diagnostic

From the PR #7 checkout:

```bash
uv run --with pytest python -m pytest -q \
  tests/test_keck_strap_apd_noise.py \
  tests/test_psfr_spectral_tt_kernel.py \
  tests/test_tt_closed_loop_formalism.py

uv run python scripts/keck_strap_apd_audit.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
  --mirror-model unity \
  --output "$PWD/results/n0013_apd_audit"
```

To compare with the published Keck II mirror response, repeat with
`--mirror-model ragland_2016_keck_ii` and a DIFFERENT output folder.
The noise estimator now fits the same rejection model that is used for
the physical jitter. This remains a **model sensitivity experiment**.

Optional detector assumptions can be configured in TOML or explicitly
overridden by
`--photoelectrons-per-count`,
`--apd-excess-noise-factor`,
`--read-noise-variance-counts2` and
`--sky-estimation-variance-counts2`.
Changing these is a sensitivity experiment, **not a fitting procedure**.

Outputs:
- `strap_apd_noise_audit.json`: centroid reproduction, detector
  assumptions, Poisson delta/MC, van Dam/Gendron, covariance comparisons;
- `strap_apd_noise_comparison.png`: per-axis sensor-noise variance;
- `strap_apd_noise_report.md`: tabular scientific summary.

No NIRC2 image intensity, source separation or Strehl is used to select
detector parameters.

## 6. Acceptance tests and remaining unknowns

- Strict independent reproduction of `DTTCENTROIDS` from APD channels;
- Jacobian verified by finite differences;
- Poisson analytic and Monte-Carlo agreement in the high-flux limit;
- explicit sensitivity to gain, avalanche excess and read/sky variance;
- nonpositive normalization denominators and invalid calibrations rejected;
- van Dam estimation uses the *selected* mirror's rejection transfer;
- no silent promotion of APD noise to default calibration.

**Still unknown / future work**:
instrumentally verified 2020 APD count-to-photon gain, avalanche excess
noise, background readout statistics, optical centroid gain, TT command
units and non-common-path guide/science vibration terms. These are
necessary for quantitative confidence, but lack of them should be
documented, not hidden by forcing a fit to LP415-20.

## Primary references

- van Dam et al., PASP 118 (2006), 310–318, §2.4 and Eq. (4–5),
  Keck LGS AO performance, quad-cell TT sensor, controller,
  PSD and Monte-Carlo:
  https://doi.org/10.1086/499498
  https://www2.keck.hawaii.edu/optics/aodocs/PASP06_MvD.pdf
- Looze, van Dam & Johansson (2006), SPIE 6272 62724L, §2,
  normalized four-APD quad-cell centroid:
  https://doi.org/10.1117/12.672427
- Ragland et al. (2016), Proc. SPIE 9909, 99091P, §5.2,
  experimentally identified Keck II TT mirror model:
  https://arodes.hes-so.ch/record/8385/files/Published%20version.pdf
- Core S/W/K/P formalism, source locations and tests:
  `docs/tt-closed-loop-theoretical-contract.md`.
