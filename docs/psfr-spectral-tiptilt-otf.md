# PRIME+ — Optional closed-loop spectral tip-tilt PSF-R kernel (PR #6)

## Scientific objective

Connect the independently documented closed-loop PSD formalism in PR #5
to the Gaussian tip-tilt OTF in PRIME+, **without** fitting the observed
NIRC2 binary, forcing a target Strehl, or inventing instrument gains.

This PR is stacked on PR #5. Its changes must not merge until
`tests/test_tt_closed_loop_formalism.py` passes and the observation
diagnostics have been reviewed.

## Two modes, no automatic calibration

The configuration in `config/keck_tt_noise.toml` contains:

```toml
[tip_tilt_jitter]
method = "legacy_psfr"
mirror_model = "unity"
welch_nperseg = 512
```

- `legacy_psfr` remains the default for backward-compatible scientific
  regression. It uses P3's historical covariance algebra and the
  separately recorded PSD projection introduced earlier.
- `spectral_closed_loop` is opt-in. It computes physical atmospheric
  and sensor-noise contributions from the measured STRAP Welch
  cospectra, a chosen W/K/P controller/plant model, and the *same*
  calibrated sensor-noise covariance estimated by the configured
  van Dam/Gendron method.
- `unity` reproduces P3's pure-delay actuator assumption; the
  alternative `ragland_2016_keck_ii` applies the published
  experimentally identified Keck II mirror dynamic response, **not
  independently verified on the 2020 observation**.

The actual choice and source of the covariance are recorded in
`tiptilt_diagnostics.json`. `noise_model_validated=false` and
`command_units_verified=false` remain explicit.

## Equations and signal locations

The exact linear equations and units are derived in
`docs/tt-closed-loop-theoretical-contract.md`, §2–3.

Using `e=d-Pu`, `y=We+n`, `u=Ky`:

```
L = W P K,       S = 1/(1+L)
P_y = |W S|² P_d + |S|² P_n
P_e = |S|² P_d + |P K S|² P_n
P_e,atmos = (P_y - |S|² P_n) / |W|²
P_e,noise = |P K S|² P_n
C_e = integral_0^(fs/2) (P_e,atmos+P_e,noise) df
```

All input STRAP quantities are expressed in RMS Zernike OPD metres,
including `Cn` in m² and Welch PSDs in m²/Hz. The covariance `C_e`
is thus a physical residual covariance in m², not simply a sensor
centroid covariance.

The OTF kernel uses the PR #2 geometric relationship:

```
C_theta = (4/D)² C_e                           [rad²]
D_phi,TT = (2*pi*D/lambda)² [u,v] C_theta [u,v]^T
OTF_TT = exp(-D_phi,TT / 2)
```

When `C_e` is the covariance of the physical image-motion residual,
this is the Gaussian long-exposure OTF with the correct angular
convention, independent of photometric fitting.

`TT NOISE` in the WFE decomposition derives only from
`C_e,noise`; `TIP-TILT` derives only from `C_e,atmos`.
The total OTF uses **their sum once**; there is no double counting.

A negative eigenvalue inconsistent with roundoff in any covariance
component raises a `ValueError` instructing the user to review the
loop/noise model. Unlike the legacy path, the new spectral mode never
silently clips negative components to make the kernel positive.

## Sources and calibration caveats

- van Dam et al., PASP 118 (2006), §2.4: STRAP noise fitting and
  feedback error-budget terms.
  https://www2.keck.hawaii.edu/optics/aodocs/PASP06_MvD.pdf
- Ragland et al., Proc. SPIE 9909, 99091P (2016), §5.2:
  experimentally fitted Keck II TT mirror dynamics; applies only
  conditionally to 2020 STRAP data.
  https://arodes.hes-so.ch/record/8385/files/Published%20version.pdf
- AO controller PSD derivation and independent causal tests:
  `docs/tt-closed-loop-theoretical-contract.md`.
- Angular/Zernike/OTF conventions:
  `docs/keck-strap-tt-and-lgs-anisoplanatism.md`.

No assumption that Ragland is automatically more accurate is made.
Actual RTC program, centroid gain, command units and Keck mirror
parameters on 2020-12-02 are still subject to instrument validation.

## Reproduce on n0013

After checking out PR #6's branch and ensuring PR #5 tests pass:

```bash
uv run --with pytest python -m pytest -q \
  tests/test_tt_closed_loop_formalism.py \
  tests/test_psfr_spectral_tt_kernel.py

uv run python scripts/keck_n0013_smoke.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
  --tt-jitter-model spectral_closed_loop \
  --tt-mirror-model unity \
  --output "$PWD/results/n0013_psfr_tt_spectral_unity"
```

Repeat with `--tt-mirror-model ragland_2016_keck_ii` and a
**different output directory**, without modifying the atmospheric
profile, pixel OTF, NCPA, or any other PSF-R component.

The existing `comparison_nirc2_psfr.png` is illustrative only:
LP415-20 is a binary, there is no binary-source fit, and the
candidate PSFs should **not** be ranked by closeness to that image.

Expected reference physical covariance traces from the two PR #5 audits:
`5.495599e-13 m²` (pure-delay unity) and `4.151848e-13 m²`
(Ragland). The PR #6 direct TT covariance should reproduce these
within the precision of the same telemetry, PSD grid and noise method.

## Acceptance gates

1. New numerical tests pass, including a known synthetic controller
   with a known sensor noise and disturbance.
2. Legacy PSF kernel remains unchanged.
3. New mode reports an identical physical covariance to PR #5's
   independent audit, with no extra noise component or hidden scaling.
4. No physical covariance projection, no fitting to NIRC2 observations.
5. Do not merge into main until PR #5 has completed review and the
   new live Keck results are inspected.


## PR #6 validation extension — PSF blur isolation and model consistency

For the Keck LP415-20 binary, both the historical and the spectral PSF
kernels may produce visibly broad cores; visual agreement cannot validate
an individual star PSF without a two-source forward model.

An optional **zero-TT ablation**, strictly for model diagnosis, is
available with `--disable-tt-jitter`. It does not modify source data,
fit the PSF, or change fitting, aliasing, LGS focal anisoplanatism,
pixel OTF, atmosphere or calibration. It sets `compute_dphi_jitter=False`
in the reconstruction engine and records
`tt_jitter_ablated_for_diagnostics=true` in `metadata.json`.
The PSF with TT removed is **not** offered as a physically correct
alternative; it is a control to quantify how much broadening comes
from the TT contribution.

Example:

```bash
uv run python scripts/keck_n0013_smoke.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
  --tt-jitter-model spectral_closed_loop \
  --tt-mirror-model unity \
  --disable-tt-jitter \
  --output "$PWD/results/n0013_psfr_tt_disabled"
```

Three scenarios (spectral unity, spectral Ragland, TT ablation)
must be evaluated with identical configuration and input data. Useful
**diagnostic**, not observational fitness, metrics: normalized
encircled energy and central FWHM, peak and covariance trace. NIRC2's
LP415-20 observation is a *binary*, not a single point source, so its
core cannot be used as a fair single-star FWHM target without
explicit binary fitting.

**Important internal mismatch (recorded, not silently corrected):**
`systemDiagnosis` currently estimates the van Dam white sensor
noise through the *historical* P3 rejection transfer `ttloop.tf.rtf`,
which assumes `P=lag` (unit-modulus mirror response). Reusing that
noise covariance for the Ragland mirror while changing `S(f)` is
**not a fully self-consistent noise identification**. The spectral
Ragland mode emits a `RuntimeWarning` and saves
`noise_transfer_mismatch=true`, `van_dam_rejection_model=p3_pure_delay`.
Its numeric covariance is a sensitivity result, not a calibrated
reconstruction. A subsequent change should re-estimate the van Dam
noise with the same chosen `W/K/P` transfer, then re-assess sensitivity
to fit band. No parameter may be tuned to match the NIRC2 binary.
