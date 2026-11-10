# Keck/NIRC2 PSF-R — Physical coefficients and provenance contract (V1)

Status: **implementation audit, not absolute Keck calibration**. Scope: current
`feat/psfr-spatial-physics-audit` pipeline. No science-PSF fitting.

## 1. Core units and wavelengths

| Quantity | Physical definition | Source | Verified? |
|---|---|---|---|
| `r0_tel` | m at `atm.wvl=500e-9 m` | WFS pseudo-open-loop Zernike fit | **not** independently validated |
| `r0_dimm` | 0.976 lambda / seeing(DIMM) | site monitor, lambda=500nm | conversion verified, spatiotemporal correspondence unverified |
| Seeing coefficient | `epsilon_radians=0.976*lambda/r0` | Fried/Kolmogorov long exposure | physical model |
| Wavelength conversion | `r0(lambda2)=r0(lambda1)*(lambda2/lambda1)^(6/5)` | Kolmogorov | unit-tested |
| Cn2 integral | `r0^-5/3=0.423*(2*pi/lambda)^2*integral Cn2(h) dh` | Kolmogorov | source quality not guaranteed |
| `L0` | von Karman outer scale, m | joint fit with r0 on reconstructed Zernike variance | often degenerate / fits may reach boundary |
| `Cn2Weights` | normalized bin integrals, sum=1 | MASS profiles + DIMM ground complement, or median fallback | source-labelled, not calibrated by this code |
| `Cn2Heights` | altitude of MASS bins, metres above observatory | [0,500,1000,2000,4000,8000,16000] | discretization approximation |

## 2. Reconstruction pseudo-open-loop

`rec.res` and `dm.com` are 2-D (time, actuator), in metres OPD.
`delay_frames = holoop.latency_seconds * holoop.sample_rate_hz`.
Retain original P3 temporal *sign* assumption until independently confirmed:
`u_ol(t)=u_dm(t)+interpolate(rec_residual,t+delay_frames)`.

**Fix**: interpolate **along axis=0 only** and use only time indices whose
interpolation support is inside this exposure. Previous `np.roll(x,(shift,0))`
with `axis=None` flattened a 2-D array and mixed actuators. Recompute all
Zernike fit results from this point forward. The sign of RTC latency and precise
sensor/command synchronization have not yet been verified from timestamps.

The pseudo-open-loop Zernike variance is projected using `mat.u2z`; `r0/L0`
are fit to the theoretical radial-order variance, with an optional model of
subtracted WFS noise. `r0_tel` then replaces the initial DIMM seeing in the
forward AO model. Therefore DIMM is an **external comparison**, not the
final r0 used in current PSF-R.

### 2.1. DIMM zenith and science line-of-sight

For an atmosphere represented by a plane-parallel optical turbulence column,
at science airmass `X=sec(z)`,
`r0_los = r0_zenith * X^(-3/5)`, and
`seeing_los = seeing_zenith * X^(3/5)`.

Comparing raw site-monitor DIMM seeing to AO r0 along a science line-of-sight
can therefore introduce a substantial systematic difference. At n0120,
`EL=47.40723 deg`, `X≈1.358`, and a zenith-normalized `0.74 arcsec` DIMM
would predict `0.889 arcsec` on the science line-of-sight, near the previous
`0.906 arcsec` AO fit.

**This comparison is conditional:** the exact MKWC per-record reduction
convention (zenith-normalized or raw) is not certified by the archived
`.dimm.dat` fields. The smoke writes both direct and conditional
airmass-adjusted ratios, with the assumption in metadata. Do not replace
the independent site seeing by the conditional estimate without provenance.

### 2.2. WFS r0 is already a line-of-sight measurement

The WFS-based Zernike fit in `systemDiagnosis` estimates `r0_tel` directly
along the Keck science line of sight. The generated INI formerly exported
this as `[atmosphere] Seeing`, while generic `aoSystem` treated all INI
seeings as zenith-normalized and multiplied the corresponding `r0` by
`airmass^(-3/5)` again. This was a physical double correction.

`configFile` now emits `SeeingReference='line_of_sight'` for telemetry-
derived Keck values. `aoSystem` retains a default `SeeingReference='zenith'`
for old manually-authored INIs; it only applies `airmass^(-3/5)` to those
zenith references. The automatic science FOV follows the same convention.

Example n0120 (X=1.35835884), pre-fix/model reference lambda=1.530µm:
- `r0_tel(500nm)=0.11474041m` from the WFS fit (already LOS);
- expected `r0(1.530µm)=0.43911889m` after lambda^(6/5) scaling;
- old AO model `r0(1.530µm)=0.36540410m` after erroneous extra
  factor `X^(-3/5)=0.83213`.

The smoke now asserts this invariant after model construction and writes
both r0s plus the declared seeing reference. All 1/3/7-layer PSFs after
this fix must be compared with one another on the same commit.

### 2.3. Analytical cone effect versus OTF-equivalent anisoplanatism

Hardy (1998), eq. 16, as reproduced in Beltramo-Martin et al. (2019)
*PRIME*, MNRAS 487, 5450, eq. 12:
https://academic.oup.com/mnras/article/487/4/5450/5519866

For a normalized vertical turbulence fraction `w_l`, r0 at a consistent
line-of-sight wavelength, and `x_l=h_l/H_LGS`:

```
sigma_phi^2 = (D/r0)^(5/3) * sum_l w_l *
              [0.5*x_l^(5/3) - 0.452*x_l^2] / 0.423
WFE_cone_nm = sqrt(sigma_phi^2) * wavelength_m*1e9/(2*pi)
```

The full-layer above-beacon clipping (max Noll piston-removed
variance `1.0299`) is a model approximation, not Hardy's small-height
series formula. The `0.423` maps integrated Cn2 dh to Fried r0;
`0.452` comes from the published cone expression. An old duplicate
function used `0.425` and omitted `/0.423`, while returning WFE nm
under the misleading name `focal_anisoplanatism_variance`. That name is
now a backwards-compatible alias of `focal_anisoplanatism_wfe`;
`focal_anisoplanatism_phase_variance` explicitly returns radians².

**Do not equate** this Kolmogorov analytical phase variance to the
OTF-derived diffraction-weighted Maréchal-equivalent WFE, which includes
the model's finite-L0 and DM/frequency filtering assumptions. Their
trends can differ, and should be compared, not forced to coincide by
tweaking coefficients.

With n0120's actual seven-layer MASS/DIMM weights and H_LGS=90km,
the standalone Hardy estimator at `r0_tel(500nm)=11.474cm` gives
about `312.4nm` for the 1-layer equivalent altitude `6.043km`
and `295.0nm` for 7 layers. The older 1-vs-7 PSF comparison had a
10.78% higher peak with 7 layers but **different Git revisions** and
**missing 7-layer error-budget diagnostics**. It is not a controlled
test of this physical formula. P2 now saves
`anisoplanatism_diagnostics.json` with both Hardy and OTF results.

### 2.4. P2 finding: zero-baseline anisoplanatic SF and Strehl

The focal-angular anisoplanatic kernel is made by averaging a spatially
varying covariance over actuator baselines, then resampling it from an
odd-size support onto an even-size OTF grid. In the n0120 benchmark before
DC normalization, the **center sample was not zero**:
- 1 equivalent layer: `Dphi_ani(0)=0.304339917734454 rad^2`;
- 7 MASS-DIMM layers: `Dphi_ani(0)=0.1762538469780601 rad^2`.

A true phase structure function satisfies `Dphi(0)=0`; the atmospheric
OTF must have `OTF_turb(0)=1`. An additive constant `c` in Dphi
multiplies the complete OTF by `exp(-c/2)` and hence does **not**
affect the flux-normalized PSF, but incorrectly depresses Strehl values
calculated directly from the unnormalized OTF and inflates any
WFE inferred from the integral of that OTF.

The n0120 results gave:
- apparent `PSFR_SR_OTF_7 / PSFR_SR_OTF_1 = 1.1483352555`;
- normalized PSF peak ratio `= 1.07709787065`;
- DC-corrected OTF Strehl ratio:
  `1.1483352555 * exp(-(0.3043399177-0.1762538470)/2)
    = 1.0770978565`, agreement to 1.4e-8.

**Correction**: `psfR.phase_structure_function` now computes the
anisoplanatic SF for each science source and subtracts its central
OTF sample (independently by source) **after** the atmospheric
wavelength scaling and weighted sum. The forward PSF morphology is
unchanged, the OTF DC is one, and the SR/WFE budget is normalized.
`get_error_breakdown` uses the same zero-lag convention for total,
angular, tip-tilt anisokinetic, and focal components. The pre-subtraction
center value is saved to `anisoplanatism_diagnostics.json` for audit.

This is an optical normalization requirement, not a calibration knob,
and does not establish the accuracy of the underlying focal-angular
spatial covariance interpolation away from zero baseline. Separate
synthetic tests must check the anisoplanatic SF shape under finite L0.


### 2.5. P2 n0120 user-Mac confirmation (1 vs 7 layers)

The user reran the actual n0120 LGS science FITS and TRS SAV with the
zero-lag SF normalization patch at **identical Git SHA**
`5718a88b62c6c00f3003f0c31472c8c7b25bca71`, and no comparison flags
or changed physical parameters (`layer_comparison(2).json`).

| Measured quantity | 1 equivalent layer | 7 MASS-DIMM layers |
|---|---:|---:|
| Hardy cone RMS OPD | 312.4173 nm | 295.0075 nm |
| OTF-equivalent focal anisoplanatism | 236.4439 nm | 228.0546 nm |
| Total Maréchal-equivalent WFE | 333.2224 nm | 327.2942 nm |
| PSF-R OTF Strehl | 15.90305% | 17.12914% |
| Flux-normalized image peak | 0.01005435 | 0.01082951 |
| Anisoplanatic SF at zero lag | 0 rad² | 0 rad² |
| Unnormalized anisoplanatic zero-lag offset | 0.30433992 rad² | 0.17625385 rad² |

`SR_OTF_7/SR_OTF_1 = 1.0770978565` and
`normalized_peak_7/normalized_peak_1 = 1.0770978707`,
a relative difference of about 1.3e-8. The corrected WFE decrease is
8.3893 nm for the focal term, compared with Hardy's 17.4098 nm;
these are different estimators and do not have to be numerically equal.
Normalized PSF morphology does not change on this DC-only correction,
but the seven-layer peak remains **7.71%** higher than one-layer,
with total variation distance **0.0530756** and additional encircled
energy of **5.224 percentage points** at 16 pixels.

**P2 numerical implementation status for n0120: validated against real
Keck telemetry and controlled 1/7 contrast.** Full physical validation
across other LGS geometries, finite-`L0` angular-covariance treatment,
and absolute NIRC2 science-image calibration remain open. This case
is not a reason to tune `0.452` or another analytical coefficient.

### 2.6. P2 unresolved: why Hardy RMS ≠ Strehl-equivalent RMS

**Do not claim physical closure merely because 1- and 7-layer PSFs have a
similar trend.** On the controlled n0120 rebaseline, Hardy gives
312.4173/295.0075 nm (7/1 = 0.94427), while the OTF-derived
focal-anisoplanatic WFE gives 236.4439/228.0546 nm (7/1 = 0.96452).
Those ratios are not expected to be identical automatically.

A three-way comparison is required, all at the same wavelength:

1. Hardy analytical Kolmogorov cone phase variance `sigma_Hardy²`.
2. Pupil-piston-removed variance computed from the **same focal-only
   pupil-averaged phase SF used by P3**:
   `sigma_SF² = (1/2) sum(OTF_DL * Dphi_focal) / sum(OTF_DL)`.
3. Nonlinear Strehl-equivalent variance from that same SF:
   `sigma_SR² = -log(sum(OTF_DL * exp(-Dphi_focal/2)) / sum(OTF_DL))`.

For a nonnegative diffraction-limited OTF, Jensen's inequality yields
`sigma_SR² <= sigma_SF²`. The difference between 2 and 3 is a
mathematical consequence of the nonlinear mapping from SF to Strehl,
even if there is no physical-model bug. The difference between 1 and 2
does **not** have this explanation; it diagnoses finite-L0 effects,
pupil geometry, covariance modeling, filtering/normalization errors,
or inconsistent heights/laser geometry.

A noteworthy external check: PRIME 2019, Eq. 12 and Section 4.2,
reported ~158 nm from Hardy and ~162 nm from its fitted anisoplanatism
transfer function for a 2017 Keck observation
(https://academic.oup.com/mnras/article/487/4/5450/5519866).
The roughly 25% absolute RMS discrepancy in n0120 therefore deserves
a quantitative independent audit. Conditions are different, but
"different WFE definitions" is not a sufficient closing argument.

**Implementation observations, not assumptions**:
- `focal_angular_function` uses a finite outer-scale von Karman
  covariance (`f0=2*pi/L0`; L0 ~31m on n0120).
- The code constructs `Hfilter` from influence functions in SLAO
  `psfR.__init__`, but the SLAO branch of
  `frequencyDomain.anisoplanatismPhaseStructureFunction` calls
  `anisoplanatism_structure_function` **without forwarding Hfilter**.
  Consequently the present n0120 cone kernel has `h_filt=1`: DM
  filtering cannot yet be invoked as an explanation for this run.
- Atmospheric/lgs geometries are both converted to line-of-sight
  heights inside `aoSystem`, making their ratio `h/H` invariant to
  airmass, while `r0` is separately a line-of-sight estimate.
- The focal-angular SF is baseline averaged and resampled on the OTF
  support; small negative OTF samples from interpolation are clipped
  **only in the diagnostic**, never in the forward model. Negative
  clipped weight mass is explicitly reported.
- The true phase-SF shape away from zero baseline has not been
  independently validated by phase-screen Monte Carlo.

The smoke now emits `anisoplanatism_diagnostics.json` with
`pupil_sf_bridge_focal_only`, `pupil_sf_bridge_total_anisoplanatism`,
Jensen gap, Hardy-to-linear-SF ratio and model-condition flags, plus
`focal_anisoplanatism_sf.npy` and `diffraction_otf.npy` to
enable independent analysis **without a repeated forward model run**.
`keck_compare_layer_counts.py` compares the bridge quantities from
the one- and seven-layer outputs.

**Acceptance for P2 theoretical closure**:
- Jensen inequality passes for both n0120 layer counts.
- OTF-derived focal-only WFE recomputed from the SF matches the
  reported error budget after accounting for any signed OTF artifacts.
- If Hardy and the *linear SF pupil variance* still disagree,
  investigate matched L0→infinity, identical pupil, laser height and
  actuator/IF filter assumptions before changing any `0.452`,
  `0.423` or r0 coefficients.
- Independent synthetic phase-screen Monte Carlo validates cone
  covariance and spatial SF with a known Cn2 profile.
- Only then consider P2 physically closed and move to P1 r0 fitting.

### 2.7. P2: pupil weighting and exact L0=infinity test

**Important correction to the audit:** the true NIRC2 pupil does enter
`OTF_DL` during the optical propagation. For a **stationary** phase SF
this is all the pupil filtering required:

```
OTF(rho)=OTF_DL(rho)*exp[-Dphi(rho)/2]
```

However, the Keck LGS focal-cone phase SF from Flicker (2008) is
spatially *nonstationary*, depending on the two pupil coordinates
separately. For an amplitude pupil P(r), the correct expression is

```
OTF(rho) =
    int P(r)P(r+rho) exp[-Dphi(r,r+rho)/2] dr /
    int P(r)^2 dr
```

`FourierUtils.matrix_to_map` averages the original pairwise covariance
over **all Cartesian grid pairs** with a fixed separation, without
the actual NIRC2 pupil mask. The subsequent `OTF_DL` multiplication
correctly introduces the total number of illuminated pairs at each
separation, but cannot reconstruct their individual nonstationary
covariances or in general interchange `<exp(-D/2)>` with
`exp(-<D>/2)`. This is an existing **stationarisation
approximation**, not proof of a defect large enough to explain the
Hardy difference. No modification to `matrix_to_map` or the forward
pupil is applied in the current P2 diagnostic stage.

### Controlled L0 sweep (without touching the science PSF)
`scripts/keck_l0_cone_sweep.py` loads the completed Keck smoke run's
`anisoplanatism_diagnostics.json`,
`frequency_grid_diagnostics.json`,
`diffraction_otf.npy`, and the generated `.ini` to determine the
actual 1-D DM actuator grid; it fixes the science on-axis r0, projected
MASS-DIMM heights, LGS altitude and the NIRC2 OTF. It calls the
*same native* `focal_angular_function` used by the reconstruction, for
L0=31.04569, 100, 300, 1000m and exact Kolmogorov infinity, computing
the linear pupil-weighted SF moment and the nonlinear Strehl-equivalent
variance separately. Hardy's Kolmogorov analytical reference is shown
unchanged across L0.

Unlike simply using L0=10^8m, `focal_angular_function` implements an
analytically stable L0=infinity branch using the small-argument
McDonald-function expansion:

```
Ialpha(z) = 3/5 - a*z^(5/3) + O(z^2)
a = -Gamma(-5/6)/(2^(8/3)*Gamma(11/6))
```

The six corresponding terms in Flicker's covariance are evaluated as
a direct combination of absolute separations^(5/3), with no
cancellation of nearly equal large-outer-scale covariance terms.
The finite-L0 branch remains bitwise unchanged.

First run 1-layer, then 7-layer if the 1-layer result is convincing.
The test's **baseline L0 numerical result must reproduce the original
focal SF WFE** from the run's JSON; otherwise the source/layer/pupil
geometry is not identical and the sweep is not comparable.

Decision rules:
- If the linear pupil SF variance tends to Hardy at L0=infinity,
  finite outer scale accounts for the bulk of the discrepancy.
- If a large difference remains, inspect the exact pupil-pair
  weighting, finite actuator grid and sampled SF interpolation.
- Do not change 0.452/0.423, TT gains or science WFS r0 to force
  the analytic and numerical WFE to match.
- Even in the infinite-L0 case, exact agreement is not mandated:
  Hardy assumes an analytical ideal pupil and this implementation
  uses the segmented/masked Keck pupil and a finite square actuator
  grid in its nonstationary stationarisation.

### 2.8. n0120 infinite-L0 sweep: finite outer scale does not close Hardy gap

User-executed `l0_cone_sweep.json`, with a one-layer Keck n0120 source,
records **baseline relative error=0** at the production L0=31.045692m.
The comparison holds WFS `r0=0.4391188896m` at 1.53um, actuator grid
nActu1D=21, nOtf=128, sampling=3.0231069, and actual NIRC2
diffraction OTF fixed.

| L0 [m] | Flicker pupil-SF WFE [nm] | Flicker SR-equivalent [nm] | Hardy [nm] |
|---:|---:|---:|---:|
| 31.04569 | 237.5777 | 236.4439 | 312.4173 |
| 100 | 245.3324 | 243.6726 | 312.4173 |
| 300 | 249.6016 | 247.5172 | 312.4173 |
| 1000 | 252.6562 | 250.2096 | 312.4173 |
| infinity | 258.6954 | 255.4047 | 312.4173 |

Thus L0 explains **25.5% of the initial variance gap** and only
**28.2% of the RMS WFE gap**, leaving 53.72nm RMS/31.4% of Hardy's
phase variance unaccounted for even in the Kolmogorov limit.
Jensen's nonlinear Strehl conversion is a small, separately accounted
effect. This does not prove a defect in either analytic or numeric
phase-covariance model because the geometries and modal content
are not yet matched.

**New likely spatial-grid inconsistency in legacy `focal_angular_function`:**
The Keck run has nActu1D=21 -> `n_ph=43`.
The code uses `umax=D/2*samp/2=7.93565564m` and its
pupil-coordinate sampling is
`x_i=umax*(i-21)/43`, covering +/-3.87555m.
The separation map sent to `interpolateSupport(cmap,128)` therefore
covers +/-7.75111m.
The downstream science OTF, built by padding the actual Keck pupil with
`pupil2otf(..., samp=3.0231069)`, has an inferred full-grid
half-width `D*samp/2=15.87131m`.
The **inferred map-to-OTF lag span ratio is 0.48837209**. This is not
metadata preserved by `scipy.ndimage.zoom`: it simply stretches array
indices and does not carry physical separation coordinates.
Consequently a stationary synthetic `Dphi(rho)~|rho|^(5/3)` that is
sampled on the legacy covariance grid and resized to 128 has only
roughly ~31-32% of its physically expected amplitude at OTF lag
offset 25-40px. That demonstration is a **grid calibration
discriminant**, NOT an estimate of the correction required for the
actual nonstationary LGS focal cone.

`keck_l0_cone_sweep.py` now reports
`flicker_to_otf_lag_geometry` with physical half-spans and warnings.
**Do not arbitrarily rescale Dphi or L0.** Next validate the exact
pupil2otf lag coordinate and interpolate the pairwise nonstationary
covariance with explicit physical lag axes. Only after this test should
the more expensive true-pupil pair weighting / average-before-exponential
issue be assessed. The production `focal_angular_function` at finite
L0 and `matrix_to_map` are unchanged in the current experiment.

## 3. Keck II NIRC2 pupil angle

KAON 1160 (Ragland, 2017), equations (1) and (2):

- `theta_wfs_pupil_deg = ROTPPOSN - EL`
- `theta_nirc2_pupil_deg = EL - ROTPPOSN`
- `theta_sky_pa_deg = ROTPOSN - INSTANGL` (different physical quantity).

Numerical pupil maps use a rotation in NIRC2 detector coordinates. We apply
`theta_nirc2_pupil_deg` to the current aperture map, but the map's own
orientation zero, optional 90-degree reference difference, camera parity, and
scipy image-row sign require a *separate calibration*, including a known PSF
with pupil diffraction spikes. A rotated matching PSF is **not** alone a
proof of the mask's reference zero.

## 4. MASS-DIMM use

The archive can provide site DIMM total seeing, free-atmosphere MASS seeing,
and six MASS Cn2-integral altitude bins. Merge them only when:

1. The DIMM nearest sample and MASS six-layer profile are present and not stale
   (>60 minutes is rejected, independently of verbose mode).
2. DIMM total seeing >= MASS free-atmosphere seeing, permitting a **nonnegative**
   ground-layer integral.
3. All six MASS bin integrals are finite and nonnegative.

If not, use a **labelled median-vertical-profile fallback**, not an invented
MASS profile. If DIMM is valid but MASS invalid, keep the measured DIMM seeing
and use only default normalized layer weights.

The smoke CLI accepts `--n-layer 1/3/7`: 1 preserves the historical
single-layer benchmark, 7 retains all MASS-DIMM heights. All original seven
weights and heights are recorded independently of the equivalent-layer
compression.

MASS/DIMM nearest-neighbour selection is still based on UTC fractional hours
of the observing day, not full ISO timestamps. The 60-minute gate prevents
obviously stale samples but does not certify cross-midnight matches or
atmospheric representativeness. A full UTC timestamp-aware join is required.

## 5. Tip-tilt

Keck PSF-R telemetry memo (Ragland) labels `DTTCENTROIDS` (LGS),
`RESIDUALWAVEFRONT` TT (NGS), and `TTCOMMANDS` in **arcsec**.
Absolute on-sky optical gain remains unverified.

The physics of a RMS Noll tilt coefficient `a` (OPD metres) is
`W(x)=4*a*x/D`, so the angular motion is `theta=4*a/D`.
The Gaussian phase structure function is therefore

```
Dphi_noll(u,v) =
  (2*pi*D/lambda)**2 * [u,v] * Ctheta * [u,v]^T,
Ctheta=(4/D)**2 * C_zernike_OPD
```

where `u,v` are **normalized pupil separations**. Substitution leads to
`16*(2*pi/lambda)^2 * [u,v]*C_zernike_OPD*[u,v]^T`.

The `--tt-kernel-convention` switch distinguishes
`noll_angular` (physical angular transfer, default) from
`legacy_unscaled_opd` (the same covariance with the phase structure
function divided by 16, for controlled historical comparison). This is
**not an empirical gain fit** and the legacy option is not automatically
physically valid just because the PSF looks sharper.

LGS TT covariance options `legacy_psfr` and `spectral_closed_loop` encode
different noise/loop assumptions, and are independently labelled.
NGS uses the legacy model and retains a native tilt conversion different
from the LGS convention pending independent verification.

## 6. Other explicit approximations and risk

- `addNCPA=False` in all historical smoke tests; phase diversity/focus is
  deliberately not interpreted as atmospheric seeing.
- Forward spectral PSF: multiple throughput bins share a **single**
  reference-wavelength spatial PSF in current smoke mode.
- `HO small_negative/tiny_negative` PSD policies are **regularizations**,
  not independent WFS noise calibrations.
- RTC command gain/latency and physical LGS/NGS TT calibration are
  instrument assumptions, not facts proven by internal numeric tests.
- The 21-case ZIP results are **pre-fix exploratory baselines**; tag each new
  run with Git commit, flags and provenance before cross-case comparison.

## 7. Acceptance tests before a quantitative physical claim

1. Synthetic known-delay, multi-actuator pseudo-open-loop alignment.
2. KAON1160 `n0120` pupil angle, then two real calibration PSFs for sign
   and map-zero verification.
3. `r0_tel` vs `r0_dimm` scatter and per-frame fit residual, with
   confidence flags for temporal alignment and L0 hitting fit bounds.
4. Seven-layer MASS-DIMM versus one-layer PSF cone-effect differential at
   fixed r0; reject stale and negative Cn2 profiles.
5. Angular synthetic Gaussian jitter reproduces image-domain sigma on
   NIRC2 pixel grid, without using a binary to set the scale.
6. All derived errors (nm OPD / rad² phase / mas angle) referenced explicitly
   to the wavelength at which they are computed.
