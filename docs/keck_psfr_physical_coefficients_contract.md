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
