# P2 — Keck/NIRC2 LGS anisoplanatism: frozen scientific reference (2026-11-10)

**Status:** P2 analytical/numerical **reference frozen** for Keck n0120.
This is **not** a declaration that the old \`focal_angular_function\`
production kernel has been replaced or instrumentally calibrated.
Further changes to P2 require a new, explicitly scoped experiment/PR.
Start P1 (pseudo-open-loop / r0 / L0) independently.

## 1. Physical contract: independent AO error terms

Adopt the Keck **LGS high-order reconstruction** modal split explicitly
confirmed by the AO researcher:

- \`K_focal\`: LGS focal/cone anisoplanatism with **piston and both
  tip/tilt modes removed**, computed from a nonstationary covariance
  on the actual illuminated pupil. Hardy's 0.452 coefficient has this
  same modal convention for an ideal circular Kolmogorov pupil.
- \`K_aniso,HO\`: angular anisoplanatism restricted to the high-order
  modal subspace **excluding piston and both tip/tilt modes**.
- \`K_anisokinetism,TT\`: angular tip/tilt anisoplanatism **only**,
  referenced to NGS/STRAP/APD measurements and their physical
  closed-loop propagation.
- The LGS HO term **must not include atmospheric tip/tilt** in the
  AO-loop decomposition. Prevent double counting between the
  focal/angular HO terms and the NGS/STRAP TT term.

**Unverified:** the detailed Keck 2020 STRAP/APD optical gain,
the NGS anisokinetism covariance and the measured angular-HO term.
These remain separate later calibration tasks, not a blocker for P1.

## 2. Spatial statistics: do not stationarize the cone residual

For nonstationary anisoplanatism the Gaussian-ensemble transfer is

\`OTF_cone(rho) = sum_(r,r+rho) A(r) A(r+rho) exp[-D_cone(r,rho)/2] / sum_r A(r)^2\`.

\`A\` is the real pupil **amplitude transmission**, with the actual
Keck pupil gaps/spiders. The approximation

\`OTF_DL(rho)*exp[-0.5*mean_pairs(D_cone(r,rho))]\`

is **not mathematically interchangeable** with averaging the
exponentials; Jensen gives \`OTF_exact >= OTF_stationarized\` for
nonnegative amplitude pair weights and a Gaussian phase model.
Retain explicit nonstationary accumulation for future P2 integration;
the user confirms the additional computational expense is acceptable.
This is the nonstationary, point-wise anisoplanatism formalism of
Beltramo-Martin et al. (2018), *Off-axis point spread function
characterization in laser guide star adaptive optics systems*,
MNRAS 478, 4642–4656, DOI:10.1093/mnras/sty1103, which is subsequently
used in the 2019 PRIME reconstruction paper. The residual AO term may
be stationarized only under a separately justified approximation; the
anisoplanatic cone covariance explicitly depends on pupil position.

## 3. Golden Keck n0120 numerical checkpoints

Reference: D=10.5m, lambda_ref=1.53µm, r0_los=0.4391188896m,
LGS H_los=122252.296m. On an ideal circular pupil at L0=infinity,
direct 43x43 Kolmogorov covariance with piston+tilt projection gives
**313.15nm (1 layer)** and **295.59nm (7 layers)**, versus Hardy
**312.42nm** and **295.01nm**. Both deviations are below 0.3%.

### Keck science mask (sampled 31x31), phase WFE

| Layers | L0 [m] | Piston-only [nm] | Piston+TT removed [nm] |
|---|---:|---:|---:|
| 1 | 31.045692 | 299.88 | **288.25** |
| 1 | infinity | 356.16 | **294.72** |
| 7 | 31.045692 | 284.76 | **269.96** |
| 7 | infinity | 353.44 | **277.80** |

Reference Hardy uses a *circular*, tilt-removed, Kolmogorov pupil; it
is not an exact instrument target for the segmented Keck mask.

### Exact nonstationary versus stationarized focal-only Strehl

Same Keck 31x31 sampled mask, piston+TT removed in both methods:

| Layers | L0 [m] | Exact SR | Stationarized SR | Relative increase |
|---|---:|---:|---:|---:|
| 1 | 31.045692 | 0.264533 | 0.249780 | **+5.907%** |
| 1 | infinity | 0.250661 | 0.234763 | **+6.772%** |
| 7 | 31.045692 | 0.308125 | 0.295638 | **+4.224%** |
| 7 | infinity | 0.289538 | 0.275541 | **+5.080%** |

These SR values are focal-only OTF diagnostics, **not full AO PSF-R
Strehl**, and the ratios are **not transfer-function correction factors**.

Convergence: Keck 1-layer, L0=31.05m, piston+TT RMS
n=21 **287.76nm** -> n=31 **288.25nm**; 7-layer
n=21 **269.64nm** -> n=31 **269.96nm**.
At 1-layer measured L0, exact focal-only SR n=21
**0.264481** -> n=31 **0.264533**.

### NIRC2 pupil interpolation provenance

Pupil FITS \`keck_pupil_largeHex_272px.fits\`, \`Resolution=100\`,
science pupil angle **46.76723°**. The rotated mask's pre-clipping
min/max amplitude: -0.152881 / +1.198463; 4342 pixels outside
[0,1], of which 262 have clipping delta >0.01.
Integrated intensity change under prototype [0,1] clipping:
**-0.2545324%**. This is sufficiently small for the *global
WFE reference* but does not validate the detailed diffraction
pattern. Telescope production does not clip the rotated amplitude;
future work must compare the full-resolution diffraction OTF
before/after physical clipping and verify pupil angle/parity
against diffraction spikes.

## 4. Frozen implementation and limits

- Existing production \`p3/aoSystem/anisoplanatismModel.py\` defaults
  unchanged (legacy square-grid pair average + interpolation).
- Isolated reference scripts: \`keck_p2_cone_modal_reference.py\`,
  \`keck_p2_spatial_sampling_audit.py\`,
  \`keck_p2_masked_cone_otf.py\`; associated synthetic regression tests.
- NO science image fitting, NO empirical gain or phase correction,
  NO modification of \`r0\`, \`L0\`, Hardy, STRAP or noise estimators.
- The physically preferred nonstationary P2 *production integration*
  is **deferred and clearly separated**; it requires a planned
  performance strategy and instrument pupil/TT integration checks.
- P1 is next: validate pseudo-OL sign/timing and the joint r0/L0
  Zernike-variance fit against site DIMM, with provenance.

**Freeze criterion:** retain the values and modal conventions above as
regression references. Do not tune P1 fits to reproduce a target
NIRC2 image Strehl or alter the frozen P2 quantities.
