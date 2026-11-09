# PRIME+ — PSF-R spatial-physics verification contract (post STRAP audit)

Date: 2026-11-09. Scope: **telemetry-only** physics, no science-image fitting or parameter tuning. This first audit PR is stacked on `feat/keck-strap-apd-noise-calibration` (PR #7) until PRs #5–#7 can be merged sequentially.

## A. Frozen provisional STRAP reference (not a calibrated science-path jitter)

1. Sensor-input noise: van Dam high-frequency PSD fit with the recorded filter band, Welch length, covariance axes, cross-term, and sampling uncertainty.
2. Reference mirror: `unity` (historical P3 pure-delay assumption); published `ragland_2016_keck_ii` used **only** for independent sensitivity, with *its own* consistent van Dam rejection.
3. Four-APD delta method and pure-Poisson Monte Carlo: uncalibrated conditional baseline, **not** an absolute detector calibration.
4. Keep physical TT spectral-OTF propagation experimental, opt-in, explicitly distinct from the legacy OTF and from measured STRAP centroids. No tuning against `n0013.fits`.

In the existing **unity** audit, `tr(C_y)=3.227752e-12 m²`, `g_S=0.97454715`, `tr(C_n,van Dam)=2.904730e-12 m²`; thus `tr(C_y-g_S C_n)=3.96956e-13 m²`. A ±10% perturbation of `tr(C_n)` changes this *measured-centroid residual* trace by ∓71%, with `C_y` and `g_S` fixed. This is **not** a calibrated uncertainty on NIRC2 science-path jitter.

Required report: independent uncertainty ranges from high-frequency band, Welch, independent time blocks, and conditional APD assumptions. Do not call a model-choice envelope a confidence interval.

## B. Review gates — in this order

Every source term must have: **T** (theoretical reference/equation), **N** (independent synthetic/numerical checks), **I** (Keck instrument calibration when available). Record unsupported I separately rather than force agreement with the observed binary.

### B0 — Common Fourier / covariance / OTF layer

Primary: `p3/psfr/psfrUtils.py:modes2Otf`, `p3/aoSystem/FourierUtils.py`.

- Zero modal covariance must produce `D_phi=0` and the diffraction-limited, normalized OTF. This PR fixes an uninitialized `dphi` in that branch and adds a regression.
- **Open scientific risk:** `modes2Otf(basis="Vii")` uses `np.linalg.svd(Cmm)`; for symmetric indefinite `Cmm`, **singular values discard eigenvalue signs** and can conceal a noise-subtracted non-PSD covariance. Do not silently treat it as physically valid. Evaluate `eigh` diagnostics/PSD tolerance and a fail-closed or explicitly flagged legacy policy in a subsequent PR, with an independent synthetic indefinite fixture. Do **not** replace by abs eigenvalues.
- Check `D_phi(0)=0`, PSD >=0, normalized `OTF(0)=1`, Hermitian symmetry, nonnegative/finite PSF, spectral sampling and the OPD(m²)-to-phase(rad²) conversion.
- Check contribution addition in structure-function space, not double counting of terms already contained in empirical HO covariance.

### B1 — DM fitting error (next full audit)

Primary: `p3/aoSystem/frequencyDomain.py:pitch`, `p3/psfr/psfR.py:fitting_phase_structure_function`.

- Current model: `Phi_fit(k)=0.0229 r0^(-5/3) (k²+L0^(-2))^(-11/6) M_out(k)`, where `M_out` is from DM pitch and square/circular AO cutoff selection. The **normalization and geometric cutoff are assumptions to verify** against the original literature and the instrument.
- Test explicit independent PSD quadrature versus FFT-derived covariance and phase SF; r0 scaling at fixed L0 and geometry; limits for varying DM pitch, sampling and grid extent.
- Compare to published `~0.23(d/r0)^(5/3)` or `~0.28(d/r0)^(5/3)` **only after matching the actual cutoff/DM influence-function approximation**; the coefficient is not a universal pass/fail threshold.
- Output per-term `D_phi`/OTF, computed variance, sampling convergence, source and sensitivity. No NIRC2 fitting.

### B2 — Spatial aliasing

Primary: `p3/aoSystem/fourierModel.py:aliasingPSD`, `p3/psfr/psfR.py:aliasing_phase_structure_function`.

- Audit independent roles of WFS subaperture pitch versus DM actuator pitch; comb folding, reconstruction filter, temporal weighting and sensed-layer mask.
- Compare chunked and streaming algorithms on identical synthetic geometry; test vanishing aliasing under a *properly specified* ideal anti-alias filter, nonnegative in-band PSD, r0 scaling, and absence of fitting/HO double-counting.
- Check the current `aliasingPSD()/atm.r0**(-5/3)` followed by `r0**(-5/3)` normalization (conditional on consistent atmospheric profile and reference wavelength).

### B3 — LGS focal-angular anisoplanatism and anisokinetism

Primary: `p3/aoSystem/anisoplanatismModel.py`, `p3/aoSystem/frequencyDomain.py:anisoplanatismPhaseStructureFunction`, `p3/psfr/psfR.py:phase_structure_function`.

- The code calculates a **combined focal-angular** term and a separate anisokinetism term. Audit joint covariance/cross terms; do not separately sum focal and angular variances when correlations matter.
- Compare full Fourier/SF prediction with published Keck LGS angular/focal anisoplanatism derivations and their limiting cases.
- Unit/limit tests: `h=0`, zero guide-science angle, `H_LGS→∞`, layers above LGS, reversed angle, pupil coordinate zero lag, one/multiple layers, telescope airmass scaling.
- Trace `Cn2 = weights * r0**(-5/3)`, reference science/atmospheric wavelength scaling, dimensions, source `direction` frame, and no TT-jitter double-counting.
- Distinguish equality of functions under geometric limits from validation of the 2020 layer-height/profile inputs.

### B4 — Empirical high-order residual (HO)

Primary: `p3/psfr/psfR.py:residual_phase_structure_function`, `psfrUtils.modes2Otf`.

- Verify DM-command difference/gain transfer; unit conventions of `dmIF`; modal noise subtraction and sign/PSD diagnostics; use synthetic injected commands and noise.
- Audit whether `C_ao`, aliasing, fitting and anisoplanatism are disjoint under the assumed reconstruction; no implicit tuning of `gho`.

### B5 — Pixel transfer, NCPA, pupil, and error-budget consistency

- Pixel sinc convention, detector pixel scale and orientation, PSF polychromatic integration.
- Keck pupil, spiders/segmentation, NCPA and non-common-path disturbances; zero NCPA must be explicit hypothesis, not certainty.
- Fix/check the separate tomographic warning `sqrt(wfeST**2 - wfeS**2)` when negative: no arbitrary clipping of a physically significant invalid decomposition.

## C. Conditions before multi-acquisition benchmark

1. Focused + full tests green, and deterministic per-term ablations.
2. Fixed reference configuration and uncertainty-report schema; traceable code SHA, dataset metadata, configuration, units and source.
3. Every major term has T and N evidence; missing I remains explicitly marked and sensitivity quantified.
4. First evaluate a single isolated-star reference and only then generalize across multiple observations; `n0013` binary is not a quantitative single-star PSF reference.

## Scientific anchors

- Beltramo-Martin et al. (2019), *PRIME*, MNRAS 487, 5450: fitting/aliasing/aniso structure-function decomposition, original Keck PSF-R.
  https://academic.oup.com/mnras/article/487/4/5450/5519866
- van Dam et al. (2006), *Angular anisoplanatism in laser guide star adaptive optics*, SPIE 6272-106, joint angular/focal effect.
  https://www2.keck.hawaii.edu/optics/lgsao/docs/spie2006-6272-106.pdf
- Ferreira et al. (2018), *Numerical estimation of wavefront error breakdown in adaptive optics*; fitting coefficient ~0.23 is model dependent.
  https://doi.org/10.1051/0004-6361/201832579
- PRIME+ repository: `docs/psfr-physics-traceability.md` is the authoritative T/N/I ledger. This contract complements rather than replaces it.
