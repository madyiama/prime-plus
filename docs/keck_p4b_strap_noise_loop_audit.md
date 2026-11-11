# P4-B — STRAP noise and closed-loop physical validity

**Status: draft research PR stacked on P4-A (#19), not a calibrated
Keck 2020 jitter model.** Do not merge or replace the physical TT
covariance based on these outputs alone.

This is the three-part experiment agreed after the same-acquisition
n0013 P4-A benchmark. P4-A verified the *numerical* chain
`DTTCENTROIDS arcsec -> C_theta rad² -> OTF_TT` to near
floating-point precision; it did **not** identify the real
sensor-noise floor or Keck 2020 servo transfer.

## How to run locally

From the existing P4-A checkout:

```bash
git fetch origin
git switch --track origin/feat/psfr-p4b-strap-noise-robustness

uv run --with pytest python -m pytest -q \
  tests/test_keck_p4b_strap_audit.py \
  tests/test_keck_strap_apd_noise.py \
  tests/test_keck_p4_angular_otf_chain.py \
  tests/test_keck_p4_tt_frame.py

uv run python scripts/keck_p4b_strap_audit.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --output "results/n0013_P4B" \
  --segment-length 512 --segment-welch 256
```

The script only uses the science FITS for instrument headers and
telemetry association. **No FITS image intensity is used to fit, choose
or rank a STRAP model.** The output includes
`p4b_strap_noise_audit.json` and
`p4b_strap_noise_audit.png`.

## Gate 1 — van Dam robustness (prespecified, no tuning)

Full-archive `DTTCENTROIDS`, in native **arcsec**, analyzed with
the existing `estimate_tt_noise` algorithm, **without changing it**.

| Parameter | Prespecified alternatives |
|---|---|
| High-frequency band / Nyquist | 0.55–0.85, 0.65–0.95, 0.75–0.95 |
| Welch window length (frames) | 256, 512, 1024 |
| Plant mirror model | unity; Ragland 2016 Keck II (exploratory) |
| Reference / no selected winner | unity, 0.65–0.95, Welch 512 |

For each pair, report `C_n` in arcsec², min eigenvalue, trace,
legacy `C_y-C_n`, Welch fit bins/segments, negative signed residual
spectral fraction, Welch integral vs time-domain variance, plus
physical atmospheric/noise covariance obtained by the existing
`ttClosedLoop.physical_tt_covariance_from_telemetry`.

A raw non-PSD noise estimate may be automatically projected by the
*existing* estimator. P4-B records the **unprojected** matrix and
whether that repair happened. It then **refuses to interpret**
the repaired value as physical TT (or to accept an invalid baseline).
For other failed physical covariances it records the error, without
zeroing negative modes.

Analyze robustness across the whole **prespecified grid**; do not
select a band because its PSF best resembles a visually elongated
binary image. The range is a *sensitivity* measure, **not** a
statistical confidence interval.

## Gate 2 — APD photon statistics versus flux

Read B.APDCOUNTS (four channels), APD_SKY_BACK, APD_IMAT and
B.DTTCENTROIDS; require the independent four-channel reconstruction
to agree with the archive to the existing stringent tolerance.

**Critical methodological safeguard:** segment the *chronological*
record into non-overlapping windows of exactly 512 frames **before**
looking at flux. Do NOT sort individual frames into flux quantiles:
conditioning on observed APD counts changes the shot-noise
distribution and can induce a false flux dependence.

Within each complete segment:

1. Record mean raw and net (background-subtracted) count rates;
   count-rate variance and discarded incomplete trailing frames.
2. Propagate the existing conditional APD Poisson model through
   the quad-cell Jacobian, with p=1 photoelectron-equivalent/count,
   avalanche excess factor=1, and no read noise; flag all these
   settings as *unverified*.
3. Estimate white sensor noise from STRAP centroids via van Dam,
   **within the same fixed chronological segment**, Welch 256,
   0.65–0.95 Nyquist, identical loop transfer.
4. Compare both covariances in **arcsec² at the sensor input**,
   not with the closed-loop measured-centroid covariance or physical
   NIRC2 image motion.
5. Preserve each block's raw unprojected noise eigenvalues
   and whether van Dam internally repaired the PSD matrix.

A rank correlation is **only exploratory** and only computed
when at least six complete blocks exist and the between-block
net count range is >=10% of its mean, all thresholds declared
before running. Otherwise status is **undetermined**. A correlation
does not calibrate APD gain; atmospheric fluctuations, optical
quad-cell gain, nonstationarity and vibration may co-vary with flux.

## Gate 3 — W/K/P/S closed-loop sensitivity and algebra

Reuse `from_p3_tiptilt_loop` and the **existing** P3 construction
of sensor W, controller K, lag, and optional Ragland 2016 Keck II
mirror transfer; do not create empirical 2020 coefficients.

Check `S(1+WPK)=1`,
`H_(y,d)=WS`, `H_(y,n)=S`,
`H_(e,d)=S`, `H_(e,n)=-PKS`.
Report `|W|_min`, loop denominator modulus,
sensitivity maximum, sensor→measured and
sensor→physical noise gains.

Compare mirror alternatives in **two separate experiments**:

1. **Fixed noise:** hold the sensor noise covariance at the
   unity baseline (0.65–0.95, 512) for both mirror hypotheses.
   Any change isolates W/K/P/S in the *conditional model*.
2. **Refitted noise:** estimate van Dam again using
   each model's own rejection `S(f)`. The difference then
   contains both changed transfers and changed estimated noise.

The physical on-camera covariance cannot be validated without an
independent Keck 2020 TTM loop identification and STRAP optical
gain. Negative atmospheric/total covariance is recorded as failure,
not projected.

## What P4-B can and cannot conclude

Our initial P4-A n0013 reference gave:
- raw centroid trace `3.227752e-12 m²` in RMS Noll OPD,
- van Dam noise trace `2.904730e-12 m²`: **~90%** of raw
  measurement variance in the historical unit convention;
- 31.6 mas RMS/axis (legacy), 41.1 mas RMS/axis
  (spectral), both **conditional on uncertain sensor noise**
  and on an assumed W/K/P model.

The new audit can identify **band/segment sensitivity**,
check the *direction* and *strength* of APD flux dependence
and quantify *model-conditional* plant differences. It cannot
claim that the residual jitter has a known true amplitude,
or that an APD photoelectron calibration is available.

**Do not modify P1 r0/L0, the P3 pupil, the P4-A angular OTF,
noise levels, or the TT calibration based on this study.**
Future simulation against independently known jitter plus
science-image fitting with a fixed PSF will supply another
validation channel.

## Merge requirements

- Synthetic tests and all relevant P4-A tests pass.
- Real n0013 P4-B JSON and figure reviewed, including
  negative-bin and non-PSD diagnostics.
- Same audit repeated on at least two other Keck LGS
  telemetry sets, or explicitly documented as single-case
  exploratory evidence.
- No code path silently changes the production sensor noise,
  angular TT covariance, model selection or OTF.
- Instrument-gain uncertainty and 2016-vs-2020 mirror
  calibration clearly disclosed.


## Correction de l'initialisation Keck RTC (11/11/2026)

Un premier test n0013 s'arrêtait avant le balayage van Dam :

```
AttributeError: 'structtype' object has no attribute 'freq'
  ... from_p3_tiptilt_loop -> tf.freq
```

`telemetryKeck` charge le taux d'échantillonnage et les paramètres
du contrôleur STRAP (`DT_SERVO`) mais ne calcule pas les fonctions
de transfert complexes. Dans la reconstruction complète, c'est
`systemDiagnosis.temporal_transfer_function()` qui initialise
`ttloop.tf.freq/wfs/servo/lag/rtf`.

P4-B utilise désormais `initialize_keck_loop_transfers(trs)`,
qui appelle **exactement cette même méthode P3** par une instance
de diagnostic dépourvue de calcul atmosphérique. Ainsi, on construit
les transferts instrumentaux sans déclencher la soustraction HO,
la reconstruction Zernike, ni les fits `r0/L0`.

Des tests synthétiques vérifient cette initialisation sur des
coefficients RTC bruts, l'identité du rejet avec le modèle
`unity` et la gestion d'un contrôleur incomplet.

**Il s'agissait d'un défaut d'orchestration du script, pas
d'une divergence démontrée du modèle STRAP.** Le message
`Cupy import failed; P3 will fall back to CPU use` ne bloque
pas l'exécution ; le calcul CPU est prévu par P3.

Après mise à jour de la branche, relancer les mêmes tests puis
la commande n0013 ci-dessus. Le nouveau rapport expose
`loop_transfer_initialization` et sa provenance.
