# PRIME+ — Audit théorique V1 (périmètre borné)

Date : 2026-11-09. Base de la PR #8 : `main`, après fusion des PR #5, #6 et #7.
Objectif : vérifier que les **formules, unités et implémentations essentielles** sont plausibles, que les bugs manifestes sont corrigés et que les hypothèses non calibrées restent explicites. **Pas** d'audit exhaustif de tous les postes WFE avant les essais Keck multi-acquisitions.

## Référence de PSF-R à figer

- STRAP : bruit capteur **van Dam + `unity`** (hypothèse P3) par défaut. `ragland_2016_keck_ii` uniquement en sensibilité, avec rejet van Dam recalculé pour le miroir choisi.
- APD analytique / MC Poisson : comparaisons conditionnelles, **pas** des paramètres instrumentaux 2020 mesurés.
- Jitter TT spectral physique : mode **opt-in expérimental** ; ne pas assimiler variance des centroïdes STRAP et jitter NIRC2.
- Aucune optimisation de la PSF sur l'image binaire `n0013.fits`. Le fitting scientifique restera un module distinct.

## Résultat de revue V1 du code (inspection statique)

| Terme | Formulation / diagnostic | Verdict code |
| --- | --- | --- |
| OTF/structure de phase | `OTF=OTF_DL exp(-Σ Dφ/2)`, Véran et al. (1997) | Structure théorique adaptée. Contrôler le zéro de SF, la normalisation et la PSD modale. |
| Résidu HO | `psfR.residual_phase_structure_function`, `psfrUtils.modes2Otf` | **Deux bugs corrigés dans #8** : SF non initialisée si `Cmm=0`; SVD et orientation d'eigenmodes sur covariance potentiellement indéfinie. `eigh`/refus des valeurs propres significativement négatives, tests ajoutés. |
| DM fitting | `0.0229 r0^{-5/3}(k²+1/L0²)^{-11/6} M_out(k)` | Hypothèse théoriquement classique : turbulence hors bande de contrôle avec coupure spatiale DM idéalisée. Tests d'exposant et de quadrature PSD/SF ajoutés, encore à exécuter. |
| Aliasing spatial | `fourierModel.aliasingPSD` et `psfR.aliasing_phase_structure_function` | Répliques de fréquences WFS + filtrage de reconstruction : architecture plausible. Normalisation `r0^{-5/3}` compatible avec le code lu ; géométrie/gain Keck restent hypothétiques. |
| Anisoplanétisme LGS | focal-angulaire **conjoint** + anisokinétisme, `Cn2` et conversion de longueur d'onde | Architecture conforme à la décomposition PRIME. **Bug de terme croisé XY corrigé dans #8**, test synthétique ajouté. Les conditions de turbulence, angle et hauteur 2020 restent conditionnelles. |
| Pixel, NCPA, pupille | sinc pixel et OTF statique | Formulations standard; ne pas confondre `NCPA=0` dans un audit avec une calibration instrumentale. Pas de développement approfondi en V1. |

La revue est **statique**. Elle ne démontre ni que les tests nouveaux passent, ni que le modèle reproduit fidèlement l'instrument 2020. Les corrections #8 modifient le code scientifique et doivent être validées localement avant merge.

## Conditions de clôture V1 (cinq contrôles, pas un programme de recherche)

1. Tests ciblés `tests/test_psfr_spatial_physics_contract.py` verts ; ensemble de la suite `pytest -q` vert. Le seul échec antérieur (`__new__` sans `tt_covariance_mode`) a déjà été corrigé en PR #7.
2. Exécution d'un `n0013` en ablation TT et en mode TT de référence, avec diagnostics des termes, sans nouveau réglage sur la binaire.
3. Aucune covariance HO/TT significativement négative **silencieusement rendue positive**. Si la nouvelle vérification HO échoue, c'est une non-conformité de modèle/bruit à examiner ; elle bloque cette PR, pas le développement de l'archive Keck.
4. Conserver les unités, la provenance, les modes et la configuration utilisés avec les résultats.
5. Fusionner #8 sur `main` et marquer V1 **théoriquement sourcée / numériquement vérifiée**, mais **instrumentalement non calibrée**.

## Ce qu'on reporte après V1

Calibration physique des gains STRAP et du miroir TT 2020, modèle de ciel APD temporel, identification complète des profils `Cn2(h)`, chaque terme détaillé du budget WFE, réglage NCPA, intégration PSF fitting, étude des effets de champ et du Strehl sur plusieurs images. Ces travaux ne sont **pas** des prérequis pour lancer un premier benchmark multi-acquisitions, qui doit rester conditionnel et documenté.

## Références principales

- Véran et al., JOSA A 14 (1997), 3057 : reconstruction PSF depuis les mesures de boucle, DOI 10.1364/JOSAA.14.003057.
- Beltramo-Martin et al., MNRAS 487 (2019), 5450 : PRIME/Keck, séparation fitting, aliasing et anisoplanétisme, https://academic.oup.com/mnras/article/487/4/5450/5519866
- Ferreira et al., A&A (2018), DOI 10.1051/0004-6361/201832579 : approximation du fitting `~0.23(d/r0)^(5/3)` sous hypothèses spécifiques.
- `docs/psfr-physics-traceability.md` : inventaire des sources et niveaux théorie/numérique/instrument.


## Complément d'audit V1 : code TT / HO et cohérence des unités (2026-11-09)

**Important : aucune justification théorique pour réintroduire les anciens
coefficients empiriques STRAP `0.18` et `3.2`.** Le noyau gaussien a
sa propre dérivation et ses propres tests ; l'éventuel écart à NIRC2 ne
justifie pas d'ajustement de gain sur une image binaire.

### Autres termes de PSF-R

- Le `r0` du modèle direct est **déjà converti** à la longueur d'onde
  scientifique : `fourierModel.__init__` affecte `ao.atm.wvl=freq.wvlRef`
  et le setter `atmosphere.wvl` applique `r0 ∝ lambda^(6/5)`. Ne
  PAS ajouter de nouveau facteur `(lambda_atm/lambda_sci)^2` au fitting
  ou à l'aliasing. Les commentaires de `psfR` sont moins explicites
  que cette mutation et doivent être interprétés dans leur contexte.
- Fitting : PSD von Kármán intégrée en dehors du cutoff DM; aliasing :
  PSD repliée dans la bande de correction ; anisoplanétisme :
  approximation focal-angulaire + anisokinétisme ; pixel :
  sinc de réponse de pixel. Ils suivent les constructions PRIME. Le
  caractère exact du filtre DM et les paramètres de couches Keck ne sont
  pas validés par une seule observation.
- HO temporel : deux erreurs analytiques constatées dans
  `systemDiagnosis.temporal_transfer_function` (ancienne intégration
  `sinc(f/(πfs)) exp(-if/fs)` et délai `exp(-2if τ)` au lieu de
  `sinc(f/fs) exp(-iπf/fs)` et `exp(-2πifτ)`) ont été corrigées dans
  cette PR en réutilisant les helpers STRAP testés.
- Les grilles fréquentielles HO et TT commencent maintenant à
  `f_min=fs/N` via `np.fft.rfftfreq(...)[1:]` ; auparavant la fréquence
  initiale subissait une seconde division par `N/fs` (~0.00085 Hz
  contre ~0.0292 Hz pour STRAP n0013). Les tests analytiques ont été
  ajoutés ; impact observationnel à mesurer, probablement limité pour
  la TT, mais l'ancienne convention était mathématiquement incorrecte.

### Chaîne STRAP vérifiée par étapes

1. `DTTCENTROIDS` reproduit numériquement depuis les quatre APD et
   `APD_IMAT`. `a_OPD = D * theta / 4` [m] si `theta` est une
   véritable position angulaire [rad]; cela ne valide pas, à lui seul,
   le gain optique de la quad-cell sur le ciel.
2. `C_y=Y^T Y/N` avec `Y` centré : covariance des centroïdes mesurés,
   et NON covariance du mouvement physique de la cible.
3. Van Dam : spectre Welch haute fréquence `P_y ≈ |S|² P_n` où
   `P_n=2 C_n / fs`. L'implémentation de la densité unilatérale
   et de la covariance hors-diagonale est cohérente sous l'hypothèse
   bruit blanc et faible turbulence dans la bande de fit.
4. En mode `legacy_psfr`, quand `C_y-(1+pn)C_n` est PSD,
   `C_legacy=[C_y-(1+pn)C_n] + pn C_n = C_y-C_n`.
   Ce résultat **n'est pas** le résidu physique exact pour une boucle
   dont le capteur mesure `S n`.
5. En mode spectral, `y = W S d + S n`, `e=S d-PKS n` impliquent
   `P_e=[P_y-|S|²P_n]/|W|² + |PKS|²P_n`. Ce calcul, ses unités,
   et les tests synthétiques sont corrects ; sa validité instrumentale
   est conditionnelle à W, K, P et au gain du centroïde.
6. `C_theta=(4/D)^2 C_e`, `Dphi=(2πD/λ)^2 u^T C_theta u`,
   `OTF_tt=exp(-Dphi/2)`. Cette transformation est exactement la
   transformée de Fourier de la convolution par un jitter gaussien.
   Les tests géométriques préexistants la contrôlent directement.

### Mesures conditionnelles n0013 (modèle `unity`, van Dam)

- `tr(C_y)=3.2277523e-12 m²`,
  `tr(C_n)=2.9047296e-12 m²`,
  `g_S=0.97454715`, `g_physical=0.05057986`.
- Le bruit filtré attribué aux centroïdes vaut 87.7 % de leur trace.
- `tr(C_legacy)=3.2302275e-13 m²` => jitter RMS moyen par axe
  `31.58 mas`, FWHM gaussienne équivalente `74.36 mas`.
- `tr(C_spectral)=5.4955993e-13 m²` => `41.19 mas` par axe,
  FWHM équivalente `96.99 mas`.
- Approximation `W=1` : `tr(C_e)≈tr(C_y)-(g_S-g_physical)tr(C_n)
  =5.43877e-13 m²`, proche de l'intégrale spectrale complète.
  Cette cohérence **écarte un bug évident de facteur 4/16** dans la
  propagation géométrique ; elle ne certifie pas le bruit mesuré.
- À `C_y` fixée, réduire `C_n` **augmente** le jitter spectral
  reconstruit (sensibilité `g_physical-g_S≈-0.924`) : ne pas
  supposer qu'une surestimation du bruit explique un jitter trop
  grand par défaut.
- L'image `n0013` représente une binaire, donc le `NIRC2 SR`
  dérivé sans modèle de source n'est pas une vérité de Strehl
  isolé. Une comparaison qualitative ne valide aucun gain STRAP.

La V1 peut conserver le mode de TT spectral comme conditionnel et
le mode legacy comme référence reproductible. Ne pas les confondre
avec des estimations instrumentales certifiées. Suivi : issue #10.

### Tests de cette mise à jour

```bash
uv run --with pytest python -m pytest -q tests/test_tt_transfer_functions.py \
  tests/test_tt_closed_loop_formalism.py \
  tests/test_psfr_spatial_physics_contract.py
uv run --with pytest python -m pytest -q
```

Aucun test de ce nouveau commit n'a été exécuté par l'assistant.


## Contrôle complémentaire — espace fréquentiel et longueur d'onde (09/11/2026)

Périmètre : `psfR` Keck `psdExpansion=False`, sans fitting d'image.

- `frequencyDomain.PSDstep = 1/(D*sampRef)` [m⁻¹],
  `nOtf=nPix*kRef`, `U=2(q-nOtf/2)/nOtf`.
- Échantillonnage caméra `p` [rad/pixel] :
  `sampRef = kRef*lambda_ref/(D*p)`, donc la fréquence angulaire
  `(D/lambda_ref)*(sampRef/2)*U =
  (q-nOtf/2)/(nPix*p)`.
  **Le diamètre et la longueur d'onde se compensent exactement dans
  l'échantillonnage du noyau de jitter gaussien à échelle angulaire
  de pixel fixée.** Ce dernier n'est pas une WFE en phase à
  re-normaliser comme `r0`.
- `DTTCENTROIDS` présumés en arcsec ; `a = D theta / 4` [m RMS
  Zernike], puis `Ctheta = (4/D)^2 C_OPD`. Ces deux conversions
  s'annulent algébriquement, sous réserve que l'échelle réelle des
  centroïdes soit angulaire. L'APD_IMAT n'établit pas cette calibration.
- `psfR` traite aujourd'hui ses 3 composantes de bande avec
  **une seule SF/OTF calculée à `wvlRef=min(wvl_bins)`**, puis
  des poids photométriques. C'est une approximation monochromatique
  à la longueur d'onde de référence, pas une vraie intégration de
  trois PSF chromatiques. Ce n'est pas l'origine du surplus de
  jitter angulaire, mais cela limite la précision du Strehl/premier
  anneau de la PSF en bande large.
- Correction scientifique additionnelle : `atmosphere.wvl` modifiait
  `atm.r0` sans modifier les Fried lengths `atm.layer[l].r0`.
  L'appel `atm.slab()` après la conversion 500 nm -> bande K
  devenait incohérent. Le setter convertit désormais simultanément
  tous les `r0`, avec contrôle aller-retour 500 nm <-> K.
  Sur le chemin Keck actuel, les noyaux anisoplanatiques normalisés
  sont préparés avant le changement de wavelength ; ce bug n'est
  donc pas établi comme cause du flou TT de `n0013`.
- Diagnostic WFE corrigé : pour `TOTAL ANISOPLANATISM`, la longueur
  d'onde au dénominateur doit être `freq.wvlRef` et non la
  longueur d'onde centrale `ao.src.wvl[0]`, afin de correspondre
  au forward SF.

Le script `keck_n0013_smoke.py` écrit maintenant
`frequency_grid_diagnostics.json` avec les paramètres de grille,
la longueur d'onde de référence, le pixel scale prédit/mesuré et
la conversion STRAP. Le test synthétique
`test_jitter_otf_frequency_grid_has_correct_camera_pixel_scale_at_any_wavelength`
vérifie indépendamment le noyau aux longueurs d'onde 1.65,
2.055501 et 2.35 µm.

**Limite instrumentale non levée :** la preuve des unités en arcsec
et du gain optique STRAP en 2020 requiert une calibration de
l'interaction miroir/centroïdes ou une méthode indépendante ; le
simple accord avec l'image binaire ne suffit pas.
