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
