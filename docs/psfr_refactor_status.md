# État de la refactorisation PSF-R et bilan des optimisations — 14 novembre 2026

**Référence :** branche \`refactor/psfr-core\` (PR #30 vers \`main\`, actuellement **draft**).
**Optimisations récentes :** branche expérimentale
\`perf/psfr-focal-cache-profiling\` (PR #42, **draft** et base
\`feat/psfr-ao-parameter-fit\`, PR #39).
**Principe :** ne pas importer implicitement la pile de fitting dans
\`refactor/psfr-core\`. Les résultats de tests sur la pile performance ne
prouvent pas encore la validité scientifique absolue sur Keck.

## 1. État du code et des PR

| PR | Branche / rôle | État et décision |
| --- | --- | --- |
| [#30](https://github.com/madyiama/prime-plus/pull/30) | \`refactor/psfr-core\` → \`main\` | Draft ; conservée pour revue de périmètre et validations Keck multi-acquisitions |
| [#31](https://github.com/madyiama/prime-plus/pull/31) / [#32](https://github.com/madyiama/prime-plus/pull/32) | HO pur et assemblage de SF | Fusionnées **dans la branche de refactorisation**, pas nécessairement dans \`main\` |
| [#33](https://github.com/madyiama/prime-plus/pull/33) / [#35](https://github.com/madyiama/prime-plus/pull/35) | Profils spectraux et noyau optique chromatique isolé | Fusionnées dans la branche de refactorisation ; raccordement physique à valider |
| [#34](https://github.com/madyiama/prime-plus/pull/34) | Mesures d'initialisation, diagnostics \`psfr_timing_report\`, cProfile optionnel | **Fusionnée dans \`refactor/psfr-core\`** après CI verte ; aucune modification du modèle physique |
| [#36–#39](https://github.com/madyiama/prime-plus/pull/39) | Templates et fitting stellaire/AO | **Drafts séparées**, pas de fusion automatique avant validation des données et de la physique |
| [#41](https://github.com/madyiama/prime-plus/pull/41) | Mémoire HO et traces RSS | **Draft séparée** : hypothèse de SIGKILL, non confirmée par un test réel concluant |
| [#42](https://github.com/madyiama/prime-plus/pull/42) | Optimisations PSF-R expérimentales | **Draft** : inclut #44–#47, mais hérite du code de #39. Ne pas fusionner telle quelle dans \`refactor/psfr-core\` |
| [#43](https://github.com/madyiama/prime-plus/pull/43) | FFT HO en lots | Fermée sans fusion, abandonnée faute de gain strict démontré |
| [#44–#47](https://github.com/madyiama/prime-plus/pull/47) | Vii factorisé, FFT réelle, quadrant direct, CPU Ialpha | **Fusionnées dans #42** (branche expérimentale). Pas fusionnées dans \`main\` ni dans \`refactor/psfr-core\` |

Les PR fusionnées entre branches ne sont pas équivalentes à une publication
dans \`main\`. Ne pas supprimer les branches encore sources de PR drafts :
la reprise scientifique et l'intégration ciblée doivent rester traçables.

## 2. Architecture et améliorations effectives

### Déjà dans \`refactor/psfr-core\`

- Séparation graduelle des statistiques HO et de l'assemblage de la
  structure de fonction en fonctions réutilisables (#31, #32).
- Profils spectrophotométriques NIRC2 et noyau chromatique isolé
  (#33, #35) ; **ce noyau ne remplace pas encore le chemin V1 de Keck**.
- Mesures par étapes de l'initialisation PSF-R, helper
  \`p3.psfr.timing.psfr_timing_report\`, profilage cProfile opt-in
  (\`--profile-psfr\`, #34).
- Le modèle \`psfR\` historique reste l'adaptateur de production numérique ;
  l'architecture cible de \`docs/psfr_core_refactor.md\` n'est pas intégralement
  réalisée.

### Seulement dans la pile performance expérimentale (#42)

Les commandes et API décrites ci-dessous **ne sont pas encore dans**
\`refactor/psfr-core\` :

1. **#40 — six termes Flicker / symétrie :**
   \`p3/aoSystem/focalFast.py:symmetric_focal_layer\` évite des évaluations
   redondantes de la covariance radiale ; le chemin \`legacy\` reste disponible.
2. **#42 — cache de géométrie focale et chronométrage fin :**
   \`p3/aoSystem/focalCache.py\` ne met en cache que la structure de fonction
   géométrique LGS **non filtrée**, indexée par les paramètres
   physiques et le choix d'algorithme. Il ne stocke jamais la pupille
   NIRC2 tournée, l'OTF optique, les termes HO/TT, la NCPA ni une PSF finale.
   Un filtre spatial matriciel interdit ce cache.
3. **#44 — factorisation de Vii :**
   \`p3/psfr/hoFactorized.py:accumulate_vii_factorized\` regroupe
   algébriquement \`FFT(sum λ M²)\` sans éliminer les contributions de
   covariance. L'ordre des opérations change : vérifier la parité au niveau
   SF HO **et** PSF, et non promettre une identité bit à bit.
4. **#45 — FFT réelles unilatérales :**
   \`accumulate_vii_factorized_real\` utilise
   \`numpy.fft.rfft2/irfft2\` et des spectres hermitiens pour réduire
   le calcul/mémoire. Le chemin \`legacy\` reste la référence.
5. **#46 — quadrant focal direct :**
   \`direct_focal_quadrant_map\` évalue seulement les paires réellement
   agrégées, suivant les conventions historiques X/Y de
   \`vector_to_separation\`, sans allouer la matrice complète en
   \`n_ph**4\`. Équations Flicker inchangées ; toute matrice
   \`Hfilter\` non scalaire impose un retour au calcul complet.
6. **#47 — CPU Ialpha multi-threads :**
   \`parallel_ialpha\` conserve \`FourierUtils.Ialpha\` en float64,
   parallélise seulement les blocs indépendants de lignes et réassemble
   les résultats dans l'ordre avant les sommes. Valeurs disponibles :
   \`--focal-workers 1|2|4\`, **1 par défaut** ; le cas
   \`L0=+inf\` n'ouvre aucun pool car le noyau est analytique.
   La parallélisation ne s'applique qu'au mode focal \`quadrant\`.

**Aucune approximation interpolée de Bessel, projection PSD HO cachée,
modification du TT van Dam ou accélération GPU n'a été ajoutée au mode
nominal.** Les méthodes accélérées sont explicitement optionnelles.

## 3. Résultats disponibles et leurs limites

*Mesures fournies pour Keck/NIRC2 n0120 et benchmarks sur Mac ; ne pas
présenter comme des propriétés universelles du modèle.*

| Test | Configuration | Observation |
| --- | --- | --- |
| #46 — focal synthétique | nActu=7, 7 couches ; symmetry vs quadrant | **2,05×** sur la routine focale ; erreur L2 relative 3,14e-15 ; \`tracemalloc\` ≈81 % inférieur (pas RSS) |
| #47 — synthétique | nActu=10 ; 1, 2, 4 workers | 393 / 225 / 146 ms ; **2,69×** à 4 ; résultat identique bit à bit |
| #47 — synthétique | nActu=21 ; 1, 2, 4 workers | 6,56 / 3,75 / 2,50 s ; **2,62×** à 4 ; résultat identique bit à bit |
| #47 — Keck n0120 | 4 vs 1 worker, même méthode quadrant, aucun cache | 5,800 → 2,575 s pour le noyau focal (**2,25×**) ; cube PSF **128×128×3 identique bit à bit** ; flux/pics/centroïdes identiques |
| #47 — Keck n0120 | 1 vs 4 workers, échantillon unique de chaque | init PSF-R 13,388 → 11,796 s ; **ne pas inférer un gain global reproductible avec deux exécutions isolées** |
| Test V1 plus récent | n0120, HO FFT réelle / quadrant / 4 workers | ventilation d'initialisation ≈ **8,287 s**, dont HO SF 5,135 s et frequency-domain 2,327 s ; autre exécution, pas une comparaison contrôlée |
| Chronométrage externe | script \`keck_psfr_v1_e2e.py\` | **29,257 s** de temps écoulé mesuré ; inclut chargement, fichiers, validation et visualisation ; ne pas confondre avec \`psfr_init_total\` |

Ces chiffres viennent des rapports Mac transmis pendant la revue. Ils
ne sont pas des artefacts de CI committés au dépôt. La CI Python de #47
était verte (**856 passed, 97 skipped, 88 subtests**, run 43586962920),
ce qui ne remplace pas les essais instrumentaux sur plusieurs observations.

## 4. Reproduction

### Branche de refactorisation, protocole nominal V1

\`\`\`bash
git fetch origin
git switch refactor/psfr-core
git pull --ff-only
uv sync
uv run --with pytest python -m pytest -q
uv run python scripts/keck_psfr_v1_e2e.py \
  --fits /path/to/n0120.fits \
  --sav /path/to/n0120_LGS_trs.sav \
  --calib p3/aoSystem/data/KECK_CALIBRATION \
  --config config/psfr_keck_v1.toml \
  --output results/psfr_v1_nominal
\`\`\`

La branche cœur utilise ici **les options nominales**. Les options
\`--ho-vii-method\`, \`--focal-algorithm\` et \`--focal-workers\`
décrites ci-dessous appartiennent à la pile expérimentale.

### Branche expérimentale, réglages explicitement optimisés

\`\`\`bash
git switch perf/psfr-focal-cache-profiling
git pull --ff-only
uv sync
export P3_DISABLE_GPU=TRUE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
time uv run python scripts/keck_psfr_v1_e2e.py \
  --fits /path/to/n0120.fits \
  --sav /path/to/n0120_LGS_trs.sav \
  --calib p3/aoSystem/data/KECK_CALIBRATION \
  --config config/psfr_keck_v1.toml \
  --ho-vii-method factorized_real \
  --focal-algorithm quadrant \
  --focal-workers 4 \
  --output results/psfr_v1_optimized
\`\`\`

Ne pas utiliser \`--profile-psfr\` pour comparer les temps absolus.
Pour le profiling et les tests A/B : voir
\`docs/psfr_focal_cache_profiling.md\`,
\`docs/psfr_focal_quadrant.md\`,
\`docs/psfr_focal_ialpha_workers.md\` sur la branche expérimentale.
Le \`psfr_init_total\` est un temps **interne** au constructeur ;
le \`total\` de \`time\` est le processus complet.

## 5. Règles de clôture et reprise

- **Performance : chantier mis en pause.** Pas de PR de GPU,
  d'approximation de Bessel ou de nouveaux workers par défaut.
- **Ne pas fusionner #42 telle quelle vers #30** : #42 est empilée
  sur #39, elle-même sur les PR de fitting #36–#38. Pour intégrer les
  gains validés dans le cœur, ouvrir ultérieurement une PR **distincte
  et isolée**, ciblant \`refactor/psfr-core\`, en excluant fitting,
  puis vérifier CI et PSF-R Keck sur plusieurs rotations.
- **Ne pas clôturer prématurément #30 ou #41** : #30 requiert revue
  scientifique et contrôle des suppressions héritées ;
  #41 est une hypothèse de robustesse mémoire non démontrée.
- **Validation scientifique à poursuivre :** NGS/LGS, orientation
  pupille NIRC2, covariances HO non PSD sans correction silencieuse,
  bruit STRAP/TT et chromatisme. Une parité de calcul n'est pas une
  calibration instrumentale ni une preuve de Strehl/FWHM absolus.
- Les nouvelles PR doivent préciser **base**, **dépendances**,
  **config nominale**, **ce qui a réellement été testé**, et
  **critères d'acceptation restants**. Ne pas confondre fusion dans
  une branche expérimentale et fusion dans \`main\`.

Références techniques du cœur :
\`docs/psfr_core_refactor.md\`,
\`docs/psfr_keck_v1_release_acceptance.md\`,
\`docs/psfr_performance_profiling.md\`.
