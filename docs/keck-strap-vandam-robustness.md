# Sprint STRAP / van Dam — Robustesse statistique

## Situation

La PR #3 a introduit les estimateurs configurables van_dam (défaut),
gendron et legacy_autocorrelation. Le LGS Keck `n0013` donne après
correction TT des fonctions de transfert :

- `tr(C_measured)=3.22775e-12 m²`;
- `tr(C_n,sensor)=2.90473e-12 m²` (van Dam, bande 0.65–0.95 Nyquist,
  Welch 512);
- `g_S=0.97455` pour la propagation du bruit capteur vers les
  centroïdes mesurés;
- `pn=0.05058` pour la réponse de bruit utilisée par la PSF-R P3.

**Attention** : avoir une covariance PSD positive ne suffit pas à
démontrer la validité de l'estimateur; une petite variation de bruit
peut déplacer fortement le bilan atmosphérique après soustraction.

## Objectif de cette PR

Quantifier la dépendance des estimations de bruit à trois réglages
indépendants : bande haute fréquence, longueur des segments Welch,
et choix d'une sous-série temporelle. Cette PR ne modifie ni
`estimate_tt_noise`, ni la reconstruction `psfR` ni les
valeurs par défaut de `config/keck_tt_noise.toml`.

### Grilles par défaut

- bandes (fraction de la fréquence de Nyquist, pas de `fs`) :
  `[0.50,0.80]`, `[0.60,0.85]`, `[0.65,0.95]`,
  `[0.70,0.90]`, `[0.80,0.98]`;
- longueurs Welch : 256, 512, 1024 échantillons;
- séries temporelles disjointes : deux moitiés, trois tiers,
  quatre quarts du jeu complet.

Une estimation par bloc requiert au moins trois segments Welch,
avec recouvrement de 50 %. Sinon elle est enregistrée comme
`skipped` et n'entre **pas** dans les statistiques.

## Variables scientifiques produites

Toutes les covariances sont exprimées en `m²` de coefficient
Zernike RMS OPD; les entrées sont les centroïdes STRAP déjà
convertis en unités physiques par la PR #2.

Pour chaque configuration:

- `C_n,sensor` : covariance du bruit blanc de mesure **avant boucle**,
  inférée par le plateau de PSD après correction par `|S(f)|²`;
- `g_S C_n,sensor` : contribution prédite du bruit **dans les centroïdes
  résiduels**, avec `g_S=(2/fs) integral(|S|²df)`;
- `C_y-g_S C_n,sensor` : contribution attribuable au signal physique
  **dans les centroïdes mesurés**, pas encore le jitter angulaire
  sur la caméra;
- `C_y-(1+pn) C_n,sensor` : expression historique P3, conservée
  **exclusivement pour diagnostic**. Son statut n'est pas validé.

Les valeurs propres minimales sont enregistrées, **sans projection
PSD silencieuse dans le benchmark**.

## Métriques et incertitudes

La covariance van Dam est estimée indépendamment pour chaque bande
et longueur Welch sur la série entière et sur des sous-séries
disjointes. Les blocs produisent dispersion interblocs (écart-type)
et percentiles 16/84; il ne s'agit **pas** d'intervalles de confiance
à 68 % car les blocs peuvent être dépendants, le flux variable,
et certaines séries sont trop courtes. Le bruit spectral a été
ajusté sur les mêmes données, sans validation APD indépendante.

Aucune bande n'est automatiquement sélectionnée selon le Strehl
ou l'accord à LP415-20 (binaire).

## Reproduction sur Mac

Depuis la racine du dépôt :

```bash
git fetch origin
git switch --track origin/feat/strap-vandam-robustness

uv run --with pytest python -m pytest -q tests/test_tt_noise_robustness.py

uv run python scripts/keck_tt_noise_robustness.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
  --output "$PWD/results/n0013_vandam_robustness"
```

Options disponibles : `--band LOW HIGH` (répéter autant que
nécessaire), `--welch 256 512 1024`, `--partitions 2 3 4`,
`--min-segments 3` et `--config` pour lire la configuration
van Dam de référence.

Quatre fichiers sortent dans le répertoire demandé :

- `van_dam_robustness.png` : deux cartes d'évolution des traces,
  selon bande et Welch;
- `van_dam_robustness.csv` : toutes les configurations et
  blocs, les cas invalides marqués explicitement;
- `van_dam_robustness.json` : séries complètes des estimations
  et dispersions interblocs;
- `van_dam_robustness.md` : tableau des estimations et limites.

Les fichiers FITS/SAV restent dans le répertoire externe de l'utilisateur.

## Critères de validation du sprint

1. Tests sur données synthétiques à bruit connu et évolution
   temporelle lente; résultats déterministes.
2. Évaluation des mêmes données sans refaire la PSF.
3. Mesure des variations bande / Welch / sous-séries avec avertissement
   lorsque le nombre de segments est insuffisant.
4. Comparaison séparée des composantes capteur / mesure filtrée et
   du calcul P3 historique.
5. Une analyse des variations observées sur `n0013` avant de décider
   si les paramètres van Dam par défaut sont robustes.

## Étapes scientifiques suivantes, hors PR

- Simulateur de boucle fermée : prouver les fonctions de transfert
  `S(f)`, `T(f)`, `pn` et la séparation bruit de mesure /
  jitter angulaire de la PSF, sans hypothèse de covariance scalaire
  non validée;
- Validation indépendante de bruit de comptage et excès de bruit
  APD, avec données de calibration instrumentale;
- Comparaison multi-objets Keck seulement après ces validations.
