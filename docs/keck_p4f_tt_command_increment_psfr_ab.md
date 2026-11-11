# P4-F — Test PSF-R Keck réel : TT centroïdes vs incréments de commandes

## Question scientifique

Pour une reconstruction Keck LGS/NIRC2 d'une **PSF longue pose**,
remplacer explicitement les slopes tip-tilt par

```
y_cmd[k] = (DTTCOMMANDS[k] - DTTCOMMANDS[k-1]) / DT_SERVO[0]
```

puis recalculer l'estimateur de bruit STRAP choisi, la covariance TT,
la fonction de structure TT, l'OTF et enfin la PSF.

**Résultat déjà attendu sur n0120 :** P4-E prouve
`ΔDTTCOMMANDS = 0.1*DTTCENTROIDS` avec une erreur RMS relative de
`2.564e-7`. En conséquence, les deux PSF devraient être identiques
à la précision numérique : **ce test ne constitue pas un nouvel
estimateur indépendant de l'erreur turbulente** et ne peut, à lui
seul, résoudre le sur-élargissement du jitter.

## Implémentation

Deux variantes sur **exactement les mêmes trames STRAP** :

- `centroids_aligned` : `B.DTTCENTROIDS[1:]` (trame zéro supprimée).
- `command_increment` : `np.diff(B.DTTCOMMANDS, axis=0)/0.1`.

La première trame est exclue des deux branches, car la commande
précédente n'existe pas dans la fenêtre d'archive. La variante
`centroids` reste disponible en tant que **chemin d'origine inchangé**
et demeure la valeur par défaut de `keck_n0013_smoke.py`.

Le module `p3/telemetry/ttCommandReplay.py` :
- impose le contrôleur `[g,0,0,0,-1,0,0]` ;
- valide l'identité numérique `Δu/g≈y` (RMS relatif < 1e−4) avant
  de qualifier les commandes en arcsec *nominalement équivalents* ;
- aligne les trames `slopes_arcsec`, `slopes`, `com`,
  `com_raw` et l'intensité APD si compatible ;
- utilise **`D*arcsec_to_rad/4`**, conversion Noll géométrique
  sans facteur historique 0,576 ni gain instrument non vérifié ;
- met à jour `tipTilt.nExp` et reste isolé du HO, du r0, de la
  pupille, de la calibration NIRC2 ;
- applique la substitution **avant** `systemDiagnosis`, de sorte
  que van Dam/Gendron calcule le bruit de la *nouvelle* série,
  pas le bruit de l'ancien signal.

Les résultats de la PSF, la covariance, le budget WFE et les
diagnostics sont générés normalement par `psfR`.

## Commande unique sur n0120

```bash
git fetch origin
git switch --track origin/feat/psfr-p4f-tt-command-increment-ablation

uv run --with pytest python -m pytest -q \
  tests/test_keck_p4f_tt_command_replay.py \
  tests/test_keck_p4e_tt_long_exposure_audit.py

uv run python scripts/keck_p4f_tt_source_ab.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --n-layer 7 \
  --p3-pupil-resampling physical_binary \
  --tt-profile reduced_jitter \
  --output results/n0120_tt_source_ab
```

L'expérience lance automatiquement **deux PSF-R complets** (même
FITS, SAV, pupille, r0, profil de bruit van Dam, HO, bandes) puis
enregistre :
- `results/n0120_tt_source_ab/centroids_aligned/` : reconstruction A
  et diagnostics complets ;
- `results/n0120_tt_source_ab/command_increment/` : reconstruction B
  et diagnostics complets ;
- `results/n0120_tt_source_ab/tt_source_ab_comparison.json` :
  écart relatif TT source, covariance d'entrée, covariance bruit
  van Dam, OTF TT, PSF relative L2, pic PSF ;
- `results/n0120_tt_source_ab/tt_source_ab_comparison.png` :
  modèles A et B à normalisation identique, carte de différence.

Pour reproduire la dernière expérience **Gendron** à paramètres
inchangés sauf la source, utiliser
`--tt-profile gendron_comparison`.

Il est aussi possible de passer les options du pipeline original,
par exemple `--tt-axis-parity flip_x`, qui seront transmises aux
deux exécutions.

## Interprétation

- **A ≈ B** : le bruit, la covariance résiduelle et la PSF sont
  invariants au choix des deux représentations : confirmé que les
  incréments des commandes ne donnent pas une mesure physique
  indépendante. Le gain optique STRAP et le bruit restent les
  incertitudes dominantes.
- **A ≠ B** : la relation numérique d'entrée est validée mais la
  PSF change. Il faut rechercher une erreur de normalisation,
  une divergence d'alignement, ou une dépendance insoupçonnée
  dans `systemDiagnosis`/`psfR`. Le script émet un rapport et
  sort avec un code non nul si la différence PSF L2 > 1e−4.
  Ce n'est pas une preuve que la PSF B est meilleure.
- **Le rapport de replay échoue** : les commandes ne reproduisent
  pas les centroïdes pour cette observation, on refuse alors de
  les convertir en arcsec sans calibration complémentaire.

**P4-F est une vérification A/B sur un cas réel, pas une correction
instrumentale.** Pas de calcul de gain optique ni de modification
du modèle PSF-R de production.

## État

PR exploratoire empilée sur P4-E (PR23). Les tests synthétiques et
l'exécution sur le SAV n0120 du Mac sont à valider avant merge.
