# P4 — vérification numérique complète de DTTCENTROIDS à OTF_TT

**Objet :** exclure un bug de conversion d'unité, de facteur 4/16,
d'échelle Fourier, d'angle, de covariance ou de normalisation dans
le moteur TT. **Ne démontre pas** que la covariance de bruit ou
le gain optique STRAP sont physiquement corrects.

## 1. Contrat de données et dimensions

La branche P4 utilise la télémétrie Keck LGS :

- `B.DTTCENTROIDS`: mesure des centroïdes STRAP en arcsec,
  deux colonnes. Le format d'archive et `APD_IMAT`
  reproduisent numériquement les centroïdes à ~1.5e-8 arcsec
  sur n0013. Il reste à étalonner l'optical gain.
- `tipTilt.slopes_arcsec`: copie intacte des données en arcsec.
- `tipTilt.slopes`: ancienne représentation Zernike RMS OPD
  centrée, utilisée par les diagnostics et estimateurs de
  bruit historiques ; `a_2,a_3=(D/4) theta_rad` en mètres.
- `tipTilt.Cn_tt`: bruit capteur **à son entrée**, en m²
  Zernike OPD, identifié selon van Dam/Gendron.
- `tipTilt.Cn_tt_arcsec2`: estimation **indépendamment
  répétée sur le tableau de centroïdes bruts** avec le même
  modèle spectral et le même transfert, en arcsec².
  Identité exigée : `Cn_arcsec2 = Cn_opd / gain²`.
- `C_theta`: covariance finale du mouvement résiduel
  sous le modèle choisi, en rad², dans le repère du capteur.
- `nu_row, nu_col`: fréquences de l'OTF en cycles/radian
  de l'image, explicitement transformées vers les axes STRAP.

Les deux voies sont comparées **à chaque reconstruction LGS**
où le TT est activé ; elles doivent coïncider à tolérance
numérique annoncée, sinon une erreur est levée.

## 2. Loi physique de l'OTF du mouvement d'image

Le mouvement d'image angulaire `theta` applique à la PSF
une translation. La moyenne sur une distribution gaussienne
de moyenne nulle et covariance `C_theta` en rad² donne :

```
OTF_TT(nu) = E(exp(-2πi nu·theta))
           = exp(-2π² nu.T C_theta nu)

Dphi_TT(nu) = 4π² nu.T C_theta nu

OTF_TT = exp(-Dphi_TT/2)
```

Ainsi, le noyau analytique n'utilise ni Zernike ni
`D` ni `lambda` **une fois les fréquences angulaires
et la covariance exprimées dans leurs unités finales**.
Le diamètre et la longueur d'onde ne servent qu'à
construire la correspondance entre échelle de pupille
et grille fréquentielle de l'OTF.

Si l'on passe par les Zernike RMS Noll,
`a = D theta/4` puis `theta = 4a/D`,
le produit des deux conversions est exactement l'identité.
En particulier, **le facteur 16 n'amplifie pas un
déplacement mesuré**. Supprimer le facteur 16 **sans
supprimer la première conversion D/4** conduit à un
jitter effectif quatre fois trop petit.

Le mode `legacy_unscaled_opd` ne sert plus que de
comparaison diagnostique; `noll_angular` est la
formulation physiquement cohérente si les centroïdes
sont bien en arcsec, sous l'hypothèse de gain optique.

## 3. Contrôle indépendant de la grille de fréquences

P3 définit `U_=2(row-N/2)/N`, `V_=2(col-N/2)/N`,
avec `N=nPix*kRef_integer`.
La séparation normalisée est `(sampRef/2)*(U,V)`.

```
nu = (D/lambda_ref)*(sampRef/2)*(U,V)
```

Avec `sampRef=kRef_float * lambda_ref/(D * pixel_rad)`,
la fréquence détecteur devient :

```
nu_row = (row-N/2)/(nPix*pixel_rad)
          * (kRef_float/kRef_integer)

nu_col = (col-N/2)/(nPix*pixel_rad)
          * (kRef_float/kRef_integer)
```

Le code vérifie cette égalité à l'exécution, y compris
si l'oversampling flottant diffère de l'entier
`kRef_`. Le calcul du pixel OTF doit utiliser
les mêmes `nu_row,nu_col` :
`sinc(nu_row*pixel_rad)*sinc(nu_col*pixel_rad)`.

Dans les tableaux NumPy, le premier axe correspond
aux **lignes**, le second aux **colonnes**. La
transformation physique des axes STRAP vers les
axes caméra est gérée séparément par
`tt_axis_angle_deg` et `tt_axis_parity`.
L'angle nominal de pupille peut servir de
référence de compatibilité, mais sa validité
STRAP→NIRC2 n'est pas acquise.

## 4. Comparaison indépendante des voies et invariants

Des tests couvrent :

1. `DTTCENTROIDS[arcsec]` → covariance temporelle
   biaisée `C_theta[rad²]`, indépendante du piston
   angulaire moyen, avec les deux termes croisés.
2. aller-retour Noll `D/4` → `4/D`, pour plusieurs
   diamètres, sans multiplication résiduelle par 16.
3. estimation van Dam sur les mêmes échantillons
   en arcsec et en OPD, y compris rejet W/K/P,
   vérification à `<2e-9` relatif.
4. calcul du jitter résiduel `legacy_psfr`
   directement en angles, puis du jitter
   `spectral_closed_loop` en reprenant la
   covariance native, avec contrôle exact de
   leur équivalence en unités OPD.
5. identité de l'OTF analytique gaussienne et
   de la formule historique `angular_jitter_structure_function`,
   et test sur une covariance anisotrope/corrélée.
6. `OTF_TT(0)=1`, `0<=OTF_TT<=1`, et
   `Dphi_TT(0)=0`, sans facteur caché.
7. transformée inverse de la gaussienne : une
   covariance injectée `2.6²` et `4.3²`
   pixels² est récupérée par moments de PSF à
   précision `1e-3` (simulation Fourier indépendante).
8. égalité des grilles de l'OTF pixel et TT.
9. passage une seule fois de `Dphi_TT` dans
   `SF=...+g_tt*Dphi_TT`, puis
   `OTF_Turb=exp(-SF/2)` à la longueur d'onde
   scientifique de référence.
10. détection positive d'une erreur volontaire
    de calibration `1.01` sur OPD/centroïdes,
    d'une fausse échelle pixel ou d'un gain
    de bruit injecté `+3%`.

Les invariants ne présument pas l'intensité des images
NIRC2 et n'utilisent jamais les PSF observées pour
choisir un gain de bruit.

## 5. Exécution sur données Keck

```bash
git fetch origin
git switch feat/psfr-p4-strap-jitter-validation
git pull --ff-only

uv run --with pytest python -m pytest -q \
  tests/test_keck_p4_angular_otf_chain.py \
  tests/test_keck_p4_tt_frame.py \
  tests/test_psfr_spectral_tt_kernel.py \
  tests/test_keck_tt_geometry_and_anisoplanatism.py

# Les quatre modèles restent strictement sur le même n0013.
uv run python scripts/keck_p4_tt_benchmark.py \
  --fits "/path/to/n0013.fits" \
  --sav "/path/to/n0013_LGS_trs.sav" \
  --calib p3/aoSystem/data/KECK_CALIBRATION \
  --output results/n0013_P4_angular_chain \
  --n-layer 7 --pupil-resampling physical_binary
```

Les sorties par variante avec TT activé comprennent
`p4_tt_angular_chain.json`, `p4_otf_tt.npy`,
`tiptilt_diagnostics.json`, `tt_noise_diagnostics.json`
et `p4_tt_metrics.json`. Le variant A0 sans TT
ne dispose pas d'un `OTF_TT` propre et indique
l'ablation explicitement.

Une erreur sur les échelles, les conversions ou la
norme OTF lève une exception, au lieu de pousser
une PSF dont le noyau est incertain.

**Exception volontaire :** les anciennes données
Keck décimées avec `decimation>1` ne sont pas
acceptées dans le replay LGS, parce que
`telemetryKeck` ne décime pas historiquement
`slopes_arcsec` et ne réajuste pas la
fréquence/les transferts de boucle. Cette
voie exige une correction séparée.

## 6. Ce qui reste incertain après les tests

Ces tests peuvent établir l'absence d'une incohérence
*algébrique et numérique* de la chaîne angulaire,
**pas** la valeur physique réelle de la covariance.

Restent non calibrés :
- gain optique des quad-cellules APD;
- relation entre comptages APD et photoélectrons,
  facteur de bruit d'avalanche et fond;
- modèle dynamique réel du miroir et de la boucle 2020;
- calibration absolue des axes STRAP→NIRC2;
- vibrations non communes entre WFS et NIRC2;
- validité de l'approximation gaussienne/stationnaire
  pour la distribution de mouvement;
- gain `gtt` si ajusté au lieu de 1;
- décomposition `C_y-(1+p_n)C_n` legacy, qui
  n'est pas en général identique à la reconstruction
  spectrale de la boucle.

Une fois les tests numériques passés, le prochain
chantier sera de vérifier **le bruit et les transferts
de boucle**, sans ajouter de facteur libre pour
forcer l'accord avec la PSF d'une binaire.


## 7. Échec réel sur n0013 : conversion float32 → OPD (11 novembre 2026)

Le premier benchmark P4 sur `n0013` s'arrêtait dans
`systemDiagnosis`, y compris pour `A0_no_tt`, avec :

```
P4 native-angular van Dam/Gendron noise and P3 Noll OPD
noise are inconsistent by 2.269615e-08 relative
```

Cette différence était petite mais supérieure au seuil
strict `2e-9`; elle n'indiquait **pas** un problème
physique dans le bruit du détecteur.

Cause reproductible : `scipy.io.readsav` peut fournir
`DTTCENTROIDS` en `float32`. Dans NumPy, multiplier
un tel tableau par un scalaire Python `float` de gain
mètres/arcsec peut **conserver le résultat en float32**,
puis la soustraction de la moyenne introduit des
arrondis supplémentaires. En parallèle, l'estimateur
du bruit natif P4 convertissait les données en float64.
L'homogénéité mathématique attendue des deux estimateurs
était alors perdue à ~2e-8 relatif.

Un essai indépendant avec 4 096 échantillons synthétiques
`float32` et Welch 512 a donné ~`2.51e-8` relatif
pour le chemin historique, contre une égalité au niveau
des arrondis machine pour la promotion avant multiplication.

Correctif P4 :
- `strap_centroids_arcsec_float64` conserve les valeurs
  archivées, en les copiant en float64 avant tout gain
  et avant centrage ;
- `telemetryKeck.restoringTelemetry` utilise ce tableau
  pour `slopes_arcsec` et `slopes` ;
- nouveau test avec archive synthétique float32,
  van Dam sur angles et OPD, comparaison stricte
  `<2e-9`, ainsi que contrôle que la version
  historique en float32 reproduit l'écart.

**Le seuil de validation n'a pas été relâché.**
La conversion ne peut pas récupérer la précision déjà
perdue lors de l'enregistrement initial en float32,
mais évite une seconde quantification artificielle
dans le modèle. Cette correction d'arrondi ne valide
ni le gain optique APD ni le budget de bruit physique.

Les quatre variantes doivent être relancées sur les mêmes
données avec le code corrigé. Les tests réels et la CI
de ce commit restent à confirmer.
