# Keck P1 final — audit scientifique de r0/L0 et procédure corrigée

**Décision projet du 10/11/2026** : ne pas changer maintenant
l'estimateur opérationnel Zernike. P3 (pupille) se poursuit en priorité.
Le présent audit préparatoire est **read-only** ; l'implémentation
définitive du r0/L0 sera reprise **en fin de refonte**, une fois
les autres termes stabilisés.

## 1. État exact et mesure n0120

Résultats fournis après PR16, 500nm en ligne de visée :

| Estimateur | r0 | L0 | RMS log modale | Statut |
|---|---:|---:|---:|---|
| Zernike historique, Noll 4..264 | 11.4740 cm | 31.05 m | biais hauts ordres constaté | provisoire |
| Jolissaint modifié, grille 100, 75 modes | 10.1559 cm | 10 m (borne) | 0.17625 | expérimental |
| Jolissaint modifié, grille 201, 75 mêmes modes | 11.0682 cm | 22 m | 0.09526 | expérimental |

La convergence 100/201 vaut **8.9836% en r0**, > seuil
exploratoire de 5%. 201 améliore la forme du fit, mais le
`L0` est faiblement identifiable : coût 15m vs 22m
écart ~0.00018236 seulement. La PSD Welch des commandes
intégrée donne -16.0%/-16.1% par rapport à la variance
empirique sur deux grilles : vérifier le contenu spectral,
la fenêtre/detrending et la stationnarité avant d'inverser
un transfert temporel. La fraction d'énergie des commandes
dans la zone `|CTF|² < 0.5` est seulement ~0.00042
sur cette observation : le déficit Welch **ne démontre pas**
en lui-même une atténuation par boucle de 16%.

Le facteur Keck historique de conversion DM est
`0.4095e-6 m/volt` dans `telemetryKeck.py` ; Ragland
2017 (AO Telemetry for PSF-R, tableau 1) donne
`0.6e-6 µm/volt` pour les unités DM. Ne jamais
remplacer ce nombre sans étude de la convention
mouvement de surface/OPD, du RTC et de l'époque.

## 2. Les trois tests exécutables préparés

### Gate A — signe et décalage pseudo-open-loop

`sample_shifted_residual` teste les hypothèses
`u_pol(t)=u_dm(t) +/- res(t+delta)`,
avec `delta = delta_nominal + {-1,0,+1}` trame.
Interpolation sans wrap, mêmes 4096 trames
sous-échantillonnées de toute l'exposition pour 6 tests ;
mêmes Zernike et L0 fixe. Il produit :
- signe, retard, variance modale, r0 correspondant *à L0 fixé*,
  métrique de forme, nombre de trames réellement utilisées ;
- politique `rtc_sign_independently_validated=False` ;
- aucun choix automatique sur meilleur coût.

**Critère A** : l'interpolation est numériquement exacte sur
injection synthétique et ne mélange aucun actionneur ;
l'analyse démontre la sensibilité du résultat à la convention.
Pour fermer *physiquement*, obtenir les timings du RTC Keck
(commands, WFS residual, integration, latence) et établir le
signe/calibration via une impulsion ou une série IMAT
indépendante. Sinon statut `unverified`, même si les coûts
varient peu.

### Gate B — plage des ordres et pondération du fit

À `L0` du fit historique figé, comparer les ordres
3–10, 3–12, 3–14, 15–22 et 2–22 :
- médiane de variance par ordre radial comme la production ;
- fit amplitude en pertes absolues puis logarithmiques,
  avec loi physique `Var ∝ r0^(-5/3)` ;
- distribution des rapports observé/théorie par ordre ;
- RMS des log-ratios après amplitude optimale.

**Critère B (exploratoire, PAS seuil de calibration)** :
les plages centrales 3–10/3–12/3–14 et les fonctions de
perte ne doivent pas déplacer r0 de plus de 5% si l'on
souhaite déclarer stable ce modèle. Si le seuil est
dépassé, conserver le diagnostic `order-sensitive` et
abandonner le 2–22 sans restriction pour une estimation
calibrée. Ne jamais supprimer de modes uniquement pour
faire correspondre le DIMM.

Cette vérification utilise le modèle **historique** ;
une validation finale doit répéter le fit sur le même
espace de Zernike **modifiés par les fonctions
d'influence DM**, un anneau 9m/2.65m et un modèle
de covariance projeté identiquement à la mesure.
Le masque 75 modes de Jolissaint 2018 est une **prior**
historique à revérifier sur une campagne 2020.

### Gate C — unités et projection DM/Zernike

- consigner gain OPD/volt et ratio à l'alternative publiée ;
- montrer la sensibilité `r0 ∝ gain^(-6/5)`
  **si toutes les amplitudes de commande et résidus
  changent avec le même gain** ;
- consigner dimensions du DM, 11.25m de support
  IF, 10.5m de `tel.D`, cond. de Gram, projection
  d'une commande DM uniforme dans les modes ;
- rappeler que les IF 100/201 se réfèrent au même
  modèle physique Keck Xinetics mais ont une
  convergence numérique insuffisante.

**Critère C** : `OPD m/volt` indépendant, cohérent
entre DMCOMMAND et RESIDUALWAVEFRONT, conversion
phase `2pi/lambda` une fois et uniquement une fois,
pupille physique confirmée, absence de fuite
piston/TT sous la projection. Sinon pas de
validation absolue r0, même si le fit est beau.

## 3. Exécution demain : un seul checkout

La branche d'audit est une fille de PR17/P3 : elle
contient aussi les tests P3, sans altérer la production.

```bash
git fetch origin
git switch --track origin/feat/psfr-scientific-review-r0-final-gates

uv run --with pytest python -m pytest -q \
  tests/test_keck_p3_pupil_audit.py \
  tests/test_keck_r0_final_three_gates.py

# P3 : lancer la diffraction sur une observation
uv run python scripts/keck_p3_pupil_audit.py \
  --fits "/chemin/vers/n0120.fits" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --output "results/n0120_P3_pupil"

# Contrôles P1 restants, recherche uniquement
uv run python scripts/keck_r0_final_three_gates.py \
  --fits "/chemin/vers/n0120.fits" \
  --sav "/chemin/vers/n0120_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --output "results/n0120_r0_final_three_gates" \
  --n-layer 7 --max-frames 4096
```

Livrables :
- `results/n0120_P3_pupil/p3_pupil_diffraction_hypotheses.png`;
- `results/n0120_P3_pupil/p3_pupil_orientation_audit.json`;
- `results/n0120_r0_final_three_gates/r0_final_three_gates.json`.

La branche est en brouillon. Si les tests ne passent pas,
rapporter traceback et fichier de sortie. Ne pas merger
comme nouvelle calibration physique.

## 4. Procédure r0/L0 corrigée — cible de fin de refonte

### 4.1 Choisir une observation réellement exploitable

Vérifier FITS science et `.sav` correspondants, temps
UTC/exposition, mode NGS/LGS, fréquence WFS/STRAP,
`WSSMPRG`, état boucle, matrice de reconstruction,
gains, référence absolue du DM, marqueurs de saturation,
qualité subaps et longueur d'onde. Garder 500nm
comme référence de publication et préciser la ligne
de visée (pas une deuxième correction airmass).

### 4.2 Obtenir une chaîne instrumentale identifiée

1. Valider `DMCOMMAND`, `RESIDUALWAVEFRONT`,
   `OFFSETCENTROID` et `TTCOMMANDS` : unités
   calibrées, OPD vs surface, signe, actuateurs valides,
   et ordres temporels (timestamps).
2. Reconstruire une véritable WFS pseudo-boucle-ouverte
   indépendante si la matrice d'interaction et le gain
   centroïde sont disponibles ; sinon conserver une
   approche **DM-command** explicite, sans l'appeler
   pente pseudo-open-loop.
3. Si pseudo-OL choisie :
   `s_POL(t)=s_res(t)+/- D_interaction u_DM(t-τ)`
   avec le **signe et le retard validés** par la convention
   RTC, pas par le Strehl de n0120.
4. Projeter un masque de pupille instrument exact sur
   une base de Zernike **DM-modifiés** (Jolissaint) ;
   imposer same IF / mêmes filtres spatiaux au
   modèle et aux données.

### 4.3 Un seul modèle statistique mesuré/prédit

Proposer pour les variances d'un ensemble de modes
admissibles (indice i) :

`V_i(obs) = (D/r0)^(5/3) M_i(L0; IF,pupil,RTC)
            + V_i(noise) + V_i(alias) + V_i(vib)
            + V_i(other)`.

Le modèle `M_i` doit inclure la projection
sur les Zernike modifiés et la réponse de boucle
**réellement propagée**. Un spectre temporel ne
peut pas être relié à une covariance purement
spatiale sans un modèle de vent et de dynamique.
Pour la voie DM-commands :
- calculer et contrôler les PSD/cross-PSD des commandes
  par blocs stationnaires ;
- reconstruire le niveau de bruit depuis la réponse
  WFS/RTC et un plateau HF défendable, puis le
  propager **dans la même base modale** ;
- ne pas utiliser une covariance de différences de
  commandes DM comme bruit du capteur indépendant ;
- déduire l'aliasing soit du modèle Shack–Hartmann
  instrument calibré, soit d'une simulation adaptée
  aux époques Keck; l'ancien polynôme de 2018 ne
  doit pas être automatiquement appliqué à 2020 ;
- contrôler `C_observed-C_noise-C_alias` :
  symétrie et PSD, pas de réparation silencieuse.

### 4.4 Fit identifiable et erreurs de calibration

- `r0` à `L0` fixé sur plusieurs valeurs
  physiquement plausibles, pour séparer amplitude
  et forme ; puis profil de vraisemblance en `L0`.
- Pondérer selon l'incertitude des variances par
  blocs temporels/cross-modal (pas selon les
  seules grandes variances du bas ordre).
- Tester 3–10,3–12,3–14 et masque 75 modes 2018
  requalifié, en suivant les modes exclus et leur
  SNR/response instrumentale.
- Tests de convergence 100/201/261 ou autre
  équivalent ; seuil 5% sur r0 n'est qu'un
  critère numérique, jamais une précision absolue.
- Rééchantillonnage **par blocs temporels**
  pour les intervalles de confiance ; modéliser
  les corrélations modales. Les seules erreurs
  `least_squares(...).jac` ne garantissent
  pas des barres physiques en présence de
  résidus corrélés/mal spécifiés.
- Si `L0` profile reste plat / borne active,
  produire `L0_unconstrained` ou estimer
  r0 conditionnellement à des `L0` plausibles.
  Ne pas publier 22m comme mesure précise.

### 4.5 Validation indépendante et multi-acquisitions

1. Injecter turbulence von Kármán simulée de
   `r0,L0` connus dans une simulation AO avec
   bruit, alias et latence ; récupération
   `r0,L0` sans biais sur différentes magnitudes,
   SNR, vitesses de vent et modes NGS/LGS.
2. Comparer idéalement à des PSF en **boucle ouverte
   quasi simultanées dans le même axe**, comme dans
   Jolissaint 2018, **et non à un fit de la
   science PSF NIRC2 que l'on reconstruit**.
3. Ensuite utiliser MASS/DIMM datés comme
   témoins externes : corrélation, biais,
   dispersion et bootstrap sur de nombreuses
   acquisitions ; vérifier météo/coupole/airmass.
   Le site DIMM n'est pas une cible d'égalité.
4. Ne promouvoir l'estimateur en production que
   si résidus stationnaires, covariance PSD,
   observabilité L0, robustesse aux modes,
   paramètres instrumentaux calibrés et erreur
   empirique mesurée sont documentés.

## 5. Conditions de clôture

**Gate tests demain :** tests unitaires verts,
rapports produits, toutes les anomalies
visibles et quantifiées. Cela clôture la **préparation
de l'audit**, pas la calibration absolue.

**Gate correction du code (fin projet) :**
supprimer/encapsuler les erreurs connues (lag
wrap LGS, soustraction de bruit constante à
tous les retards de `get_wind_speed`, projection
idéalement incohérente), avec tests de
non-régression avant/après et nouveau modèle
Keck validé. Les valeurs de r0 issues de
l'ancien code restent archivées.

**Gate physique :** comparaisons simulations
et observations indépendantes sur une
campagne multi-acquisitions ; résultats
séparés NGS/LGS ; incertitude établie.
