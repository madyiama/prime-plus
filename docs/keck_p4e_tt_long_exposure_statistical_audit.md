# P4-E — Audit statistique du jitter, contrôleur TT et bruit STRAP

## Décision

Pour n0120, les sept coefficients observés sont

```
DT_SERVO = [0.1, 0, 0, 0, -1, 0, 0]
```

Avec la convention `H(z)=B(z)/(1+a1 z^-1+...)` : `K(z)=0.1/(1-z^-1)`.

En négligeant les composants additionnels qui ne figurent pas dans
ce filtre numérique, le contrôleur satisfait

```
DTTCOMMANDS[k] - DTTCOMMANDS[k-1] = 0.1 * DTTCENTROIDS[k-d]
```

à un décalage d'horodatage `d` et à la conversion d'unités/axes près.

**Un résultat exact sur cette identité valide les unités relatives
et la chaîne de contrôle ; il n'est PAS une mesure indépendante du gain
optique STRAP.** Les deux signaux sont liés par le contrôleur.
Ne pas confondre `d` (offset d'enregistrement) avec le délai
physique miroir+WFS de Ragland.

## Pourquoi ne pas calculer un scalaire de correction depuis Δcommandes ?

Pour tout gain optique effectif `g`, le contrôleur voit les
centroïdes **déjà calibrés** `y=g*e+n` et produit les commandes
`u=K*y`. La loi `Δu=0.1*y` continue à être vérifiée, que
`g` soit égal à un ou non. Une estimation du facteur scalaire
sur la base de cette identité serait tautologique.

Pour reconstruire le jitter *physique* longue pose, on veut
`C_e=∫Phi_ee(f)df`. Avec `S_g=(1+gWPK)^-1` et une atmosphère
`d` indépendante du bruit capteur `n` :

```
Phi_ee = |S_g|² * Phi_dd + |P K S_g|² * Phi_nn
Phi_yy = |g W S_g|² * Phi_dd + |S_g|² * Phi_nn
```

Ces expressions doivent être évaluées sous hypothèses explicites
de `g`, `P`, `W`, de turbulence et de bruit. `Phi_dd` est
un paramètre de nuisance : si on le laisse arbitraire, les données
de centroïdes seules ne déterminent pas `g`. Un prior
**indépendant** sur la turbulence (p.ex. r0 issu de la voie
haut ordre, et un modèle de spectre temporel incluant vents et L0)
ou une calibration physique du DTT serait nécessaire.

Le `r0` Keck provenant déjà d'une POL HO est potentiellement
une contrainte externe utile si cette estimation n'exploite pas
elle-même l'amplitude TT à calibrer ; il ne faut pas utiliser un
`r0` ré-estimé des mêmes mesures TT comme étalon indépendant.

## Expérience livrée (sans modification du moteur PSF-R)

1. Lire `DTTCENTROIDS`, `DTTCOMMANDS`, `TIMESTAMP` et les
   coefficients `DT_SERVO` depuis le SAV ; refuser les
   filtres qui ne sont pas intégrateurs simples.
2. Sur les 70 % initiaux des données, ajuster
   `Δu[k]=b+B*y[k-d]` en balayant les offsets -10 à 10 trames.
   Évaluer sur les 30 % finaux, et **comparer au filtre nominal**
   `Δu[k]=0.1*y[k-d]` sans ajuster `B`.
3. Rapporter covariances de `Δu` et de `y` au même alignement,
   notamment `C_Δu` vs `0.1²*C_y`. Ces deux covariances
   ne sont pas indépendantes si le contrôleur suit sa loi.
4. Pour `g ∈ [0.5, 0.75, 1, 1.25, 1.5, 2]`, estimer le bruit
   de mesure avec **van Dam**, en re-calculant le facteur
   de rejet supposé pour chaque `g`, puis en refaisant
   l'estimation pour trois bandes hautes fréquences.
5. Les diagnostics rapportent la covariance du bruit et la composante
   atmosphérique **dans les unités de centroïdes enregistrées
   (arcsec² nominales)**. En complément, la fonction
   `ttConditionalCovariance.py` calcule une **famille de covariances
   résiduelles conditionnelles bande limitée**, pour chaque `g`
   supposé, selon le modèle mécanique provisoire ci-dessous.
   Elles ne sont PAS des covariances physiques calibrées du miroir Keck ;
   aucune n'est sélectionnée ni injectée dans le moteur PSF-R.

### Hypothèses strictes du scan de bruit

La fonction d'hypothèse est

```
q = exp(-i 2*pi*f/fs)
S_n(f;g) = (1-q)/(1+(0.1*g-1)*q)
```

qui correspond à `y[k]=g*(d[k]-u[k-1])+n[k]`,
`u[k]=u[k-1]+0.1*y[k]`. Elle suppose :
- WFS instantané `W=1` ;
- une unité de commande égale à une unité physique de correction ;
- un miroir parfait retardé d'une trame ;
- aucune perturbation hors du modèle ;
- bruit blanc dans la bande retenue.

**Ce ne sont pas les calibrations Keck 2020.** Le scan est
un diagnostic d'incertitude et non une estimation de `g`.
Le modèle spectraI physique Keck nécessite des transferts W/K/P
calibrés indépendamment, notamment pour éviter de confondre
le gain, le bruit et le transfert mécanique.

Les fonctions de `ttNoise.py` sont linéaires en unité ; leurs
diagnostics originaux sont étiquetés OPD m² pour leur usage P3.
Dans cet audit, l'entrée est directement `DTTCENTROIDS`
en arcsec, donc les valeurs sorties sont explicitement
étiquetées **arcsec² enregistrés**, pas m².

### Covariance conditionnelle longue pose : identité spectrale

Sous les hypothèses spécifiées et en supposant `Phi_dd` et `Phi_nn`
indépendants, les deux PSD satisfont :

```
Phi_yy = g² |S|² Phi_dd + |S|² Phi_nn
Phi_ee = |S|² Phi_dd + |T|² Phi_nn
S = (1-q)/(1+(g*k-1)*q)
T = -q*k/(1+(g*k-1)*q)
```

En éliminant `Phi_dd`, nous obtenons exactement :

```
Phi_ee = Phi_yy/g² + (|T|² - |S|²/g²) Phi_nn
```

L'intégration fournit `C_e(g)` par bande, en arcsec² conditionnels,
avec **aucune projection PSD**. Le code conserve les valeurs propres
négatives et le nombre de bins non physiques de `Phi_yy-|S|²Phi_nn`.
Cette formule évite le problème de division par `S(f=0)=0`.

**Avertissement longue pose :** l'intégration Welch couvre seulement
`[fs/nperseg, fs/2]`, pas la bande proche de DC. Le terme turbulent
lent, potentiellement important sur une exposition longue, est absent.
Il serait trompeur d'assimiler cette intégrale à la covariance absolue
de jitter longue pose. Les valeurs sont des *tests de sensibilité*.
Pour une PSF prédictive, il faudra compléter la basse fréquence et
étalonner les transferts optiques/mécaniques.

## Commande

```bash
git fetch origin
git switch --track origin/feat/psfr-p4e-tt-command-statistical-audit

uv run --with pytest python -m pytest -q \
  tests/test_keck_p4e_tt_long_exposure_audit.py \
  tests/test_keck_p4d_dtt_mechanical_audit.py

uv run python scripts/keck_p4e_tt_long_exposure_audit.py \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
  --output results/n0120_tt_statistics
```

Fichiers :
- `tt_statistical_audit.json` : matrices, délais, cohérence
  avec l'intégrateur nominal, sensibilité de van Dam aux hypothèses.
- `tt_statistical_audit.png` : qualité de reproduction du
  contrôleur et évolution des estimations de bruit en fonction
  des hypothèses de gain.

**Interprétation :** si `R²_nominal` est proche de 1, la
transformation incréments-commandes/centroïdes est confirmée
et ne livre pas de nouveau scalaire. Sinon, contrôler la
convention de signes, les unités, la dynamique du contrôleur,
les retards et les éventuelles composantes feedforward
avant d'interpréter un écart comme un gain optique.

## Après cet audit

Construire un **modèle de covariance TT longue pose conditionnel
à un gain `g`**, à partir des deux spectres observés, du niveau
de bruit van Dam, des transferts W/K/P et d'un modèle
indépendant `Phi_dd`. Un gain n'est déclaré mesuré que s'il
est *identifiable* et validé sur une autre acquisition, sans
sélectionner sa valeur sur la PSF NIRC2.
