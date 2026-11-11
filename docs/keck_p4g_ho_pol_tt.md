# P4-G — Contrainte HO POL → tilt atmosphérique → résidu TT Keck

## Objectif physique

Évaluer si la turbulence reconstruite par la **POL haut ordre LGS**
permet de contraindre **la covariance atmosphérique résiduelle TT dans la
direction STRAP (NGS)**, sans calibrer à l'aveugle les angles Keck et
sans ajuster la PSF NIRC2.

**Invariant scientifique absolu :** P3 gère déjà séparément
l'anisocinétisme NGS–science, via `psfR.dani_tt` dans
`psfR.dphi_ani`. Nous ne devons **jamais** ajouter une covariance
anisocinétique dans la covariance TT résiduelle STRAP, sans quoi on
compte deux fois le même terme. G5 est diagnostic uniquement.

Le code **ne modifie aucun composant de la PSF-R** :
`dphi_tt`, `dani_tt`, `dphi_ani`, `Cn_tt` ou paramètres `gtt`.

## Hypothèses et formules

### G1 — Tilt incident dans la direction STRAP

Nous utilisons directement les valeurs `r0_tel`, `L0_tel`
estimées dans `systemDiagnosis` à partir des covariances de Zernike
POL haut ordre. La phase turbulente a pour PSD spatiale
(cycles/m) :

```
W_phi(k) = 0.023 r0^(-5/3) [|k|² + (1/L0)²]^(-11/6)
```

Pour la pupille *réelle* Keck stockée dans le FITS
`NIRC2_MASK/`, le tilt est le coefficient du meilleur plan
de phase pondéré par la surface éclairée. Sa réponse spatiale
est `H_i(k)=M^-1_{ij} <x_j exp(-i 2π k·x)>_pupil`.
La covariance physique d'angle est

```
C_OL = [λ/(2π)]² ∫∫ W_phi(k) H(k)H*(k)ᵀ d²k.
```

Le facteur `λ²r0(λ)^(-5/3)` est indépendant de la longueur
d'onde utilisée pour exprimer `r0` lorsqu'on la convertit
cohérente­ment. Le code utilise celle de la POL (`atm.wvl`,
typiquement 500 nm), non la longueur d'onde K/H de NIRC2.

La quadrature spatiale **log-polaire** conserve les fréquences
sous la première case FFT, essentielles pour le tilt. Une somme
FFT rectangulaire simple à `512²` peut omettre la majorité de
la variance de basse fréquence. La normalisation est testée
contre l'intégrale circulaire indépendante en `J2` (Noll),
et le rapport Keck / cercle est indiqué. Il est conseillé de
comparer `--pupil-fft-pad 1024` et `2048` en convergence.

### G2 — Turbulence résiduelle TT en direction NGS

La boucle `W,K,P` provient de
`systemDiagnosis` et `ttClosedLoop.closed_loop_transfers` ;
la réponse physique et le gain optique restent des **hypothèses**.
Pour chaque couche (j), la fréquence temporelle du mode
spatial (k) est `f = |k · v_j|` sous l'hypothèse de Taylor
(*frozen flow*). On intègre `|S_g(f)|²` contre le spectre tilt
pour obtenir la **covariance atmosphérique résiduelle**, sans
anisocinétisme :

```
C_res,atm,NGS = Σ_j w_j ∫∫ W_phi(k) H(k)H*(k)ᵀ
                            |S_g(k·v_j)|² d²k * [λ/(2π)]².
```

La réponse `S_g=(1+g WPK)^-1` utilise le miroir `unity`
par défaut ; le modèle historique Ragland 2016 est une
comparaison possible, NON une calibration prouvée pour 2020.

Les profils de vitesse proviennent du code P3, mais **les
directions du vent `wDir` de Keck ont été initialisées à zéro
par défaut**, elles ne sont pas des mesures atmosphériques
indépendantes. G4 inclut donc une rotation commune des vents.
Au-delà de Nyquist, l'approximation utilisée ne prétend
aucune correction effective `S=1`.

### G3 — Covariance mesurée de STRAP sur la même bande

À partir des centroïdes archivés `y`, van Dam ré-estime
le bruit du capteur pour chaque modèle `g,W,K,P`. Le
cospectre mesuré est `P_yy`, et

```
P_res,atm,STRAP(f) =
    [P_yy(f) - |S_g(f)|² P_nn(f)] / [g² |W(f)|²].
```

Cette expression n'inverse pas `S_g` et s'applique aux
covariances **atmosphériques résiduelles**.
Le modèle HO et STRAP sont comparés uniquement sur
`[fs/nperseg, fs/2]`, pas sur leurs intégrales de bande
différentes. Les bins négatifs de cospectre et la
non-PSD éventuelle sont **conservés**, jamais projetés.

La fonction `C_res,atm` exclut le bruit capteur propagé.
G2 fournit séparément la **covariance totale prédite**
`C_res,atm + C_res,bruit` comme indication sous modèle
hypothétique. Le ratio G4 compare **atmosphère contre
atmosphère**, pas une covariance totale à une covariance bruitée.

### G4 — Facteur de cohérence, pas gain optique identifié

```
alpha(g) = tr[C_res,atm,HO(g)] /
           tr[C_res,atm,STRAP(g)] .
```

Le résultat `alpha(g=1)` est un **ratio conditionnel**
à comparer sur plusieurs acquisitions, et non un scalaire
à appliquer immédiatement à la PSF.
La sensibilité balaie des gains supposés `0.7,1.0,1.3`,
trois bandes van Dam, les incertitudes `dr0,dL0`
rapportées par la POL et différentes rotations de vent.
Les bornes de sensibilité ne sont pas des intervalles de
confiance postérieurs.

Ne **pas** assimiler `sqrt(alpha)` à un gain optique
mesuré : le vrai gain modifie aussi `S_g` et la propagation
du bruit. Aucun résultat G4 n'est injecté dans
`psfR.jitter_phase_structure_function`.

### G5 — Anisocinétisme séparé, aucune double comptabilisation

Si et seulement si la séparation NGS→science **en axes de la
pupille FITS** est fournie, G5 calcule :

```
C_delta = [λ/(2π)]² Σ_j w_j ∫∫ W_phi(k) H H*
                     2[1-cos(2π k·h_j Δθ)] d²k
```

`C_delta` est la **covariance du tilt DIFFERENTIEL**,
pas le résidu de correction du miroir. Elle n'est jamais
ajoutée à `C_res,NGS`, et n'est pas injectée dans le
jitter P3. **P3 calcule déjà `dani_tt`** et son
anisoplanétisme focal `dani_focang` séparément.

Lorsque la direction scientifique est éloignée de STRAP, la
covariance complète du résidu scientifique pourrait aussi
contenir un terme croisé
`Cov(res_NGS, tilt_sci - tilt_NGS)` ; additionner
`C_res_NGS+C_delta` sans l'analyser n'est pas exact.
Le terme de G5 est donc **diagnostique**, pour vérifier
l'ordre de grandeur existant, pas pour le substituer.

Aucune rotation mécanique/sky → axes du masque FITS n'est
calibrée par le script : G5 nécessite un `--science-offset-pupil-arcsec`
**déjà exprimé dans ce repère**. En l'absence de cet
offset, le fichier précise `science_NGS_separation_not_provided`,
plutôt que d'inventer une géométrie.

## Commandes pour n0120

```bash
git fetch origin
git switch --track origin/feat/psfr-p4g-ho-pol-tt-covariance

uv run --with pytest python -m pytest -q \
  tests/test_keck_p4g_ho_to_tt.py

uv run python scripts/keck_p4g_ho_to_tt.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --n-layer 7 \
  --mirror-model unity \
  --output results/n0120_p4g_ho_tt
```

Le fichier `results/n0120_p4g_ho_tt/p4g_ho_to_tt.json`
contient G1, G2, G3, G4, G5. La figure
`results/n0120_p4g_ho_tt/p4g_ho_to_tt.png` présente le
RMS atmosphérique prédit/mesuré sur bande commune, le ratio
`alpha(g)`, les sensibilités `r0/L0` et vents, ainsi que le
statut G5. **Le tilt de meilleur plan de Noll n'est pas
nécessairement identique à la mesure angulaire de la quad-cellule
STRAP** ; ce choix de projection constitue lui aussi une hypothèse
à valider avant calibration du gain.

Si l'offset est **déjà connu dans le repère de la pupille** :

```bash
# Exemple fictif : NE PAS copier des valeurs non calibrées !
# Ajouter uniquement vos véritables composantes de séparation :
--science-offset-pupil-arcsec DX DY
```

Pour étudier le modèle mécanique historique :

```bash
# Reprendre la commande complète ci-dessus en remplaçant :
--mirror-model ragland_2016_keck_ii
```

Et pour la convergence de la pupille :

```bash
# Reprendre la commande complète ci-dessus en ajoutant :
--pupil-fft-pad 2048 --radial-points 700 --azimuth-points 64
```

## Conditions avant d'interpréter alpha physiquement

1. Vérifier que les modes HO utilisés pour `r0,L0`
   n'ont pas de dépendance non modélisée au tilt LGS et que
   la focalisation à 90 km n'induit pas de biais important.
2. Valider la réponse instrumentale Keck `W,K,P`
   et les directions de vent, actuellement approximatives.
3. Refaire la comparaison de G3 avec plusieurs fenêtres Welch
   (`nperseg`) et bandes de bruit, sans sélection sur la PSF.
4. Comparer les ratios entre n0120, n0013 et plusieurs nuits ;
   les facteurs qui évoluent comme la météo ne sont pas
   des constantes de gain instrumentales.
5. Conserver l'anisocinétisme `dani_tt` et l'anisoplanétisme
   focal `dani_focang` existants, sans réinjecter `C_delta`
   dans le jitter STRAP.

## Références

- R. Noll, *Zernike polynomials and atmospheric turbulence*, JOSA 66 (1976).
- J.-M. Conan, G. Rousset & P.-Y. Madec, *Wave-front temporal spectra
  in high-resolution imaging through turbulence*, JOSA A 12,
  1559–1570 (1995), DOI:10.1364/JOSAA.12.001559.
- van Dam et al., *Angular anisoplanatism in laser guide star
  adaptive optics*, Proc. SPIE 6272 (2006).
- Beltramo-Martin et al., *Off-axis point spread function
  characterization in laser guide star adaptive optics systems*,
  MNRAS 478, 4642–4655 (2018).
