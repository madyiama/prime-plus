# PRIME+ — Contrat scientifique de la PSF-R et boucle fermée tip-tilt STRAP

## Principe de projet

Une PSF-R doit être **mathématiquement et dimensionnellement correcte dans
le cadre de ses hypothèses**, même en présence d'écarts aux observations.
On ne corrige pas ces écarts en ajustant un facteur instrumental arbitraire.
Les paramètres inconnus sont déclarés comme tels, et leur contribution
à l'incertitude doit pouvoir être analysée. La calibration empirique via
fitting de la PSF scientifique relève d'une étape ultérieure, séparée,
dont la fonction est d'estimer des paramètres à partir des images sans
modifier silencieusement la définition du modèle physique.

Chaque contribution du budget PSF-R doit à terme comporter une fiche :
1. grandeur physique et domaine (capteur, commande, miroir, caméra);
2. unités et convention de phase ou de Fourier;
3. équation et dérivation ou référence bibliographique avec section;
4. hypothèses instrumentales nécessaires et paramètres identifiés;
5. domaine de validité, données de calibration requises, inconnues;
6. tests unitaires analytiques, tests synthétiques et cas de non-régression;
7. état de validation (théorique / numérique / instrumentale / observationnelle).

Cette PR traite la **boucle fermée tip-tilt STRAP seulement**.
Elle n'altère pas `p3/psfr/psfR.py` : le changement de covariance du noyau
physique nécessitera la validation des transferts et des unités commande.

## 1. Variables et unités

Toutes les quantités angulaires sont converties en coefficients OPD
RMS Zernike [m], comme établi dans la PR #2
(`a = D*theta/4` avec `theta` en radians).

| Variable | Signification | Domaine | Unité |
|---|---|---|---|
| d(t) | Perturbation TT entrante sur l'axe science | pupille | m RMS OPD |
| e(t) | Résidu TT physique après correction | pupille/image | m RMS OPD |
| y(t) | Résidu mesuré par STRAP après intégration et bruit | capteur | m RMS OPD |
| n(t) | Bruit additif équivalent capteur | capteur | m RMS OPD |
| u(t) | Signal de commande | sortie contrôleur | unité de commande, à calibrer |
| W(f) | Transfert capteur et temps d'intégration | capteur | réponse calibrée |
| K(f) | Transfert de contrôleur IIR | RTC | réponse calibrée |
| P(f) | Transfert commande -> correction physique, avec délai et miroir | actionneur | inclut la conversion commande |
| C_n | Covariance du bruit blanc capteur, avant boucle | capteur | m² RMS OPD |
| P_n(f) | Densité spectrale unilatérale de n | capteur | m²/Hz |

**Attention aux unités de `DTTCOMMANDS`** : PR #2 les marque
`command_units_verified=False`. Le calcul spectral n'utilise pas
directement les commandes comme une grandeur physique certifiée.
Son bloc `P(f)` reste conditionnel à cette identification.

## 2. Dérivation des transferts

Pour une boucle linéaire et invariante dans le temps, avec convention
de rétroaction négative :

    e(f) = d(f) - P(f)*u(f)
    y(f) = W(f)*e(f) + n(f)
    u(f) = K(f)*y(f)

En posant `L(f)=W(f)*P(f)*K(f)` et `S(f)=1/(1+L(f))` :

    y(f) = W(f)*S(f)*d(f) + S(f)*n(f)
    e(f) = S(f)*d(f) - P(f)*K(f)*S(f)*n(f)

Les quatre fonctions de transfert, selon le **point où le signal
est prélevé**, sont alors :

| Chemin | Fonction de transfert | Source |
|---|---|---|
| turbulence -> mesure | W S | déduction des 3 équations |
| bruit capteur -> mesure | S | déduction des 3 équations |
| turbulence -> résidu physique | S | déduction des 3 équations |
| bruit capteur -> résidu physique | -P K S | déduction des 3 équations |
| bruit capteur -> commande | K S | déduction des 3 équations |

La base théorique est celle des budgets d'erreur de contrôleurs AO
linéaires de Juvénal et al. (2018), et du budget TT de Keck de
van Dam et al. (2006). Les équations ci-dessus sont **notre dérivation
algébrique explicite** à partir du schéma W/K/P ; elles ne prétendent
pas être une citation littérale d'une équation numérotée de ces articles.

Références :
- van Dam et al. (2006), PASP 118, 310–318,
  « The W. M. Keck Observatory Laser Guide Star Adaptive Optics System:
  Performance Characterization », section 2.4, équations (4–5) du
  contrôleur à quatre coefficients et estimation par PSD :
  https://www2.keck.hawaii.edu/optics/aodocs/PASP06_MvD.pdf
- Juvénal, Kulcsár, Raynaud & Conan (2018), JOSA A 35, 1465–1476,
  « Linear controller error budget assessment for classical
  adaptive optics systems », DOI 10.1364/JOSAA.35.001465 :
  https://doi.org/10.1364/JOSAA.35.001465
- van Dam, Le Mignant & Macintosh (2004), Applied Optics 43,
  5458–5467, « Performance of the Keck Observatory
  adaptive-optics system » :
  https://doi.org/10.1364/AO.43.005458

## 3. PSD et covariances

Sous les hypothèses d'indépendance statistique du bruit et de la
perturbation, et pour un bruit de mesure blanc :

    P_y(f) = |W(f)*S(f)|² P_d(f) + |S(f)|² P_n(f)
    P_e(f) = |S(f)|² P_d(f) + |P(f)*K(f)*S(f)|² P_n(f)

Pour une covariance d'entrée `C_n` de bruit échantillonné blanc
à la cadence `fs`, sa densité spectrale unilatérale est
`P_n(f) = 2 C_n / fs`, à l'exception des points d'extrémité
de mesure nulle dans l'intégrale continue.

La contribution du bruit aux centroïdes résiduels n'est PAS égale
à `C_n` : elle résulte de l'intégrale sur `|S|² P_n`.
La contribution au résidu *physique* est l'intégrale sur
`|P K S|² P_n`. Les gains normalisés sont :

    g_sensor_to_measure = (2/fs) integral_0^(fs/2) |S(f)|² df
    g_sensor_to_physical = (2/fs) integral_0^(fs/2) |P K S|² df
    g_sensor_to_command = (2/fs) integral_0^(fs/2) |K S|² df

La contribution atmosphérique **dans les mesures** est
`P_y(f)-|S(f)|² P_n(f)`.
Pour retrouver sa contribution au résidu **physique**, il faut
en plus inverser l'amplitude du transfert du capteur :

    P_e,atmos(f) = [P_y(f)-|S(f)|² P_n(f)]/|W(f)|²

Sous réserve que `W` soit suffisamment loin de zéro.
On ajoute ensuite le terme physique de bruit propagé :

    P_e,total(f) = P_e,atmos(f) + |P K S|² P_n(f)

Il est ainsi possible d'estimer une covariance de résidu physique
à partir de la PSD *mesurée* sans passer par la formule scalaire
historique `Cy-(1+pn)Cn`, dont l'interprétation doit être vérifiée.

Aucune PSD de résidu négative n'est tronquée arbitrairement :
une estimation Welch imparfaite peut conduire à une covariance de
composante non positive, qui doit rester diagnostiquée. L'OTF
exigera plus tard une politique physique explicite, validée
indépendamment, au lieu d'une projection destinée à améliorer la PSF.

## 4. Cas de P3 et modèle instrumental Keck

P3 fournit :

    W(f) = ttloop.tf.wfs
    K(f) = ttloop.tf.servo
    P(f) = ttloop.tf.lag                     # modèle historique
    L(f) = W K P
    S(f) = ttloop.tf.rtf

Dans cette hypothèse `|P(f)|=1` (actionneur réduit à un pur
retard), `|P K S|²=|K S|²`, et donc l'ancien
`pn = (2/fs) integral |ntf|²` peut être égal au facteur
physique de bruit réinjecté. Cela **ne valide pas**
`Cy-(1+pn)Cn` pour les centroïdes mesurés, car la contribution
du bruit à `Cy` dépend de `|S|²`, et la partie atmosphérique
dépend de `|W|²`.

La calibration instrumentale réelle demeure incertaine.
Ragland et al. (2016), section 5.2, publient pour **Keck II**
le transfert du miroir tip-tilt :

    H_TTM(s) = 2.55e10 / (2.55e10 + 4.5e7*s + 4e3*s² + s³)
    s = 2*pi*i*f

Référence : « Point Spread Function Determination for Keck
Adaptive Optics », Proc. SPIE 9909, 99091P :
https://arodes.hes-so.ch/record/8385/files/Published%20version.pdf

La fonction `keck_ii_mirror_ragland_2016()` est donc disponible pour
une **étude de sensibilité**, mais **pas sélectionnée par défaut**.
On ne sait pas si sa calibration s'applique telle quelle à
l'observation du 2020-12-02 ni si les unités de commande enregistrées
permettent l'application directe du modèle.

Autres points non vérifiés :
- gain effectif du centroïde STRAP sur l'étoile guide ;
- conversion commande -> mouvement du miroir;
- réponse du miroir en 2020, retard variable, programme IIR actif;
- hypothèse de bruit blanc et indépendance avec turbulence/vibrations;
- cohérence entre la fenêtre d'intégration du STRAP et les
  timestamps d'échantillonnage.

## 5. Tests indépendants

Le cas de contrôle synthétique, **sans coefficient instrumental Keck** :

    e[k] = d[k] - u[k-1]
    y[k] = e[k] + n[k]
    u[k] = u[k-1] + gain*y[k]

Avec `q=exp(-2*pi*i*f/fs)` et `den=1+(gain-1)q` :

    S = (1-q)/den
    P K S = gain*q/den
    S + P K S = 1

Ces fonctions de transfert exactes sont comparées à une simulation
causale en temps discret, sur bruit blanc synthétique de variance
connue. L'algorithme d'intégration PSD est en plus confronté à
des PSD analytiques multiaxes, y compris hors diagonale.

Ces tests prouvent la cohérence du **formalisme mathématique sous ses
hypothèses**. Ils ne prouvent pas les paramètres physiques de Keck.

## 6. Reproduction sur le Mac

Tests :

```bash
git fetch origin
git switch --track origin/feat/strap-closed-loop-formalism
uv run --with pytest python -m pytest -q tests/test_tt_closed_loop_formalism.py
uv run --with pytest python -m pytest -q
```

Audit spectro-temporel observationnel, sans fitting ni PSF :

```bash
uv run python scripts/keck_tt_loop_audit.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
  --output "$PWD/results/n0013_tt_loop"
```

Pour comparer le miroir publié (sans supposer la calibration vérifiée) :

```bash
uv run python scripts/keck_tt_loop_audit.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
  --mirror-model ragland_2016_keck_ii \
  --output "$PWD/results/n0013_tt_loop_ragland"
```

Livrables : `tt_closed_loop_audit.json`,
`tt_closed_loop_audit.png`, avec distinction des covariances
physiques, mesurées, bruit, turbulence, et gains de transfert.

## Décision de validation prévue

Ne modifier le noyau jitter de `psfR` qu'après :
- tests analytiques et simulations causales réussis ;
- vérification des unités de commande STRAP et de la convention `P` ;
- comparaison spectrale quantitative sur `n0013` avec le modèle
  historique et avec le modèle miroir publié ;
- formalisation du périmètre de validité et des inconnues.

Aucune cible de Strehl observé ni de largeur de PSF ne sert de critère
d'acceptation. Le fitting photométrique et spatial est un lot futur,
distinct du moteur physique.
