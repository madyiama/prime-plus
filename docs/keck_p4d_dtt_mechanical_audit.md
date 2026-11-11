# P4-D — Identification du mouvement mécanique du DTT depuis la télémétrie

## Objectif

Caractériser la **réponse mécanique relative** entre les deux voies de
`B.DTTCOMMANDS` et les trois voies de `B.DTTSTRAINGAUGE` de Keck II
sans supposer que les valeurs des jauges sont des angles ciel. Il
ne s'agit **ni** d'une estimation du jitter résiduel de NIRC2
**ni** d'une calibration du gain optique STRAP.

Le premier essai n0120 avait donné :
- `13556` trames ; cadence estimée selon la médiane `500.801 Hz` ;
- 3 jauges différentielles, 2 commandes ;
- décalage d'environ 2 trames dans une régression sur les **niveaux** ;
- `R²_train=0.99914`, `R²_validation=0.99888` ;
- 29 valeurs différentes de `CONF_ID`, 28 transitions ;
- `DTTCOMMANDSCLIPPED` nul dans toutes les trames.

La précision des niveaux peut être dominée par les variations lentes,
elle **ne prouve pas** la réponse dynamique rapide ni une latence de 4 ms.

## Expérience indépendante du modèle de PSF

`p3/telemetry/ttMechanicalAudit.py` et
`scripts/keck_p4d_dtt_mechanical_audit.py` réalisent :

1. **Audit d'horodatage** en ticks `uint64/int64`, avant conversion
   en temps flottant. L'écart moyenne/médiane ne doit pas devenir un
   faux décalage temporel. Les intervalles >1.5 fois ou <2/3 fois la
   médiane délimitent des segments continus. Ce critère détecte les
   anomalies d'espacement sans prouver leur cause (trames perdues,
   cadence variable ou autres).
2. **Identification en niveaux** du gain relatif `g_diff = u @ A + b`
   avec un décalage de -10 à +10 trames, en conservant la validation
   temporelle sur les derniers 30% des données.
3. **Identification sur différences finies** `Δg_diff` vs `Δu`
   et sélection du retard sur les **70% d'apprentissage** uniquement ;
   une paire franchissant un intervalle d'horodatage anormal est exclue.
4. **PSD et cohérence** entre commandes et jauges projetées dans les
   coordonnées des commandes via la pseudo-inverse de `A`.
   Les spectres de Welch de segments **temporellement continus**
   sont moyennés avec pondération par nombre de fenêtres; les
   segments disjoints ne sont jamais recollés.
   Avec moins de six fenêtres Welch, aucune cohérence n'est publiée
   (une fenêtre donne artificiellement une cohérence égale à 1).
   Si nécessaire la résolution Welch descend de 1024 à 512,
   256 ou 128 trames pour s'adapter aux données.
5. **Évolution par segment de `CONF_ID`** : écart relatif
   `||A_local - A_train||_F / ||A_train||_F`,
   qualité de l'ajustement et excitation des deux commandes.
   Les segments courts/non identifiables sont explicitement ignorés.

### Commande

Depuis le repo, branche P4-D :

```bash
git fetch origin
git switch --track origin/feat/psfr-p4d-dtt-mechanical-telemetry-audit

uv run python scripts/keck_p4d_dtt_mechanical_audit.py \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
  --nperseg 1024 \
  --max-lag 10 \
  --output results/n0120_dtt_mechanical
```

Les livrables sont :
- `dtt_mechanical_audit.json` (détail des modèles, cohérences,
   transitions de configuration, délais et avertissements) ;
- `dtt_mechanical_audit.png` (R² retard en niveaux/différences,
   cohérence et gain relatif en fonction de la fréquence).

Tests (sans fichiers Keck) :

```bash
uv run --with pytest python -m pytest -q \
  tests/test_keck_p4d_dtt_mechanical_audit.py
```

### Interprétation

**Si le R² des différences reste élevé** près de 2 trames,
le suivi des variations rapides est mieux établi que par le premier
essai. Il peut néanmoins inclure des délais de la chaîne d'acquisition.

**Si le R² des niveaux reste proche de 1 mais celui des différences
s'effondre**, la corrélation provient surtout des variations lentes.

**Une cohérence spectrale élevée** confirme un suivi relatif
commande–jauge dans les bandes où les commandes excitent le miroir.
Une chute de cohérence, une phase non linéaire, ou une décroissance
de gain peut suggérer une limite de transfert, sous réserve du
bruit des jauges et de l'excitation disponible.

**Les modifications de matrices lors des transitions de `CONF_ID`**
peuvent refléter une configuration différente, un changement de
niveau d'excitation ou un artefact statistique. Ce n'est pas une
mesure directe d'un gain optique STRAP.

### Limitations à respecter

- La transformation `A` est ajustée à partir de commandes archivées
  et de jauges mécaniques, **pas** par référence à un angle sur le ciel.
- L'identification reste purement relative; sans matrice
  jauges → mouvement du miroir (incluant conversion optique), elle ne
  donne **aucune** valeur absolue du jitter TT.
- La cohérence par axe après projection suppose implicitement que
  les deux voies commandées sont suffisamment indépendantes;
  une excitation fortement corrélée requiert une vraie analyse
  MIMO avec une estimation matricielle des spectres croisés.
- Le retard de phase ajusté est une approximation par **retard pur**
  du transfert complexe; les résonances et la fréquence peuvent
  provoquer un modèle beaucoup plus complexe.
- La cadence d'échantillonnage est celle de la télémétrie, pas
  nécessairement les 60 kHz de la boucle interne DTT.
- Aucune mesure n'est utilisée pour ajuster la PSF NIRC2;
  les modèles `unity` et `ragland_2016_keck_ii` restent
  des **hypothèses séparées**, non remplacées automatiquement.

## Prochaine décision

Comparer la réponse réelle avec un modèle de transfert de miroir
avant d'ajouter un gain optique STRAP à P3. Ne sélectionner
aucun facteur multiplicatif depuis une comparaison d'images.
