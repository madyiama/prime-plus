# P4-C — Framework PSF-R : configuration finale provisoire STRAP TT

## Décision de baseline

Pour converger vers un framework de reconstruction exploitable **sans
présumer d'une calibration Keck 2020**, conserver les composants P1/P2/P3,
l'OTF géométriquement normalisée de P4-A et choisir une covariance TT
plus modérée pour l'usage courant :

| Hypothèse de P4-A sur la même acquisition n0013 | RMS/axe | Statut |
|---|---:|---|
| A0 : pas de TT | 0 mas | ablation, non physique |
| **A1 : van Dam, legacy_psfr, unity, Noll angular** | **31.58 mas** | **baseline opérationnelle provisoire** |
| A2 : noyau `Dphi/16` | 7.89 mas effectif | ablation géométriquement non cohérente |
| A3 : van Dam, spectral_closed_loop, unity, Noll angular | 41.13 mas | alternative expérimentale |

Le but n'est pas d'identifier une vérité depuis une binaire NIRC2.
A1 est un choix pratique pour éviter de sur-élargir systématiquement
les images pendant la consolidation du framework. Il ne doit pas être
présenté comme le jitter réel du télescope.

P4-B n0013 a montré que :
- `tr(C_y)=0.0199293 arcsec²` ;
- `tr(C_n)=0.0179348 arcsec²`, soit ~90% de `C_y` ;
- la sensibilité de bruit à la bande/Welch dépasse la précision physique
  nécessaire à une PSF-R calibrée ;
- Ragland 2016 n'est pas un modèle validé du miroir Keck 2020 ;
- l'expérience de flux APD n'a qu'une plage inter-fenêtres de ~3% :
  elle ne permet pas de calibrer le gain photoélectronique.

## Contrat d'architecture retenu

Les deux décisions doivent être **indépendantes** :

1. **Mesure du bruit** `Cn_tt` à l'entrée du capteur : van Dam (baseline),
   Gendron (extrapolation de covariance temporelle),
   `legacy_autocorrelation` (comparaison historique).
2. **Reconstruction du mouvement physique** `C_TT` : `legacy_psfr`
   (baseline), ou `spectral_closed_loop` (W/K/P/S explicites).

Les deux routes utilisent **la même** conversion
`DTTCENTROIDS[arcsec] -> C_theta[rad²]` et la même
`OTF_TT=exp(-2 pi² nu^T C_theta nu)`, avec `nu` en cycles/radian,
toutes deux déjà vérifiées numériquement par P4-A sur données réelles.

**Aucun réglage de bruit ne doit modifier** les paramètres de turbulence
`r0,L0`, le profil `Cn²`, l'optique de pupille, la correction HO ou la
formule TT d'OTF. La reconfiguration ne touche que `Cn_tt`, la
décomposition résiduelle/propageée TT et le noyau calculé à partir de
cette covariance.

### Interface du framework

La configuration `config/keck_tt_noise.toml` reste la source autoritaire
si `--tt-profile` est omis. Elle contient par défaut :

```toml
[tip_tilt_noise]
method = "van_dam"
high_frequency_band = [0.65, 0.95]
welch_nperseg = 512

[tip_tilt_jitter]
method = "legacy_psfr"
mirror_model = "unity"
welch_nperseg = 512
```

Trois **profils nommés et reproductibles** existent maintenant :

| Profil CLI | Estimateur du bruit | Covariance physique |
|---|---|---|
| `reduced_jitter` | van Dam 0.65–0.95, Welch512 | legacy_psfr, unity |
| `spectral_comparison` | van Dam 0.65–0.95, Welch512 | spectral_closed_loop, unity |
| `gendron_comparison` | Gendron temporel | legacy_psfr, unity |

Ils ne modifient ni le HO ni r0. Les valeurs de bruit APD idéal
restent réservées à `scripts/keck_strap_apd_audit.py` et P4-B :
en l'absence de calibration indépendante du gain photoélectronique
et du facteur de bruit d'avalanche, il serait prématuré de mettre
le Poisson conditionnel dans le parcours scientifique par défaut.

**Priorité des valeurs** :
`option CLI individuelle > --tt-profile explicitement fourni > TOML`.
Un profil n'écrase donc pas silencieusement un TOML personnalisé.

Si `legacy_psfr` est choisi avec un miroir autre que `unity`, le
framework **refuse la combinaison**, car cette méthode ne se sert
pas du transfert du miroir ; elle ne doit pas ignorer silencieusement
la demande de calibration. Pour tester Ragland 2016, utiliser
`spectral_closed_loop` et marquer la sortie comme expérimentale.

### Commandes n0013

Exemple : reconstruire une PSF au réglage de clôture :

```bash
uv run python scripts/keck_n0013_smoke.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --output "results/n0013_framework_A1" \
  --n-layer 7 --p3-pupil-resampling physical_binary \
  --tt-profile reduced_jitter
```

Pour comparer *sur la même acquisition* :

```bash
# Bruit inchangé, covariance atmosphérique spectrale
... --tt-profile spectral_comparison

# Covariance legacy inchangée, méthode de bruit différente
... --tt-profile gendron_comparison

# Ou changer séparément les méthodes sans profil:
... --tt-noise-method gendron --tt-jitter-model legacy_psfr
```

La sortie ajoute `tt_selection.json` dès le début de chaque run,
avec paramètres, provenance, méthode physique active, et avertissements
de calibration. Le fichier final `smoke_summary.json` récapitule
la méthode effective après politique LGS/NGS.

Le noyau `legacy_unscaled_opd` est réservé à la comparaison
historique `/16`. Il n'est **jamais** sélectionné par les profils
standards et reste explicitement non physique pour les centroïdes
archivés déjà exprimés en angle.

## Critères de clôture de la refonte framework

Clôturer la **capacité logicielle** lorsque :

- les tests P1/P2/P3/P4-A et P4-C passent sur Mac et CI quand disponible ;
- les trois profils reconstruisent les PSF voulues, ou signalent sans
  ambiguïté une covariance non PSD dans une méthode expérimentale ;
- A1 reproduit n0013 sans modification du `r0/L0`, de la pupille ou
  de l'OTF ; A3 correspond à la référence spectrale P4-A ;
- les diagnostics incluent les unités, la bande/Welch, la provenance
  du profil et le statut de calibration ;
- le framework peut être invoqué à partir d'une configuration TOML ou
  d'arguments CLI, sans écrire dans les fonctions physiques pour
  sélectionner un autre bruit ;
- aucun ajustement sur la binaire n'a été utilisé pour choisir le
  facteur 16, les paramètres du bruit ou de la boucle.

**Chantiers différés sans bloquer la clôture informatique** :
calibration Keck 2020 du gain optique STRAP, photoélectrons par compte,
excès de bruit d'avalanche, transfert du miroir et délai W/K/P/S,
validation en simulation avec bruit/jitter connus, plusieurs jeux
télescope et robustesse sur le champ, estimation physique r0/L0.
Ne jamais qualifier A1 de modèle instrument validé tant que ces
contrôles ne sont pas réalisés.
