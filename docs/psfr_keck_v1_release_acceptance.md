# PSF-R Keck/NIRC2 — recette V1 (baseline instrumentale non calibrée)

**Statut** : framework numérique reproductible et tests de régression ;
**pas** une calibration absolue de gain optique STRAP, du bruit APD, des
directions de vent, de la polarité STRAP→caméra, des constantes de WFS
haut ordre ou du modèle mécanique TT Keck 2020.

## 1. Configuration nominale et invariants

**Source de vérité** : `config/psfr_keck_v1.toml`. Pour la reconstruction
scientifique nominale V1 (Keck II NIRC2 + LGS STRAP) :

| Élément | Paramètre V1 |
|---|---|
| HO : pseudo-boucle ouverte | P3 `systemDiagnosis` et ajustement Zernike `r0,L0` |
| HO : covariance | `strict` (échec explicite si non-PSD ; pas de clipping) |
| Profil atmosphérique | 7 couches MASS/DIMM, si données valides ; provenance tracée |
| Pupille | `physical_binary`, `header`, parité `native` |
| NCPA | désactivée pour la recette télémetrie seule |
| Bruit STRAP | `van_dam`, 65–95 % Nyquist, Welch 512 |
| TT covariance | `legacy_psfr`, miroir `unity` |
| TT kernel | `noll_angular` et a = D θ / 4, **jamais** `/16` |
| Anisocinétisme | **`dani_tt` déjà dans P3**, ne pas ajouter à `Ctt` |
| GHO/GTT | 1 / 1 pour la PSF-R nominale ; pas de fitting de PSF |
| Gain optique STRAP | non calibré indépendamment — pas de correction empirique |

Les profils `spectral_comparison` et `gendron_comparison` sont
des expériences, non des alternatives automatiques. Ne jamais
sélectionner l'une d'elles sur la seule netteté NIRC2.

## 2. Préparation

Dans la racine du dépôt et après fusion des PR du noyau :

```bash
git switch main
git pull --ff-only origin main
uv sync
```

Si une installation scientifique additionnelle est nécessaire,
les tests ne doivent pas être rendus verts artificiellement en
supprimant leurs dépendances ; rétablir les dépendances dans
l'environnement.

### Tests synthétiques (aucune archive nécessaire)

```bash
uv run --with pytest python -m pytest -q \
    tests/test_keck_psfr_v1_release.py \
    tests/test_keck_p4c_tt_selection.py \
    tests/test_keck_p4_angular_otf_chain.py \
    tests/test_keck_p4_tt_frame.py \
    tests/test_keck_tt_geometry_and_anisoplanatism.py \
    tests/test_psfr_spectral_tt_kernel.py
```

L'ensemble de la suite se lance ensuite par :

```bash
uv run --with pytest python -m pytest -q
```

Certains tests Keck ont besoin de données/ressources locales ;
les exécuter seulement avec les prérequis présents, en enregistrant
tous les échecs et skips.

## 3. Recette bout en bout sur vraies acquisitions

Le programme `scripts/keck_psfr_v1_e2e.py` accepte N couples
FITS/SAV **alignés par position**. Il exécute le même moteur
`keck_n0013_smoke.py`, avec des arguments strictement déclarés
dans `config/psfr_keck_v1.toml` :

```bash
uv run python scripts/keck_psfr_v1_e2e.py \
  --fits \
    "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0120.fits" \
    "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav \
    "/Users/obm/Documents/KECK DATA/20201202/TRS/n0120_LGS_trs.sav" \
    "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "p3/aoSystem/data/KECK_CALIBRATION" \
  --config "config/psfr_keck_v1.toml" \
  --output "results/psfr_v1_acceptance"
```

**Adapter les chemins FITS/SAV à leur emplacement réel.**

La commande produit, dans `results/psfr_v1_acceptance/` :

- `psfr_v1_acceptance.json` : synthèse machine-readable des cas
  PASS/FAIL et des contrôles de recette.
- `n0120/`, `n0013/` : `telemetry_only_psf.npy`,
  `p4_otf_tt.npy`, `comparison_nirc2_psfr.png`,
  `tt_selection.json`, `metadata.json`,
  `tiptilt_diagnostics.json`, `anisoplanatism_diagnostics.json`,
  `ho_covariance_diagnostics.json`, et
  `psfr_v1_run.log` (stdout+stderr de l'acquisition).

Si le run a déjà produit ces fichiers, on peut refaire les
contrôles sans tout reconstruire :

```bash
uv run python scripts/keck_psfr_v1_e2e.py \
  --fits n0120.fits n0013.fits \
  --sav n0120_LGS_trs.sav n0013_LGS_trs.sav \
  --calib p3/aoSystem/data/KECK_CALIBRATION \
  --output results/psfr_v1_acceptance --check-only
```

(`--check-only` utilise uniquement le **nom de fichier FITS**
pour localiser les résultats ; les données FITS/SAV peuvent
donc ne pas être accessibles à ce moment.)

## 4. Critères de validation

Le contrôle automatisé refuse :

- un mode AO autre que LGS ;
- une covariance HO réparée/projetée malgré `strict` ;
- une PSF contenant NaN/Inf/flux négatif ou nul ;
- une OTF TT non physique, ou `OTF(0)≠1` ;
- une incohérence DTTCENTROIDS → cov. rad² → OTF ;
- un profil TT différent de la baseline choisie ;
- la normalisation historique de phase de TT divisée par 16 ;
- les sorties indispensables manquantes.

Il ne **prouve pas** à lui seul la précision scientifique sur
des objets astrophysiques, ni la justesse absolue de (r_0),
(L_0) et du jitter TT.

Les cas `non-PSD` ne doivent pas être forcés à passer :
enregistrer et isoler l'observation, inspecter le bruit HO,
puis décider de son traitement dans une future version.

## 5. Fin de recette et décision d'étiquetage

La V1 peut être qualifiée **fonctionnelle reproductible** lorsque :
(1) les tests automatisés passent sur main ;
(2) les deux acquisitions traversent toute la chaîne sans exceptions ;
(3) `psfr_v1_acceptance.json` indique 2/2 PASS ;
(4) la provenance du profil et les limitations physiques
sont documentées ; (5) les images reconstruites sont examinées
visuellement et leur différence avec NIRC2 enregistrée,
**sans utiliser ce constat pour modifier le gain TT**.

Ne pas appeler cette qualification une validation absolue
instrumentale du Keck.

## 6. Gestion des travaux de recherche P4-B → P4-G

Les branches/PR d'audit peuvent être conservées ou fusionnées
comme outillage de diagnostic si leurs tests passent. Elles ne
doivent pas modifier les méthodes, constantes ni corrections
physiques de la configuration nominale V1.

- P4-B : sensibilité van Dam/APD (diagnostic).
- P4-D : jauges/commands (diagnostic).
- P4-E : statistiques et modèle conditionnel de boucle (diagnostic).
- P4-F : expérimentation A/B centroïdes/commandes validée numériquement.
- P4-G : prédiction HO→TT (discordance physique non résolue,
  **ne jamais injecter son coefficient alpha dans la PSF**).
