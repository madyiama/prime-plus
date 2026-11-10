# PRIME+ — revue scientifique transversale (10 novembre 2026)

**Statut : inspection statique des implémentations, des audits P1/P2 et
de n0120, pas certification de l'instrument Keck 2020.**
Branche d'audit seule ; **aucune modification du moteur PSF-R**.
Dépend de PR #17 (P3 pupille), PR #16 (P1, estimateur atmosphérique
figé provisoirement), PR #8 (P2 référence analytique).

## Sources et conventions

1. Véran, Rigaut, Maître & Rouan, JOSA A 14, 3057 (1997),
   DOI 10.1364/JOSAA.14.003057 : séparation télémétrie / OTF.
2. Jolissaint, Ragland, Christou & Wizinowich, Applied Optics 57,
   7837 (2018), DOI 10.1364/AO.57.007837 : estimation turbulente
   Keck DM, IF, modes, WFS noise, aliasing; validation sur PSF
   open-loop quasi simultanées. Les DIMM/MASS externes ne sont
   pas une vérité équivalente pour l'axe Keck.
3. O. Beltramo-Martin, `PSFR_KECK.pdf` (LAM, 2020), §2–3 :
   décomposition contrôlé/non contrôlé, propagation hors axe
   non stationnaire et géométrie LGS/NGS.
4. S. Ragland, `AO Telemetry for PSF-R` (Keck note 2017),
   pages 1–3 : structure A/ B et unités, guide 0.6 µm/Volt.
5. Ragland, KAON 1160 (2017) : WFS rotation / NIRC2 rotation.
6. `docs/keck_p2_frozen_reference_2026-11-10.md`,
   `docs/psfr-spatial-physics-audit-contract.md`,
   `docs/keck_psfr_physical_coefficients_contract.md`.

Classification : **BUG** = contradiction directe du code avec un
invariant physique/mathématique; **RISQUE** = choix d'implémentation
biaisant potentiellement le résultat; **HYPOTHÈSE** = modèle
standard mais non calibré; **CALIB** = donnée instrumentale
manquante; **DIAGNOSTIC** = quantification non équivalente
à la PSF physique. Les priorités n'impliquent pas qu'il faille
modifier immédiatement la baseline.

## 1. Bilan par terme

| Terme | Fonction(s) | Constat vérifiable | Classe | Priorité | Suite |
|---|---|---|---|---|---|
| HO modal covariance | `psfR.residual_phase_structure_function`, `psfrUtils.modes2Otf` | `du=diff(DMCOMMAND)/gain` puis `Cao-Cn_ao`. Le `Cn_ao` actuel provient précisément des **différences de DMCOMMAND**, pas d'une mesure indépendante de bruit ; les deux covariances sont fortement liées. Des cas NGS déclenchent le rejet PSD attendu. | RISQUE modèle/bruit | **P0** | Tester séparément différence, bruit, gain, spectres propres ; aucune projection PSD cachée |
| HO conversion unités | `telemetryKeck.restoringTelemetry`, `psfR` | `volt2meter=0.4095e-6` tandis que Ragland 2017 indique `0.6e-6` pour DMCOMMAND / residual. Ce ne sont pas nécessairement les mêmes conventions/époques (OPD vs surface). | CALIB | **P0** | Identifier signification et provenance du gain avant changement |
| Anisoplanétisme focal LGS | `anisoplanatismModel.focal_angular_function`, `frequencyDomain` | Kernel de production stationnarisé/interpolé ; P2 montre 4–7% de Strehl relatif entre non-stationnaire exact et moyenne exponentielle. Hfilter pas relayé dans le chemin SLAO historique. | RISQUE | **P0** | Intégrer référence non stationnaire avec grille de convergence et masque P3 |
| TT STRAP/NGS | `ttNoise`, `ttClosedLoop`, `psfR.jitter_phase_structure_function` | Van Dam suppose plancher HF et rejet W/K/P ; optical gain et commandes TT non certifiés. Le mode legacy `C_y-C_n` ne représente pas généralement le mouvement réel. NGS convertit les unités différemment de LGS. | CALIB | **P0** | Étalonnage indépendant et épreuve NGS/LGS ; mode spectral expérimental |
| Pupil / static | `telescope.load_and_process_data` | Interpolation spline du binaire rotatif produit des valeurs de transmission négatives et >1 sur n0120; P2 documente min -0.153/max 1.198. La carte NCPA suit la rotation mais son repère zéro n'est pas calibré. | BUG pour masque binaire si traité comme transmission physique | **P0** | PR #17 puis pupille amplitude physiquement bornée et repère calibré |
| Fitting DM | `psfR.fitting_phase_structure_function` | PSD von Kármán au-delà d'un cutoff DM idéalisé, sans réponse réelle complète des fonctions d'influence ni gain modal. Exposant `r0^-5/3` correct. | HYPOTHÈSE | P1 | Confronter cutoff dur à filtre DM réel, convergence PSD |
| Aliasing WFS | `fourierModel.aliasingPSD`, `psfR.aliasing_phase_structure_function` | Architecture de repliement de fréquences compatible avec PRIME, mais réponse WFS/RTC, gain et termes croisés ne sont pas vérifiés sur Keck 2020. | CALIB/HYPOTHÈSE | P1 | Vérification unités et simulation Shack–Hartmann |
| Spectral | `psfR`, `FourierUtils.sf_3D_to_psf_3D` | Une seule SF/OTF à `wvlRef=min(wvl_bins)` avec pondération de bins ; ce n'est pas une véritable intégration de PSF chromatiques. | APPROXIMATION | P1 | PSF par lambda puis intégration transmission |
| Cn²(h) | `telemetryKeck.restoringCalibrationData` | MASS/DIMM non colocalisés, profil de fallback possible, directions des vents initialisées à zéro, plusieurs couches équivalentes. `nLayer=1` et `nLayer=7` ne sont PAS des validations indépendantes du r0 WFS. | HYPOTHÈSE/CALIB | P1 | Provenance temporelle, validité seeing, sensible aux couches et vents |
| NCPA / focus | `telemetryKeck`, smoke `addNCPA=False` | Aucune aberration statique NIRC2 calibrée dans les smokes actuels ; séquences NGS focus / phase diversity possibles, dates de cartes NCPA différentes. | CALIB | P1 | Classifier science/PD ; étalonner carte OPD et orientation |
| Jitter + anisokinétisme | `anisoplanatismModel`, `psfR` | TT LGS mesuré par STRAP et anisokinétisme NGS ; absence de double-compte exigée par P2. Cross-termes HO/TT souvent négligés avec justification de faible amplitude dans les simulations du rapport 2020, pas une identité universelle. | HYPOTHÈSE | P1 | Tests géométrie et bornes en dehors de n0120 |
| Phase structure / OTF | `psfrUtils.modes2Otf`, `psfR` | Conditions `Dphi(0)=0` / `OTF(0)=1` corrigées ; `eigh` plutôt que SVD sur covariance symétrique; refus de valeurs propres négatives matérielles. | Correctif numériquement fondé | Contrôle continu | Tests zéro covariance et positivité sans clipping tolérant arbitraire |
| WFE budget | `psfR.get_error_breakdown` | Racine de somme de carrés et Maréchal supposent indépendance des contributions; un budget de WFE par terme n'est pas en général égal au Strehl total non stationnaire. | DIAGNOSTIC | P2 | Distinguer WFE de phase, WFE Strehl-equivalente, termes corrélés |
| Wind/tau0 | `systemDiagnosis.get_wind_speed` | Soustrait `noise*var_noise_zer` **à tous les lags** du processus Zernike. Pour bruit blanc, la covariance n'est non nulle qu'à lag zéro ; pour bruit filtré sa forme doit être calculée. Le facteur `coeff=1.15` et les fenêtres autocorrélation sont des paramètres empiriques. | **BUG de modèle de bruit** | P1 | Modéliser `C_noise(tau)`, tests white/colored, ne pas retoucher tau0 pour image |
| Helper pseudo-OL historique | `psfrUtils.getOLslopes` | `np.roll(u,(-2,2))` sans `axis` sur tenseur; et `MI*...` élément par élément. La fonction n'est pas certifiée sur le chemin actif. | BUG latent | P1 | Déprécier ou corriger avec contraction calibrée et axes explicites |

## 2. Notes analytiques

### 2.1 Covariance HO : forte dépendance entre signal et bruit

Pour `autocorrelation`, le proxy
`Cn_ao=0.5 * mean((u[t+1]-u[t]) (u[t+1]-u[t])^T)`
(avec biais bord LGS périodique) est construit depuis le
**même u** que
`Cao = mean(diff(u)/gain * diff(u)/gain^T)`.
En dehors des effets d'extrémité et des erreurs de normalisation,
`Cao ≈ 2 Cn_ao / gain²`.

En conséquence, `Cao-Cn_ao` n'est **pas** un estimateur de
covariance de turbulence ayant une décomposition du bruit
indépendante. Cela ne prouve pas que la SF HO construite est
entièrement fausse : des hypothèses historiques d'étalonnage
peuvent exister, mais elles doivent être explicitées, pas supposées.
La non-PSD de certaines acquisitions est un signal d'alerte
physique, pas à masquer en prenant des valeurs singulières.

### 2.2 Anisoplanétisme : moyenne de l'exponentielle

En milieu non stationnaire, la réponse correcte dans le modèle
gaussien est
`OTF(rho) = sum_r A(r)A(r+rho) exp(-D(r,rho)/2) / sum_r A(r)^2`.
Elle n'est pas égale à
`OTF_DL(rho)exp(-mean_pairs(D(r,rho))/2)`.
La baseline P2 donne pour Keck n0120 environ +4.2% à +6.8%
sur le Strehl focal-only exact relativement à l'approximation
stationnarisée, selon L0/couches. Ces nombres ne sont ni
des gains à appliquer, ni des corrections de r0.

### 2.3 Bruit STRAP et HO : ne pas fusionner les modèles

Le TT utilise une covariance de centroïdes angulaires et une
propagation W/K/P (avec hypothèse de bruit blanc HF); le HO
utilise actuellement des différences des commandes DM.
Leurs variances ne sont pas interchangeables et un même
modèle de bruit de capteur ne peut être soustrait deux fois.

### 2.4 Les paramètres empiriques à inventorier

- `dm.volt2meter=0.4095e-6` : calibration instrumentale à étayer.
- `wfs.nSubap=[20]`, `wfs.pixel_scale=800`, `wfs.gain=1`,
  `dm.mechCoupling=[0.11]`, `dm.opt_cond=1e2` :
  valeurs instrumentales/par défaut à historiser.
- `estimateLoopDelay` : table `WSSMPRG` et retard RTC;
  bonne interpolation mais signe/latence pas vérifiés.
- `get_wind_speed(...coeff=1.15)` : paramètre empirique.
- `tt_noise_high_band=(0.65,0.95)`, Welch 512 et
  `spectral_command_noise_variance(high_band=(0.75,0.95))` :
  choix expérimentaux de plateau.
- `jolissaint mode_selection='idl_2018'` :
  liste de 75 modes issue d'une autre campagne, pas certifiée n0120.
- `seeingEstimator.alias_correction` : polynôme empirique
  publié pour le Keck historique, `False` par défaut.
- `L0_grid=(10,15,22,31,45,65,100,150,250,inf)` :
  grille de recherche, non intervalle de confiance.
- `pupilAngle`, `PMSANGL`, `PMRANGL`, `PathPupil` :
  mapping mécano-optique et zéro non calibrés.
- `nLayer`, hauteurs/poids MASS, `wDir=0`, r0 seeing
  à 500nm / science wvlRef : provenance à conserver.
- `NCPA=off` et modèles `TT=legacy_psfr/spectral_closed_loop` :
  politiques de benchmark, pas calibrations.

## 3. Garde-fous immédiats (sans changer P3)

1. Aucun terme HO/TT significativement non-PSD ne doit être
   rendu positif silencieusement pour obtenir une image.
2. Chaque résultat PSF doit étiqueter le commit, les méthodes
   HO/TT, l'option NCPA, la longueur d'onde, le profil
   atmosphérique, le masque, la rotation et l'estimateur r0.
3. Ne pas choisir une valeur de gain, `L0`, angle ou retard
   parce qu'elle améliore une image binaire.
4. Dès que P3 a fixé le masque et l'orientation, comparer des
   acquisitions science similaires à géométrie constante.
5. Reporter la calibration absolue r0/L0 à la dernière phase,
   mais garder les trois tests diagnostics autonomes prêts
   (voir `docs/keck_final_r0_l0_audit_procedure.md`).

## 4. Priorités correctives après les audits

**P0 scientifiquement critiques avant une claim de précision** :
bruit HO/PSD, gain OPD, zéro et transmission de pupille,
non-stationnarité LGS, référence TT sensor→ciel.
**P1 importantes pour PSF photométrique** : PSF chromatique,
NCPA/phase diversity, alias WFS, profils Cn², vent.
**P2 d'ingénierie / métriques** : WFE budget, vieux helpers non
utilisés, documentation des valeurs par défaut.

Tous les points cités sont à confirmer par tests des fonctions
concernées ; une inspection statique ne constitue pas à elle seule
une démonstration expérimentale.
