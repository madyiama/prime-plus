# PRIME+ — Registre de traçabilité des modèles physiques

**Objectif : une PSF reconstruite correctement dans un cadre explicité,
pas un Strehl ajusté a posteriori.** Un écart avec NIRC2 est admissible
si les hypothèses, les inconnues et la sensibilité sont documentées.

Ce registre distingue trois validations **non interchangeables** :

- **T — Théorie sourcée** : équation dérivée, convention et bibliographie;
- **N — Numérique** : tests analytiques, conservation des unités, simulation indépendante;
- **I — Instrumentale** : coefficients, gains et délais vérifiés sur Keck.

Le fait qu'un résultat soit proche d'une image ne confère aucun
statut T, N ou I. Les anciennes expressions sont maintenues lorsque leur
remplacement n'est pas encore validé; les limites sont indiquées.

| Terme | Code principal | Formulation et source | T | N | I | À faire |
|---|---|---|---|---|---|---|
| TT angle/Zernike | `aoSystem/tipTiltGeometry.py` | Coefficients Noll RMS, `a=D*theta/4`, noyau de jitter gaussien; voir `docs/keck-strap-tt-and-lgs-anisoplanatism.md` | Oui | Oui | Partielle | gain STRAP et unités commandes |
| Contrôleur TT | `telemetry/systemDiagnosis.py` | 4-tap IIR Keck, van Dam et al. (2006) §2.4; fonction intégration W et retard | Oui | Partielle | Non | confirmer coefficients du RTC et latence 2020 |
| Bruit TT capteur | `telemetry/ttNoise.py` | van Dam (2006) : plateau PSD après rejet; Gendron & Léna (1995) : extrapolation temporelle | Oui | Oui | Non | validation APD, gain de centroïdes |
| Robustesse TT | `telemetry/ttNoiseRobustness.py` | Welch et sous-séries; comparaison de biais, sans IC formel | Oui | Oui | Non | valider stabilité sur plusieurs acquisitions |
| Propagation TT | `telemetry/ttClosedLoop.py` | Dérivation W/K/P/S documentée dans `docs/tt-closed-loop-theoretical-contract.md`; Juvénal et al. (2018), van Dam et al. (2006) | Oui | À vérifier sur Mac | Non | confirmer P et comparer modèles Keck |
| TT mirror | `telemetry/ttClosedLoop.py` | Modèle Keck II publié par Ragland et al. (2016) §5.2, utilisé seulement en sensibilité | Oui | À vérifier sur Mac | Non pour 2020 | identification fréquence/gain/retard 2020 |
| Erreur haute fréquence HO | `psfr/psfR.py`, `telemetry/systemDiagnosis.py` | Formel historique P3, **audit de provenance incomplet** | À auditer | À auditer | Non | unités HO, PSD, projection, aliasing |
| Fitting spatial AO | `aoSystem/frequencyDomain.py` | Spectres turbulence/von Kármán et réponse du miroir, **équations à tracer** | À auditer | À auditer | Partielle | DOI et numéro d'équation par terme |
| Aliasing spatial | `aoSystem/frequencyDomain.py` | Spectres spatialement repliés, **audit de provenance incomplet** | À auditer | À auditer | Non | schéma de sous-échantillonnage WFS |
| Anisoplanétisme LGS | `aoSystem/anisoplanatismModel.py` | Cone-effect et phase structure function; correctif `nLayer=1` en PR #2, **provenance complète à auditer** | À auditer | Partielle | Non | géométrie des étoiles, profil couches, normalisation |
| Pixel / détecteur | `psfr/psfR.py` | OTF de pixel rectangulaire en produit de `sinc`, **convention à auditer** | À auditer | À auditer | Non | échantillonnage/pixel scale/orientation |
| NCPA et pupille Keck | `psfr/psfR.py`, fichiers calibration | Calibrations instrumentales, NCPA désactivée pour `n0013` | À auditer | À auditer | Non | pupille, pistons de segments, masques |
| Strehl et Maréchal | `psfr/psfR.py` | Approximation RMS -> Strehl; **usage et domaine à auditer** | À auditer | À auditer | Non | ne pas prendre l'écart au Strehl NIRC2 comme biais à corriger |

Références déjà vérifiées pour TT :
- van Dam et al., PASP 118 (2006) §2.4 :
  https://www2.keck.hawaii.edu/optics/aodocs/PASP06_MvD.pdf
- Juvénal, Kulcsár, Raynaud, Conan, JOSA A 35 (2018), DOI
  10.1364/JOSAA.35.001465 :
  https://doi.org/10.1364/JOSAA.35.001465
- Ragland et al., SPIE 9909 (2016) §5.2 :
  https://arodes.hes-so.ch/record/8385/files/Published%20version.pdf

## Règle pour les prochaines PR scientifiques

Une PR qui modifie la PSF-R doit :
1. identifier le terme du registre et joindre la source exacte ou
   une dérivation complète sourcée;
2. préciser le **signal concerné** (mesure, miroir, image), le modèle,
   les unités et la convention de Fourier;
3. documenter les constantes instrumentales : valeur/source,
   gamme de validité ou état `unverified`;
4. apporter un test analytique et un test synthétique indépendant;
5. conserver les anciens résultats de référence pour audit de
   régression, y compris les valeurs propres négatives non physiques;
6. expliquer pourquoi toute différence n'est **pas** une calibration
   empirique cachée sur l'observation scientifique.

Le **PSF fitting** est un produit de la prochaine phase : il pourra
ajuster des paramètres explicitement définis avec une incertitude,
**sans redéfinir la théorie ni modifier le noyau de reconstruction**.
