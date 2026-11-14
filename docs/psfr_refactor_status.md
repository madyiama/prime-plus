# Consolidation PSF-R et fitting — 14 novembre 2026

**Cible :** refactor/psfr-core (PR #30 vers main, encore draft).
**PR d'intégration :** [#50](https://github.com/madyiama/prime-plus/pull/50), fusionnée dans cette branche (commit 75d235f8).
Le contenu ci-dessous décrit l'état effectif de refactor/psfr-core ; main reste inchangée.

## PR et dépendances

| PR | Fonction | Statut |
| --- | --- | --- |
| #31/#32 | Statistiques HO et structure de fonction pure | Fusionnées dans refactor/psfr-core |
| #33/#35 | Profils NIRC2 et noyau optique monochromatique isolé | Fusionnées dans refactor/psfr-core |
| #34 | Profilage et horodatages structurés | Fusionnée et conservée |
| #36 | Export de template PSF-R et provenance | Fusionnée |
| #37 | Fitting stellaire, flux/positions/fond | Fusionnée |
| #38 | Détection guidée par PSF et rejet des pics isolés | Fusionnée |
| #39 | Fitting AO contraint sur base gelée | Fermée, remplacée par PR #50 fusionnée |
| #40/#42 | Géométrie focale symétrique/cache | PR42 fermée et remplacée par PR #50 fusionnée |
| #44–#47 | HO Vii factorisé/FFT réelle, quadrant et 4 workers | Code reporté par PR #50 |
| #41 | Débogage SIGKILL et tentative de réduction de mémoire Vii | Draft, non intégrée sans validation Linux réelle |
| #43 | FFT HO en lots | Fermée sans fusion |
| #48/#49 | Documentation technique et scientifique | Intégrée/actualisée dans PR #50 |

Les anciennes PR #39 et #42 ont été fermées comme remplacées
après la fusion de PR #50. Elles n'ont pas été fusionnées historiquement,
mais leur code a été consolidé via PR #50.

## Paramètres par défaut de PSF-R

- Covariance HO : strict PSD, aucune projection silencieuse.
- Calcul Vii HO : **factorized_real**, FFT réelles unilatérales.
- Focal : **quadrant**, six termes Flicker complets, même noyau Ialpha.
- CPU focal : **4 workers** quand quadrant est choisi ; 1 pour legacy/symmetry.
- Cache disque : OFF par défaut ; seules les SF focales géométriques
  indépendantes de la pupille sont éligibles si activées.
- Les chemins legacy HO et focal restent disponibles par options explicites.
- Le fitting stellaire et AO est présent mais n'est **jamais déclenché**
  par défaut au cours de la reconstruction télémétrique.

La recette physique Keck V1 conserve van Dam, legacy_psfr TT,
plante unity, noyau Noll angular, sept couches, pupille
physical_binary/header/native et tests de covariance HO stricts.
Les optimisations numériques ne représentent pas une calibration
instrumentale absolue de STRAP ou de Keck.

## Architecture livrée pour le Keck PSF Lab

- p3/psfr/psfR.py : reconstruction et diagnostics.
- p3/psfr/ho.py, hoFactorized.py, structure.py : HO et SF.
- p3/aoSystem/focalFast.py, focalCache.py : géométrie rapide et cache.
- p3/psfr/template.py : template de PSF et provenance.
- p3/psfr/stellar.py, detection.py : détection et fitting des étoiles.
- p3/psfr/ao_basis.py, ao_fit.py : ajustement de r0 et HO sur SF gelées.
- scripts/keck_psfr_v1_e2e.py : validation numérique complète.
- Scripts export_psfr_template.py, fit_psfr_stars.py et fit_psfr_ao.py :
  étapes de fitting explicites et indépendantes.

Le changement de défaut est spécifique à la pile consolidée :
on peut toujours reconstruire avec les méthodes historiques.

## Résultats et points encore ouverts

- Mesure Keck n0120, 4 vs 1 worker : calcul focal 5,800 → 2,575 s
  (2,25×), PSF finale 128×128×3 identique bit à bit.
- Autre mesure optimisée n0120 : initialisation du modèle 8,287 s,
  HO SF 5,135 s, domaine fréquentiel 2,327 s.
- Temps E2E externe dans une autre exécution : 29,257 s.
  Les mesures internes sont incluses dans ce temps, non additionnables.
- Ancienne CI de PR #42 : 856 tests réussis ; la CI du nouvel
  assemblage #50 est une validation indépendante indispensable.
- La validation scientifique sur plusieurs acquisitions LGS/NGS,
  orientations NIRC2, niveaux HO et TT, et le chromatisme réel
  reste à poursuivre.
- PR41 n'est pas une correction de SIGKILL démontrée. En Docker Linux,
  mesurer RSS et cgroup memory.events avant de conclure à un OOM.

Pour l'exécution Mac, les options legacy et le raccordement avec
Keck PSF Lab, voir [le guide d'intégration](psfr_keck_lab_integration.md).
Pour les algorithmes, voir [le guide de performance](psfr_perf_implementation_guide.md).
