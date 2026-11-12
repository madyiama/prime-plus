# Refactor PSF-R — périmètre, paradigme, invariants

**Branche :** `refactor/psfr-core`; base = `main` au 12/11/2026.
**Statut :** isolation technique initiale, **aucune modification de la physique**.

## Architecture fonctionnelle cible

À terme, remplacer les objets à état mutable par des fonctions explicites
sur des structures de données simples (dataclasses uniquement pour les
données validées, si cela apporte de la lisibilité) :

1. `load_observation(fits, sav, calibrations, config) -> observation`
2. `estimate_ao_statistics(observation) -> covariance_and_atmosphere`
3. `prepare_otf_terms(statistics, geometry) -> precomputed_terms`
4. `reconstruct_psf(precomputed_terms, wavelengths, directions) -> psf_cube`
5. `render_stars(psf_cube, fluxes, positions, sed, background) -> image`
6. `fit_stars(image, ..., parameters) -> fit_result` (étape future)

**Important :** les cinq fonctions ci-dessus décrivent la cible et ne sont
pas encore toutes implémentées. Le constructeur `psfR` actuel demeure un
adaptateur historique pendant la migration. Aucun modèle purement
paramétrique `psfao21` / MAOPPY n'est nécessaire pour la PSF-R.

## Périmètre initial de cette PR

- Supprimer `p3/deepLoop`, `p3/psfao21` et les tests exploratoires
  `p3/testing` qui dépendent de MAOPPY.
- Conserver `aoSystem`, `telemetry`, `psfr` et, temporairement,
  `psfFitting`, tant que les vrais liens de dépendance n'ont pas été
  remplacés et validés.
- Conserver `psfrRun.py` comme script hérité non public, puis le
  retirer quand l'exécution via API/CLI sera vérifiée.
- Introduire `p3.psfr.products.normalize_psf`, transformation pure
  vers flux unité **sur le support de pixels fourni**. Ce n'est ni un
  modèle chromatique ni un fitter.
- Préparer un benchmark synthétique du noyau Vii sans optimisation
  mathématique ni modification des résultats.

## Invariants physiques / numériques à protéger

1. Covariances HO et TT hermitiennes/symétriques, PSD ou erreur explicite ;
   `strict` reste le comportement nominal Keck.
2. OTF(0)=1 ; aucune double application de la transmission spectrale,
   de l'OTF pixel ou de l'anisocinétisme (`dani_tt`).
3. Conservation du flux d'une PSF, non-négativité, convention FFT, pupille,
   orientation et unités.
4. Résultat identique à la baseline avant toute optimisation (comparaison
   élément par élément + SR/FWHM/encircled energy et diagnostics).
5. Paramètres stellaires (flux/position/fond) ne doivent plus déclencher
   l'estimation des statistiques AO.
6. Les tests sur données réelles doivent inclure plusieurs couples
   FITS/SAV Keck en NGS et LGS ; ne jamais traiter la conformité au
   modèle de télémétrie comme une calibration absolue de STRAP.

## Risque scientifique majeur : spectre

`psfR.__call__` passe encore par
`FourierUtils.sf_3D_to_psf_3D` : cette fonction construit essentiellement
une PSF à `freq.wvlRef` et la repondère ensuite avec les flux des bandes.
Le chemin `sf_3D_to_psf_4D` est distinct et comporte lui-même des
hypothèses d'unités. Ne pas affirmer que la PSF large bande est juste
avant tests analytiques : Airy(λ), phase ~(λref/λ)^2, loi de pixel,
dispersion, F(λ) T(λ) Δλ, conservation de flux et convergence en
nombre de bins. Les tests de normalisation seuls sont insuffisants.

## Performance : protocole, puis optimisation

1. Mesurer temps, mémoire RSS et précision du pipeline sur les mêmes
   observations NIRC2 et la même configuration.
2. Profilage : lecture SAV / estimation HO+TT / Vii / PSD-FFT /
   anisoplanétisme / projection PSF.
3. Extraire les invariants de la boucle d'évaluation : matrices de
   modes, pupil autocorrelation, géométrie, grilles FFT, interpolation.
4. Optimiser Vii par FFT en lots/chunks avec taille mémoire contrôlée,
   puis benchmark CPU ; envisager GPU seulement avec bénéfice mesuré.
5. Garder `main` et la configuration `psfr_keck_v1.toml` comme
   référence numérique. Aucune approximation de physique pour gagner
   du temps sans audit quantitatif.

Commandes :

```bash
uv run --with pytest python -m pytest -q tests/test_psfr_products.py
uv run python scripts/bench_psfr_modes2otf.py --n-pupil 64 --n-modes 64 --trials 5
uv run --with pytest python -m pytest -q
```
