# P3 - fourier PSF

P3 is a collection of libraries for Adaptive Optics (AO) Point Spread Function (PSF) modeling, simulation, fitting and reconstruction.

## Keck/NIRC2 telemetry PSF-R V1 (MADYIAMA)

The stable **numerical** baseline for Keck LGS/NIRC2 is described in
[docs/psfr_keck_v1_release_acceptance.md](docs/psfr_keck_v1_release_acceptance.md).
The single locked recipe lives in [config/psfr_keck_v1.toml](config/psfr_keck_v1.toml)
and its end-to-end gate is [scripts/keck_psfr_v1_e2e.py](scripts/keck_psfr_v1_e2e.py).

For one real observation:

```bash
uv run --with pytest python -m pytest -q tests/test_keck_psfr_v1_release.py
uv run python scripts/keck_psfr_v1_e2e.py \\
  --fits /path/to/n0120.fits \\
  --sav /path/to/n0120_LGS_trs.sav \\
  --calib p3/aoSystem/data/KECK_CALIBRATION \\
  --output results/psfr_v1_acceptance
```

This baseline uses **van Dam (65–95% Nyquist, Welch 512), legacy_psfr,
unity TT plant, Noll angular OTF, 7 atmospheric layers, physical Keck pupil
resampling and strict HO covariance**. Angular anisokinetism is already
modelled by P3 `dani_tt`; do **not** inject it into the NGS residual jitter.
No science-image fitting or unverified STRAP optical gain correction is applied.
Instrument-specific absolute TT calibration and HO atmospheric inference
remain scientifically provisional. Exploratory audit scripts P4-B–G are not
part of the production recipe.


The list of libraries is:

* [aoSystem](https://github.com/astro-tiptop/P3/tree/main/p3/aoSystem) → The main purpose of this library is to model an AO PSF. It is an analytical simulator based on a Fourier appoach.
Supported wavefront sensors are Shack-Hartmann and Pyramid.
Several kind of AO are supported: single conjugate AO (SCAO), single laser AO (SLAO), laser tomography AO (LTAO), ground layer AO (GLAO), multi conjugate AO (MCAO), but only one kind of guide star can be used (natural or laser).
In [TIPTOP](https://github.com/astro-tiptop/TIPTOP) it is used to compute the so-called high order PSF.<br>
Examples and tests can be found in [p3/aoSystem/testing](https://github.com/astro-tiptop/P3/tree/main/p3/aoSystem/testing).<br>
Examples of parameters file can be found in [p3/aoSystem/parFiles](https://github.com/astro-tiptop/P3/tree/main/p3/aoSystem/parFiles).
* [deepLoop](https://github.com/astro-tiptop/P3/tree/main/p3/deepLoop)
* [psfFitting](https://github.com/astro-tiptop/P3/tree/main/p3/psfFitting) → This library can be used to fit parameters on a PSF model (aoSystem/fourierModel or psfao21).<br>
Examples and tests can be found in [p3/testing](https://github.com/astro-tiptop/P3/tree/main/p3/testing).
* [psfao21](https://github.com/astro-tiptop/P3/tree/main/p3/psfao21) → This library can be used to model with a small number of parameters the AO PSF. Its main purpose is PSF fitting.<br>
**Note**: This library **requires** the maoppy repo from [gitlab.lam.fr/lam-grd-public/maoppy](https://gitlab.lam.fr/lam-grd-public/maoppy).<br>
Examples and tests can be found in [p3/testing](https://github.com/astro-tiptop/P3/tree/main/p3/testing).
* [psfr](https://github.com/astro-tiptop/P3/tree/main/p3/psfr) → The main purpose of this library is to reconstruct PSF from telemetry data or from image fitting.<br>
Examples and tests can be found in [p3/testing](https://github.com/astro-tiptop/P3/tree/main/p3/testing).
* [telemetry](https://github.com/astro-tiptop/P3/tree/main/p3/telemetry) → This library is used to manage telemetry data for PSFR.<br>
Examples and tests can be found in [p3/testing](https://github.com/astro-tiptop/P3/tree/main/p3/testing).
