# PRIME+ — PSF reconstruction (P3 core)

Research framework for AO telemetry-based point spread function
reconstruction, initially for Keck II/NIRC2. This refactor removes the
DeepLoop and PSFAO21 research packages; the legacy P3 engine is retained
until an equivalent, validated functional pipeline replaces it.

## Scientific baseline: Keck/NIRC2 V1

The stable **numerical** Keck LGS baseline is described in
[docs/psfr_keck_v1_release_acceptance.md](docs/psfr_keck_v1_release_acceptance.md).
Use the locked recipe [config/psfr_keck_v1.toml](config/psfr_keck_v1.toml)
and end-to-end gate [scripts/keck_psfr_v1_e2e.py](scripts/keck_psfr_v1_e2e.py).

```bash
uv run --with pytest python -m pytest -q tests/test_keck_psfr_v1_release.py
uv run python scripts/keck_psfr_v1_e2e.py \
  --fits /path/to/n0120.fits \
  --sav /path/to/n0120_LGS_trs.sav \
  --calib p3/aoSystem/data/KECK_CALIBRATION \
  --output results/psfr_v1_acceptance
```

Nominal V1: van Dam STRAP noise (65–95% Nyquist, Welch 512),
legacy_psfr TT covariance, unity TT plant, Noll angular TT OTF,
7-layer atmosphere, physical Keck pupil, and strict HO PSD checks.
No science-image fitting or unverified instrument gain calibration is
applied. This is **not** a validated absolute Keck STRAP calibration.

## Package boundaries (transition state)

- `p3/psfr` — telemetry reconstruction engine + pure PSF product helpers.
- `p3/telemetry` — instrument telemetry and diagnostics.
- `p3/aoSystem` — optical physics, Fourier operators, pupil and filters
  currently required by PSF-R; not yet narrowed to only necessary code.
- `p3/psfFitting` — retained legacy fitting utilities; future refactor
  will separate physical PSF production from stellar image rendering.
- `scripts/` — reproducible Keck end-to-end validation and diagnostics.

[Functional architecture, scientific contracts and performance plan](docs/psfr_core_refactor.md).

**Experimental performance implementation and measured Keck results:**
[algorithm map, code paths, numerical acceptance, remaining scientific
limits and PR dependency policy](docs/psfr_perf_implementation_guide.md).
These opt-in methods are developed on `perf/psfr-focal-cache-profiling`
(PR #42), which currently includes the unfinished fitting stack (#36–#39).
They are **not part of the independent `refactor/psfr-core` branch or `main`**.
The historical `legacy` HO/focal algorithms remain the defaults.


### Flux-normalized PSF product (new, representation only)

```python
from p3.psfr.products import normalize_psf

# Input is a forward *kernel* with zero background, e.g.
# legacy P3's (y,x,source,wavelength) or (y,x,wavelength).
product = normalize_psf(psf_cube)
assert abs(product.image.sum() - 1.0) < 1e-12
```

Spectral weights in the input are retained, not re-applied. The legacy
chromatic PSF model still needs independent physical validation.
For performance baseline (synthetic Vii only):

```bash
uv run python scripts/bench_psfr_modes2otf.py --trials 5
```

The original P3 project authors and BSD-3-Clause license remain credited
in the package metadata and LICENSE file.
