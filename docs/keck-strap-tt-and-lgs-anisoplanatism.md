# Keck STRAP tip-tilt and LGS anisoplanatism: physical conventions

This note accompanies the experimental reconstruction fix on the Keck
NIRC2 2020-12-02 LGS dataset. **It is not a validation of the detector-noise
estimator**, and the FITS target LP415-20 is a binary, not a single reference
point source.

## STRAP telemetry units

The Keck STRAP `DTTCENTROIDS` stream is treated as an angular displacement in
**arcsec** after the quad-cell `APD_IMAT` calibration. The empirical factors
`0.18` and `3.2` previously multiplied the centroid stream. They are
removed here. The 2020 telemetry reproduces `DTTCENTROIDS` from
`APDCOUNTS` and `APD_IMAT` to numerical precision, supporting the conclusion
that the instrument matrix is already applied.

For RMS-normalized Noll Zernike tilts
`Z2 = 2*rho*cos(phi)`, `Z3 = 2*rho*sin(phi)`,
the RMS OPD coefficient `a` for angular displacement `theta` is
`a = D*theta/4`. Therefore the input conversion from arcsec to metres of
RMS OPD is `D * pi/(180*3600) / 4`. With `D=10.5m` this is
`12.7264 micrometres RMS OPD / arcsec`.

**Caveat:** `DTTCOMMANDS` is now explicitly retained in a raw attribute and
is temporarily assumed to share the arcsec convention when converted to
RMS OPD. This command-unit assumption has **not** yet been independently
verified against the Keck RTC documentation. No calibration factor is invented
to force agreement. Validate this before interpreting command-derived
covariances scientifically. The historical NGS telemetry path likewise has
not been separately recalibrated.

## Angular jitter OTF

Convert RMS OPD coefficient covariance `C_a [m^2]` into angular covariance
`C_theta [rad^2] = (4/D)^2 C_a`.

For pupil-baseline coordinates `(u,v)` in units of aperture diameter, the
Gaussian jitter phase structure function is

`D_phi = (2*pi*D/lambda)^2 [u,v] C_theta [u,v]^T`.

The corresponding OTF factor is `exp(-D_phi/2)`. This is exactly
`exp(-2*pi^2 * f^T C_theta f)` for angular spatial frequency
`f=(D/lambda)[u,v]`.

P3's `freq.U_`, `freq.V_` are multiplied by `sampRef/2` and rotated by
the pupil angle before insertion; analytical unit tests check the OTF against
the Gaussian Fourier transform. This avoids inserting an empirical optical
gain. The factor `4` arises only when expressing the same tilt in normalized
Zernike OPD, not when the covariance is already angular.

## Covariance and unresolved detector noise

The existing one-frame autocorrelation estimate `Cn_tt` and noise
propagation factor `pn` are *unchanged*. The historical decomposition is:

- `C_atmos_raw = C_meas - (1+pn)*Cn_tt`
- `C_noise_prop = pn*Cn_tt`
- `C_jitter = C_atmos + C_noise_prop`

On `n0013`, `C_atmos_raw` has negative eigenvalues. This is not a
physical covariance. The implementation explicitly warns and projects each
covariance term onto the positive semidefinite (PSD) cone by clipping negative
eigenvalues to zero. It records the **raw** traces and eigenvalues as well as
the projected terms, marks `noise_model_validated=False`, and uses
the same PSD components in both the OTF and the WFE breakdown.

**PSD projection is a numerical/physical constraint only**, not a repaired
estimate of the true atmospheric tip-tilt. It can strongly underestimate
the real jitter when `Cn_tt` is biased. Future work must calibrate the STRAP
shot/read noise and excess noise against APD counts and inspect the temporal
PSD with the loop transfer function. The covariance estimate is not declared
scientifically reliable merely because the output is finite.

## LGS cone-effect anisoplanatism

The LGS focal-anisoplanatism contribution was computed but skipped by
`psfR.phase_structure_function` whenever `nLayer=1`, since the one-layer
`sort_params_from_labels` returns `Cn2=None`.

For one layer, derive `Cn2 = atm.weights * r0^(-5/3)` from the fitted
`r0`. For a multilayer fit use the supplied per-layer `Cn2`.
Before adding the anisoplanatic phase-structure function to the science OTF,
scale the coefficient by `(lambda_atm/lambda_ref)^2`, as in P3's
`get_error_breakdown`. This affects both one- and multilayer PSF-R
consistently.

## Validation commands

```bash
uv run --with pytest python -m pytest -q tests/test_keck_tt_geometry_and_anisoplanatism.py
uv run --with pytest python -m pytest -q
uv run python scripts/keck_n0013_smoke.py \
  --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
  --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
  --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
  --output "$PWD/results/n0013"
```

Compare `comparison_nirc2_psfr.png`, `tiptilt_diagnostics.json`, and
`telemetry_only_psf.npy` with the earlier run. This comparison remains
qualitative until the binary is fitted, background is removed rigorously,
and a single-star reference dataset is included.
