# Keck STRAP noise-estimation sprint — van Dam / Gendron

## Scope

The high-order WFS estimator is unchanged. The LGS/STRAP tip-tilt
measurement noise is selectable from the project configuration
config/keck_tt_noise.toml. The default is van_dam. The chosen method and
PSD settings are recorded in the generated Keck INI file.

This is an *estimator implementation and benchmark*, not a validated
noise budget. The STRAP centroid units were corrected in PR #2; this PR
operates only on their calibrated RMS-Zernike OPD timeseries.

## Methods

### van_dam (default)

Motivated by van Dam et al. (2006), PASP 118, 310–318, section 2.4.
For the two-axis residual centroid series y(t), compute a one-sided Welch
PSD P_y(f) (m²/Hz) and cross-PSD. Use the rejection response of the
actual STRAP controller from the P3 telemetry, S(f)=1/(1+H(f)).

In a high-frequency band assumed to be dominated by white measurement
noise, P_y(f) ~= |S(f)|² N_0. Divide P_y by |S|², average the high-frequency
spectral floor using a 10% trimmed mean to reduce narrow peaks, and infer

    C_n = (fs / 2) * N_0,

where N_0 is the fitted 2x2 matrix of one-sided white spectral densities.
All m²/Hz and m² units are explicit. Negative eigenvalues in the spectral
noise-covariance estimate are projected to PSD with a warning and the raw
eigenvalues are exported.

**Assumptions and limitations:**

- The transfer S(f) and centroid gain have to be calibrated for the
  observed Keck control loop. This PR corrects the TT-specific boxcar
  integration response to sinc(f/fs)*exp(-i*pi*f/fs) and the pure delay
  to exp(-2*pi*i*f*delay_seconds) (analytic unit tests included).
  The servo filter, effective centroid gain and timing/latency calibration
  remain to be verified with STRAP observations.
- The HO transfer function remains unchanged in this TT-specific sprint;
  its historical frequency/phase conventions must be audited separately.
- Consistency warning compares the *predicted filtered noise contribution*
  integral(|S|² * N0 df) with the observed covariance. A conservative
  spectral-sampling tolerance is heuristic, not a formal confidence interval;
  diagnostics retain the comparison and its threshold.
- The selected high-frequency band must be dominated by measurement
  noise; atmospheric power and vibration peaks can bias the result.
- The procedure is not the complete van Dam calibration: it does not
  fit the centroid gain, model APD excess noise, or perform the
  Monte-Carlo consistency check from the original paper.
- The existing downstream model C_atmos = Ctt - (1+pn)*Cn_tt is not
  modified. Its interpretation must still be checked against the
  closed-loop PSD and the meaning of Cn_tt.

### gendron

Gendron & Lena (1995) and subsequent AO telemetry work estimate the
zero-lag noise spike by assuming the atmospheric covariance is smooth
and even near zero lag. Fit an even parabola through C(1), C(2):

    C_atmos(0) ~= (4*C(1) - C(2)) / 3,
    C_noise ~= C(0) - C_atmos(0).

This requires turbulent residuals that remain sufficiently correlated
over two frames. It is distinct from the historical P3 one-lag method.

### legacy_autocorrelation

The exact previous P3 one-frame cyclic estimator:

    C_noise = C(0) - C_circular(1).

Kept for bit-for-bit scientific regression comparison; do not label
this method a full Gendron-Lena estimator.

## Configuration

Edit config/keck_tt_noise.toml:

    [tip_tilt_noise]
    method = "van_dam"
    high_frequency_band = [0.65, 0.95]
    welch_nperseg = 512

The band is expressed as fractions of fs/2 (Nyquist). Choose gendron
or legacy_autocorrelation with the method key. CLI overrides:

    --tt-noise-method van_dam|gendron|legacy_autocorrelation
    --tt-noise-band 0.65 0.95
    --tt-welch-nperseg 512
    --config path/to/keck_tt_noise.toml

This selection happens in systemDiagnosis, *before* configFile writes
the generated instrument INI. The generated sensor_LO/NoiseEstimationMethod
entry is an audit record; manually editing the generated INI after
systemDiagnosis will not change the already estimated covariance.

## Test and observation runs

    uv run --with pytest python -m pytest -q tests/test_tt_noise_estimators.py tests/test_keck_tt_noise_configuration.py
    uv run python scripts/keck_n0013_smoke.py \
      --fits "/Users/obm/Documents/KECK DATA/20201202/IMAG/n0013.fits" \
      --sav "/Users/obm/Documents/KECK DATA/20201202/TRS/n0013_LGS_trs.sav" \
      --calib "$PWD/p3/aoSystem/data/KECK_CALIBRATION" \
      --output "$PWD/results/n0013"

Outputs include tt_noise_diagnostics.json and (van_dam only)
tt_noise_psd.png, in addition to the pre-existing PSF comparison and
tiptilt_diagnostics.json.

Compare both van_dam and gendron without interpreting a better fit to
the LP415-20 *binary* as proof of physical validity. Assess spectral
stability across several fitting bands, transfer-function gain
sensitivity, corrected covariance eigenvalues, and uncertainty from
multiple temporal segments before selecting a calibrated estimator.

## References

- van Dam et al. 2006, "The W. M. Keck Observatory Laser Guide Star
  Adaptive Optics System: Performance Characterization", PASP 118, 310.
  https://www2.keck.hawaii.edu/optics/aodocs/PASP06_MvD.pdf
- Gendron & Lena (1995), temporal slope noise from zero-lag
  autocorrelation peak and quadratic extrapolation.
- Guesalaga et al. (2017), "Online estimation of the wavefront outer
  scale profile from adaptive optics telemetry", MNRAS 465, Appendix:
  high-frequency spectral contamination and limitations of lag-1
  estimates.
  https://doi.org/10.1093/mnras/stw2548
