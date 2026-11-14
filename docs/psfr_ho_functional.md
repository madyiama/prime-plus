# PSF-R functional HO boundary (stage 2)

Branch: refactor/psfr-functional-ho (stacked on refactor/psfr-core).
No change to main, Keck V1 configuration, modal basis, sampling, or noise model.

- Pure estimate_ho_statistics converts actuator OPD measurements to a raw
  second-moment covariance [m^2] and a noise-subtracted phase covariance
  [rad^2] at the reference wavelength.
- DM method uses differences of consecutive commands / RTC gain, without
  removing the mean. Slopes method takes **already reconstructed** DM-basis
  residual OPD. Inputs are not raw WFS centroids.
- Pure reconstruct_ho_otf keeps the exact existing modes2Otf Vii/Uij
  Fourier implementation and the strict PSD validation by default.
- Legacy psfR.residual_phase_structure_function adapts the telemetry objects
  to this API. Its self.Cao diagnostics remain the original raw OPD covariance.
- No HO noise estimator, eigenvalue policy, or Keck physical calibration
  is changed. This stage is a software boundary, NOT a new instrument model.
- No end-to-end Keck FITS/SAV science validation is claimed. The tests
  reproduce historical algebra and exercise the real modal Fourier kernel.

Commands:

    uv run --with pytest python -m pytest -q tests/test_psfr_ho_functional.py
    uv run --with pytest python -m pytest -q

Next: separate prepared AO kernels from functional SF assembly and then
develop the chromatic PSF renderer under independent analytic tests.
