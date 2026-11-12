Stage 3: pure PSF-R phase structure function assembly

Branch refactor/psfr-functional-sf stacks on refactor/psfr-core, never main.
Precomputed HO, TT, fit, alias and anisoplanatism kernels are explicit inputs.
Existing r0^-5/3, wavelength scaling, 1/3/7 layers, source axes, and
per-source OTF central zero-lag subtraction are unchanged.
Legacy psfR remains an adapter and retains anisoplanatism diagnostics.
This is not yet a polychromatic PSF renderer or physical Keck validation.
