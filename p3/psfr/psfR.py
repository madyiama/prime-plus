#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:20:28 2018

@author: omartin
"""
#%% MANAGE PYTHON LIBRAIRIES
import time
import warnings
import sys as sys

import numpy as np
import numpy.fft as fft

from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.fourierModel import fourierModel
import p3.aoSystem.FourierUtils as FourierUtils
import p3.psfr.psfrUtils as psfrUtils
from p3.psfr.ho import estimate_ho_statistics, reconstruct_ho_otf
from p3.psfr.structure import (
    assemble_phase_structure_function,
    normalize_anisoplanatism_zero_lag as _anisoplanatism_zero_lag,
)
from p3.aoSystem.frequencyDomain import frequencyDomain as frequencyDomain
from p3.aoSystem.ttFrame import sensor_frame_frequencies
from p3.aoSystem.ttAngular import (
    direct_angular_tt_covariance,
    assert_equivalent_angular_covariances,
    angular_otf_frequencies,
    gaussian_tt_otf,
)
from p3.aoSystem.tipTiltGeometry import (
    opd_covariance_to_angular,
    project_covariance_to_psd,
    angular_jitter_structure_function,
)
from p3.telemetry.ttClosedLoop import physical_tt_covariance_from_telemetry

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000


def _sqrt_checked_variance(variance, label):
    """Keep negative noise-subtracted variances invalid rather than masking them."""
    variance = float(variance)
    if not np.isfinite(variance) or variance < 0:
        warnings.warn(
            f"{label}: invalid noise-subtracted variance {variance:.6e}; "
            "check tip-tilt noise estimation and loop transfer model.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan
    return np.sqrt(variance)


def _integrate_spectral_psf(psf):
    """Combine spectral slices into the 2-D broadband image for a Strehl check."""
    if psf.ndim == 2:
        return psf
    if psf.ndim == 3:
        return np.sum(psf, axis=-1)
    raise ValueError("Expected a 2-D PSF or a 3-D wavelength cube")


class psfR:
    """
    """
    # INIT
    def __init__(self, trs, path_root='', nLayer=None, theta_ext=0,
                 compute_dphi_jitter=True, tt_covariance_mode='legacy_psfr',
                 tt_mirror_model='unity', tt_welch_nperseg=512,
                 ho_covariance_policy='strict',
                 tt_kernel_convention='noll_angular',
                 tt_axis_angle_deg=None, tt_axis_parity='native',
                 focal_algorithm='legacy', focal_cache_dir=None,
                 ho_vii_method='legacy'):
        """
        Instantiating the psf-reconstruction model from a telemetry object
        """

        # INSTANTIATING THE MODEL OF THE AOSYSTEM
        tstart = time.time()
        _start_ao_perf = time.perf_counter()

        if not hasattr(trs,'path_ini'):
            raise ValueError("no .ini file attached with the telemetry object")

        self.path_ini = trs.path_ini
        self.trs = trs
        self.theta_ext = theta_ext
        self.compute_dphi_jitter = compute_dphi_jitter
        if tt_covariance_mode not in ('legacy_psfr', 'spectral_closed_loop'):
            raise ValueError("Unknown TT covariance mode: " + str(tt_covariance_mode))
        if tt_mirror_model not in ('unity', 'ragland_2016_keck_ii'):
            raise ValueError("Unknown TT mirror model: " + str(tt_mirror_model))
        if not isinstance(tt_welch_nperseg, (int, np.integer)) or tt_welch_nperseg < 128:
            raise ValueError("TT Welch segment length must be integer >= 128")
        # Independent TT axes: None preserves historical coupling to the
        # telescope's pupilAngle for bitwise-comparable old PSFs. An explicit
        # angle is a hypothesis pending STRAP/NIRC2 frame calibration.
        if tt_axis_angle_deg is not None and not np.isfinite(tt_axis_angle_deg):
            raise ValueError("TT sensor-to-camera axis angle must be finite")
        if tt_axis_parity not in ('native', 'flip_x', 'flip_y'):
            raise ValueError("Unknown TT sensor-to-camera parity")
        self.tt_axis_angle_deg = (None if tt_axis_angle_deg is None
                                  else float(tt_axis_angle_deg))
        self.tt_axis_parity = tt_axis_parity
        self.tt_covariance_mode = tt_covariance_mode
        self.tt_mirror_model = tt_mirror_model
        self.tt_welch_nperseg = int(tt_welch_nperseg)
        if tt_kernel_convention not in ('noll_angular', 'legacy_unscaled_opd'):
            raise ValueError("Unknown TT OTF kernel convention: " + str(tt_kernel_convention))
        self.tt_kernel_convention = tt_kernel_convention
        # Diagnostic-only alternative: the original P3 kernel omits the
        # factor 16 when Ctt is interpreted as RMS Noll tilt coefficients.
        # NEVER select based on science-image fit without angular calibration.
        self.tt_kernel_dphi_multiplier_vs_noll = (
            1.0 if tt_kernel_convention == 'noll_angular' else 1.0/16.0
        )
        if ho_covariance_policy not in ('strict', 'tiny_negative', 'small_negative'):
            raise ValueError("Unknown HO covariance policy: " + str(ho_covariance_policy))
        if ho_covariance_policy == 'small_negative' and str(trs.aoMode).upper() != 'NGS':
            raise ValueError(
                "HO small_negative regularization is exploratory and NGS-only; "
                "use tiny_negative for the tighter LGS/NGS diagnostic policy"
            )
        self.ho_covariance_policy = ho_covariance_policy
        self.ho_covariance_diagnostics = {}
        if ho_vii_method not in ('legacy','factorized','factorized_real'):
            raise ValueError("Unknown HO Vii method")
        self.ho_vii_method = ho_vii_method
        self.ho_vii_diagnostics = {}
        self.ao = aoSystem(self.path_ini, path_root=path_root)

        # defining the number of model parameters
        self.tag = 'PSF-R'
        self.param_labels = ['Cn2', 'gho', 'gtt', 'F', 'dx', 'dy', 'bkg', 'stat']
        self.n_param_atm = min(self.ao.atm.nL, self.ao.dms.nRecLayers)
        self.n_param_dphi = 2

        self.wvl, self.nwvl = FourierUtils.create_wavelength_vector(self.ao)

        # checking the pupil
        if self.ao.tel.path_pupil is None and np.any(trs.tel.pupil):
            self.ao.tel.pupil = self.trs.tel.pupil

        if self.ao.tel.path_static_on and hasattr(trs.tel,'map_ncpa')and np.any(trs.tel.map_ncpa):
            self.ao.tel.opdMap_on = trs.tel.map_ncpa
        self.t_initAO = 1000*(time.time() - tstart)
        self.timing_breakdown_ms = {
            "ao_system": 1000*(time.perf_counter()-_start_ao_perf)
        }


        # INSTANTIATING THE SPATIAL FREQUENCY AND THE FOURIER MODEL
        if focal_algorithm not in ("legacy", "symmetry"):
            raise ValueError("Unknown focal covariance algorithm")
        self.focal_algorithm = focal_algorithm
        self.focal_cache_dir = focal_cache_dir
        _start_dm_filter = time.perf_counter()
        self.Hfilter = 1
        if self.ao.aoMode=="SLAO":
            n_ph = 2*self.ao.dms.nActu1D[0] + 1
            modes = self.ao.dms.setInfluenceFunction(n_ph)
            #val_act = np.argwhere(modes.max(axis=0)==0)
            self.Hfilter = np.dot(modes, modes.T)
        self.timing_breakdown_ms["dm_influence_filter"] = 1000*(
            time.perf_counter()-_start_dm_filter
        )

        _start_frequency = time.perf_counter()
        self.freq = frequencyDomain(
            self.ao, Hfilter=self.Hfilter,
            focal_algorithm=self.focal_algorithm,
            focal_cache_dir=self.focal_cache_dir,
        )
        self.timing_breakdown_ms["frequency_domain"] = 1000*(time.perf_counter()-_start_frequency)
        _start_fourier = time.perf_counter()
        self.fao = fourierModel(self.path_ini, calcPSF=False, display=False,
                                freq=self.freq, ao=self.ao)
        self.timing_breakdown_ms["fourier_model"] = 1000*(time.perf_counter()-_start_fourier)

        self.t_freq = 1000*(time.time() - tstart) - self.t_initAO

        # INSTANTIATING THE FITTING PHASE STRUCTURE FUNCTION FOR r0=1m
        _start_fitting_sf = time.perf_counter()
        self.dphi_fit = self.fitting_phase_structure_function(1)
        self.timing_breakdown_ms["fitting_sf"] = 1000*(time.perf_counter()-_start_fitting_sf)
        self.t_dfit = 1000*(time.time() - tstart) - self.t_freq

        # INSTANTIATING THE ALIASING PHASE STRUCTURE FUNCTION FOR r0=1m
        _start_aliasing_sf = time.perf_counter()
        self.dphi_alias = self.aliasing_phase_structure_function(1)
        self.timing_breakdown_ms["aliasing_sf"] = 1000*(time.perf_counter()-_start_aliasing_sf)
        self.t_dalias = 1000*(time.time() - tstart) - self.t_dfit

        # INSTANTIATING THE TT RESIDUAL PHASE STRUCTURE FUNCTION IN LGS MODE
        _start_tt = time.perf_counter()
        if compute_dphi_jitter:
            self.dphi_tt = self.jitter_phase_structure_function()
        else:
            self.dphi_tt = 0*self.dphi_fit
        self.timing_breakdown_ms["tiptilt_sf"] = 1000*(time.perf_counter()-_start_tt)
        self.t_dtt = 1000*(time.time() - tstart) - self.t_dalias

        # INSTANTIATING THE AO RESIDUAL PHASE STRUCTURE FUNCTION
        _start_ho_sf = time.perf_counter()
        self.dphi_ao = self.residual_phase_structure_function()
        self.timing_breakdown_ms["ho_sf"] = 1000*(time.perf_counter()-_start_ho_sf)
        self.t_dao = 1000*(time.time() - tstart) - self.t_dtt

        # INSTANTIATING THE ANISOPLANATISM PHASE STRUCTURE FUNCTION IF ANY
        _start_aniso_transfer = time.perf_counter()
        if self.ao.lgs is None or self.ao.lgs.height==0:
            if self.freq.isAniso:
                self.dphi_ani = self.freq.dani_ang
        else:
            self.dani_focang = self.fao.freq.dani_focang
            self.dani_ang = self.freq.dani_ang
            self.dani_tt = self.freq.dani_tt
            self.dphi_ani = self.dani_focang + self.dani_tt

        self.timing_breakdown_ms["anisoplanatism_transfer"] = 1000*(
            time.perf_counter()-_start_aniso_transfer
        )
        # COMPUTING THE DETECTOR PIXEL TRANSFER FUNCTION
        _start_detector_otf = time.perf_counter()
        if self.trs.tel.name!='OOMAO':
            self.otfPixel = self.pixel_optical_transfer_function()
        else:
            self.otfPixel = 1.0

        self.timing_breakdown_ms["detector_pixel_otf"] = 1000*(
            time.perf_counter()-_start_detector_otf
        )
        # COMPUTING THE ERROR BREAKDOWN:
        _start_budget = time.perf_counter()
        self.get_error_breakdown()
        self.timing_breakdown_ms["error_budget_including_forward"] = 1000*(time.perf_counter()-_start_budget)

         # DEFINING BOUNDS
        self.bounds = self.define_bounds()

        self.t_init = 1000*(time.time()  - tstart)
        self.timing_breakdown_ms["psfr_init_total"] = float(self.t_init)

    def _repr__(self):
        return 'PSF-Reconstruction model'

    def define_bounds(self):
        """
            Defines the bounds for the PSF model parameters :
                Cn2/r0, gho, gtt, F, dx, dy, bg, stat
        """
        _EPSILON = np.sqrt(sys.float_info.epsilon)

        # Bounds on r0
        bounds_down = list(np.ones(self.ao.atm.nL)*_EPSILON)
        bounds_up = list(np.inf * np.ones(self.ao.atm.nL))
        # optical gains
        bounds_down += [0,0]
        bounds_up += [np.inf, np.inf]
        # Photometry
        bounds_down += list(np.zeros(self.ao.src.nSrc))
        bounds_up += list(np.inf*np.ones(self.ao.src.nSrc))
        # Astrometry
        bounds_down += list(-self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc))
        bounds_up += list( self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc))
        # Background
        bounds_down += [-np.inf]
        bounds_up += [np.inf]
        # Static aberrations
        bounds_down += list(-self.freq.wvlRef/2*1e9 * np.ones(self.ao.tel.nModes))
        bounds_up += list(self.freq.wvlRef/2 *1e9 * np.ones(self.ao.tel.nModes))
        return (bounds_down,bounds_up)

    def fitting_phase_structure_function(self, r0):
        """
        Computes the phase structure function of the DM fitting error.
        """
        psd = r0**(-5/3) * self.freq.psdKolmo_
        pix2freq = 1/(self.ao.tel.D * self.freq.sampRef)#2*self.freq.kc_/self.freq.resAO
        cov = FourierUtils.psd2cov(psd, pix2freq)
        return np.real(fft.fftshift(FourierUtils.cov2sf(cov)))

    def aliasing_phase_structure_function(self, r0):
        """
        Computes the phase structure function of the WFS aliasing error.
        """
        # computing the aliasing PSD over the AO-corrected area
        self.psdAlias_ = self.fao.aliasingPSD()/self.fao.ao.atm.r0**(-5/3)

        # zero-padding the PSD
        self.psdAlias_ = FourierUtils.enlargeSupport(self.psdAlias_,
                                                     self.freq.nOtf/self.freq.resAO)

        # computing the aliasing phase structure function
        pix2freq = 1/(self.ao.tel.D * self.freq.sampRef)
        cov = FourierUtils.psd2cov(self.psdAlias_, pix2freq)
        dphi_alias = r0**(-5/3) * np.real(fft.fftshift(FourierUtils.cov2sf(cov)))

        # interpolating the phase structure function if required
        if dphi_alias.shape[0] != self.freq.nOtf:
            dphi_alias = FourierUtils.interpolateSupport(dphi_alias,
                                                         self.freq.nOtf,
                                                         kind='spline')
        return dphi_alias

    def residual_phase_structure_function(self, method='dm-based', basis='Vii'):
        """HO SF via pure statistics plus the existing modal Fourier kernel.

        self.Cao remains the uncorrected actuator covariance [m**2] for
        the legacy error budget, and PSD diagnostics remain unchanged.
        """
        if method == 'dm-based':
            measurements = self.trs.dm.com
            gain = self.ao.rtc.holoop['gain']
        elif method == 'slopes-based':
            measurements = self.trs.rec.res
            gain = None
        else:
            raise ValueError("Unknown HO residual method: " + str(method))

        stats = estimate_ho_statistics(
            measurements, self.trs.wfs.Cn_ao,
            reference_wavelength_m=self.freq.wvlRef,
            method=method, loop_gain=gain,
        )
        self.Cao = stats.empirical_covariance_m2
        _, dphi_ao = reconstruct_ho_otf(
            stats, self.trs.mat.dmIF, self.ao.tel.pupil, self.freq.nOtf,
            sampling=self.freq.sampRef, basis=basis,
            covariance_psd_policy=getattr(self, 'ho_covariance_policy', 'strict'),
            covariance_diagnostics=getattr(self, 'ho_covariance_diagnostics', None),
            vii_method=getattr(self,'ho_vii_method','legacy'),
            vii_diagnostics=getattr(self,'ho_vii_diagnostics',None),
        )
        return dphi_ao

    def jitter_phase_structure_function(self, Ctt=None):
        """Compute the Gaussian image-motion structure function from angular jitter.

        Internal covariance inputs are RMS-normalized Zernike tilt OPD [m^2].
        Only the final OTF kernel is expressed in angle [rad^2]. Retain the
        current noise estimator, but make its non-PSD decomposition explicit.
        """
        initial_tt_model = Ctt is None
        if Ctt is None:
            s = self.trs.tipTilt.slopes
            self.Ctt = np.dot(s.T, s)/self.trs.tipTilt.nExp
            noise_raw = np.asarray(self.trs.tipTilt.Cn_tt, dtype=float)
            if self.tt_covariance_mode == 'spectral_closed_loop':
                if self.trs.aoMode != 'LGS':
                    raise ValueError(
                        "Spectral closed-loop jitter currently requires Keck LGS STRAP"
                    )
                self.tt_atmos_cov, self.tt_noise_cov, self.tt_model_cov, spectral = (
                    physical_tt_covariance_from_telemetry(
                        s, noise_raw, self.trs.ttloop,
                        mirror_model=self.tt_mirror_model,
                        nperseg=self.tt_welch_nperseg,
                    )
                )
                # Van Dam must identify Cn using the SAME W/K/P model as
                # this physical covariance calculation. Legacy objects
                # without model metadata are conservatively treated as
                # having used the original pure-delay P3 rejection.
                noise_method = getattr(
                    self.trs.tipTilt, 'noise_estimation_method', 'unknown'
                )
                rejection_mirror = getattr(
                    self.trs.tipTilt, 'noise_rejection_mirror_model', 'unity'
                )
                noise_transfer_mismatch = (
                    noise_method == 'van_dam'
                    and self.tt_mirror_model != rejection_mirror
                )
                if noise_transfer_mismatch:
                    warnings.warn(
                        "van Dam measurement-noise covariance and physical "
                        "TT propagation use different mirror models "
                        f"({rejection_mirror} vs {self.tt_mirror_model}); "
                        "the PSF is a sensitivity study only. Re-estimate "
                        "sensor noise with the selected W/K/P response.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                # Explicitly distinguish physical jitter from centroid noise.
                # No covariance clipping; component validity is checked upstream.
                self.tiptilt_diagnostics = {
                    'jitter_covariance_method': 'spectral_closed_loop',
                    'tt_mirror_model': self.tt_mirror_model,
                    'mirror_model_verified_for_2020': False,
                    'measurement_noise_estimator': noise_method,
                    'van_dam_rejection_model': (
                        'p3_pure_delay' if rejection_mirror == 'unity'
                        else rejection_mirror
                    ) if noise_method == 'van_dam' else None,
                    'noise_transfer_mismatch': noise_transfer_mismatch,
                    'tt_welch_nperseg': self.tt_welch_nperseg,
                    'raw_trace_m2': float(np.trace(self.Ctt)),
                    'noise_trace_m2': float(np.trace(noise_raw)),
                    'historical_pn_for_comparison': float(
                        self.trs.ttloop.tf.pn
                    ),
                    'physical_atmos_trace_m2': float(
                        np.trace(self.tt_atmos_cov)
                    ),
                    'physical_noise_trace_m2': float(
                        np.trace(self.tt_noise_cov)
                    ),
                    'physical_total_trace_m2': float(
                        np.trace(self.tt_model_cov)
                    ),
                    'physical_atmos_eigenvalues_m2': np.linalg.eigvalsh(
                        self.tt_atmos_cov
                    ).tolist(),
                    'physical_total_eigenvalues_m2': np.linalg.eigvalsh(
                        self.tt_model_cov
                    ).tolist(),
                    'covariance_projection_applied': False,
                    'noise_model_validated': False,
                    'command_units_verified': bool(
                        getattr(self.trs.tipTilt, 'command_units_verified', False)
                    ),
                    'gain_sensor_to_measured_noise':
                        spectral['gain_sensor_to_measured_noise'],
                    'gain_sensor_to_physical_noise':
                        spectral['gain_sensor_to_physical_noise'],
                    'physical_atmos_min_eig_m2':
                        spectral['physical_atmos_min_eig_m2'],
                    'physical_total_min_eig_m2':
                        spectral['physical_total_min_eig_m2'],
                    'spectral_covariance_audit': spectral,
                }
                Ctt = self.tt_model_cov
            else:
                Ctt = self._legacy_tiptilt_covariance(noise_raw)
        else:
            Ctt, _ = project_covariance_to_psd(
                Ctt, label="Explicit TT kernel covariance"
            )

        # Independent path: retain the historical OPD noise estimator, but
        # independently replay the complete physical covariance directly from
        # the archived Keck STRAP angular centroids. The D/4 and 4/D pair
        # must cancel numerically; otherwise do not emit a science PSF.
        Ctheta = opd_covariance_to_angular(Ctt, self.ao.tel.D)
        angular_audit = None
        if (initial_tt_model and str(getattr(self.trs, 'aoMode', '')).upper() == 'LGS'
                and hasattr(self.trs.tipTilt, 'slopes_arcsec')):
            from p3.aoSystem.tipTiltGeometry import zernike_tilt_opd_per_arcsec
            raw_arcsec = np.asarray(self.trs.tipTilt.slopes_arcsec, dtype=float)
            opd_measured = np.asarray(self.trs.tipTilt.slopes, dtype=float)
            if raw_arcsec.shape != opd_measured.shape:
                raise ValueError(
                    "P4 STRAP raw angular samples and OPD samples have different "
                    "shapes. Keck telemetry decimation currently does not decimate "
                    "slopes_arcsec or update the TT sample rate/loop transfer; "
                    "run P4 with decimation=1."
                )
            opd_per_arcsec = zernike_tilt_opd_per_arcsec(self.trs.tel.D)
            expected_opd = (raw_arcsec - raw_arcsec.mean(axis=0))*opd_per_arcsec
            scale = max(float(np.max(np.abs(expected_opd))), 1e-24)
            unit_error = float(np.max(np.abs(expected_opd-opd_measured)))/scale
            if raw_arcsec.shape != opd_measured.shape or unit_error > 2e-10:
                raise ValueError(
                    "DTTCENTROIDS angular-to-OPD conversion inconsistent: "
                    f"relative error={unit_error:.3e}; check D/4, arcsec and telemetry units"
                )
            direct_cov, direct_diag = direct_angular_tt_covariance(
                raw_arcsec, noise_raw, self.trs.tel.D,
                method=self.tt_covariance_mode,
                pn=float(self.trs.ttloop.tf.pn),
                ttloop=self.trs.ttloop,
                mirror_model=self.tt_mirror_model,
                nperseg=self.tt_welch_nperseg,
                native_noise_arcsec2=getattr(
                    self.trs.tipTilt, 'Cn_tt_arcsec2', None,
                ),
            )
            angular_audit = assert_equivalent_angular_covariances(
                Ctt, direct_cov, self.ao.tel.D,
            )
            angular_audit.update({
                'dt_centroids_input_unit': 'arcsec',
                'opd_intermediate_unit': 'm_rms_noll',
                'opd_intermediate_gain_m_per_arcsec': float(opd_per_arcsec),
                'centroid_opd_reencode_relative_error': unit_error,
                'direct_angular_replay': direct_diag,
            })
            Ctheta = direct_cov

        axis_override = getattr(self, 'tt_axis_angle_deg', None)
        tt_axis_angle = (
            float(self.trs.tel.pupilAngle) if axis_override is None
            else float(axis_override)
        )
        tt_axis_parity = getattr(self, 'tt_axis_parity', 'native')

        # Pupil-baseline and detector-pixel Fourier grids are checked
        # against each other. The returned frequencies are cycles/radian.
        pixel_scale = getattr(getattr(self.ao, 'cam', None), 'psInMas', None)
        nu_row, nu_col = angular_otf_frequencies(
            self.freq, self.ao.tel.D, self.freq.wvlRef,
            pixel_scale_mas=pixel_scale,
        )
        nu_sensor_row, nu_sensor_col = sensor_frame_frequencies(
            nu_row, nu_col, tt_axis_angle, parity=tt_axis_parity,
        )
        dphi, otf_tt = gaussian_tt_otf(
            Ctheta, nu_sensor_row, nu_sensor_col,
        )

        # Cross-check the previous Noll OPD/phase expression rather than
        # quietly changing the physical model during a noise audit.
        unit = self.freq.sampRef/2
        u_old, v_old = sensor_frame_frequencies(
            unit*self.freq.U_, unit*self.freq.V_,
            tt_axis_angle, parity=tt_axis_parity,
        )
        old_dphi = angular_jitter_structure_function(
            Ctheta, self.ao.tel.D, self.freq.wvlRef, u_old, v_old
        )
        model_scale = max(float(np.max(np.abs(dphi))), 1e-30)
        kernel_error = float(np.max(np.abs(dphi-old_dphi)))/model_scale
        if kernel_error > 2e-10:
            raise ValueError(
                "Direct-angular Gaussian TT OTF disagrees with the "
                f"Noll/OPD path: relative error={kernel_error:.3e}"
            )

        legacy_multiplier = getattr(
            self, 'tt_kernel_dphi_multiplier_vs_noll', 1.0
        )
        dphi_used = legacy_multiplier * dphi
        # get_error_breakdown() calls this function with component-specific
        # covariances (noise and atmosphere). Do not replace a previously
        # computed science TT OTF with the final diagnostic component OTF.
        # Keep standalone explicit-covariance use working for synthetic tests.
        if initial_tt_model or not hasattr(self, 'otf_tt'):
            self.otf_tt = np.exp(-dphi_used/2)
        if initial_tt_model:
            self.tiptilt_diagnostics.update({
                'tt_axis_camera_angle_deg': tt_axis_angle,
                'tt_axis_camera_parity': tt_axis_parity,
                'tt_axis_uses_legacy_pupil_angle': axis_override is None,
                'tt_axis_independently_calibrated': False,
                'physical_total_covariance_opd_m2': np.asarray(Ctt).tolist(),
                'tt_otf_direct_angular_noll_relative_error': kernel_error,
                'tt_otf_zero_lag': float(self.otf_tt[
                    self.otf_tt.shape[0]//2, self.otf_tt.shape[1]//2
                ]),
                'tt_otf_min': float(np.min(self.otf_tt)),
                'tt_otf_max': float(np.max(self.otf_tt)),
                'tt_source_angle_replay': angular_audit,
            })
        return dphi_used

    def _legacy_tiptilt_covariance(self, noise_raw):
        """Preserve original PR #2 behaviour for reproducible comparisons."""
        noise_cov, noise_eigenvalues = project_covariance_to_psd(
            noise_raw, label="STRAP noise covariance"
        )
        pn = float(self.trs.ttloop.tf.pn)
        if not np.isfinite(pn) or pn < 0:
            raise ValueError("Invalid tip-tilt noise transfer factor")

        # Atmospheric residual and servo-propagated detector noise.
        # Measured residuals contain instantaneous readout noise, while
        # servo feedback transmits an additional pn*Cn contribution.
        # This decomposition uses the historical P3 estimator unchanged.
        atmospheric_raw = self.Ctt - (1.0 + pn) * noise_cov
        self.tt_atmos_cov, atmospheric_eigs = project_covariance_to_psd(
            atmospheric_raw, label="TT atmospheric residual covariance"
        )
        self.tt_noise_cov, _ = project_covariance_to_psd(
            pn * noise_cov, label="TT propagated noise covariance"
        )
        # A covariance must be PSD to represent a Gaussian PSF blur.
        # Projection provides a usable physical kernel, but may be biased
        # low when the detector noise is overestimated.
        self.tt_model_cov = self.tt_atmos_cov + self.tt_noise_cov
        self.tiptilt_diagnostics = {
            "raw_trace_m2": float(np.trace(self.Ctt)),
            "noise_trace_m2": float(np.trace(noise_raw)),
            "noise_transfer_factor": pn,
            "corrected_trace_m2": float(np.trace(atmospheric_raw)),
            "corrected_eigenvalues_m2": atmospheric_eigs.tolist(),
            "noise_eigenvalues_m2": noise_eigenvalues.tolist(),
            "atmospheric_projected_trace_m2": float(np.trace(self.tt_atmos_cov)),
            "noise_propagated_trace_m2": float(np.trace(self.tt_noise_cov)),
            "model_jitter_trace_m2": float(np.trace(self.tt_model_cov)),
            "negative_variance_projected": bool(np.any(atmospheric_eigs < 0)),
            "noise_model_validated": False,
            "command_units_verified": bool(
                getattr(self.trs.tipTilt, "command_units_verified", False)
            ),
        }
        self.tiptilt_diagnostics['jitter_covariance_method'] = 'legacy_psfr'
        self.tiptilt_diagnostics['tt_mirror_model'] = 'unity'
        return self.tt_model_cov

    def pixel_optical_transfer_function(self):
        """
        Computes the OTF of the pixel.
        """
        #note : self.U_/V_ ranges from -1 to 1
        umax = self.ao.tel.D/self.freq.wvlRef * self.freq.sampRef/2
        ps = self.ao.cam.psInMas * np.pi*1e-3/3600/180
        otfPixel = np.sinc(self.freq.U_ * umax * ps)\
                  *np.sinc(self.freq.V_ * umax * ps)
        return otfPixel

    def phase_structure_function(self, r0, x0_dphi, Cn2=None):
        """Compatibility adapter for the pure phase-SF assembly."""
        # Validate gains before touching the AO object: legacy error contract.
        if (x0_dphi is None or len(x0_dphi) != 2
                or not np.all(np.isfinite(x0_dphi))):
            raise ValueError(
                "PSF-R requires two finite HO/TT gains [gho, gtt] after "
                "the atmospheric parameters: use [r0, gho, gtt] for one "
                "layer or [w_i*r0**(-5/3) for each layer, gho, gtt] "
                "for multiple atmospheric/reconstructed layers."
            )
        is_aniso = self.freq.isAniso
        result = assemble_phase_structure_function(
            r0_m=r0, gains=x0_dphi,
            dphi_ao=self.dphi_ao, dphi_tt=self.dphi_tt,
            dphi_fit=self.dphi_fit, dphi_alias=self.dphi_alias,
            n_sources=self.ao.src.nSrc,
            anisoplanatism_kernel=self.dphi_ani if is_aniso else None,
            layer_strengths=Cn2 if is_aniso else None,
            layer_weights=self.ao.atm.weights if is_aniso else None,
            atmosphere_wavelength_m=self.ao.atm.wvl if is_aniso else None,
            science_wavelength_m=self.freq.wvlRef if is_aniso else None,
        )
        if is_aniso:
            self.dani = result.anisoplanatism_rad2
            self.dani_raw_zero_lag_rad2 = result.raw_zero_lag_rad2.copy()
            self.dani_zero_lag_reference = result.zero_lag_reference
        return result.dphi_rad2

    def __call__(self, x0, nPix=None):

        # ----------------- GETTING THE PARAMETERS
        (Cn2, r0, x0_dphi, x0_jitter, x0_stellar, x0_stat) = FourierUtils.sort_params_from_labels(self, x0)

        # ----------------- GETTING THE PHASE STRUCTURE FUNCTION
        self.SF = self.phase_structure_function(r0, x0_dphi, Cn2=Cn2)

        # ----------------- COMPUTING THE PSF
        PSF, self.SR = FourierUtils.sf_3D_to_psf_3D(self.SF,
                                               self.freq,
                                               self.ao,
                                               x_jitter = x0_jitter,
                                               x_stat = x0_stat,
                                               x_stellar = x0_stellar,
                                               nPix = nPix,
                                               otfPixel = self.otfPixel)
        return PSF

    def get_error_breakdown(self, r0=None, gho=1, gtt=1):
        '''
        Computing the AO error breakdown from the variance of
        each individual covariance terms.
        INPUTS:
            - an instance of the psfr object
            - the specific r0 to which the fitting error must be computed. If
            None, takes the value in the ao.atm object
            - gho and gtt, set to 1 by default
        '''
        sr2fwe = lambda x: np.sqrt(-np.log(x))* self.freq.wvlRef*1e9/2/np.pi
        otf_dl = self.freq.otfDL
        S = otf_dl.sum()
        self.wfe = dict()

        #1. STATIC ABERRATIONS
        if np.any(self.ao.tel.opdMap_on):
            self.wfe['NCPA'] = np.std(self.ao.tel.opdMap_on[self.ao.tel.pupil.astype(bool)])
        else:
            self.wfe['NCPA'] = 0

        #2. DM FITTING ERROR
        if not r0:
            r0 = self.ao.atm.r0 * (self.freq.wvlRef/self.ao.atm.wvl)**1.2
        otf_fit = np.exp(-0.5 * r0**(-5/3) * self.dphi_fit)
        sr_fit  = np.sum(otf_dl * otf_fit)/S
        self.wfe['FITTING'] = sr2fwe(sr_fit)

        #3. WFS ALIASING ERROR
        otf_alias = np.exp(-0.5 * r0**(-5/3) * self.dphi_alias)
        sr_alias  = np.sum(otf_dl * otf_alias)/S
        self.wfe['ALIASING'] = sr2fwe(sr_alias)

        #4. WFS NOISE ERROR
        # noise on high-order modes
        msk = self.trs.dm.validActuators.reshape(-1)
        self.wfe['HO NOISE'] = 1e9 * np.sqrt(self.trs.holoop.tf.pn * np.mean(self.trs.wfs.Cn_ao[msk,msk]))
        # noise on tip-tilt modes
        if self.compute_dphi_jitter:
            noise_cov = self.tt_noise_cov
            dphi_tt_n = self.jitter_phase_structure_function(Ctt=noise_cov)
            sr_tt_n = np.sum(otf_dl * np.exp(-0.5*dphi_tt_n))/S
            self.wfe['TT NOISE'] = sr2fwe(sr_tt_n)
        else:
            # A diagnostic TT ablation removes both atmospheric and noise
            # jitter from the forward model; its error budget must do likewise.
            self.wfe['TT NOISE'] = 0.0

        #5. AO BANDWIDTH ERROR
        C = self.Cao - (1+self.trs.holoop.tf.pn)*self.trs.wfs.Cn_ao
        self.wfe['SERVO-LAG'] = 1e9*np.sqrt(C.trace()/self.trs.dm.nCom)

        #6. RESIDUAL TIP-TILT (after the same PSD projection as in the OTF).
        if self.compute_dphi_jitter:
            dphi_tt_atmos = self.jitter_phase_structure_function(
                Ctt=self.tt_atmos_cov
            )
            sr_tt = np.sum(otf_dl * np.exp(-0.5*dphi_tt_atmos))/S
            self.wfe['TIP-TILT'] = sr2fwe(sr_tt)
            self.wfe['TIP-TILT-WFE'] = (
                np.sqrt(np.trace(self.tt_atmos_cov)/2) * 1e9
            )
        else:
            self.wfe['TIP-TILT'] = 0
            self.wfe['TIP-TILT-WFE'] = 0

        #7. PIXEL TF
        sr_pixel = np.sum(otf_dl * self.otfPixel)/S
        self.wfe['PIXEL TF'] = np.sqrt(-np.log(sr_pixel))* self.freq.wvlRef*1e9/2/np.pi

        #8. ANISOPLANATISM
        if self.freq.isAniso:
            # Use the same reference wavelength as phase_structure_function;
            # ao.src.wvl[0] is the *central* science wavelength, whereas
            # freq.wvlRef may be the shortest spectral bin.
            Cn2     = self.ao.atm.weights * self.ao.atm.r0**(-5/3) * (self.ao.atm.wvl/self.freq.wvlRef)**2
            dani = _anisoplanatism_zero_lag(
                (self.dphi_ani[0].transpose(1,2,0) * Cn2).sum(axis=2)
            )
            otf_ani = np.exp(-0.5 * dani)
            sr_ani  = np.sum(otf_dl * otf_ani)/S
            self.wfe['TOTAL ANISOPLANATISM'] = sr2fwe(sr_ani)
        else:
            self.wfe['TOTAL ANISOPLANATISM'] = 0

        if self.trs.aoMode == 'LGS' and self.freq.isAniso:
            # Angular
            dani = _anisoplanatism_zero_lag(
                (self.dani_ang[0].transpose(1,2,0) * Cn2).sum(axis=2)
            )
            otf_ani = np.exp(-0.5 * dani)
            sr_ani  = np.sum(otf_dl * otf_ani)/S
            self.wfe['ANGULAR ANISOPLANATISM'] = sr2fwe(sr_ani)
            #tiptilt
            dani = _anisoplanatism_zero_lag(
                (self.dani_tt[0].transpose(1,2,0) * Cn2).sum(axis=2)
            )
            otf_ani = np.exp(-0.5 * dani)
            sr_ani  = np.sum(otf_dl * otf_ani)/S
            self.wfe['ANISOKINETISM'] = sr2fwe(sr_ani)
            # focal
            dani = _anisoplanatism_zero_lag(
                (self.dani_focang[0].transpose(1,2,0) * Cn2).sum(axis=2)
            )
            otf_ani = np.exp(-0.5 * dani)
            sr_ani  = np.sum(otf_dl * otf_ani)/S
            self.wfe['FOCAL ANISOPLANATISM'] = np.sqrt(sr2fwe(sr_ani) **2 - self.wfe['ANGULAR ANISOPLANATISM']**2)
        else:
            self.wfe['ANGULAR ANISOPLANATISM'] = 0
            if self.freq.isAniso:
                self.wfe['ANGULAR ANISOPLANATISM'] = self.wfe['TOTAL ANISOPLANATISM']
            self.wfe['ANISOKINETISM']          = 0
            self.wfe['FOCAL ANISOPLANATISM']   = 0

        #9. TOTAL WFE
        # Approximate Maréchal-equivalent WFE budget: include every spatial
        # term used in the forward OTF, including WFS aliasing.
        self.wfe['TOTAL WFE'] = np.sqrt(self.wfe['NCPA']**2 + self.wfe['FITTING']**2
                + self.wfe['ALIASING']**2 + self.wfe['HO NOISE']**2
                + self.wfe['TT NOISE']**2 + self.wfe['SERVO-LAG']**2
                + self.wfe['TIP-TILT']**2 + self.wfe['TOTAL ANISOPLANATISM']**2)

        self.wfe['TOTAL WFE WITH PIXEL'] = np.hypot(self.wfe['TOTAL WFE'],self.wfe['PIXEL TF'])

        #10. TOTAL STREHL-RATIO
        self.wfe['REF WAVELENGTH'] = self.fao.freq.wvlRef
        self.wfe['MARECHAL SR'] = 1e2*np.exp(-(self.wfe['TOTAL WFE'] * 2*np.pi*1e-9/self.fao.freq.wvlRef)**2 )
        self.wfe['MARECHAL SR WITH PIXEL'] = 1e2*np.exp(-(self.wfe['TOTAL WFE WITH PIXEL'] * 2*np.pi*1e-9/self.fao.freq.wvlRef)**2 )

        #11. STREHL RATIO FROM THE IMAGE
        self.wfe[self.ao.cam.tag + " SR"] = 1e2*FourierUtils.getStrehl(self.trs.cam.image,
                                                                       self.ao.tel.pupil,
                                                                       self.freq.sampRef)
        if self.ao.atm.nL==1:
            self.psf = self([self.ao.atm.r0, 1, 1])
        else:
            self.psf = self(list(self.ao.atm.weights*self.ao.atm.r0**(-5/3)) +  [1, 1])
        self.wfe['PSFR SR OTF'] = self.SR[0]
        self.wfe["PSFR SR PEAK"] = 1e2*FourierUtils.getStrehl(
            _integrate_spectral_psf(self.psf), self.ao.tel.pupil,
            self.freq.sampRef, method='max',
        )













