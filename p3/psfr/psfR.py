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
from p3.aoSystem.frequencyDomain import frequencyDomain as frequencyDomain
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
                 tt_mirror_model='unity', tt_welch_nperseg=512):
        """
        Instantiating the psf-reconstruction model from a telemetry object
        """

        # INSTANTIATING THE MODEL OF THE AOSYSTEM
        tstart = time.time()

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
        self.tt_covariance_mode = tt_covariance_mode
        self.tt_mirror_model = tt_mirror_model
        self.tt_welch_nperseg = int(tt_welch_nperseg)
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


        # INSTANTIATING THE SPATIAL FREQUENCY AND THE FOURIER MODEL
        self.Hfilter = 1
        if self.ao.aoMode=="SLAO":
            n_ph = 2*self.ao.dms.nActu1D[0] + 1
            modes = self.ao.dms.setInfluenceFunction(n_ph)
            #val_act = np.argwhere(modes.max(axis=0)==0)
            self.Hfilter = np.dot(modes, modes.T)

        self.freq = frequencyDomain(self.ao, Hfilter=self.Hfilter)
        self.fao = fourierModel(self.path_ini, calcPSF=False, display=False,
                                freq=self.freq, ao=self.ao)

        self.t_freq = 1000*(time.time() - tstart) - self.t_initAO

        # INSTANTIATING THE FITTING PHASE STRUCTURE FUNCTION FOR r0=1m
        self.dphi_fit = self.fitting_phase_structure_function(1)
        self.t_dfit = 1000*(time.time() - tstart) - self.t_freq

        # INSTANTIATING THE ALIASING PHASE STRUCTURE FUNCTION FOR r0=1m
        self.dphi_alias = self.aliasing_phase_structure_function(1)
        self.t_dalias = 1000*(time.time() - tstart) - self.t_dfit

        # INSTANTIATING THE TT RESIDUAL PHASE STRUCTURE FUNCTION IN LGS MODE
        if compute_dphi_jitter:
            self.dphi_tt = self.jitter_phase_structure_function()
        else:
            self.dphi_tt = 0*self.dphi_fit
        self.t_dtt = 1000*(time.time() - tstart) - self.t_dalias

        # INSTANTIATING THE AO RESIDUAL PHASE STRUCTURE FUNCTION
        self.dphi_ao = self.residual_phase_structure_function()
        self.t_dao = 1000*(time.time() - tstart) - self.t_dtt

        # INSTANTIATING THE ANISOPLANATISM PHASE STRUCTURE FUNCTION IF ANY
        if self.ao.lgs is None or self.ao.lgs.height==0:
            if self.freq.isAniso:
                self.dphi_ani = self.freq.dani_ang
        else:
            self.dani_focang = self.fao.freq.dani_focang
            self.dani_ang = self.freq.dani_ang
            self.dani_tt = self.freq.dani_tt
            self.dphi_ani = self.dani_focang + self.dani_tt

        # COMPUTING THE DETECTOR PIXEL TRANSFER FUNCTION
        if self.trs.tel.name!='OOMAO':
            self.otfPixel = self.pixel_optical_transfer_function()
        else:
            self.otfPixel = 1.0

        # COMPUTING THE ERROR BREAKDOWN:
        self.get_error_breakdown()

         # DEFINING BOUNDS
        self.bounds = self.define_bounds()

        self.t_init = 1000*(time.time()  - tstart)

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
        """
        Computes the phase structure function of the tip-tilt-excluded
        residual phase.
        """
        # computing the empirical covariance matrix of the AO-residual OPD in the DM actuators domain
        if method == 'slopes-based':
            du = self.trs.rec.res
        elif method == 'dm-based':
            du = np.diff(self.trs.dm.com,axis=0)/self.ao.rtc.holoop['gain']

        self.Cao = np.dot(du.T,du)/du.shape[0]

        # Unbiasing noise and accounting for the wave number
        Cao = (2*np.pi/self.freq.wvlRef)**2 * (self.Cao - self.trs.wfs.Cn_ao)

        # Computing the phase structure function
        _, dphi_ao = psfrUtils.modes2Otf(Cao, self.trs.mat.dmIF,
                                         self.ao.tel.pupil,
                                         self.freq.nOtf,
                                         basis=basis,
                                         samp=self.freq.sampRef)

        return dphi_ao

    def jitter_phase_structure_function(self, Ctt=None):
        """Compute the Gaussian image-motion structure function from angular jitter.

        Internal covariance inputs are RMS-normalized Zernike tilt OPD [m^2].
        Only the final OTF kernel is expressed in angle [rad^2]. Retain the
        current noise estimator, but make its non-PSD decomposition explicit.
        """
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
                # Explicitly distinguish physical jitter from centroid noise.
                # No covariance clipping; component validity is checked upstream.
                self.tiptilt_diagnostics = {
                    'jitter_covariance_method': 'spectral_closed_loop',
                    'tt_mirror_model': self.tt_mirror_model,
                    'mirror_model_verified_for_2020': False,
                    'raw_trace_m2': float(np.trace(self.Ctt)),
                    'noise_trace_m2': float(np.trace(noise_raw)),
                    'noise_transfer_factor': float(self.trs.ttloop.tf.pn),
                    'corrected_trace_m2': float(np.trace(self.tt_atmos_cov)),
                    'corrected_eigenvalues_m2': np.linalg.eigvalsh(
                        self.tt_atmos_cov
                    ).tolist(),
                    'atmospheric_projected_trace_m2': float(
                        np.trace(self.tt_atmos_cov)
                    ),
                    'noise_propagated_trace_m2': float(
                        np.trace(self.tt_noise_cov)
                    ),
                    'model_jitter_trace_m2': float(
                        np.trace(self.tt_model_cov)
                    ),
                    'negative_variance_projected': False,
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

        Ctheta = opd_covariance_to_angular(Ctt, self.ao.tel.D)
        ang = self.trs.tel.pupilAngle * np.pi/180
        unit = self.freq.sampRef/2
        if ang:
            Ur = unit * (self.freq.U_*np.cos(ang) + self.freq.V_*np.sin(ang))
            Vr = unit * (-self.freq.U_*np.sin(ang) + self.freq.V_*np.cos(ang))
        else:
            Ur = unit * self.freq.U_
            Vr = unit * self.freq.V_

        return angular_jitter_structure_function(
            Ctheta, self.ao.tel.D, self.freq.wvlRef, Ur, Vr
        )

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

        # On-axis phase structure function
        SF = x0_dphi[0]*self.dphi_ao\
           + x0_dphi[1]*self.dphi_tt \
           + r0**(-5/3) * (self.dphi_fit + self.dphi_alias)

        # Anisoplanatism is present for an LGS even with one equivalent
        # atmospheric layer. In that case the fitted parameter is r0, not Cn2.
        # P3's anisoplanatic kernels are normalized by r0^(-5/3) at atm.wvl,
        # so convert their phase variance to the science reference wavelength.
        if self.freq.isAniso:
            if Cn2 is None:
                Cn2 = np.asarray(self.ao.atm.weights) * r0**(-5/3)
            Cn2 = np.asarray(Cn2, dtype=float).reshape(-1)
            if Cn2.size != self.ao.atm.nL:
                raise ValueError("Anisoplanatism: Cn2 must have one value per layer")
            Cn2_at_science_wvl = Cn2 * (
                self.ao.atm.wvl / self.freq.wvlRef
            )**2
            self.dani = np.einsum(
                'slxy,l->xys', self.dphi_ani, Cn2_at_science_wvl
            )
            SF = SF[:,:,np.newaxis] + self.dani
        else:
            SF = np.repeat(SF[:,:,np.newaxis],self.ao.src.nSrc,axis=2)

        return SF

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
        else:
            noise_cov, _ = project_covariance_to_psd(
                self.trs.ttloop.tf.pn * self.trs.tipTilt.Cn_tt,
                label="TT propagated noise covariance",
            )
        dphi_tt_n = self.jitter_phase_structure_function(Ctt=noise_cov)
        sr_tt_n = np.sum(otf_dl * np.exp(-0.5*dphi_tt_n))/S
        self.wfe['TT NOISE'] = sr2fwe(sr_tt_n)

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
            Cn2     = self.ao.atm.weights * self.ao.atm.r0**(-5/3) * (self.ao.atm.wvl/self.ao.src.wvl[0])**2
            dani    = (self.dphi_ani[0].transpose(1,2,0) * Cn2).sum(axis=2)
            otf_ani = np.exp(-0.5 * dani)
            sr_ani  = np.sum(otf_dl * otf_ani)/S
            self.wfe['TOTAL ANISOPLANATISM'] = sr2fwe(sr_ani)
        else:
            self.wfe['TOTAL ANISOPLANATISM'] = 0

        if self.trs.aoMode == 'LGS' and self.freq.isAniso:
            # Angular
            dani    = (self.dani_ang[0].transpose(1,2,0) * Cn2).sum(axis=2)
            otf_ani = np.exp(-0.5 * dani)
            sr_ani  = np.sum(otf_dl * otf_ani)/S
            self.wfe['ANGULAR ANISOPLANATISM'] = sr2fwe(sr_ani)
            #tiptilt
            dani    = (self.dani_tt[0].transpose(1,2,0) * Cn2).sum(axis=2)
            otf_ani = np.exp(-0.5 * dani)
            sr_ani  = np.sum(otf_dl * otf_ani)/S
            self.wfe['ANISOKINETISM'] = sr2fwe(sr_ani)
            # focal
            dani    = (self.dani_focang[0].transpose(1,2,0) * Cn2).sum(axis=2)
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
        self.wfe['TOTAL WFE'] =  np.sqrt(self.wfe['NCPA']**2 +  self.wfe['FITTING']**2
                + self.wfe['HO NOISE']**2 + self.wfe['TT NOISE']**2 + self.wfe['SERVO-LAG']**2
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













