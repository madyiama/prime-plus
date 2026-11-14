#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:57:49 2018

@author: omartin
"""

#%% Libraries
import numpy as np
import numpy.fft as fft
from p3.psfr.hoFactorized import accumulate_vii_factorized
import p3.aoSystem.FourierUtils as FourierUtils
import urllib.request
import os

#%% DATA TRANSFER
def get_data_file(path_sav,filename):
    if not os.path.isfile(path_sav + f"/{filename}"):
        print('Downloading of the file %s in the folder %s'%(filename,path_sav))
        urllib.request.urlretrieve(f"https://nuage.osupytheas.fr/s/WjeQ8BB3wp2mEyL/download?files={filename}",
                                   path_sav + f"{filename}")

#%%  PSF-R FACILITIES
def compute_psd_from_slopes(trs):
    """
    Compute the spatial PSD of the reconstructed phase and the DM influence function.
    """
    # get the residual wavefront in the actuator domain
    com_res = trs.rec.res
    # get the DM influence function
    modes = trs.mat.dmIF
    n_px = int(np.sqrt(modes.shape[0]))
    n_frames = com_res.shape[0]
    # get the cube of phase in randian
    phase = 2*np.pi/trs.wfs.wvl * np.dot(modes, com_res.T).reshape(n_px, n_px, n_frames)
    phase = FourierUtils.enlargeSupport(phase, 2)
    # get the psd
    psd = np.fft.fftshift(np.mean(abs(np.fft.fft2(phase, axes=(0,1)))**2, axis=2))
    return psd

def getOLslopes(s,u,MI,dt):
    return s + MI*(dt*np.roll(u,(-2,2)) + (1-dt)*np.roll(u,(-1,2)))

def mkotf(indptsc,indptsc2,nU1d,ampl,dp,C_phi):
    #Instantiation
    otf         = np.zeros((nU1d,nU1d))
    C_phip_diag = np.exp(np.diag(C_phi))
    C_phipi     = np.exp(-2*C_phi)
    C_phi_size2 = C_phi.shape[1]

    for iu in np.arange(0,nU1d,1):
        for ju in np.arange(0,nU1d,1):
            indpts  = indptsc[iu, ju]
            indpts2 = indptsc2[iu, ju]
            indpts  = indpts[0]
            indpts2 = indpts2[0]
            if len(indpts)== 0:
                otf[iu,ju] = 0
            else:
                msk        = np.unravel_index(C_phi_size2*indpts + indpts2, C_phipi.shape)
                myarg      = C_phip_diag[indpts2]*C_phip_diag[indpts]*C_phipi[msk]
                kernel     = np.dot(myarg,np.conjugate(ampl[indpts2])*ampl[indpts])
                otf[iu,ju] = kernel*dp**2

    dc  = np.sum(abs(ampl[:])**2) * dp**2
    otf = otf/dc;
    return otf

def mkotf_indpts(nU1d,nPh,u1D,loc,dp):
    # index pts in a 3x bigger array
    locInPitch = loc/dp
    minLoc     = np.array([locInPitch[0,:].min(),locInPitch[1,:].min()])
    ninloc     = loc.shape[1]
    nc         = 3*nPh
    n          = nc-nPh
    n1         = int((n-n%2)/2 + 1 + nPh%2)
    minLoc2    = minLoc -(n1-1)
    loc2       = np.round(locInPitch.T - np.ones((ninloc,2))*minLoc2+1)
    loc2       = loc2.T
    #embed the loc2 inside ncxnc array.
    indx_emb       = loc2[0,:] + (loc2[1,:]-1)*nc-1
    indx_emb       = indx_emb.astype(int)
    mask           = np.zeros((nc,nc))
    indx_emb       = np.unravel_index(indx_emb,mask.shape)
    mask[indx_emb] = 1
    indptsc        = np.zeros((nU1d,nU1d,), dtype=np.object)
    indptsc2       = np.zeros((nU1d,nU1d,), dtype=np.object)

    for iu in np.arange(0,nU1d,1):
        for ju in np.arange(0,nU1d,1):
            u2D        = np.array([u1D[iu],u1D[ju]])
            uInPitch   = u2D/dp
            iniloc_sh2 = np.round(loc2.T + np.ones((ninloc,2))*np.round(uInPitch))
            iniloc_sh2 = iniloc_sh2.T
            iniloc_sh2 = iniloc_sh2.astype(int)
            indxsh     = iniloc_sh2[0,:] + (iniloc_sh2[1,:]-1)*nc-1
            indxsh     = np.unravel_index(indxsh,mask.shape)
            #index of points in iniloc_sh2 that are intersect with iniloc_sh
            #indpts is the overlapping points in the shifted array
            indptsc[ju,iu]  = mask[indxsh].nonzero()
            mask2           = np.zeros((nc,nc),dtype=bool)
            mask2[indxsh]   = 1
            indptsc2[ju,iu] = mask2[indx_emb].nonzero()

    return indptsc,indptsc2

def modes2Otf(Cmm, modes, pupil, n_otf, samp=2, basis='Vii',
              covariance_psd_policy='strict', covariance_diagnostics=None,
              vii_method='legacy', vii_diagnostics=None):
    """
    Computes the OTF and phase structure function from the covariance matrix
    of given coeffcients and the corresponding modal basis.
    INPUTS:
        - Cmm, the n_m x n_M covariance matrix
        - modes, the n_pup**2 x n_m matrix concatenating 1D vector of modes
        - pupil, the n_pup x n_pup telescope pupil
        - nOtf, the output shape for the otf
        - samp, the oversampling factor; samp=2 means Nyquist sampling
        - basis, either Vii (Gendron+06) or Uij (Veran97)
    OUTPUTS:
        - otf, the nOtf x nOtf optical transfer function
        - dphi, the corresponding phase structure function in rad**2
    """
    if vii_method not in ('legacy','factorized'):
        raise ValueError("Unknown Vii mode reconstruction method")
    if basis!='Vii' and vii_method!='legacy':
        raise ValueError("Vii factorization is only available for basis Vii")
    if vii_diagnostics is not None:
        vii_diagnostics["method"]=vii_method
        vii_diagnostics["pupil_or_ho_otf_cached"]=False
    #Autocorrelation of the pupil expressed in pupil
    n_pup = int(np.sqrt(modes.shape[0]))
    pup_pad = FourierUtils.enlargeSupport(pupil, samp)
    fft_pup = fft.fft2(pup_pad)
    conj_pup = np.conjugate(fft_pup)
    G = fft.fftshift(np.real(fft.ifft2(abs(fft_pup)**2)))

    #Defining the inverse
    den = np.zeros_like(G)
    msk = G/G.max() > 1e-7
    den[msk] = 1/G[msk]

    if np.any(Cmm) and basis=="Vii":
        # Cmm is a covariance, not an arbitrary matrix. Using SVD singular
        # values would silently turn a negative eigenvalue into positive
        # variance; Vh would also apply eigenvectors in the wrong direction.
        covariance = np.asarray(Cmm, dtype=float)
        if covariance.shape != (modes.shape[1], modes.shape[1]):
            raise ValueError("Modal covariance and influence modes have incompatible shapes")
        if not np.isfinite(covariance).all():
            raise ValueError("Modal covariance contains non-finite values")
        scale = max(float(np.max(np.abs(covariance))), 1e-30)
        if np.max(np.abs(covariance-covariance.T)) > 1e-10*scale:
            raise ValueError("Modal covariance must be symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5*(covariance+covariance.T)
        )
        # 'strict' rejects any negative beyond numerical eigensolver tolerance.
        # 'tiny_negative' is an explicit, tightly bounded option for LGS/NGS
        # finite-sample HO covariance estimates; it is not machine roundoff.
        # 'small_negative' retains the prior wider exploratory NGS-only option.
        # Check both the worst eigenvalue and accumulated negative mass.
        if covariance_psd_policy not in (
            'strict', 'tiny_negative', 'small_negative'
        ):
            raise ValueError("Unknown modal covariance PSD policy")
        tolerance = max(1e-12*float(np.max(np.abs(eigenvalues))), 1e-24)
        negative_values = -eigenvalues[eigenvalues < -tolerance]
        negative_mass = float(np.sum(negative_values))
        positive_mass = float(np.sum(eigenvalues[eigenvalues > 0]))
        negative_mode_fraction = (
            float(np.max(negative_values)) / max(float(np.max(eigenvalues)), 1e-30)
            if negative_values.size else 0.0
        )
        negative_trace_fraction = negative_mass / max(positive_mass, 1e-30)
        # Explicit engineering limits, NOT statistically validated confidence
        # intervals. 1e-5 (0.001%) is deliberately much tighter than the
        # NGS-only 0.005 (0.5%) exploratory policy.
        projection_limits = {
            'strict': 0.0,
            'tiny_negative': 1e-5,
            'small_negative': 5e-3,
        }
        projection_limit = projection_limits[covariance_psd_policy]
        can_regularize = (
            covariance_psd_policy != 'strict'
            and negative_mode_fraction <= projection_limit
            and negative_trace_fraction <= projection_limit
        )
        if covariance_diagnostics is not None:
            covariance_diagnostics.update({
                'policy': covariance_psd_policy,
                'lambda_min': float(np.min(eigenvalues)),
                'lambda_max': float(np.max(eigenvalues)),
                'negative_mode_count': int(negative_values.size),
                'negative_eigenvalue_mass': negative_mass,
                'negative_mode_fraction_of_lambda_max': negative_mode_fraction,
                'negative_trace_fraction_of_positive_mass': negative_trace_fraction,
                'maximum_allowed_fraction': projection_limit,
                'projection_applied': bool(can_regularize and negative_values.size),
                'units': 'input covariance units',
                'model_validated_by_projection': False,
            })
        if negative_values.size and not can_regularize:
            raise ValueError(
                "Modal covariance is not positive-semidefinite: "
                f"lambda_min={np.min(eigenvalues):.6e}, "
                f"lambda_max={np.max(eigenvalues):.6e}, "
                f"tolerance={tolerance:.6e}, "
                f"negative_mode_fraction={negative_mode_fraction:.6g}, "
                f"negative_trace_fraction={negative_trace_fraction:.6g}. "
                "Review HO noise subtraction; explicit optional policies "
                "are tiny_negative (0.001% each, LGS/NGS) and "
                "small_negative (0.5% each, NGS-only)."
            )
        if can_regularize and negative_values.size:
            import warnings
            warnings.warn(
                "HO covariance estimated from telemetry has small negative "
                f"eigenvalues: clipped {negative_values.size} modes, "
                f"negative_trace_fraction={negative_trace_fraction:.4g}, "
                f"worst_relative_eigenvalue={negative_mode_fraction:.4g}, "
                f"policy={covariance_psd_policy}, cap={projection_limit:.4g}; "
                "explicit PSD regularization, NOT instrument validation.",
                RuntimeWarning,
                stacklevel=2,
            )
        ss = np.clip(eigenvalues, 0, None)
        nModes = len(ss)
        M_all = np.dot(modes, eigenvectors)
        M_all = M_all.reshape(n_pup, n_pup, nModes)
        M_all = FourierUtils.enlargeSupport(M_all, samp)*pup_pad[:, :, np.newaxis]

        if vii_method=='factorized':
            buf=accumulate_vii_factorized(
                M_all,ss,conj_pup,diagnostics=vii_diagnostics,
            )
        else:
            buf = np.zeros_like(G)
            for k in range(nModes):
                Mk = M_all[:, :, k]
                # Historical Vii calculation: preserve bitwise baseline.
                Vk = np.real(fft.fft2(Mk**2)*conj_pup) - abs(fft.fft2(Mk))**2
                # Summing modes into dm basis
                buf = buf + ss[k] * Vk
            if vii_diagnostics is not None:
                vii_diagnostics["mode_count"]=int(nModes)
                vii_diagnostics["fft2_count_nominal"]=2*int(nModes)
                vii_diagnostics["numerical_reordering"]=False

        dphi = den*fft.fftshift(np.real(fft.ifft2(2*buf)))
        otf = G*np.exp(-0.5*dphi)

    elif np.any(Cmm) and basis=="Uij":
        nm   = modes.shape[1]
        dphi = np.zeros_like(pup_pad)

        #Double loops on modes
        for i in np.arange(1,nm,1):
            Mi = np.reshape(modes[:,i],(n_pup, n_pup))
            Mi = FourierUtils.enlargeSupport(Mi,samp)
            for j in range(1,i):
                #Getting modes + interpolation to twice resolution
                Mj = np.reshape(modes[:,j],(n_pup, n_pup))
                Mj = FourierUtils.enlargeSupport(Mj,samp)
                term1 = np.real(fft.fft2(Mi*Mj*pup_pad)*conj_pup)
                term2 = np.real(fft.fft2(Mi*pup_pad)*np.conjugate(fft.fft2(Mj*pup_pad)))
                # Uij computation
                Uij = np.real(fft.ifft2(term1-term2))
                #Summing terms
                fact = np.double(i!=j) + 1
                dphi = dphi + fact*Cmm[i,j]*Uij
                dphi = fft.fftshift(2*dphi)*den*msk
        otf = G*np.exp(-0.5*dphi)
    else:
        # Diffraction-limited case: a zero modal covariance must return
        # exactly zero phase structure function (without undefined locals).
        G = G/G.max()
        otf = G
        dphi = np.zeros_like(G)

    # Interpolation of the OTF => determination of the PSF fov
    otf = otf*(G>1e-5) #3.6 ms
    otf = FourierUtils.interpolateSupport(otf, n_otf) #84ms
    dphi = FourierUtils.interpolateSupport(dphi, n_otf) #86ms
    otf = otf/otf.max()
    return otf, dphi

def pointWiseLocation(D, dp, idxValid=None):
    # Defining the point-wise locations
    xloc = np.arange(-D/2,D/2+dp,dp)
    actuLocY, actuLocX = np.meshgrid(xloc,xloc)
    if idxValid is not None:
        actuLocX = actuLocX[idxValid]
        actuLocY = actuLocY[idxValid]

    return np.array([actuLocX,actuLocY])

def sr2wfe(Strehl,wvl):
    return 1e9*np.sqrt(-np.log(Strehl))*wvl/2/np.pi

def wfe2sr(wfe,wvl):
    return np.exp(-(2*np.pi*wfe*1e-9/wvl)**2)

def zonalCovarianceToOtf(Cphi, npsf, D, dp, idxValid=None):
    # Grabbing the valid actuators positions in meters
    loc  = pointWiseLocation(D, dp, idxValid=idxValid)
    #OTF sampling
    nPh  = round(D/dp+1)
    #nU1d = 2*nPh;
    u1D  = np.arange(-nPh,nPh,1)*dp
    nU1d = len(u1D)
    # Determining couples of point with the same separation
    shiftX,shiftY = mkotf_indpts(nU1d,nPh,u1D,loc,dp)
    # WF Amplitude
    if idxValid is None:
        idxValid = np.ones(Cphi.shape[0])
    amp0 = np.ones((np.count_nonzero(idxValid),1))
    # Long-exposure OTF
    otf = mkotf(shiftX,shiftY,nU1d,amp0,dp,-0.5*Cphi)
    #Interpolation
    otf = FourierUtils.interpolateSupport(otf,npsf)
    otf = otf/otf.max()
    return otf