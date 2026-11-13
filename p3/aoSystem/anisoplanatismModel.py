#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:54:57 2021

@author: omartin
"""
import numpy as np
import scipy.special as spc
import p3.aoSystem.FourierUtils as FourierUtils
from p3.aoSystem.zernike import zernike
from p3.aoSystem.focalFast import symmetric_focal_layer, grouped_matrix_to_map

#%% VARIANCE
def anisoplanatism_wfe(tel, atm, src, ngs, dtype=np.float64):
    """
    Returns the anisoplanatism wavefront error.
    """

    nLayer = atm.nL
    Hs = atm.heights * tel.airmass
    ax = src.direction[0] - ngs.direction[0]
    ay = src.direction[1] - ngs.direction[1]
    nSrc = len(ax)
    wfe = np.zeros(nSrc)

    # definition of the spatial frequency domain
    PSDstep = 1/(tel.D * 2)
    nOtf = tel.resolution*2
    ky, kx = FourierUtils.freq_array(nOtf, offset=1e-10, L=PSDstep, dtype=dtype)
    k2 = kx**2 + ky**2

    # PSD for the atmospheric turbulence
    cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
    Watm = cte*(1/atm.L0**2 + k2)**(-11/6)
    Watm[nOtf//2, nOtf//2] = 0

    for iSrc in range(nSrc):
        thx = ax[iSrc]
        thy = ay[iSrc]
        psd = np.zeros((nOtf, nOtf))
        for l in range(nLayer):
            zl = Hs[l]
            if zl and (thx or thy):
                psd += 2*(1 - np.cos(2*np.pi*zl*(kx*thx + ky*thy))) * Watm

        wfe[iSrc] = np.sqrt(np.trapz(np.trapz(psd, kx[0]), kx[0]))* atm.wvl*1e9/2/np.pi

    return wfe

# Piston-removed phase variance of a full layer, in units of (D/r0)^(5/3) (Noll 1976).
_FULL_LAYER_VARIANCE = 1.0299

def _height_ratio(atm, lgs) -> np.ndarray:
    """h/z_LGS for each layer."""
    return np.asarray(FourierUtils.cpuArray(atm.heights), dtype=float) / float(FourierUtils.cpuArray(lgs.height[0]))

def _clip_layer_variance(var_sensed: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-layer cone variance: the small-h/z expansion is clipped to [0, full layer]
    (it goes negative for h/z > 1.6), and layers at or above the LGS (x >= 1) are
    not sensed, so they contribute the full open-loop variance."""
    var = np.clip(var_sensed, 0, _FULL_LAYER_VARIANCE)
    return np.where(x >= 1, _FULL_LAYER_VARIANCE, var)

def focal_anisoplanatism_phase_variance(tel, atm, lgs):
    """Kolmogorov cone-effect piston-AND-tilt-removed variance [rad^2].

    Hardy (1998), eq. 16, as reproduced by Beltramo-Martin et al.
    (2019, MNRAS 487, 5450), eq. 12:

      sigma_phi^2 = (D/r0)^(5/3) sum_l w_l *
          [0.5 (h_l/H)^(5/3) - 0.452 (h_l/H)^2] / 0.423.

    The published 0.452 term corresponds to removal of both global
    piston and x/y wavefront tilt on an ideal circular pupil (Hardy;
    Sasiela, Electromagnetic Wave Propagation in Turbulence, Eq 6.197).
    It must NOT be directly equated to the legacy Flicker SF/OTF
    integration which removes piston but still contains cone tip-tilt.

    Here r0 is defined for the same line of sight and wavelength as atm,
    the weights are fractions of integrated Cn2 dh, h/H is along the same
    geometrical line of sight, and 0.423 comes from Fried's r0 definition.

    The finite-beacon clipping and the full-layer cap are *modeling
    approximations*. This Kolmogorov variance does NOT model finite L0,
    a filtered DM, or the diffraction-weighted OTF-equivalent WFE.
    """
    x = _height_ratio(atm, lgs)
    weights = np.asarray(FourierUtils.cpuArray(atm.weights), dtype=float).reshape(-1)
    if (x.shape != weights.shape or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("Invalid normalized layer weights for focal anisoplanatism")
    kernel = (0.5 * x**(5.0/3.0) - 0.452 * x**2) / 0.423
    variance_normalized = np.sum(weights * _clip_layer_variance(kernel, x))
    return float(variance_normalized * (tel.D/atm.r0)**(5.0/3.0))


def focal_anisoplanatism_wfe(tel, atm, lgs):
    """Analytical Kolmogorov cone-effect wavefront RMS [nm OPD].

    See focal_anisoplanatism_phase_variance for the reference formula
    and assumptions. Not interchangeable with the OTF-derived Maréchal
    equivalent error from psfR.get_error_breakdown().
    """
    phase_variance = focal_anisoplanatism_phase_variance(tel, atm, lgs)
    return np.sqrt(phase_variance) * (atm.wvl*1e9/(2*np.pi))


def anisokinetism_wfe(tel, atm, src, ngs, method="Sasiela"):
    """
    Computes the wavefront error of the anisokinetism.
    """

    if method == "Sasiela":
        z_l = atm.heights
        cn2 = atm.r0**(-5/3) * atm.weights
        D = tel.D
        cx = np.array([2.67*3, -3.68*5, 2.35*(17/3), 0.304*7, 0.306*(23/3)])
        cy = np.array([2.67, -3.68, 2.35, 0.304, 0.306])
        thx = abs(src.direction[0] - ngs.direction[0])
        thy = abs(src.direction[1] - ngs.direction[1])

        mu002 = np.sum(cn2*z_l**2)
        mu004 = np.sum(cn2*z_l**4)
        mu006 = np.sum(cn2*z_l**6)
        mu143 = np.sum(cn2*z_l**(14/3))
        mu203 = np.sum(cn2*z_l**(20/3))

        var_x = cx[0]*mu002/D**(1/3)*(thx/D)**2\
               +cx[1]*mu004/D**(1/3)*(thx/D)**4\
               +cx[2]*mu143/D**(1/3)*(thx/D)**(14/3)\
               +cx[3]*mu006/D**(1/3)*(thx/D)**6\
               +cx[4]*mu203/D**(1/3)*(thx/D)**(20/3)

        var_y = cy[0]*mu002/D**(1/3)*(thy/D)**2\
               +cy[1]*mu004/D**(1/3)*(thy/D)**4\
               +cy[2]*mu143/D**(1/3)*(thy/D)**(14/3)\
               +cy[3]*mu006/D**(1/3)*(thy/D)**6\
               +cy[4]*mu203/D**(1/3)*(thy/D)**(20/3)

        return np.sqrt(var_x + var_y)* atm.wvl*1e9/2/np.pi

    elif method == "covariance":
        # get the covariance matrix of Zernike
        zern = zernike([2,3], tel.resolution)
        cov_zer = zern.anisokinetism(tel, atm, src, ngs)

        # get the variance
        return np.sqrt(cov_zer.trace()) * atm.wvl*1e9/2/np.pi

    else:
        raise ValueError("The method " + method + "is not supported." )

def focal_anisoplanatism_variance(tel, atm, lgs):
    """Legacy misnamed alias, returns WFE in nm (NOT a variance).

    Previously used an inconsistent 0.425 coefficient and lacked /0.423
    despite returning the same quantity as focal_anisoplanatism_wfe.
    Prefer focal_anisoplanatism_phase_variance for phase variance [rad^2]
    or focal_anisoplanatism_wfe for RMS OPD [nm].
    """
    return focal_anisoplanatism_wfe(tel, atm, lgs)


#%% PHASE STRUCTURE FUNCTION
def anisoplanatism_structure_function(tel, atm, src, lgs, ngs,
                                      nOtf, samp, nActu, msk_in=1, Hfilter=1, focal_algorithm='legacy'):
    """
    Computes the anisoplanatic phase structure functions
    INPUTS:
        - a telescope and atmosphere object
        - src, a source object that defines the target positions
        - lgs, ngs, laser guide star and natural guide star source objects
        - nOtf, the size of the angular frequency domain in pixels
        - samp, the sampling factor (2=Nyquist)
        - nActu, the 1D number DM actuators
        - Hfilter, a matrix to filter out the high spatial frequencies
    OUTPUTS:
        - NGS case : dani_ang, the angular anisoplanatism function
        - LGS case :
            - dfocani_ang, the focal-angular function
            - dani_ang
            - dani_tt, the tip-tilt anisoplanatism function
    """

    if lgs is None or lgs.height==0:
        # NGS mode, angular anisoplanatism
        dani_ang = angular_function(tel, atm, src, ngs,
                                    nOtf, samp, msk_in=msk_in)
        return dani_ang, 0, 0

    else:
        # angular + focal anisoplanatism
        lgs_dir = [lgs.zenith, lgs.azimuth]
        lgs.azimuth = 0
        lgs.zenith = 0
        dani_foc = focal_angular_function(tel, atm, src, lgs,
                                          nActu, nOtf, samp,
                                          h_filt=Hfilter,
                                          covariance_algorithm=focal_algorithm)
        # angular anisoplanatism only
        lgs.zenith = lgs_dir[0]
        lgs.azimuth = lgs_dir[1]
        dani_ang = angular_function(tel, atm, src, lgs,
                                    nOtf, samp, msk_in=msk_in)

        dani_focang = dani_foc + dani_ang
        # anisokinetism
        dani_tt = anisokinetism_function(tel, atm, src, ngs, nOtf, samp)

        return dani_focang, dani_ang, dani_tt

def angular_function(tel, atm, src, gs, nOtf, samp, method="Fourier", msk_in=1, dtype=np.float64):
    """
    Returns the nOtf x nOtf phase structure function for the angular anisoplanatism
    by following Flicker's 2008 report or the Fourier approach (Rigaut+98)
    """

    nLayer = atm.nL
    Hs = atm.heights * tel.airmass
    ax = src.direction[0] - gs.direction[0]
    ay = src.direction[1] - gs.direction[1]
    nSrc = len(ax)
    Dani_l = np.zeros((nSrc,nLayer,nOtf,nOtf))

    if method == "Flicker":
        umax = tel.D * samp
        f0 = 2*np.pi/atm.L0
        # Angular frequencies
        x = umax/nOtf * (np.array(range(nOtf)) - nOtf//2)
        rhoX, rhoY = np.meshgrid(x, x)

        # Instantiation
        I0 = 3.0/5.0
        I1 = (FourierUtils.Ialpha(f0*rhoX, f0*rhoY)
              if covariance_algorithm == 'legacy' else None)
        cte = 0.12184*0.06*(2*np.pi)**2*atm.L0**(5/3)

        # Anisoplanatism Structure Function
        for iSrc in range(nSrc):
            thx = ax[iSrc]
            thy = ay[iSrc]
            for l in range(nLayer):
                zl = Hs[l]
                if zl !=0:
                    if thx or thy:
                        I2 = FourierUtils.Ialpha(f0*(zl*thx),
                                                 f0*(zl*thy) )
                        I3 = FourierUtils.Ialpha(f0*(rhoX+zl*thx),
                                                 f0*(rhoY+zl*thy) )
                        I4 = FourierUtils.Ialpha(f0*(rhoX-zl*thx),
                                                 f0*(rhoY-zl*thy) )
                        Dani_l[iSrc, l] = (2*I0 - 2*I1 - 2*I2 + I3  + I4)
        Dani_l *= cte

    elif method == "Fourier":
        # definition of the spatial frequency domain
        PSDstep = 1/(tel.D * samp)
        ky, kx = FourierUtils.freq_array(nOtf, offset=1e-10, L=PSDstep, dtype=dtype)
        k2 = kx**2 + ky**2
        # PSD for the atmospheric turbulence
        cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
        Watm = cte*(1/atm.L0**2 + k2)**(-11/6)
        Watm[nOtf//2, nOtf//2] = 0

        for iSrc in range(nSrc):
            thx = ax[iSrc]
            thy = ay[iSrc]
            for l in range(nLayer):
                zl = Hs[l]
                if zl and (thx or thy):
                    psd = 2*(1 - np.cos(2*np.pi*zl*(kx*thx + ky*thy))) * Watm * msk_in
                    Bphi = np.fft.fft2(np.fft.fftshift(psd)) * PSDstep**2
                    Dani_l[iSrc, l] = np.fft.fftshift(np.real(2*(Bphi.max() - Bphi)))

    else:
        raise ValueError("The  method " + method + " is not supported")

    return Dani_l

def focal_angular_function(
    tel, atm, src, gs, nActu, nOtf, samp, h_filt=1,
    *, pupil_grid_mode='legacy', output_grid='otf',
    covariance_algorithm='legacy',
):
    """
    Returns the nActu**2 x nActu**2 point wise phase structure function
    for the focal-angular anisoplanatism  by following Flicker's 2008 report.
    """

    if gs.height == 0 or gs.height==np.inf:
        return angular_function(tel, atm, src, gs, nOtf, samp,
                                method="Fourier", msk_in=h_filt)


    #1\ Grabbing inputs
    # Defaults preserve the production algorithm exactly. Two opt-in
    # audit-only choices expose the raw per-baseline SF and sample
    # actual physical telescope-pupil coordinates. Neither changes
    # matrix_to_map's square-grid, unmasked pair averaging.
    if pupil_grid_mode not in ('legacy', 'physical_diameter'):
        raise ValueError("Unknown focal pupil_grid_mode: " + str(pupil_grid_mode))
    if output_grid not in ('otf', 'native'):
        raise ValueError("Unknown focal output_grid: " + str(output_grid))
    if covariance_algorithm not in ("legacy", "symmetry"):
        raise ValueError("Unknown focal covariance algorithm: " + str(covariance_algorithm))
    umax = tel.D/2 * max([1,samp])/2
    # The exact Kolmogorov limit must be evaluated analytically, NOT by
    # setting L0 to an enormous finite value: the McDonald covariance
    # terms individually approach 3/5 and their alternating sum loses
    # precision after multiplication by L0**(5/3).
    outer_scale = float(atm.L0)
    if not (outer_scale > 0 or np.isposinf(outer_scale)):
        raise ValueError("Outer scale must be positive or +infinity")
    kolmogorov_limit = bool(np.isposinf(outer_scale))
    if not kolmogorov_limit:
        f0 = 2*np.pi/outer_scale
    dist = [ src.direction[0] - gs.direction[0],
              src.direction[1] - gs.direction[1]]

    if np.isscalar(h_filt):
        h_filt = 1
    else:
        h_filt_t = h_filt.T

    def vector_to_separation(u):
        x, y = np.meshgrid(u, u)
        X = np.ones((n_ph**2, 1))*x.T.reshape(-1)
        Y = np.tile(y,[n_ph, n_ph])
        # Samples separation in the pupil
        rho_x = np.transpose(X.T - x.T.reshape(-1))
        rho_y = Y - y.T.reshape(-1)
        return X, Y, rho_x, rho_y

    #2\ SF Calculation
    # SLAO uses one deformable mirror; nActu1D can be a one-element
    # NumPy array even though the geometrical calculation needs a scalar.
    n_actu_values = np.asarray(nActu)
    if n_actu_values.size != 1:
        raise ValueError("Focal anisoplanatism requires a single DM actuator count")
    n_actu = n_actu_values.item()
    if not np.isfinite(n_actu) or n_actu <= 0 or n_actu != int(n_actu):
        raise ValueError("DM actuator count must be a positive integer")
    n_ph = 2 * int(n_actu) + 1

    # All coordinates below are physical metres in the telescope pupil.
    # Legacy grid is preserved bit-for-bit for default reconstructions.
    # The physical alternative spans [-D/2, D/2] on each pupil axis.
    if pupil_grid_mode == 'legacy':
        u = umax/n_ph * (np.arange(n_ph) - n_ph//2)
    else:
        u = np.linspace(-tel.D/2, tel.D/2, n_ph)
    X1, Y1, rhoX, rhoY = vector_to_separation(u)
    _, _, sep_x, sep_y = vector_to_separation(np.arange(n_ph))
    grid_size = (2*n_ph - 1) if output_grid == 'native' else nOtf
    Dani_l = np.zeros((src.nSrc, atm.nL, grid_size, grid_size))

    # Instantiation
    if kolmogorov_limit:
        # Ialpha(z) = 3/5 - a*z**(5/3) + higher-order terms,
        # a = -Gamma(-5/6)/(2**(8/3)*Gamma(11/6)).
        # Thus L0**(5/3)*(Ialpha(0)-Ialpha(2*pi*r/L0))
        # tends exactly to a*(2*pi*r)**(5/3).
        nu = 5.0/6.0
        a = (-spc.gamma(-nu)
             / (2.0**(2.0*nu+1.0)*spc.gamma(1.0+nu)))
        cte = 0.12184*0.06*(2*np.pi)**(2.0+5.0/3.0)*a
    else:
        I0 = 3.0/5.0
        I1 = FourierUtils.Ialpha(f0*rhoX, f0*rhoY)
        cte = 0.12184*0.06*(2*np.pi)**2 * outer_scale**(5.0/3.0)

    # Opt-in algebraic acceleration. The existing six Flicker terms and
    # matrix_to_map remain the default; direct full Keck image parity is
    # required before enabling this for production telemetry.
    if covariance_algorithm == 'symmetry':
        for iSrc in range(src.nSrc):
            thx = dist[0][iSrc]
            thy = dist[1][iSrc]
            for l, zl in enumerate(atm.heights):
                if zl:
                    tmp = symmetric_focal_layer(
                        u, X1, Y1, rhoX, rhoY, sep_x, sep_y,
                        cone_fraction=zl/gs.height,
                        theta_shift_x_m=zl*thx,
                        theta_shift_y_m=zl*thy,
                        finite_outer_scale_m=outer_scale,
                    )
                    if not np.isscalar(h_filt):
                        tmp = np.dot(h_filt, np.dot(tmp, h_filt_t))
                    cmap = grouped_matrix_to_map(tmp, sep_x, sep_y)
                    Dani_l[iSrc, l] = (
                        cmap if output_grid == 'native'
                        else FourierUtils.interpolateSupport(cmap, nOtf)
                    )
        return cte*Dani_l

    # Anisoplanatism Structure Function
    for iSrc in range(src.nSrc):
        thx = dist[0][iSrc]
        thy = dist[1][iSrc]
        for l, zl in enumerate(atm.heights):
            if zl:
                g = zl/gs.height
                if kolmogorov_limit:
                    # Exact L0 -> infinity limit of the same six-term
                    # Flicker covariance, preserving its spatial dependence
                    # on both pupil coordinates and point-pair baseline.
                    p = lambda x, y: np.hypot(x, y)**(5.0/3.0)
                    tmp = (
                        p(rhoX, rhoY)
                        + p((1-g)*rhoX, (1-g)*rhoY)
                        - p(rhoX - g*X1 + zl*thx,
                            rhoY - g*Y1 + zl*thy)
                        + p(g*X1 - zl*thx, g*Y1 - zl*thy)
                        + p(g*(rhoX-X1) - zl*thx,
                            g*(rhoY-Y1) - zl*thy)
                        - p((1-g)*rhoX + g*X1 - zl*thx,
                            (1-g)*rhoY + g*Y1 - zl*thy)
                    )
                else:
                    I2 = FourierUtils.Ialpha(f0*(rhoX*(1-g)),
                                             f0*(rhoY*(1-g)))
                    I3 = FourierUtils.Ialpha(f0*(rhoX - g*X1 + zl*thx),
                                             f0*(rhoY - g*Y1 + zl*thy))
                    I4 = FourierUtils.Ialpha(f0*(g*X1 - zl*thx),
                                             f0*(g*Y1 - zl*thy))
                    I5 = FourierUtils.Ialpha(f0*(g*(rhoX-X1) - zl*thx),
                                             f0*(g*(rhoY-Y1) - zl*thy))
                    I6 = FourierUtils.Ialpha(
                        f0*((1-g)*rhoX + g*X1 - zl*thx),
                        f0*((1-g)*rhoY + g*Y1 - zl*thy)
                    )
                    tmp = 2*I0 - I1 - I2 + I3 - I4 - I5 + I6
                if not np.isscalar(h_filt):
                    tmp = np.dot(h_filt, np.dot(tmp, h_filt_t))

                # ---- get the maps
                cmap = FourierUtils.matrix_to_map(tmp, sep_x=sep_x, sep_y=sep_y)

                # ---- interpolation
                # Native output exposes the physical covariance-lag axis
                # to audit scripts. Production still does the original zoom.
                Dani_l[iSrc, l] = (
                    cmap if output_grid == 'native'
                    else FourierUtils.interpolateSupport(cmap, nOtf)
                )

    return cte*Dani_l

def anisokinetism_function(tel, atm, src, gs, nOtf, samp):
    """
    Returns the nOtf x nOtf anisokinetism phase structure function as a Gaussian
    Kernel.
    """

    #1\ Defining the spatial filters
    dist_x = src.direction[0] - gs.direction[0]
    dist_y = src.direction[1] - gs.direction[1]
    nSrc = len(dist_x)

    #2\ defining tip-tilt modes
    zern = zernike([2,3], nOtf)
    X = zern.modes[0]/2
    Y = zern.modes[1]/2
    X2 = X**2
    Y2 = Y**2
    # Off-diagonal covariance is the local 2-D quadratic cross term XY.
    # Transposing either coordinate changes XY into an incorrect square
    # on the usual Cartesian Zernike grid.
    XY = X*Y
    YX = Y*X

    # instantiating the phase structure function
    Dani_l = np.zeros((nSrc, atm.nL, nOtf, nOtf))

    # computing the phase structure function for each layers and src
    for iSrc in range(nSrc):
        thx = dist_x[iSrc]
        thy = dist_y[iSrc]
        if thx or thy:
            for l, zl in enumerate(atm.heights):
                if zl !=0:
                    # update the atmosphere
                    atm_l = atm.slab(l)
                    fr0 = atm_l.r0**(-5/3)
                    # get the 2x2 anisokinetism covariance matrix
                    covAniso = zern.anisokinetism(tel, atm_l, src, gs)
                    # defining the Gaussian kernel
                    Dani_l[iSrc,l] = (covAniso[0, 0]*X2\
                                    + covAniso[1,1]*Y2\
                                    + covAniso[0,1]*XY\
                                    + covAniso[1,0]*YX)/fr0

    return Dani_l

