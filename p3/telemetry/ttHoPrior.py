"""P4-G: independent HO POL von-Karman prior for Keck STRAP tilt statistics.

G1: atmospheric incident tilt covariance in NGS direction from HO r0,L0.
G2: frozen-flow atmospheric tilt filtered through STRAP closed loop.
G3: compare that same-direction *atmospheric residual* to STRAP PSD with
    van Dam sensor noise removed, without projecting negative spectra.
G4: gain/band/HO-prior sensitivity; report only conditional consistency.
G5: differential NGS-science anisokinetic covariance as a SEPARATE
    diagnostic; never insert into the on-axis residual covariance or PSF.

Critically, G5's anisokinetic term is already represented by P3 dani_tt.
This module does NOT alter P3 dphi_tt, dani_tt, dphi_ani or source PSF.
It makes no assertion that r0,L0 alone determine the temporal PSD: wind
directions in Keck P3 are often default assumptions, not measurements.

References: Noll (1976); Conan/Rousset/Madec JOSA A (1995);
Keck LGS angular anisoplanatism (van Dam et al. 2006);
Beltramo-Martin et al. MNRAS 478, 4642-4655 (2018).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import map_coordinates
from scipy.special import jv
from scipy.signal import welch, csd

ARCSEC_TO_RAD=np.pi/(180*3600.0)
RAD_TO_MAS=1e3/ARCSEC_TO_RAD


def _covariance(matrix):
    c=np.asarray(matrix,dtype=float)
    if c.shape!=(2,2) or not np.all(np.isfinite(c)):
        raise ValueError("Expected finite covariance 2x2")
    return (c+c.T)/2


@dataclass
class TiltKernel:
    """Spatial frequency grid and 2x2 real angular-tilt power kernel.

    kernel_ij(k) is dimensionless spatial phase→angular-tilt transfer
    with a λ/(2π) factor applied at integration. Pixel pupil weights
    are fractional *illuminated areas*, not squared optical amplitudes.
    """
    radius:np.ndarray                # cycles/m
    kx:np.ndarray                    # (nrad,nazimuth)
    ky:np.ndarray
    hh:np.ndarray                    # (2,2,nrad,nazimuth), units m^-2
    diameter_m:float
    pupil_grid:int
    fft_pad:int
    mask_fill_fraction:float
    max_frequency_cycles_per_m:float


def kernel_from_pupil(mask,diameter_m, *,pad=1024,n_rad=450,n_az=40,
                      kmax_diameter=15.0):
    """Compute best-fit plane tilt from the *actual* Keck pupil mask.

    We project phase on (x,y) in the intensity-weighted illuminated
    pupil, using the 2x2 coordinate moment matrix. Fourier transforms
    are sampled on a log-polar spatial frequency grid to retain low
    frequency turbulence; a flat fft2 sum drops most tilt variance.
    Grid resolution/padding must be tested in the scientific audit.
    """
    p=np.asarray(mask,dtype=float)
    if (p.ndim!=2 or p.shape[0]!=p.shape[1] or p.shape[0]<32
        or not np.isfinite(p).all() or p.min()<0 or p.max()>1
        or not (0<p.sum()<p.size) or pad<2*p.shape[0]
        or n_rad<64 or n_az<16 or diameter_m<=0
        or kmax_diameter<=1):
        raise ValueError("Pupil must be a nonempty bounded square mask; "
                         "pad>=2*n, n_rad>=64, n_az>=16")
    n=p.shape[0]
    dx=diameter_m/n
    ticks=(np.arange(n)-(n-1)/2)*dx
    yy,xx=np.meshgrid(ticks,ticks,indexing="ij")
    xy=np.stack([xx,yy])
    mu=np.einsum("ij,aij->a",p,xy)/p.sum()
    coord=xy-mu[:,None,None]
    moment=np.einsum("ij,aij,bij->ab",p,coord,coord)/p.sum()
    if np.linalg.eigvalsh(moment).min()<=1e-10*diameter_m**2:
        raise ValueError("Degenerate pupil-coordinate moment matrix")
    invmom=np.linalg.inv(moment)
    ft=[]
    for axis in range(2):
        z=np.zeros((pad,pad),dtype=float)
        start=(pad-n)//2
        z[start:start+n,start:start+n]=p*coord[axis]
        ft.append(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(z)))/p.sum())
    # Near-zero frequencies matter strongly, especially when L0 is large.
    # Log-polar integration avoids the catastrophic missing-subharmonic
    # bias of summing a finite FFT grid directly.
    rad=np.geomspace(1e-7/diameter_m,kmax_diameter/diameter_m,n_rad)
    theta=np.arange(n_az)*2*np.pi/n_az
    kx=rad[:,None]*np.cos(theta)[None,:]
    ky=rad[:,None]*np.sin(theta)[None,:]
    df=1/(pad*dx)
    coords=np.stack([ky/df+pad//2,kx/df+pad//2])
    # Complex interpolation matters for cross-terms between x/y.
    fk=np.stack([
        map_coordinates(z.real,coords,order=1,mode="constant")+
        1j*map_coordinates(z.imag,coords,order=1,mode="constant")
        for z in ft
    ])
    # Fourier form factors of best-fit physical gradients; dimensions 1/m.
    H=np.einsum("ab,brt->art",invmom,fk)
    hh=(H[:,None,:,:]*H[None,:,:,:].conj()).real
    return TiltKernel(
        radius=rad,kx=kx,ky=ky,hh=hh,diameter_m=float(diameter_m),
        pupil_grid=n,fft_pad=pad,mask_fill_fraction=float(p.mean()),
        max_frequency_cycles_per_m=float(rad[-1]),
    )


def _von_karman(radius,r0_m,L0_m):
    if (not np.isfinite(r0_m) or r0_m<=0
        or not np.isfinite(L0_m) or L0_m<=0):
        raise ValueError("HO r0 and L0 must be finite positive")
    return 0.023*r0_m**(-5/3)*(
        radius**2+L0_m**(-2)
    )**(-11/6)


def _radial_covariance(kernel,atmosphere, *,multiplier=None,wavelength_m=500e-9):
    """Integrate over polar angle and log-spaced radius, returning rad²."""
    if not 0<wavelength_m<1e-3:
        raise ValueError("Reference wavelength should be SI metres")
    if multiplier is None:
        multiplier=1.0
    m=np.asarray(multiplier,dtype=float)
    if not np.isfinite(m).all():
        raise ValueError("Nonfinite spatial weighting function")
    power=(kernel.hh*(
        atmosphere*kernel.radius
    )[None,None,:,None]*m)
    angle_mean=power.mean(axis=-1)*2*np.pi
    c=(wavelength_m/(2*np.pi))**2*trapezoid(
        angle_mean,x=kernel.radius,axis=-1
    )
    return _covariance(c)


def circular_vk_tilt_variance(r0,L0,D,wavelength_m=500e-9, *,
                             nrad=4000):
    """Independent circular-pupil Bessel integral, per-axis rad².

    With L0→infty: 0.1816*λ²*r0^(-5/3)*D^(-1/3), up to rounding.
    Keck pupil does NOT generally have this coefficient.
    """
    r=np.geomspace(1e-8/D,40/D,nrad)
    x=np.pi*D*r
    H=(16/D)*jv(2,x)/x
    return float(
        (wavelength_m/(2*np.pi))**2*np.pi*
        trapezoid(_von_karman(r,r0,L0)*H**2*r,x=r)
    )


def incident_and_residual_tilt(kernel, *,r0_m,L0_m,wavelength_m,
                                weights,wind_speeds_m_s,wind_dirs_deg,
                                loop_frequency_hz=None,
                                loop_sensitivity_power=None,
                                low_band_hz=0.0,high_band_hz=None,
                                science_offset_pupil_arcsec=None,
                                altitudes_m=None):
    """G1/G2/G5 using the SAME von Karman pupil and layer hypotheses.

    Wind frozen-flow temporal frequencies are |κ·v|. Every layer
    receives its Cn2 fraction. G5 is a DIFFERENTIAL tilt covariance
    NGS-science and is NEVER added to the returned G2 residual.
    """
    weights=np.asarray(weights,dtype=float)
    speeds=np.asarray(wind_speeds_m_s,dtype=float)
    dirs=np.asarray(wind_dirs_deg,dtype=float)
    if (weights.ndim!=1 or len(weights)==0
        or weights.shape!=speeds.shape or weights.shape!=dirs.shape
        or not np.isfinite(weights).all() or np.any(weights<0)
        or weights.sum()<=0 or not np.isfinite(speeds).all()
        or np.any(speeds<0) or not np.isfinite(dirs).all()):
        raise ValueError("Invalid Cn2 weights / layer wind configuration")
    weights=weights/weights.sum()
    if loop_sensitivity_power is not None:
        fgrid=np.asarray(loop_frequency_hz,dtype=float)
        sgrid=np.asarray(loop_sensitivity_power,dtype=float)
        if (fgrid.ndim!=1 or sgrid.shape!=fgrid.shape or len(fgrid)<2
            or np.any(np.diff(fgrid)<=0) or not np.isfinite(sgrid).all()
            or np.any(sgrid<0)):
            raise ValueError("Invalid TT loop sensitivity spectrum")
    else:
        fgrid=sgrid=None
    if low_band_hz<0 or (high_band_hz is not None and
                         high_band_hz<=low_band_hz):
        raise ValueError("Invalid integration frequency band")
    if science_offset_pupil_arcsec is not None:
        if altitudes_m is None:
            raise ValueError("G5 anisokinetism needs layer altitudes")
        heights=np.asarray(altitudes_m,dtype=float)
        sep=np.asarray(science_offset_pupil_arcsec,dtype=float)
        if (heights.shape!=weights.shape or
            np.any(~np.isfinite(heights)) or np.any(heights<0) or
            sep.shape!=(2,) or not np.isfinite(sep).all()):
            raise ValueError("Invalid G5 pupil-frame science offset or layers")
    else:
        heights=sep=None
    psd=_von_karman(kernel.radius,r0_m,L0_m)
    c_ol=np.zeros((2,2))
    c_rej_full=np.zeros((2,2))
    c_rej_band=np.zeros((2,2))
    c_diff=np.zeros((2,2))
    for i,wt in enumerate(weights):
        if wt==0: continue
        c_ol+=wt*_radial_covariance(kernel,psd,wavelength_m=wavelength_m)
        if sgrid is not None:
            ang=np.deg2rad(dirs[i])
            temporal=np.abs(
                kernel.kx*speeds[i]*np.cos(ang)
                +kernel.ky*speeds[i]*np.sin(ang)
            )
            # Below loop sampling, the exact integrator suppresses DC;
            # above Nyquist, correction cannot be assumed effective.
            s2=np.interp(temporal,fgrid,sgrid,left=0.0,right=1.0)
            c_rej_full+=wt*_radial_covariance(
                kernel,psd,multiplier=s2,wavelength_m=wavelength_m
            )
            select=(temporal>=low_band_hz)
            if high_band_hz is not None:
                select &= temporal<=high_band_hz
            c_rej_band+=wt*_radial_covariance(
                kernel,psd,multiplier=s2*select,
                wavelength_m=wavelength_m
            )
        if heights is not None:
            # Atmospheric tilt difference s - n, NOT an extra servo
            # residual. P3 dani_tt already accounts for this term.
            shift=heights[i]*sep*ARCSEC_TO_RAD
            cos=np.cos(2*np.pi*(
                kernel.kx*shift[0]+kernel.ky*shift[1]
            ))
            c_diff+=wt*_radial_covariance(
                kernel,psd,multiplier=2*(1-cos),
                wavelength_m=wavelength_m
            )
    return {
        "incident_ngs_cov_rad2":_covariance(c_ol),
        "residual_ngs_atmos_full_cov_rad2":_covariance(c_rej_full)
            if sgrid is not None else None,
        "residual_ngs_atmos_band_cov_rad2":_covariance(c_rej_band)
            if sgrid is not None else None,
        "anisokinetism_ngs_science_cov_rad2":_covariance(c_diff)
            if heights is not None else None,
    }


def measured_strap_residual_atmosphere(centroids_arcsec,
                                      white_noise_cov_arcsec2,
                                      *,sample_rate_hz,
                                      loop_frequency_hz,
                                      loop_sensitivity_power,
                                      loop_sensor_power,
                                      gain=1.0,nperseg=512,
                                      fmin_hz=None):
    """G3: match HO band and STRAP atmospheric residual covariance.

    y = g*W*e + S*n. Hence residual-atmospheric PSD =
    (PSD_y - |S|² PSD_n)/(g² |W|²).
    No division by S, no clipping, and no PSD projection.
    NOTE: the quadrature is only Welch band [fs/nperseg, fs/2].
    """
    if not np.isfinite(gain) or gain<=0:
        raise ValueError("Optical gain must be finite positive")
    y=np.asarray(centroids_arcsec,dtype=float)
    if (y.ndim!=2 or y.shape[1]!=2 or len(y)<nperseg
        or nperseg<128 or not np.isfinite(y).all()):
        raise ValueError("Invalid STRAP series or Welch window")
    cn=_covariance(white_noise_cov_arcsec2)
    if np.linalg.eigvalsh(cn).min()< -1e-12*max(np.linalg.norm(cn),1e-30):
        raise ValueError("Van Dam covariance is non-PSD")
    kw=dict(fs=sample_rate_hz,nperseg=nperseg,noverlap=nperseg//2,
            detrend="constant",scaling="density",window="hann")
    f,pxx=welch(y[:,0],**kw)
    _,pyy=welch(y[:,1],**kw)
    _,pxy=csd(y[:,0],y[:,1],**kw)
    power=np.empty((2,2,len(f)))
    power[0,0]=pxx;power[1,1]=pyy
    power[0,1]=power[1,0]=pxy.real
    fs=float(sample_rate_hz)
    fl=np.asarray(loop_frequency_hz,dtype=float)
    S=np.interp(f,fl,np.asarray(loop_sensitivity_power,dtype=float),
                left=0.,right=1.)
    W=np.interp(f,fl,np.asarray(loop_sensor_power,dtype=float),
                left=1.,right=1.)
    if np.min(W)<1e-5:
        raise ValueError("Cannot invert zero STRAP sensor transfer")
    noise_psd=cn[:,:,None]*2/fs*S[None,None,:]
    atm_y=power-noise_psd
    atm_e=atm_y/(gain*gain*W[None,None,:])
    cutoff=max(float(fs/nperseg),float(fmin_hz or 0))
    band=(f>=cutoff)&(f<=fs/2)
    def integrate(array):
        return _covariance(trapezoid(array[:,:,band],x=f[band],axis=-1))
    cy=integrate(atm_y)
    ce=integrate(atm_e)
    negative=np.linalg.eigvalsh(
        np.moveaxis(atm_y[:,:,band],-1,0)
    )[:,0]
    return {
        "measured_centroid_atm_band_cov_arcsec2":cy,
        "inferred_physical_residual_atm_band_cov_arcsec2":ce,
        "min_residual_atm_eigenvalue_arcsec2":
            float(np.linalg.eigvalsh(ce).min()),
        "noise_subtracted_negative_cospectral_bins":int((negative<0).sum()),
        "welch_bins":int(band.sum()),
        "band_hz":(float(f[band][0]),float(f[band][-1])),
        "no_psd_projection":True,
    }


def scalar_consistency(pred_rad2,strap_arcsec2):
    """G4: ratio of traces, never a calibrated optical gain."""
    a=_covariance(pred_rad2)
    b=_covariance(strap_arcsec2)*ARCSEC_TO_RAD**2
    eb=np.linalg.eigvalsh(b)
    tr=float(np.trace(b))
    scale=max(float(np.max(np.abs(eb))),1e-40)
    if eb.min()< -1e-6*scale or tr<=0:
        return {"status":"invalid_STRAP_residual_covariance",
                "alpha":None,"angle_amplitude_proxy":None}
    alpha=float(np.trace(a)/tr)
    return {
        "status":"conditional_covariance_trace_ratio",
        "alpha":alpha,
        "angle_amplitude_proxy":float(np.sqrt(alpha)),
        "physical_optical_gain_identified":False,
        "shape_mismatch_unsolved":True,
    }


def matrix_json(matrix):
    return None if matrix is None else np.asarray(matrix,dtype=float).tolist()


def mas_rms(matrix_rad2):
    if matrix_rad2 is None:return None
    c=_covariance(matrix_rad2)
    v=np.linalg.eigvalsh(c)
    if v.min()< -1e-7*max(float(abs(v).max()),1e-36):
        return None
    return (np.sqrt(np.maximum(v,0))*RAD_TO_MAS).tolist()
