"""Conditional, band-limited, long-exposure TT residual covariance.

Exact PSD algebra for the *explicitly assumed* discrete integrator:
  y[k] = g * (d[k] - u[k-1]) + n[k]
  u[k] = u[k-1] + controller_gain*y[k]

q=exp(-i*2*pi*f/fs)
D=1+(g*controller_gain-1)*q
S=(1-q)/D
T_noise_to_residual=-q*controller_gain/D

P_yy = g^2 |S|^2 P_dd + |S|^2 P_nn
P_ee = |S|^2 P_dd + |T|^2 P_nn
     = P_yy/g^2 + (|T|^2 - |S|^2/g^2)*P_nn.

The last identity is finite at f=0; it avoids dividing by |S|^2.
The band-limited matrix from observed *recorded* centroid spectra
is NOT a calibrated Keck sky-jitter covariance, because mirror P,
optical gain g, and WFS integration W remain uncalibrated.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch, csd


def observed_centroid_cospectrum(samples_arcsec, sampling_hz, *,
                                nperseg=512):
    y=np.asarray(samples_arcsec,dtype=float)
    if (y.ndim!=2 or y.shape[1]!=2 or len(y)<nperseg
        or not np.isfinite(y).all() or nperseg<128):
        raise ValueError("Expected finite paired STRAP centroids and nperseg>=128")
    kwargs=dict(fs=sampling_hz,nperseg=nperseg,
                noverlap=nperseg//2,window="hann",
                detrend="constant",scaling="density")
    f,xx=welch(y[:,0],**kwargs)
    _,yy=welch(y[:,1],**kwargs)
    _,xy=csd(y[:,0],y[:,1],**kwargs)
    psd=np.empty((2,2,len(f)),dtype=float)
    psd[0,0]=xx; psd[1,1]=yy
    psd[0,1]=psd[1,0]=xy.real
    return f,psd


def conditional_residual_covariance(
    frequency_hz,
    measured_cospectrum_arcsec2_per_hz,
    white_sensor_noise_covariance_recorded_arcsec2,
    *,
    sampling_hz, controller_gain, optical_gain,
    min_frequency_hz=None,
):
    """Band-limited C_e under hypothesized unit-plant, 1-sample delay.

    Sensor noise covariance is white BEFORE closed-loop rejection, in
    recorded centroid arcsec². No atmosphere model is assumed or fit.
    No negative PSD is repaired: invalid/ill-conditioned scenarios
    retain their diagnostics, and no physical RMS is endorsed.
    """
    f=np.asarray(frequency_hz,dtype=float)
    pyy=np.asarray(measured_cospectrum_arcsec2_per_hz,dtype=float)
    cn=np.asarray(white_sensor_noise_covariance_recorded_arcsec2,dtype=float)
    fs=float(sampling_hz); g=float(optical_gain); k=float(controller_gain)
    if (f.ndim!=1 or len(f)<4 or not np.isfinite(f).all()
        or np.any(np.diff(f)<=0) or f[0]<0 or f[-1]>fs/2+1e-6):
        raise ValueError("Invalid frequency grid")
    if (pyy.shape!=(2,2,len(f)) or cn.shape!=(2,2)
        or not np.isfinite(pyy).all() or not np.isfinite(cn).all()):
        raise ValueError("Invalid finite 2x2 spectral covariance")
    if not np.allclose(cn,cn.T,atol=1e-16,rtol=1e-10):
        raise ValueError("Sensor noise covariance must be symmetric")
    if np.linalg.eigvalsh(cn).min() < -1e-12*max(np.linalg.norm(cn),1e-30):
        raise ValueError("Sensor noise must be PSD")
    if not 0<g or not 0<k*g<2:
        raise ValueError("Unstable hypothesized gain")
    df=float(np.median(np.diff(f)))
    # Do not declare full-exposure jitter with a Welch window.
    fmin=float(df if min_frequency_hz is None else max(df,min_frequency_hz))
    fit=f>=fmin
    if fit.sum()<8:
        raise ValueError("Insufficient non-DC bins for covariance")
    q=np.exp(-2j*np.pi*f/fs)
    den=1+(g*k-1)*q
    S=(1-q)/den
    T=-q*k/den
    noise_density=2*cn/fs
    s2=np.abs(S)**2
    t2=np.abs(T)**2
    p_atmos_y=pyy-s2[None,None,:]*noise_density[:,:,None]
    p_e=pyy/g**2 + (
        t2-s2/g**2
    )[None,None,:]*noise_density[:,:,None]
    # Symmetry and finite precision: preserve negative eigenvalues rather
    # than silently projecting a physically inconsistent candidate.
    ce=trapezoid(p_e[:,:,fit],x=f[fit],axis=-1)
    ce=(ce+ce.T)/2
    cya=trapezoid(p_atmos_y[:,:,fit],x=f[fit],axis=-1)
    cya=(cya+cya.T)/2
    physical_eig=np.linalg.eigvalsh(ce)
    spectrum_eigenvalues=np.linalg.eigvalsh(
        np.moveaxis(p_atmos_y[:,:,fit],-1,0)
    )
    n_bad=int(np.count_nonzero(spectrum_eigenvalues[:,0] < -
        1e-9*np.maximum(np.abs(spectrum_eigenvalues).max(axis=1),1e-30)))
    valid=bool(physical_eig.min()>=-
               1e-9*max(np.abs(physical_eig).max(),1e-30))
    return {
        "status":"conditional_band_limited_unverified_instrument_plant",
        "assumed_effective_optical_gain":g,
        "frequency_interval_hz":[float(f[fit][0]),float(f[fit][-1])],
        "welch_low_frequency_leakage_not_included":True,
        "filter_plant_and_command_absolute_scale_calibrated":False,
        "no_psd_projection":True,
        "centroid_atmospheric_band_covariance_recorded_arcsec2":cya.tolist(),
        "residual_band_covariance_conditional_true_arcsec2":ce.tolist(),
        "residual_band_covariance_min_eigenvalue_arcsec2":
            float(physical_eig.min()),
        "atmospheric_measured_spectral_bins_negative":n_bad,
        "band_covariance_psd":valid,
        "residual_band_rms_axes_conditional_mas":(
            (1000*np.sqrt(np.maximum(0,np.diag(ce)))).tolist()
            if valid else None
        ),
        "disclaimer":(
            "Not an absolute Keck jitter estimate. The result is a "
            "band-limited sensitivity under unity physical command scale, "
            "one-frame mirror delay, instantaneous WFS and assumed g; "
            "unknown low-frequency variance and mirror calibration remain."
        ),
    }
