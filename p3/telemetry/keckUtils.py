#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:37:55 2021

@author: omartin
"""

#%% IMPORTING LIBRARIES
import numpy as np
from pathlib import Path
import warnings
from astropy.table import Table
import matplotlib.pyplot as plt

#%% GETTING INFOR FROM THE READER
def estimateLoopDelay(hdr):
    """
    Return the loop delay in seconds for the HO and TT loops
    """

    wssmprg  = int(hdr['WSSMPRG'])
    _,aoMode = getStagePositionWFS(hdr)

    if wssmprg == 0:
        t0 = 8000e-6
    if wssmprg == 1:
        t0 = 2600e-6
    if wssmprg == 2:
        t0 = 850e-6
    if wssmprg == 3:
        t0 = 700e-6
    if wssmprg == 4:
        t0 = 28000e-6
    if wssmprg == 5:
        t0 = 9500e-6
    if wssmprg == 6:
        t0 = 2000e-6
    if wssmprg == 7:
        t0 = 1350e-6

    # Total delays
    t8    = 40e-6
    delay = t0+t8

    if aoMode == 'NGS':
        t9        = 1e-6
        t10       = 2e-6
        textra_tt = 200e-6 # retrieved from rejection transfer function fit
        delay_tt  = t0 + t9 + t10 + textra_tt
    else:
        delay_tt  = 1e-3 #STRAP

    return delay, delay_tt

def getScale(hdr):
    """
    Returns the instrument pixel scale in mas
    """
    instName = getInstName(hdr)
    if instName == 'NIRC2':
        # Setup NIRC2 plate scales
        scales = {"NARROW": 9.942,
              "MEDIUM": 19.829,
              "WIDE": 39.686}
        return scales[hdr['CAMNAME'].upper()]

    elif instName == 'OSIRIS':
        return 9.95
    else:
        return None

def getPA(hdr):
    #theta = float(hdr['ROTPOSN']) - float(hdr['EL']) - getInstAngle(hdr)
    theta = float(hdr['ROTPOSN']) - getInstAngle(hdr)
    return theta

def getPupilMask(hdr):
    return hdr['PMSNAME']

def getTelescopeAirmass(hdr):
    el = float(hdr['EL'])
    zenithAngle = 90 - el
    airmass = 1/np.cos(zenithAngle*np.pi/180)
    return zenithAngle,airmass

def getInstAngle(hdr):
    """
    Returns the instrument pupil position in degrees
    """
    instName = getInstName(hdr)
    if instName == 'NIRC2':
        return float(hdr['INSTANGL'])
    elif instName == 'OSIRIS':
        return float(hdr['INSTANGL']) - 42.5
    else:
        return 0

def getInstName(hdr):
    return hdr['CURRINST']

def getCentralWavelength(hdr):
    """
    Returns the central wavelength in meters
    """
    return float(hdr['CENWAVE'])*1e-6

def getGain(hdr):
    return hdr['DETGAIN']

def getFilterName(hdr):
    instName = getInstName(hdr)
    if instName == 'NIRC2':
        filter1 = str(hdr['fwiname']).strip()
        filter2 = str(hdr['fwoname']).strip()
        # A wheel holding an open position or blocking glass does not
        # identify the scientific passband. Select the other wheel.
        if filter1.upper().startswith(('PK', 'CLEAR', 'OPEN')):
            return filter2
        return filter1
    elif instName == 'OSIRIS':
        f = hdr['ifilter']
        return f.split('-')[0]
    else:
        return None

def _find_filter_profile_file(path_filter, spectral_filter):
    """Resolve locally tabulated NIRC2 profiles without a hard-coded whitelist."""
    folder = Path(path_filter)
    wanted = ''.join(ch for ch in str(spectral_filter).casefold() if ch.isalnum())
    if not wanted:
        raise FileNotFoundError("No NIRC2 science-filter name in the FITS header")
    if not folder.is_dir():
        raise FileNotFoundError(f"NIRC2 filter-profile directory missing: {folder}")
    available = sorted(folder.glob("*.dat"))
    for candidate in available:
        normalized = ''.join(ch for ch in candidate.stem.casefold() if ch.isalnum())
        if normalized == wanted:
            return candidate
    names = ', '.join(item.stem for item in available)
    raise FileNotFoundError(
        f"No tabulated NIRC2 transmission for {spectral_filter!r} in {folder}; "
        f"available profiles: {names or '(none)'}"
    )


def getFilterProfile(path_filter, spectral_filter):
    """Read a local NIRC2 filter curve; never return None on an unknown filter.

    The original calibration tables store wavelengths in microns and
    transmission in percent. Missing profiles raise FileNotFoundError, which
    the sampling layer may handle via a clearly-labelled FITS-header proxy.
    """
    profile_path = _find_filter_profile_file(path_filter, spectral_filter)
    table = Table.read(str(profile_path), format='ascii')
    wavelength = np.asarray(table[table.colnames[0]], dtype=float).copy()
    transmission = np.asarray(table[table.colnames[1]], dtype=float).copy()
    if wavelength.ndim != 1 or len(wavelength) < 2:
        raise ValueError(f"Invalid NIRC2 transmission table: {profile_path}")
    if not (np.all(np.isfinite(wavelength)) and np.all(np.isfinite(transmission))):
        raise ValueError(f"Non-finite NIRC2 transmission table: {profile_path}")
    # Original .dat files are percentages; accept a fraction-valued table
    # only when explicitly recognizable from the maximum throughput.
    if np.max(transmission) > 1.0:
        transmission = transmission / 100.0
    valid = transmission > 0.01
    wavelength = wavelength[valid]
    transmission = transmission[valid]
    if len(wavelength) < 2:
        raise ValueError(f"No usable NIRC2 transmission points: {profile_path}")
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    transmission = transmission[order]
    # Match original treatment of duplicate wavelength samples.
    repeated = np.where(np.diff(wavelength) <= 0)[0]
    wavelength[repeated + 1] += 1e-7
    return wavelength * 1e-6, transmission


def getDate(hdr):
    return hdr['DATE-OBS']

def getNCPAstr(hdr):
    """
    """
    date = getDate(hdr)
    numMonth = date[5:7]
    strMonth = ''
    strYear = date[0:4]
    if numMonth == '01': pass
    elif numMonth == '02':
        strMonth = 'Feb'
    elif numMonth == '03':
        strMonth = 'March'
        strDay = date [8:10]
        NCPAstr = strDay+ strMonth + strYear + '_PD1'
    elif numMonth == '04': pass
    elif numMonth == '05': pass
    elif numMonth == '06': pass
    elif numMonth == '07': pass
    elif numMonth == '08':
        strMonth = 'Aug'
    elif numMonth == '09':
        strMonth = 'Sept'
    elif numMonth == '10': pass
    elif numMonth == '11': pass
    elif numMonth == '12': pass

    if numMonth != '03':
        NCPAstr = strMonth + '_' + strYear

    return 'NCPA_Keck_' + NCPAstr + '.fits'

def getExposureConfig(hdr):

    ittime   = hdr['ITIME']
    coadds   = hdr['COADDS']
    sampmode = hdr['SAMPMODE']
    numreads = hdr['MULTISAM']
    gain     = hdr['GAIN']
    if sampmode == 2:
        ron = 60
    else:
        ron = 15.0 * (16.0 / numreads)**0.5


    return ittime,coadds,sampmode,numreads,ron,gain

def samplingFilterProfile(path_filter, hdr, nWvl=3, thresold=0.9,
                          bwMin=100e-9, display=False, return_metadata=False):
    """Compress a tabulated NIRC2 filter or use a labelled header-only proxy.

    Missing tables may use CENWAVE/MINWAVE/MAXWAVE from the science FITS
    header, with flat spectral weights. This supports exploratory PSF-R,
    not a calibrated passband or a quantitative chromatic PSF comparison.
    """
    instName = getInstName(hdr)
    filter_name = getFilterName(hdr)
    metadata = {
        'instrument': instName,
        'filter_name': filter_name,
        'composite_filter': hdr.get('FILTER'),
        'source': None,
        'tabulated_transmission_curve_available': False,
        'flat_transmission_approximation': False,
    }
    try:
        wvl_all, tr_all = getFilterProfile(path_filter, filter_name)
    except FileNotFoundError as exc:
        if not Path(path_filter).is_dir():
            raise FileNotFoundError(
                f"NIRC2 filter calibration directory does not exist: {path_filter}"
            ) from exc
        if str(filter_name).upper().startswith(('CLEAR', 'OPEN', 'PK')):
            raise ValueError(
                f"No active NIRC2 science passband identified; "
                f"inner/outer filter wheels: "
                f"{hdr.get('FWINAME')!r}/{hdr.get('FWONAME')!r}"
            ) from exc
        if instName != 'NIRC2':
            raise ValueError(
                f"Missing {instName!r} filter curve for {filter_name!r}; "
                "no instrument-agnostic fallback is defined"
            ) from exc
        try:
            minimum_um = float(hdr['MINWAVE'])
            centre_um = float(hdr['CENWAVE'])
            maximum_um = float(hdr['MAXWAVE'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Unknown NIRC2 filter {filter_name!r} and missing/invalid "
                "MINWAVE, CENWAVE or MAXWAVE for provisional reconstruction"
            ) from error
        if not (np.isfinite(minimum_um) and np.isfinite(centre_um)
                and np.isfinite(maximum_um)
                and 0 < minimum_um < centre_um < maximum_um):
            raise ValueError(
                f"Unknown NIRC2 filter {filter_name!r}; invalid wavelength "
                f"interval {minimum_um}, {centre_um}, {maximum_um} microns"
            ) from exc
        bw = (maximum_um - minimum_um) * 1e-6
        if bw <= bwMin:
            wavelengths = np.array([centre_um * 1e-6])
        else:
            wavelengths = np.linspace(minimum_um, maximum_um, nWvl) * 1e-6
        transmission = np.ones(len(wavelengths), dtype=float)
        dispersion = [list(np.zeros(len(wavelengths))),
                      list(np.zeros(len(wavelengths)))]
        if bw <= bwMin:
            bw = 0.0
        metadata.update({
            'source': 'fits_header_flat_passband',
            'flat_transmission_approximation': True,
            'header_central_wavelength_um': centre_um,
            'header_min_wavelength_um': minimum_um,
            'header_max_wavelength_um': maximum_um,
            'approximation_warning': (
                "No transmission profile in local calibration; provisional "
                "flat weights over science FITS header passband, not calibrated"
            ),
        })
        warnings.warn(
            f"NIRC2 filter {filter_name!r}: no local transmission curve. "
            f"Using an UNCALIBRATED flat passband from FITS header "
            f"({minimum_um:g}-{maximum_um:g} um). Chromatic/photometric "
            "benchmark results are provisional.",
            RuntimeWarning, stacklevel=2,
        )
        result = (wavelengths, bw, transmission, dispersion)
        return (*result, metadata) if return_metadata else result

    # Existing tabulated-profile sampling: prioritize measured curves.
    id_f = tr_all > thresold * tr_all.max()
    tr_f = tr_all[id_f]
    wvl_f = wvl_all[id_f]
    if len(wvl_f) < 2:
        raise ValueError(
            f"NIRC2 filter {filter_name!r} has fewer than two usable "
            f"transmission points above relative threshold {thresold}"
        )
    p = np.polyfit(wvl_f, tr_f, 1)
    tr_f = np.polyval(p, wvl_f)
    wvlMin = wvl_f.min()
    wvlMax = wvl_f.max()
    bw = wvlMax - wvlMin
    if bw <= bwMin:
        nWvl = 1
    if nWvl > 1:
        wvl_c = np.linspace(wvlMin, wvlMax, nWvl)
    else:
        wvl_c = np.array([getCentralWavelength(hdr)])
        bw = 0.0
    tr_c = np.polyval(p, wvl_c)
    disp = [list(np.zeros(nWvl)), list(np.zeros(nWvl))]
    if display:
        plt.figure(figsize=(7, 7))
        plt.plot(wvl_all*1e6, tr_all, 'b', label='Tabulated profile')
        plt.plot(wvl_f*1e6, tr_f, 'r', label='Fitted profile')
        plt.plot(wvl_c*1e6, tr_c, 'ks', label='Compressed profile')
        plt.xlabel('Wavelength [um]')
        plt.ylabel('Transmission')
        plt.title(instName + ' filter transmission')
        plt.legend()
    metadata.update({
        'source': 'local_transmission_table',
        'profile_file': str(_find_filter_profile_file(path_filter, filter_name)),
        'tabulated_transmission_curve_available': True,
    })
    result = (wvl_c, bw, tr_c, disp)
    return (*result, metadata) if return_metadata else result


def getSaturationLevel(instName):
        """
        Set to the 95% saturation threshold in DN.
        """
        if instName == 'NIRC2':
            return 12000.0
        elif instName == 'OSIRIS':
            return 20000.0
        else:
            return np.inf

def getStagePositionWFS(hdr):
    """

    """
    OBWF   = float(hdr['OBWF'])
    LSPROP = hdr['LSPROP'].upper()
    if LSPROP == 'YES':
        aoMode = 'LGS'
        AOFCLGFO    = float(hdr['AOFCLGFO'])*1e3 #mm
        zstage_defocus = abs(OBWF - AOFCLGFO)
    else:
        aoMode = 'NGS'
        AOFCNGFO    = float(hdr['AOFCNGFO'])*1e3
        zstage_defocus = abs(OBWF - AOFCNGFO)

    return zstage_defocus, aoMode

def getWFSwavelength(hdr):
    return float(hdr['GUIDWAVE'])*1e-6