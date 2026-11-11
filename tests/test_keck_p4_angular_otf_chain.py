"""P4 numerical science contracts: Keck DTTCENTROIDS (arcsec) -> TT OTF.

None of these tests relies on an APD count-to-electron calibration or
the observed n0013 binary PSF. They isolate numerical unit-chain errors.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from p3.aoSystem.tipTiltGeometry import (
    ARCSEC_TO_RAD, angular_jitter_structure_function,
    opd_covariance_to_angular, zernike_tilt_opd_per_arcsec,
)
from p3.aoSystem.ttAngular import (
    raw_centroid_covariance_rad2,
    noll_opd_noise_to_angular,
    direct_angular_tt_covariance,
    assert_equivalent_angular_covariances,
    angular_otf_frequencies,
    gaussian_tt_otf,
)
from p3.aoSystem.ttFrame import sensor_frame_frequencies
from p3.psfr.psfR import psfR


def _synthetic_keck_centroids(n=1000):
    rng = np.random.default_rng(2020)
    mix = np.array([[0.035, 0.012], [0.0, 0.024]])
    a = rng.standard_normal((n, 2)) @ mix
    a += np.array([0.21, -0.15])  # offset must not create jitter
    return a


def _freq(n=128, D=10.5, wavelength=2.055501e-6, pixel_mas=9.942):
    # Construct the real P3 Fourier grid equations independently.
    k_ref = 2
    k_float = float(k_ref)
    N = n*k_ref
    rows, cols = np.indices((N,N), dtype=float)
    U = (rows-N/2)*2/N
    V = (cols-N/2)*2/N
    pixel_rad = pixel_mas*1e-3*ARCSEC_TO_RAD
    samp_ref = k_float*wavelength/(D*pixel_rad)
    return NS(
        U_=U, V_=V, sampRef=samp_ref, wvlRef=wavelength,
        kRef_=k_ref, kRef_float=k_float, nPix=n, nOtf=N,
    )


def test_raw_arcsec_to_radians_covariance_matches_independent_definition():
    raw = _synthetic_keck_centroids()
    expected = np.cov(raw.T, bias=True)*ARCSEC_TO_RAD**2
    np.testing.assert_allclose(raw_centroid_covariance_rad2(raw), expected,
                               rtol=2e-13, atol=1e-26)
    offset = raw+np.array([100, -47])
    np.testing.assert_allclose(raw_centroid_covariance_rad2(offset), expected,
                               rtol=2e-12, atol=1e-26)


def test_noll_roundtrip_has_no_physical_factor_sixteen():
    raw = _synthetic_keck_centroids()
    c_arcsec2 = np.cov(raw.T, bias=True)
    for D in (8, 10.5, 11.25):
        g = zernike_tilt_opd_per_arcsec(D)
        opd_cov = c_arcsec2*g**2
        np.testing.assert_allclose(
            opd_covariance_to_angular(opd_cov, D),
            c_arcsec2*ARCSEC_TO_RAD**2, rtol=1e-13, atol=1e-26,
        )
        np.testing.assert_allclose(
            noll_opd_noise_to_angular(opd_cov, D),
            c_arcsec2*ARCSEC_TO_RAD**2, rtol=1e-13, atol=1e-26,
        )
        assert_equivalent_angular_covariances(
            opd_cov, raw_centroid_covariance_rad2(raw), D,
        )


def test_direct_legacy_covariance_matches_original_noll_opd_algebra():
    raw = _synthetic_keck_centroids()
    D = 10.5
    g = zernike_tilt_opd_per_arcsec(D)
    noise_arcsec2 = np.diag([0.00012, 0.00015])
    noise_opd = noise_arcsec2*g**2
    pn = 0.18
    opd = (raw-raw.mean(axis=0))*g
    observed_opd = opd.T@opd/len(opd)
    # Choose this synthetic fixture safely within PSD region, so the
    # decomposition can be compared without any projection.
    direct, report = direct_angular_tt_covariance(
        raw, noise_opd, D, method="legacy_psfr", pn=pn,
    )
    reference = opd_covariance_to_angular(
        observed_opd-noise_opd, D,
    )
    np.testing.assert_allclose(direct, reference, rtol=2e-12, atol=1e-26)
    assert report["method"] == "legacy_psfr"
    assert_equivalent_angular_covariances(observed_opd-noise_opd, direct, D)


def test_direct_legacy_detects_unit_mismatch_and_invalid_pn():
    raw = _synthetic_keck_centroids()
    g = zernike_tilt_opd_per_arcsec(10.5)
    m = raw_centroid_covariance_rad2(raw)
    noise = np.diag([0.00012, 0.00015])*g*g
    with pytest.raises(ValueError, match="Unknown STRAP"):
        direct_angular_tt_covariance(raw, noise, 10.5, method="unknown")
    with pytest.raises(ValueError, match="pn"):
        direct_angular_tt_covariance(raw, noise, 10.5, method="legacy_psfr", pn=-0.5)
    with pytest.raises(ValueError, match="unit-chain mismatch"):
        assert_equivalent_angular_covariances(
            m*(10.5/4)**2*1.02, m, 10.5,
        )


def test_angular_otf_grid_matches_independent_detector_fft_frequencies():
    pixel_mas = 9.942
    f = _freq(pixel_mas=pixel_mas)
    row, col = angular_otf_frequencies(
        f, 10.5, f.wvlRef, pixel_scale_mas=pixel_mas,
    )
    N = f.nOtf
    pixel_rad = pixel_mas*1e-3*ARCSEC_TO_RAD
    idx = np.arange(N)-N//2
    np.testing.assert_allclose(
        row[:, N//2], idx/(f.nPix*pixel_rad),
        rtol=1e-14, atol=1e-8,
    )
    np.testing.assert_allclose(
        col[N//2, :], idx/(f.nPix*pixel_rad),
        rtol=1e-14, atol=1e-8,
    )
    assert row[N//2, N//2] == 0
    assert col[N//2, N//2] == 0
    with pytest.raises(ValueError, match="disagrees"):
        angular_otf_frequencies(f, 10.5, f.wvlRef, pixel_scale_mas=10.2)


def test_angular_otf_grid_checks_expanded_fractional_oversampling():
    pixel_mas = 9.942
    f = _freq(pixel_mas=pixel_mas)
    f.kRef_=3
    f.kRef_float=2.4
    f.nOtf=f.nPix*f.kRef_
    rows, cols=np.indices((f.nOtf, f.nOtf), dtype=float)
    f.U_=(rows-f.nOtf/2)*2/f.nOtf
    f.V_=(cols-f.nOtf/2)*2/f.nOtf
    f.sampRef=f.kRef_float*f.wvlRef/(10.5*pixel_mas*1e-3*ARCSEC_TO_RAD)
    row,col=angular_otf_frequencies(
        f, 10.5, f.wvlRef, pixel_scale_mas=pixel_mas,
    )
    step=f.kRef_float/f.kRef_/(f.nPix*pixel_mas*1e-3*ARCSEC_TO_RAD)
    assert row[f.nOtf//2+1,f.nOtf//2] == pytest.approx(step,rel=1e-13)
    assert col[f.nOtf//2,f.nOtf//2+1] == pytest.approx(step,rel=1e-13)


def test_gaussian_otf_equals_independent_fourier_gaussian_and_has_flux_one():
    n = 256
    pixel_mas = 10.0
    rad_px = pixel_mas*1e-3*ARCSEC_TO_RAD
    idx=np.arange(n)-n//2
    nu_row,nu_col=np.meshgrid(idx/(n*rad_px),idx/(n*rad_px),indexing="ij")
    sigma_row, sigma_col = 2.6*rad_px, 4.3*rad_px
    cov=np.diag([sigma_row**2,sigma_col**2])
    dphi, kernel=gaussian_tt_otf(cov,nu_row,nu_col)
    assert kernel[n//2,n//2] == pytest.approx(1.0,abs=1e-14)
    assert kernel.min() >= 0 and kernel.max() <= 1
    np.testing.assert_allclose(np.exp(-dphi/2),kernel,rtol=1e-14,atol=1e-14)
    # A full-frequency Gaussian OTF transforms into a Gaussian PSF with
    # the same specified angular covariance. No pupil/Zernike involved.
    psf=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kernel))).real
    assert psf.min() >= -1e-14
    psf/=psf.sum()
    yy,xx=np.indices((n,n))
    center=n//2
    reconstructed_row=np.sum(psf*(yy-center)**2)
    reconstructed_col=np.sum(psf*(xx-center)**2)
    assert reconstructed_row == pytest.approx(2.6**2, rel=1e-3)
    assert reconstructed_col == pytest.approx(4.3**2, rel=1e-3)
    assert psf.sum() == pytest.approx(1,abs=1e-13)


def test_zero_tt_gaussian_kernel_is_exactly_one():
    f=_freq(n=32)
    row,col=angular_otf_frequencies(
        f, 10.5, f.wvlRef, pixel_scale_mas=9.942,
    )
    dphi,otf=gaussian_tt_otf(np.zeros((2,2)),row,col)
    np.testing.assert_array_equal(dphi,np.zeros_like(row))
    np.testing.assert_array_equal(otf,np.ones_like(row))


def test_psfr_direct_angular_explicit_kernel_agrees_with_prior_noll_formula():
    D = 10.5
    wl = 2.055501e-6
    fake=psfR.__new__(psfR)
    fake.ao=NS(tel=NS(D=D),cam=NS(psInMas=9.942))
    fake.trs=NS(tel=NS(pupilAngle=46.77))
    fake.freq=_freq(n=32,D=D,wavelength=wl)
    fake.tt_kernel_dphi_multiplier_vs_noll=1
    fake.tt_axis_angle_deg=None
    fake.tt_axis_parity="native"
    C_arcsec=np.array([[9e-4,2e-4],[2e-4,4e-4]])
    c_opd=C_arcsec*zernike_tilt_opd_per_arcsec(D)**2
    new=fake.jitter_phase_structure_function(Ctt=c_opd)
    nu0=fake.freq.sampRef/2
    us,vs=sensor_frame_frequencies(
        nu0*fake.freq.U_, nu0*fake.freq.V_, 46.77,
    )
    old=angular_jitter_structure_function(
        opd_covariance_to_angular(c_opd,D),D,wl,us,vs,
    )
    np.testing.assert_allclose(new,old,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(fake.otf_tt,np.exp(-old/2),rtol=1e-13)
    fake.tt_kernel_dphi_multiplier_vs_noll=1/16
    legacy=fake.jitter_phase_structure_function(Ctt=c_opd)
    np.testing.assert_allclose(legacy,old/16,rtol=1e-13,atol=1e-13)


def test_strap_cam_parity_changes_orientation_not_variance():
    f=_freq(n=64)
    cov=np.diag([(0.02*ARCSEC_TO_RAD)**2,(0.05*ARCSEC_TO_RAD)**2])
    row,col=angular_otf_frequencies(f,10.5,f.wvlRef)
    def get(angle,parity):
        a,b=sensor_frame_frequencies(row,col,angle,parity)
        return gaussian_tt_otf(cov,a,b)[1]
    base=get(0,"native")
    rotated=get(90,"native")
    assert base[f.nOtf//2+3,f.nOtf//2] != pytest.approx(
        rotated[f.nOtf//2+3,f.nOtf//2],rel=1e-3
    )
    # Both are equal along circularly averaged power only when the grid
    # is rotation symmetric; 90-degree rotation swaps the covariance axes.
    np.testing.assert_allclose(base,rotated.T,rtol=2e-13,atol=2e-13)


def test_spectral_tt_covariance_direct_angles_equals_opd_intermediate():
    """An LTI/Welch covariance is homogeneous in squared signal units."""
    from p3.telemetry.ttClosedLoop import (
        physical_tt_covariance_from_telemetry,
        closed_loop_transfers,
    )
    N = 4096
    fs = 600.0
    f = np.linspace(0, fs/2, 257)
    W = np.ones(len(f))
    K = np.full(len(f), 0.04)
    P = np.ones(len(f))
    model = closed_loop_transfers(f, sensor=W, controller=K, plant=P)
    tf = NS(freq=f, wfs=W, servo=K, lag=P,
            rtf=model.sensitivity)
    loop = NS(tf=tf, freq=fs)
    raw = _synthetic_keck_centroids(N)
    D = 10.5
    gain = zernike_tilt_opd_per_arcsec(D)
    samples_opd = (raw-raw.mean(axis=0))*gain
    noise_opd = np.diag([1e-6, 2e-6])*gain**2
    _,_,original,_ = physical_tt_covariance_from_telemetry(
        samples_opd, noise_opd, loop, mirror_model="unity", nperseg=512,
    )
    direct, diag = direct_angular_tt_covariance(
        raw, noise_opd, D, method="spectral_closed_loop",
        ttloop=loop, mirror_model="unity", nperseg=512,
    )
    assert diag["method"] == "spectral_closed_loop"
    assert_equivalent_angular_covariances(original, direct, D, rtol=2e-10)


def test_psfr_actual_lgs_centroid_replay_marks_physical_validation_separately():
    """The full legacy P4 function takes raw arcsec and checks its Noll path."""
    raw = _synthetic_keck_centroids(1024)
    D = 10.5
    gain = zernike_tilt_opd_per_arcsec(D)
    opd = (raw-raw.mean(axis=0))*gain
    fake = psfR.__new__(psfR)
    fake.ao = NS(tel=NS(D=D), cam=NS(psInMas=9.942))
    fake.freq = _freq(n=32, D=D)
    fake.trs = NS(
        aoMode="LGS", tel=NS(D=D, pupilAngle=46.76723),
        tipTilt=NS(slopes_arcsec=raw, slopes=opd,
                   Cn_tt=np.diag([1e-6, 2e-6])*gain**2,
                   nExp=len(raw), command_units_verified=False),
        ttloop=NS(tf=NS(pn=0.12)),
    )
    fake.tt_covariance_mode="legacy_psfr"
    fake.tt_mirror_model="unity"
    fake.tt_welch_nperseg=512
    fake.tt_axis_angle_deg=None
    fake.tt_axis_parity="native"
    fake.tt_kernel_dphi_multiplier_vs_noll=1.0
    sf=fake.jitter_phase_structure_function()
    assert sf.shape == (fake.freq.nOtf, fake.freq.nOtf)
    assert fake.tiptilt_diagnostics["tt_source_angle_replay"][
        "tt_centroid_to_otf_units_verified"
    ] is True
    assert fake.tiptilt_diagnostics["tt_source_angle_replay"][
        "angular_route_relative_error"
    ] < 2e-9
    assert fake.tiptilt_diagnostics["tt_otf_zero_lag"] == pytest.approx(1.0)
    assert fake.tiptilt_diagnostics["tt_otf_max"] <= 1+1e-14
    assert fake.tiptilt_diagnostics["tt_source_angle_replay"][
        "calibration_of_instrument_gain_verified"
    ] is False


def test_psfr_rejects_corrupted_centroid_opd_unit_chain():
    raw = _synthetic_keck_centroids(512)
    D = 10.5
    gain = zernike_tilt_opd_per_arcsec(D)
    fake = psfR.__new__(psfR)
    fake.ao = NS(tel=NS(D=D),cam=NS(psInMas=9.942))
    fake.freq = _freq(n=32,D=D)
    fake.trs = NS(
        aoMode="LGS", tel=NS(D=D,pupilAngle=0.),
        tipTilt=NS(slopes_arcsec=raw,
                   slopes=(raw-raw.mean(axis=0))*gain*1.01,
                   Cn_tt=np.zeros((2,2)), nExp=len(raw)),
        ttloop=NS(tf=NS(pn=0.0)),
    )
    fake.tt_covariance_mode="legacy_psfr"
    fake.tt_mirror_model="unity"
    fake.tt_welch_nperseg=512
    fake.tt_axis_angle_deg=None
    fake.tt_axis_parity="native"
    fake.tt_kernel_dphi_multiplier_vs_noll=1.0
    with pytest.raises(ValueError,match="angular-to-OPD conversion inconsistent"):
        fake.jitter_phase_structure_function()


def test_forward_structure_function_applies_tt_once_at_science_wavelength():
    """P4 dphi_tt is already in science-wavelength radians^2."""
    fake=psfR.__new__(psfR)
    n=32
    yy,xx=np.indices((n,n),dtype=float)
    fake.dphi_ao=np.zeros((n,n))
    fake.dphi_fit=np.zeros((n,n))
    fake.dphi_alias=np.zeros((n,n))
    fake.dphi_tt=(yy-n//2)**2*0.01+(xx-n//2)**2*0.025
    fake.freq=NS(isAniso=False)
    fake.ao=NS(src=NS(nSrc=1))
    for gain in (0.,0.4,1.):
        sf=fake.phase_structure_function(0.1,[1.0,gain])
        np.testing.assert_allclose(
            sf[:,:,0],gain*fake.dphi_tt,rtol=1e-13,atol=1e-13
        )
        np.testing.assert_allclose(
            np.exp(-0.5*sf[:,:,0]),
            np.exp(-0.5*gain*fake.dphi_tt),rtol=1e-13
        )


def test_pixel_otf_uses_same_detector_frequency_grid_as_direct_tt():
    D=10.5
    f=_freq(n=64,D=D)
    fake=psfR.__new__(psfR)
    fake.ao=NS(tel=NS(D=D),cam=NS(psInMas=9.942))
    fake.freq=f
    nu_row,nu_col=angular_otf_frequencies(
        f,D,f.wvlRef,pixel_scale_mas=9.942
    )
    pixel_rad=9.942*1e-3*ARCSEC_TO_RAD
    expected=np.sinc(nu_row*pixel_rad)*np.sinc(nu_col*pixel_rad)
    np.testing.assert_allclose(
        fake.pixel_optical_transfer_function(),expected,
        rtol=1e-13,atol=1e-13
    )


def test_independent_native_arcsec_noise_joins_source_angles_without_noll():
    raw=_synthetic_keck_centroids(1200)
    D=10.5
    gain=zernike_tilt_opd_per_arcsec(D)
    native_noise=np.array([[0.00012,0.00001],[0.00001,0.00015]])
    opd_noise=native_noise*gain**2
    direct,metadata=direct_angular_tt_covariance(
        raw,opd_noise,D,method="legacy_psfr",pn=0.14,
        native_noise_arcsec2=native_noise,
    )
    fallback,_=direct_angular_tt_covariance(
        raw,opd_noise,D,method="legacy_psfr",pn=0.14,
    )
    np.testing.assert_allclose(direct,fallback,rtol=1e-13,atol=1e-26)
    assert metadata["sensor_noise_angular_source"] == (
        "noise_estimator_directly_on_DTTCENTROIDS_arcsec"
    )
    with pytest.raises(ValueError,match="angular noise and Noll-OPD"):
        direct_angular_tt_covariance(
            raw,opd_noise,D,method="legacy_psfr",pn=0.14,
            native_noise_arcsec2=1.03*native_noise,
        )


def test_native_arcsec_and_opd_welch_noise_estimation_are_scale_identical():
    from p3.telemetry.ttNoise import estimate_tt_noise
    raw=_synthetic_keck_centroids(4096)
    D=10.5
    gain=zernike_tilt_opd_per_arcsec(D)
    fs=500.0
    f=np.linspace(0,fs/2,257)
    S=np.ones_like(f)
    kwargs=dict(
        method="van_dam",sample_rate_hz=fs,
        rejection_frequency_hz=f,rejection_transfer=S,
        high_frequency_band=(0.65,0.95),nperseg=512,
    )
    native,diagnostics=estimate_tt_noise(raw,**kwargs)
    opd,diagnostics_opd=estimate_tt_noise(
        (raw-raw.mean(axis=0))*gain,**kwargs
    )
    np.testing.assert_allclose(native*gain**2,opd,rtol=1e-12,atol=1e-26)
    assert diagnostics["fit_bin_count"] == diagnostics_opd["fit_bin_count"]
