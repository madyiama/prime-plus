"""P4-G Keck real-data HO POL → STRAP long-exposure tilt consistency audit.

G1: predict incident NGS tilt covariance from HO POL r0, L0 using the
    actual calibrated local Keck pupil mask (no measured sky tilt).
G2: predict same-direction atmospheric residual with frozen-flow wind
    layers and a P3 TT closed-loop transfer.
G3: compare it to the *noise-subtracted* measured STRAP residual on the
    SAME frequency band, without PSD clipping.
G4: calculate conditional alpha and its sensitivity to g, van-Dam band,
    HO r0/L0 uncertainty, wind direction and assumed mirror transfer.
G5: independently report differential NGS–science anisokinetism if
    the *pupil-frame* science offset is supplied. NEVER add it to the
    TT residual! P3 already computes dani_tt in dphi_ani.

No NIRC2 PSF fitting. No replacement of P3 science TT covariance.
Run from repo root, see docs/keck_p4g_ho_pol_tt.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p3.telemetry.ttHoPrior import (
    ARCSEC_TO_RAD, RAD_TO_MAS, circular_vk_tilt_variance,
    incident_and_residual_tilt,kernel_from_pupil,
    mas_rms,matrix_json,measured_strap_residual_atmosphere,
    scalar_consistency,
)


def _loop_model(ttloop,gain,mirror_model):
    from p3.telemetry.ttClosedLoop import (
        closed_loop_transfers,keck_ii_mirror_ragland_2016
    )
    tf=ttloop.tf
    f=np.asarray(tf.freq,dtype=float)
    if mirror_model=="unity":
        mirror=np.ones_like(f)
    elif mirror_model=="ragland_2016_keck_ii":
        mirror=keck_ii_mirror_ragland_2016(f)
    else:
        raise ValueError("Unrecognized Keck TT mirror hypothesis")
    return closed_loop_transfers(
        f,sensor=gain*np.asarray(tf.wfs),
        controller=np.asarray(tf.servo),
        plant=np.asarray(tf.lag)*mirror,
    )


def _finite_or_none(x):
    try:
        val=float(np.asarray(x).reshape(-1)[0])
    except (TypeError,ValueError,IndexError):
        return None
    return val if np.isfinite(val) else None


def _profile(trs):
    atm=trs.atm
    weights=np.asarray(atm.Cn2Weights,dtype=float).reshape(-1)
    heights=np.asarray(atm.Cn2Heights,dtype=float).reshape(-1)
    speeds=np.asarray(atm.wSpeed,dtype=float).reshape(-1)
    dirs=np.asarray(atm.wDir,dtype=float).reshape(-1)
    if not (len(weights)==len(heights)==len(speeds)==len(dirs)):
        raise ValueError(
            "MASS/DIMM Cn2 layers/winds not aligned: "
            "choose --n-layer 7 or inspect the equivalencing code"
        )
    if np.any(weights<0) or weights.sum()<=0:
        raise ValueError("Invalid MASS-DIMM layer Cn2 weights")
    return weights,heights,speeds,dirs


def _rejection_noise(sensor,ttloop,gain,band,welch,mirror):
    from p3.telemetry.ttNoise import estimate_tt_noise
    lm=_loop_model(ttloop,gain,mirror)
    covariance,diagnostics=estimate_tt_noise(
        sensor,method="van_dam",
        sample_rate_hz=ttloop.freq,
        rejection_frequency_hz=lm.frequency_hz,
        rejection_transfer=lm.sensitivity,
        high_frequency_band=band,
        nperseg=welch,
    )
    return covariance,lm,diagnostics


def _noise_covariance_angle(model,noise_cov_arcsec2,fs):
    from scipy.integrate import trapezoid
    f=model.frequency_hz
    n2=np.abs(model.physical_sensor_noise)**2
    # Extrapolate the integrator's finite DC noise transfer explicitly.
    gain_dc=float(np.real(model.sensor[0]))
    if not np.isfinite(gain_dc) or gain_dc<=0:
        raise ValueError("Invalid low-frequency STRAP sensor response")
    dc=1/gain_dc**2
    power=trapezoid(
        np.r_[dc,n2],x=np.r_[0.,f]
    )*2/float(fs)
    return np.asarray(noise_cov_arcsec2)*ARCSEC_TO_RAD**2*power


def _as_json(result):
    if isinstance(result,np.ndarray):return result.tolist()
    if isinstance(result,np.generic):return result.item()
    if isinstance(result,dict):
        return {str(k):_as_json(v) for k,v in result.items()}
    if isinstance(result,(list,tuple)):
        return [_as_json(v) for v in result]
    return result


def _sample_prior_uncertainties(r0,L0,dr0,dL0):
    """Deterministic sensitivity brackets, not posterior confidence limits."""
    cases=[("nominal",r0,L0)]
    if dr0 is not None and dr0>0:
        cases.extend([
            ("r0_minus_reported_error",max(r0-dr0,0.01),L0),
            ("r0_plus_reported_error",r0+dr0,L0),
        ])
    if dL0 is not None and dL0>0:
        cases.extend([
            ("L0_minus_reported_error",r0,max(L0-dL0,1.)),
            ("L0_plus_reported_error",r0,min(L0+dL0,300.)),
        ])
    return cases


def run_g1_to_g5(trs, *,kernel,science_offset_pupil_arcsec=None,
                 gains=(.7,1.,1.3),bands=((.65,.95),),
                 mirror_model="unity",nperseg=512,
                 wind_rotations_deg=(-45.,0.,45.)):
    """Perform the complete non-invasive G1–G5 scientific audit.

    The HO prior already came from systemDiagnosis/POL and is NOT
    estimated here from STRAP. No PSF-R object is mutated.
    """
    r0=float(trs.atm.r0_tel)
    L0=float(trs.atm.L0_tel)
    wav=float(trs.atm.wvl)
    dr0=_finite_or_none(getattr(trs.atm,"dr0_tel",None))
    dL0=_finite_or_none(getattr(trs.atm,"dL0_tel",None))
    weights,heights,speeds,dirs=_profile(trs)
    x=np.asarray(trs.tipTilt.slopes_arcsec,dtype=float)
    fs=float(trs.ttloop.freq)
    if (not 0.005<r0<3.0 or not 1.<=L0<=300. or
        x.ndim!=2 or x.shape[1]!=2 or len(x)<nperseg):
        raise ValueError("Unphysical HO prior or insufficient matched STRAP frames")
    fmin=fs/nperseg
    c1=incident_and_residual_tilt(
        kernel,r0_m=r0,L0_m=L0,wavelength_m=wav,
        weights=weights,wind_speeds_m_s=speeds,wind_dirs_deg=dirs
    )["incident_ngs_cov_rad2"]
    circular=circular_vk_tilt_variance(
        r0,L0,kernel.diameter_m,wavelength_m=wav
    )
    report={
        "status":"conditional_HO_POL_to_NGS_STRAP_TT_experiment",
        "G1_incident": {
            "r0_HO_POL_m":r0,
            "L0_HO_POL_m":L0,
            "wavelength_reference_m":wav,
            "r0_reported_sigma_m":dr0,
            "L0_reported_sigma_m":dL0,
            "r0_L0_from_LGS_HO_not_independent_of_cone_bias":True,
            "pupil_keck_mask_used":True,
            "pupil_diameter_model_m":kernel.diameter_m,
            "pupil_fft_grid":kernel.pupil_grid,
            "pupil_fft_pad":kernel.fft_pad,
            "spatial_radial_quadrature_points":len(kernel.radius),
            "incident_ngs_cov_rad2":matrix_json(c1),
            "incident_ngs_principal_rms_mas":mas_rms(c1),
            "reference_circular_VK_variance_per_axis_rad2":circular,
            "reference_circular_Kolmogorov_variance_per_axis_rad2":
                .1816*wav**2*r0**(-5/3)*kernel.diameter_m**(-1/3),
            "layer_weights_normalized":(weights/weights.sum()).tolist(),
            "layer_heights_m":heights.tolist(),
            "layer_winds_m_s":speeds.tolist(),
            "layer_wind_directions_deg":dirs.tolist(),
            "layer_wind_direction_status":"hypothesis_not_independently_measured",
            "MASS_DIMM_profile_provenance":
                getattr(trs.atm,"massdimm_diagnostics",{}),
        },
        "G2_closed_loop":{},
        "G3_strap_measured":{},
        "G4_sensitivity":{"cases":[],"prior_scenarios":[]},
        "G5_anisokinetism_separate":{},
        "assumptions":{
            "science_PSF_used_for_selection":False,
            "P3_TT_OTF_modified":False,
            "P3_dani_tt_modified":False,
            "anisokinetism_added_to_residual_TT":False,
            "LGS_HO_absolute_TT_directly_measured":False,
            "physical_optical_gain_identified":False,
            "Keck_2020_physical_mirror_plant_calibrated":False,
            "wind_directions_measured":False,
            "Noll_best_fit_tilt_equals_STRAP_quadcell_centroid_calibrated":False,
            "angular_TT_axes_registration_calibrated":False,
            "PSF_long_exposure_full_validation":False,
        },
    }
    base=None
    nominal_noise=None
    for g in gains:
        if g<=0:
            raise ValueError("Gain scenarios must be positive")
        for band in bands:
            noise,loop,noise_diag=_rejection_noise(
                x,trs.ttloop,float(g),tuple(band),nperseg,mirror_model
            )
            powers=np.abs(loop.sensitivity)**2
            output=incident_and_residual_tilt(
                kernel,r0_m=r0,L0_m=L0,wavelength_m=wav,
                weights=weights,wind_speeds_m_s=speeds,
                wind_dirs_deg=dirs,
                loop_frequency_hz=loop.frequency_hz,
                loop_sensitivity_power=powers,
                low_band_hz=fmin,high_band_hz=fs/2,
            )
            measured=measured_strap_residual_atmosphere(
                x,noise,sample_rate_hz=fs,
                loop_frequency_hz=loop.frequency_hz,
                loop_sensitivity_power=powers,
                loop_sensor_power=np.abs(loop.sensor/float(g))**2,
                gain=float(g),nperseg=nperseg,
                fmin_hz=fmin,
            )
            predicted=output["residual_ngs_atmos_band_cov_rad2"]
            estimate=measured["inferred_physical_residual_atm_band_cov_arcsec2"]
            consistency=scalar_consistency(predicted,estimate)
            cnoise=_noise_covariance_angle(loop,noise,fs)
            case={
                "assumed_gain":float(g),
                "van_dam_nyquist_band":list(band),
                "white_sensor_noise_covariance_recorded_arcsec2":
                    matrix_json(noise),
                "G2_predicted_atmospheric_residual_full_cov_rad2":
                    matrix_json(output["residual_ngs_atmos_full_cov_rad2"]),
                "G2_predicted_atmospheric_residual_matched_band_cov_rad2":
                    matrix_json(predicted),
                "G2_predicted_residual_full_including_propagated_noise_cov_rad2":
                    matrix_json(output["residual_ngs_atmos_full_cov_rad2"]+cnoise),
                "G2_predicted_atmospheric_band_principal_rms_mas":
                    mas_rms(predicted),
                "G2_predicted_propagated_noise_full_cov_rad2":
                    matrix_json(cnoise),
                "G3_measured_STRAP_atmospheric_band_cov_arcsec2":
                    matrix_json(estimate),
                "G3_measured_STRAP_band_principal_rms_mas":
                    mas_rms(estimate*ARCSEC_TO_RAD**2),
                "G3_noise_subtracted_negative_cospectral_bins":
                    measured["noise_subtracted_negative_cospectral_bins"],
                "G3_total_welch_bins":measured["welch_bins"],
                "frequency_comparison_hz":list(measured["band_hz"]),
                "G4_alpha_HO_prediction_over_STRAP_atmosphere":consistency,
                "notes":{
                    "one_gain_is_selected_from_PSF":False,
                    "noise_reestimated_for_this_g":True,
                    "only_band_matched_atmospheric_terms_compared":True,
                    "anisokinetism_included":False,
                },
            }
            report["G4_sensitivity"]["cases"].append(case)
            if abs(float(g)-1.)<1e-12 and tuple(band)==(.65,.95):
                base=(loop,output,measured,case)
                nominal_noise=noise
    if base is None:
        raise ValueError("Include gain=1 and van Dam band 0.65–0.95 as baseline")
    loop,output,measured,case=base
    report["G2_closed_loop"]={
        "reference_gain":1.,
        "mirror_hypothesis":mirror_model,
        "controller_gain":float(trs.ttloop.gain),
        "sampling_hz":fs,
        "modelled_sensor_and_delay_from_P3":True,
        "prior_layer_wind_directions_unverified":True,
        "predicted_atmospheric_full_cov_rad2":
            case["G2_predicted_atmospheric_residual_full_cov_rad2"],
        "predicted_atmospheric_matched_band_cov_rad2":
            case["G2_predicted_atmospheric_residual_matched_band_cov_rad2"],
        "predicted_full_with_sensor_noise_cov_rad2":
            case["G2_predicted_residual_full_including_propagated_noise_cov_rad2"],
    }
    report["G3_strap_measured"]={
        "method":"van_dam",
        "sensor_white_noise_cov_recorded_arcsec2":
            matrix_json(nominal_noise),
        "predicted_and_measured_matched_band_hz":
            case["frequency_comparison_hz"],
        "measured_atmospheric_residual_cov_arcsec2":
            case["G3_measured_STRAP_atmospheric_band_cov_arcsec2"],
        "noise_subtracted_negative_cospectral_bins":
            case["G3_noise_subtracted_negative_cospectral_bins"],
        "unphysical_spectral_bins_not_clipped":True,
        "no_anisokinetism_in_this_comparison":True,
    }
    # G4 sensitivity to HO uncertainties and unmeasured wind direction:
    # vary one input at a time to avoid implying a posterior probability.
    for prior_label,r0c,L0c in _sample_prior_uncertainties(
        r0,L0,dr0,dL0
    ):
        for rotation in wind_rotations_deg:
            pred=incident_and_residual_tilt(
                kernel,r0_m=r0c,L0_m=L0c,wavelength_m=wav,
                weights=weights,wind_speeds_m_s=speeds,
                wind_dirs_deg=dirs+rotation,
                loop_frequency_hz=loop.frequency_hz,
                loop_sensitivity_power=np.abs(loop.sensitivity)**2,
                low_band_hz=fmin,high_band_hz=fs/2,
            )["residual_ngs_atmos_band_cov_rad2"]
            report["G4_sensitivity"]["prior_scenarios"].append({
                "prior_case":prior_label,"r0_m":r0c,"L0_m":L0c,
                "wind_common_rotation_deg":float(rotation),
                "predicted_HO_atmos_band_trace_rad2":float(np.trace(pred)),
                "conditional_alpha_against_nominal_STRAP":
                    scalar_consistency(pred,
                        measured["inferred_physical_residual_atm_band_cov_arcsec2"]),
            })
    report["G4_sensitivity"]["verdict"]=(
        "The trace ratio alpha is a MODEL-CONSISTENCY diagnostic only; "
        "it is NOT a direct measurement of optical gain or an approved "
        "multiplicative correction of P3 jitter covariance."
    )
    if science_offset_pupil_arcsec is None:
        report["G5_anisokinetism_separate"]={
            "status":"science_NGS_separation_not_provided",
            "requires":"--science-offset-pupil-arcsec dx dy (calibrated "
                       "components in archived pupil coordinate frame)",
            "anisokinetism_already_in_P3":"dani_tt",
            "added_to_G2_or_G3":False,
        }
    else:
        res=incident_and_residual_tilt(
            kernel,r0_m=r0,L0_m=L0,wavelength_m=wav,
            weights=weights,wind_speeds_m_s=speeds,
            wind_dirs_deg=dirs,
            science_offset_pupil_arcsec=science_offset_pupil_arcsec,
            altitudes_m=heights,
        )
        c_diff=res["anisokinetism_ngs_science_cov_rad2"]
        report["G5_anisokinetism_separate"]={
            "status":"separate_differential_covariance_NOT_injected",
            "science_minus_ngs_offset_pupil_arcsec":
                list(science_offset_pupil_arcsec),
            "differential_ngs_science_tilt_cov_rad2":matrix_json(c_diff),
            "principal_rms_mas":mas_rms(c_diff),
            "anisokinetism_already_in_P3":"dani_tt",
            "included_in_G1_G2_G3_G4":False,
            "not_safe_to_add_to_G2_without_cross_covariance":True,
            "LGS_focal_anisoplanatism_HO_separately_in_P3":
                "dani_focang",
            "science_camera_pupil_axes_aligned_independently":False,
        }
    return _as_json(report)


def _make_summary_plot(report,destination):
    """Plot the HO/STRAP matched-band comparison without selecting gain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases=report["G4_sensitivity"]["cases"]
    prior=report["G4_sensitivity"]["prior_scenarios"]
    nominal=next(c for c in cases
                 if c["assumed_gain"]==1.
                 and c["van_dam_nyquist_band"]==[.65,.95])
    fig,ax=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)

    for case in cases:
        if case["van_dam_nyquist_band"]==[.65,.95]:
            g=case["assumed_gain"]
            C=np.asarray(
                case["G2_predicted_atmospheric_residual_matched_band_cov_rad2"],
                dtype=float)
            B=np.asarray(
                case["G3_measured_STRAP_atmospheric_band_cov_arcsec2"],
                dtype=float)*ARCSEC_TO_RAD**2
            for panel,matrix,label in (
                (ax[0,0],C,"HO-predicted atmospheric residual"),
                (ax[0,0],B,"STRAP-inferred atmospheric residual"),
            ):
                eig=np.linalg.eigvalsh(matrix)
                rms=(np.sqrt(np.trace(matrix)/2)*RAD_TO_MAS
                     if eig.min()>=-1e-6*max(abs(eig).max(),1e-40)
                     and np.trace(matrix)>0 else float("nan"))
                panel.scatter([g],[rms],label=label if g==1. else None)
    ax[0,0].set(xlabel="Assumed optical gain g (NOT fitted)",
                ylabel="Band atmospheric RMS/axis [mas]",
                title="G2 vs G3: same atmospheric band")
    ax[0,0].legend()
    ax[0,0].grid(alpha=.25)

    bands=sorted(set(tuple(c["van_dam_nyquist_band"]) for c in cases))
    for band in bands:
        ordered=sorted([c for c in cases
                        if tuple(c["van_dam_nyquist_band"])==band],
                       key=lambda c:c["assumed_gain"])
        gains=[c["assumed_gain"] for c in ordered]
        alphas=[c["G4_alpha_HO_prediction_over_STRAP_atmosphere"].get("alpha")
                for c in ordered]
        ax[0,1].plot(gains,[a if a is not None else float("nan")
                            for a in alphas],"o-",label=str(band))
    ax[0,1].axhline(1,color="0.65",ls="--")
    ax[0,1].set(xlabel="Assumed optical gain g (NOT fitted)",
                ylabel="Conditional alpha, not an optical gain",
                title="G4: HO / STRAP trace consistency")
    ax[0,1].legend(title="van Dam Nyquist bands")
    ax[0,1].grid(alpha=.25)

    for label in sorted(set(x["prior_case"] for x in prior)):
        ordered=sorted((x for x in prior if x["prior_case"]==label),
                       key=lambda x:x["wind_common_rotation_deg"])
        ax[1,0].plot(
            [x["wind_common_rotation_deg"] for x in ordered],
            [x["predicted_HO_atmos_band_trace_rad2"]**.5*RAD_TO_MAS/np.sqrt(2)
             for x in ordered],"o-",label=label,
        )
    ax[1,0].set(xlabel="Common wind-direction rotation [deg]",
                ylabel="Predicted HO residual RMS/axis [mas]",
                title="G4: HO-prior and unmeasured-wind sensitivity")
    ax[1,0].legend(fontsize="x-small")
    ax[1,0].grid(alpha=.25)

    g5=report["G5_anisokinetism_separate"]
    g1=report["G1_incident"]
    ax[1,1].axis("off")
    info=(
        "G1: HO POL atmospheric prior\\n"
        f"  r0 = {g1['r0_HO_POL_m']:.4g} m\\n"
        f"  L0 = {g1['L0_HO_POL_m']:.4g} m\\n"
        f"  incident NGS principal RMS = "
        f"{g1['incident_ngs_principal_rms_mas']} mas\\n\\n"
        "G5: Anisokinetism is already modeled in P3 dani_tt\\n"
        f"  status: {g5['status']}\\n"
        "  G5 is NEVER added to G2, G3, or production OTF.\\n\\n"
        "Interpretation: exploratory HO-constrained TT check only.\\n"
        "No physical optical-gain calibration or science PSF fit."
    )
    ax[1,1].text(0,1,info,transform=ax[1,1].transAxes,
                 ha="left",va="top",fontsize=10)
    fig.suptitle(
        "P4-G Keck: atmospheric residual in NGS STRAP direction "
        "(anisokinetism separate)"
    )
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits",required=True,type=Path)
    parser.add_argument("--sav",required=True,type=Path)
    parser.add_argument("--calib",required=True,type=Path)
    parser.add_argument("--output",type=Path,
                        default=Path("results/n0120_p4g_ho_tt"))
    parser.add_argument("--n-layer",type=int,choices=(1,3,7),default=7)
    parser.add_argument("--pupil-grid",type=int,default=96)
    parser.add_argument("--pupil-fft-pad",type=int,default=1024)
    parser.add_argument("--radial-points",type=int,default=450)
    parser.add_argument("--azimuth-points",type=int,default=40)
    parser.add_argument("--tt-noise-nperseg",type=int,default=512)
    parser.add_argument("--gain-scenarios",nargs="+",type=float,
                        default=[.7,1.,1.3])
    parser.add_argument("--mirror-model",
                        choices=("unity","ragland_2016_keck_ii"),
                        default="unity")
    parser.add_argument("--wind-rotations-deg",nargs="+",type=float,
                        default=[-45.,0.,45.])
    parser.add_argument("--science-offset-pupil-arcsec",nargs=2,type=float,
                        default=None,
                        metavar=("DX","DY"),
                        help="NGS→science offset in ARCHIVED PUPIL axes, "
                             "not raw RA/Dec or NIRC2 pixels. G5 ONLY.")
    args=parser.parse_args()
    for path in (args.fits,args.sav,args.calib):
        if not path.exists():parser.error("Missing Keck input: "+str(path))
    if (args.pupil_grid<32 or args.pupil_fft_pad<2*args.pupil_grid
        or args.radial_points<64 or args.azimuth_points<16
        or args.tt_noise_nperseg<128):
        parser.error("Invalid pupil sampling, quadrature, or Welch window")
    if any(g<=0 for g in args.gain_scenarios) or not any(
        abs(g-1.)<1e-12 for g in args.gain_scenarios
    ):
        parser.error("gain-scenarios must include the nominal Keck gain 1")
    from astropy.io import fits
    from p3.telemetry.keckPupilAudit import (
        pupil_orientation_report,read_mask,binary_pupil_on_grid
    )
    from p3.telemetry.telemetryKeck import telemetryKeck
    from p3.telemetry.systemDiagnosis import systemDiagnosis
    hdr=fits.getheader(args.fits,ignore_missing_end=True)
    pupil_report=pupil_orientation_report(hdr)
    _,raw_mask=read_mask(args.calib,pupil_report)
    # For scalar trace comparison, orientation is irrelevant for an
    # isotropic VK prior but still matters for tensor/G5 cross-terms.
    mask=binary_pupil_on_grid(
        raw_mask,args.pupil_grid,rotation_deg=0.,parity="native"
    )
    kernel=kernel_from_pupil(
        mask,10.5,pad=args.pupil_fft_pad,
        n_rad=args.radial_points,n_az=args.azimuth_points
    )
    print("[P4-G] G1 pupil Fourier kernel ready",flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    trs=telemetryKeck(
        str(args.sav.resolve()),str(args.fits.resolve()),
        str(args.calib.resolve()),
        path_save=str(args.output.resolve()),cam_fov=128,
        nLayer=args.n_layer,addNCPA=False,
    )
    if str(trs.aoMode).upper()!="LGS":
        parser.error("P4-G requires a Keck LGS acquisition with STRAP B")
    sd=systemDiagnosis(
        trs,tt_noise_method="van_dam",
        tt_noise_high_band=(.65,.95),
        tt_noise_nperseg=args.tt_noise_nperseg,
        tt_mirror_model="unity",
    )
    print(f"[P4-G] HO POL prior r0={sd.trs.atm.r0_tel:.4f} m, "
          f"L0={sd.trs.atm.L0_tel:.2f} m; "
          f"STRAP fs={sd.trs.ttloop.freq:.3f} Hz",flush=True)
    report=run_g1_to_g5(
        sd.trs,kernel=kernel,
        science_offset_pupil_arcsec=args.science_offset_pupil_arcsec,
        gains=tuple(args.gain_scenarios),
        bands=((.65,.95),(.55,.85),(.75,.95)),
        mirror_model=args.mirror_model,
        nperseg=args.tt_noise_nperseg,
        wind_rotations_deg=tuple(args.wind_rotations_deg),
    )
    report["input"]={
        "fits":str(args.fits.resolve()),
        "sav":str(args.sav.resolve()),
        "mask_name":pupil_report["pupil_mask_name"],
        "mask_FITS_reference_zero_orientation_unverified":True,
        "profile_layers":args.n_layer,
        "noise_method":"van_dam",
    }
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"p4g_ho_to_tt.json"
    path.write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"
    )
    plot_path=args.output/"p4g_ho_to_tt.png"
    _make_summary_plot(report,plot_path)
    g1=report["G1_incident"]
    g2=report["G2_closed_loop"]
    g3=report["G3_strap_measured"]
    base=next(c for c in report["G4_sensitivity"]["cases"] if
              c["assumed_gain"]==1. and
              c["van_dam_nyquist_band"]==[.65,.95])
    print("[G1] Open-loop NGS principal RMS (mas):",
          g1["incident_ngs_principal_rms_mas"])
    print("[G2] HO-predicted band-limited atmospheric residual RMS (mas):",
          base["G2_predicted_atmospheric_band_principal_rms_mas"])
    print("[G3] STRAP atmospheric band RMS (mas):",
          base["G3_measured_STRAP_band_principal_rms_mas"])
    print("[G3] Negative noise-subtracted cospectral bins:",
          g3["noise_subtracted_negative_cospectral_bins"])
    print("[G4] Model trace-consistency factor alpha (NOT gain):",
          base["G4_alpha_HO_prediction_over_STRAP_atmosphere"])
    print("[G5] Anisokinetism separate:",
          report["G5_anisokinetism_separate"]["status"])
    print("[P4-G] No production jitter, OTF, or anisokinetism modified.")
    print("[P4-G] Saved:",path)
    print("[P4-G] Figure:",plot_path)


if __name__=="__main__":
    main()
