#!/usr/bin/env python3
"""P4: four controlled STRAP jitter ablations on the SAME Keck LGS acquisition.

Run from repository root:
  uv run python scripts/keck_p4_tt_benchmark.py \
    --fits /path/n0013.fits --sav /path/n0013_LGS_trs.sav \
    --calib p3/aoSystem/data/KECK_CALIBRATION \
    --output results/P4_n0013 --n-layer 7

Outputs a per-variant smoke folder, a consolidated JSON and a model-only
four-panel figure. All science FITS remain local. No image fitting.
Do not rank the variants by agreement with the observed binary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


VARIANTS = {
    "A0_no_tt": ("legacy_psfr", "noll_angular", True),
    "A1_legacy_physical": ("legacy_psfr", "noll_angular", False),
    "A2_legacy_no16_diagnostic": ("legacy_psfr", "legacy_unscaled_opd", False),
    "A3_spectral_physical": ("spectral_closed_loop", "noll_angular", False),
}


def _model_psf(path):
    im = np.load(path, allow_pickle=False)
    if im.ndim == 3:
        im = np.sum(im, axis=-1)
    if im.ndim != 2:
        raise ValueError(f"Expected 2D model PSF, got {im.shape}")
    if not np.all(np.isfinite(im)) or np.min(im) < -1e-12:
        raise ValueError("PSF contains invalid intensity")
    im = np.maximum(im, 0)
    if im.sum() <= 0:
        raise ValueError("PSF total energy is zero")
    return im/im.sum()


def summarize_existing_outputs(root):
    """Read smoke results, verify invariants, produce machine-readable table.

    Can be tested on synthetic fixtures without running Keck telemetry.
    Does not decide which jitter covariance should be considered physical.
    """
    from p3.aoSystem.ttFrame import strap_noise_spectral_balance

    out = {}
    source_noise_a1 = None
    for label in VARIANTS:
        folder = root/label
        metadata = json.loads((folder/"metadata.json").read_text())
        diagnostic = json.loads((folder/"tiptilt_diagnostics.json").read_text())
        metrics = json.loads((folder/"p4_tt_metrics.json").read_text())
        noise = json.loads((folder/"tt_noise_diagnostics.json").read_text())
        angular_file = folder/"p4_tt_angular_chain.json"
        chain = json.loads(angular_file.read_text()) if angular_file.exists() else None
        if label != "A0_no_tt" and chain is not None:
            if (chain.get("status")
                    != "numerical_chain_verified_instrument_uncalibrated"):
                raise ValueError(
                    f"Angular OTF numerical audit not validated for {label}"
                )
        if label == "A1_legacy_physical":
            source_noise_a1 = noise
        out[label] = {
            "input_fits": metadata["input_fits"],
            "input_sav": metadata["input_sav"],
            "revision": metadata["code_revision"],
            "ao_mode": metadata["ao_mode"],
            "n_layer": metadata["forward_parameters_diagnostics"]["atmosphere_layers"],
            "r0_model_m": metadata["forward_parameters_diagnostics"]["r0_model_m"],
            "L0_tel_m": metadata["atmosphere_reconstruction_audit"]["fitted_L0_m"],
            "layer_weights": metadata["atmosphere_reconstruction_audit"]["atmosphere_layer_weights"],
            "layer_heights_m": metadata["atmosphere_reconstruction_audit"]["atmosphere_layer_heights_m"],
            "forward_layer_strengths": metadata["forward_parameters_diagnostics"]["layer_strengths_m_to_minus_5_over_3"],
            "wavelength_m": metadata["psf_reference_wavelength_m"],
            "pupil_resampling": metadata["p3_pupil_resampling"],
            "pupil_parity": metadata["p3_pupil_parity"],
            "pupil_rotation": metadata["p3_array_rotation_applied_deg"],
            "tt_axis_angle_deg": metadata["p4_tt_axis_angle_deg"],
            "tt_axis_parity": metadata["p4_tt_axis_parity"],
            "tt_axis_legacy_coupling": metadata["p4_tt_axis_uses_legacy_pupil"],
            "tt_noise_method": metadata["tt_noise_method"],
            "tt_noise_model": metadata["tt_noise_rejection_mirror_model"],
            "tt_noise_covariance_m2": noise.get("noise_covariance_m2"),
            "tt_model": metadata["tt_jitter_covariance_method"],
            "tt_kernel": metadata["tt_kernel_convention"],
            "tt_enabled": metadata["tt_jitter_otf_enabled"],
            "psf": metrics["psf"],
            "camera_jitter": metrics["camera_angular_covariance"],
            "tt_angular_chain": chain,
            "tt_angular_chain_verified": bool(
                chain is not None and
                chain.get("status") ==
                "numerical_chain_verified_instrument_uncalibrated"
            ),
            "tt_diagnostics": {
                key: diagnostic.get(key) for key in (
                    "raw_trace_m2", "noise_trace_m2",
                    "model_jitter_trace_m2", "physical_total_trace_m2",
                    "physical_atmos_trace_m2", "physical_noise_trace_m2",
                    "noise_transfer_factor", "gain_sensor_to_measured_noise",
                    "gain_sensor_to_physical_noise", "noise_transfer_mismatch",
                    "negative_variance_projected",
                    "tt_axis_independently_calibrated",
                )
            },
        }

    baseline = out["A0_no_tt"]
    invariants = [
        "input_fits", "input_sav", "revision", "ao_mode",
        "n_layer", "r0_model_m", "L0_tel_m", "layer_weights",
        "layer_heights_m", "forward_layer_strengths", "wavelength_m",
        "pupil_resampling", "pupil_parity", "pupil_rotation",
        "tt_axis_angle_deg", "tt_axis_parity", "tt_axis_legacy_coupling",
        "tt_noise_method", "tt_noise_model",
    ]
    for key in invariants:
        if any(row[key] != baseline[key] for row in out.values()):
            raise ValueError(f"P4 does not isolate jitter: inconsistent {key}")
    if any(row["ao_mode"].upper() != "LGS" for row in out.values()):
        raise ValueError("STRAP P4 experiment requires LGS observation")
    if not (out["A0_no_tt"]["tt_enabled"] is False and
            all(out[k]["tt_enabled"] for k in VARIANTS if k != "A0_no_tt")):
        raise ValueError("P4 TT enable flags do not match variant contracts")
    for label, (method, kernel, _disabled) in VARIANTS.items():
        if out[label]["tt_model"] != method or out[label]["tt_kernel"] != kernel:
            raise ValueError(f"Wrong TT model or kernel in {label}")
        if out[label]["tt_noise_covariance_m2"] is None:
            raise ValueError(f"Missing identified STRAP noise covariance: {label}")
    c0 = np.asarray(out["A0_no_tt"]["tt_noise_covariance_m2"], dtype=float)
    for label in VARIANTS:
        c = np.asarray(out[label]["tt_noise_covariance_m2"], dtype=float)
        if not np.allclose(c, c0, rtol=1e-7, atol=1e-23):
            raise ValueError(f"TT noise floor not held fixed: {label}")
    # A1/A2 change ONLY the TT OTF exponent: their underlying physical
    # covariance and angular RMS must be invariant to 1/16 kernel scaling.
    a1, a2 = out["A1_legacy_physical"], out["A2_legacy_no16_diagnostic"]
    tr1 = a1["camera_jitter"]["trace_rad2"]
    tr2 = a2["camera_jitter"]["trace_rad2"]
    if not np.isclose(tr1, tr2, rtol=1e-10, atol=1e-26):
        raise ValueError("Legacy kernel ablation changed covariance")
    for label, row in out.items():
        # The A2 comparison keeps the same inferred *physical covariance*
        # but deliberately applies 1/16 of the Noll structure function.
        # Since a Gaussian OTF exponent scales as sigma_theta^2, its
        # actual displayed kernel corresponds to sigma_theta/4.
        multiplier = (
            0.0625 if row["tt_kernel"] == "legacy_unscaled_opd" else 1.0
        )
        inferred = row["camera_jitter"]
        row["tt_kernel_dphi_multiplier_vs_noll"] = multiplier
        row["kernel_effective_rms_axis_mas"] = (
            None if inferred is None
            else float(inferred["rms_per_axis_equivalent_mas"] * np.sqrt(multiplier))
        )
        row["kernel_is_physical_noll_convention"] = (
            row["tt_kernel"] == "noll_angular"
        )
    return {
        "status": "model_only_ablation_complete",
        "all_same_observation_and_forward_atmosphere": True,
        "STRAP_noise_identical_all_four": True,
        "A1_A2_same_physical_covariance": True,
        "tt_angular_replay_verified_for_all_enabled_variants": all(
            out[label]["tt_angular_chain_verified"]
            for label in VARIANTS if label != "A0_no_tt"
        ),
        "previous_output_reconstruction_needed_for_angular_audit": not all(
            out[label]["tt_angular_chain_verified"]
            for label in VARIANTS if label != "A0_no_tt"
        ),
        "tt_noise_spectral_cancellation": strap_noise_spectral_balance(
            source_noise_a1
        ),
        "source_fitting_performed": False,
        "best_fit_model_selected": False,
        "instrument_gain_calibrated": False,
        "frame_angle_independently_calibrated": False,
        "reference": "main P1/P2/P3 provisional physics",
        "results": out,
    }


def _figures(root, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    short_title = {
        "A0_no_tt": "A0 - sans jitter",
        "A1_legacy_physical": "A1 - TT legacy, noyau Noll",
        "A2_legacy_no16_diagnostic": "A2 - noyau /16 (diagnostic)",
        "A3_spectral_physical": "A3 - TT spectral, noyau Noll",
    }
    for ax, (label, row) in zip(axes.ravel(), rows.items()):
        psf = _model_psf(root/label/"telemetry_only_psf.npy")
        c = psf.shape[0]//2
        crop = psf[c-24:c+24, c-24:c+24]
        ax.imshow(np.log10(np.maximum(crop, 1e-7)),
                  vmin=-7, vmax=-1.4, origin="lower", cmap="magma")
        peak = row["psf"]["normalized_peak"]
        rms_eff = row["kernel_effective_rms_axis_mas"]
        rms = "off" if rms_eff is None else f"{rms_eff:.1f} mas"
        ax.set_title(
            f"{short_title[label]}" + chr(10)
            + f"Noyau effectif {rms} RMS/axe | pic {peak:.4g}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("P4 STRAP: same-acquisition telemetry-only PSFs; no fit")
    fig.savefig(root/"p4_tt_ablation_model_psfs.png", dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--sav", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-layer", type=int, choices=(1, 3, 7), default=7)
    p.add_argument("--pupil-resampling", choices=("legacy", "physical_binary"),
                   default="physical_binary")
    p.add_argument("--pupil-parity", choices=("native", "flip_x", "flip_y"),
                   default="native")
    p.add_argument("--pupil-rotation",
                   choices=("header", "as_archived", "reverse_sign", "plus_90", "minus_90"),
                   default="header")
    p.add_argument("--tt-axis-angle-deg", type=float, default=None,
                   help="Experimental independent STRAP camera-axis rotation")
    p.add_argument("--tt-axis-parity", choices=("native", "flip_x", "flip_y"),
                   default="native")
    p.add_argument("--noise-band", nargs=2, type=float, default=(0.65, 0.95))
    p.add_argument("--welch-nperseg", type=int, default=512)
    p.add_argument("--summarize-only", action="store_true",
                   help="Re-read four existing output folders without rerunning Keck")
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        script = Path(__file__).with_name("keck_n0013_smoke.py")
        for label, (method, kernel, disabled) in VARIANTS.items():
            cmd = [
                sys.executable, str(script),
                "--fits", str(args.fits.resolve()),
                "--sav", str(args.sav.resolve()),
                "--calib", str(args.calib.resolve()),
                "--output", str(root/label),
                "--n-layer", str(args.n_layer),
                "--p3-pupil-resampling", args.pupil_resampling,
                "--p3-pupil-parity", args.pupil_parity,
                "--p3-pupil-rotation", args.pupil_rotation,
                "--tt-noise-method", "van_dam",
                "--tt-noise-band", *[str(x) for x in args.noise_band],
                "--tt-welch-nperseg", str(args.welch_nperseg),
                "--tt-jitter-welch-nperseg", str(args.welch_nperseg),
                "--tt-mirror-model", "unity",
                "--tt-jitter-model", method,
                "--tt-kernel-convention", kernel,
            ]
            if disabled:
                cmd.append("--disable-tt-jitter")
            if args.tt_axis_angle_deg is not None:
                cmd.extend(["--tt-axis-angle-deg", str(args.tt_axis_angle_deg)])
            cmd.extend(["--tt-axis-parity", args.tt_axis_parity])
            folder = root/label
            folder.mkdir(parents=True, exist_ok=True)
            (folder/"p4_command.json").write_text(
                json.dumps(cmd, indent=2), encoding="utf-8"
            )
            print(f"[P4] Running {label}", flush=True)
            with (folder/"p4_console.log").open("w", encoding="utf-8") as log:
                exit_code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            if exit_code != 0:
                # The former runner swallowed the underlying Python traceback
                # and only displayed a generic RuntimeError. Keep the full
                # file, but surface its last lines immediately to the user.
                import collections
                with (folder/"p4_console.log").open(
                    "r", encoding="utf-8", errors="replace"
                ) as source:
                    last_lines = list(collections.deque(source, maxlen=80))
                log_tail = "".join(last_lines).strip()
                (root/"p4_failed_variant.json").write_text(json.dumps({
                    "variant": label, "return_code": exit_code,
                    "log_file": str(folder/"p4_console.log"),
                    "log_tail": log_tail,
                    "scientific_policy": "Do not clip non-PSD physical TT covariance",
                }, indent=2), encoding="utf-8")
                print(
                    f"\n[P4] {label} failed (exit {exit_code}); "
                    f"last {len(last_lines)} log lines from "
                    f"{folder/'p4_console.log'}:\n{log_tail}\n",
                    file=sys.stderr, flush=True,
                )
                raise RuntimeError(
                    f"{label} failed (exit {exit_code}). The original "
                    "traceback is printed above and preserved in "
                    f"{folder/'p4_console.log'}."
                )
    report = summarize_existing_outputs(root)
    outfile = root/"p4_tt_ablation_report.json"
    outfile.write_text(json.dumps(report, indent=2, allow_nan=False),
                       encoding="utf-8")
    _figures(root, report["results"])
    print(f"[P4] Report: {outfile}", flush=True)
    print(f"[P4] Image: {root/'p4_tt_ablation_model_psfs.png'}", flush=True)
    print("[P4] These are conditional predictions; no calibration from science binary")


if __name__ == "__main__":
    main()
