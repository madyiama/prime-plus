"""P1-A: read-only comparison of Keck WFS r0/L0, pseudo-OL and DIMM.

Input: completed keck_n0013_smoke.py run folders containing
atmosphere_reconstruction_audit.json and frequency_grid_diagnostics.json.

This is an evidence/provenance audit, NOT a refit and NOT a validation
of the RTC sign, WFS noise calibration or site-monitor seeing reduction.

Example (from repository root):
    uv run python scripts/keck_p1_r0_baseline_audit.py \
      --cases results/n0120_P2_1layers results/n0120_P2_7layers \
      --output results/P1_n0120_baseline

For many acquisitions, pass each independent case folder to --cases.
Keep multiple 1-/7-layer outputs labelled as the SAME observation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ARCSEC_PER_RAD = 180.0*3600.0/math.pi


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run an unmodified telemetry smoke first"
        )
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return d


def _positive(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and >0; got {value!r}")
    return number


def _optional_number(value):
    if value is None:
        return None
    return float(value)


def analyse_case(case: Path) -> dict:
    """Verify already-existing P1 diagnostics without altering telemetry."""
    case = Path(case)
    atm = _json(case / "atmosphere_reconstruction_audit.json")
    freq = _json(case / "frequency_grid_diagnostics.json")
    pol = atm.get("pseudo_open_loop") or {}
    wavelength = _positive(
        atm["atmosphere_wavelength_m"], "atmosphere wavelength"
    )
    r0 = _positive(
        atm["r0_tel_m_at_atmosphere_wavelength"], "WFS LOS r0"
    )
    fit_seeing = _positive(atm["seeing_tel_arcsec"], "WFS seeing")
    airmass = _positive(atm["airmass_nirc2"], "airmass")
    if airmass < 1 - 1e-8:
        raise ValueError("Physical airmass must be >=1")
    l0 = _positive(atm["fitted_L0_m"], "fitted L0")
    predicted_seeing = (
        0.976*ARCSEC_PER_RAD*wavelength/r0
    )
    seeing_fractional_error = (
        fit_seeing/predicted_seeing - 1.0
    )
    flags = []
    if abs(seeing_fractional_error) > 5e-4:
        flags.append("WFS_R0_SEEING_CONVERSION_MISMATCH")

    sign_verified = bool(pol.get("sign_independently_verified", False))
    if not sign_verified:
        flags.append("PSEUDO_OL_RTC_SIGN_UNVERIFIED")
    alignment = pol.get("method")
    if alignment != "future_residual_linear_no_wrap":
        flags.append("PSEUDO_OL_ALIGNMENT_NOT_EXPECTED")
    used = int(pol.get("frames_used", 0))
    dropped = int(pol.get("frames_dropped", -1))
    delay = _optional_number(pol.get("delay_frames"))
    if used < 2 or dropped < 0 or delay is None or not math.isfinite(delay):
        flags.append("PSEUDO_OL_FRAME_PROVENANCE_INVALID")
    if dropped >= 0 and used > 0 and dropped/(used+dropped) > 0.05:
        flags.append("PSEUDO_OL_MORE_THAN_5_PERCENT_FRAMES_DROPPED")

    r0_model = _optional_number(freq.get("rescaled_r0_m"))
    expected_r0 = _optional_number(freq.get("r0_expected_wfs_los_m"))
    r0_ratio = (
        r0_model/expected_r0
        if r0_model is not None and expected_r0 is not None
        and expected_r0 > 0 else None
    )
    if r0_ratio is None or abs(r0_ratio-1.0) > 1e-5:
        flags.append("FORWARD_R0_LINE_OF_SIGHT_RESCALE_MISMATCH")
    if freq.get("seeing_reference") != "line_of_sight":
        flags.append("FORWARD_MODEL_SEEING_REFERENCE_NOT_LOS")

    fit_success = bool(atm.get("r0_fit_success", False))
    if not fit_success:
        flags.append("JOINT_R0_L0_FIT_FAILED")
    if bool(atm.get("r0_fit_L0_at_upper_bound")) or l0 >= 99.99:
        flags.append("L0_AT_FIT_UPPER_BOUND")
    active_bounds = atm.get("r0_fit_active_parameter_bounds") or []
    if any(int(v) != 0 for v in active_bounds):
        flags.append("ATMOSPHERE_FIT_ACTIVE_BOUND")
    sigma_r0 = _optional_number(atm.get("r0_fit_uncertainty_m"))
    sigma_l0 = _optional_number(atm.get("fitted_L0_uncertainty_m"))
    if sigma_l0 is None:
        sigma_l0 = _optional_number(atm.get("L0_fit_uncertainty_m"))
    # The smoke writes "fitted_L0_uncertainty_m" and should be the
    # sole public key; do not invent a confidence interval if absent.
    if sigma_r0 is None or sigma_l0 is None:
        flags.append("R0_L0_UNCERTAINTIES_MISSING")
    elif (not math.isfinite(sigma_r0) or not math.isfinite(sigma_l0)
          or sigma_r0 < 0 or sigma_l0 < 0):
        flags.append("R0_L0_UNCERTAINTIES_NONFINITE")

    seeing_dimm = _optional_number(atm.get("seeing_dimm_arcsec"))
    if seeing_dimm is not None and seeing_dimm <= 0:
        seeing_dimm = None
    if seeing_dimm is None:
        flags.append("DIMM_SEEING_UNAVAILABLE")
    if "UNKNOWN" not in str(atm.get(
        "dimm_seeing_reference_assumption", ""
    )).upper():
        flags.append("DIMM_REFERENCE_ASSUMPTION_CHANGED_REVIEW")

    # DIMM site reference is NOT independently verified. Report BOTH
    # possibilities without choosing either as a calibration target.
    if seeing_dimm is None:
        dimm_r0_direct = dimm_r0_if_zenith_to_los = None
        ratio_direct = ratio_if_zenith = None
        seeing_dimm_if_zenith_to_los = None
    else:
        dimm_r0_direct = 0.976*ARCSEC_PER_RAD*wavelength/seeing_dimm
        seeing_dimm_if_zenith_to_los = seeing_dimm*airmass**(3.0/5.0)
        dimm_r0_if_zenith_to_los = (
            0.976*ARCSEC_PER_RAD*wavelength
            / seeing_dimm_if_zenith_to_los
        )
        ratio_direct = r0/dimm_r0_direct
        ratio_if_zenith = r0/dimm_r0_if_zenith_to_los
        if min(ratio_direct, ratio_if_zenith) < 0.67 or max(
            ratio_direct, ratio_if_zenith
        ) > 1.5:
            flags.append("WFS_DIMM_POTENTIAL_DISCREPANCY_REVIEW")
    # Only label large differences, never tune r0 from site seeing
    # without timestamp/cross-line-of-sight provenance.
    return {
        "case": str(case),
        "wavelength_m": wavelength,
        "r0_wfs_los_m": r0,
        "seeing_wfs_los_arcsec": fit_seeing,
        "seeing_from_r0_expected_arcsec": predicted_seeing,
        "seeing_relative_error": seeing_fractional_error,
        "airmass": airmass,
        "L0_wfs_fit_m": l0,
        "r0_fit_uncertainty_m": sigma_r0,
        "L0_fit_uncertainty_m": sigma_l0,
        "r0_fit_success": fit_success,
        "r0_fit_cost": _optional_number(atm.get("r0_fit_cost")),
        "r0_fit_nfev": atm.get("r0_fit_nfev"),
        "L0_at_upper_bound": bool(atm.get("r0_fit_L0_at_upper_bound")),
        "pseudo_ol_method": alignment,
        "pseudo_ol_delay_frames": delay,
        "pseudo_ol_frames_used": used,
        "pseudo_ol_frames_dropped": dropped,
        "pseudo_ol_sign_independently_verified": sign_verified,
        "r0_forward_los_m": r0_model,
        "r0_forward_expected_los_m": expected_r0,
        "r0_forward_ratio_to_expected": r0_ratio,
        "seeing_reference_forward": freq.get("seeing_reference"),
        "seeing_dimm_arcsec": seeing_dimm,
        "seeing_dimm_los_if_zenith_arcsec": seeing_dimm_if_zenith_to_los,
        "r0_dimm_direct_m": dimm_r0_direct,
        "r0_dimm_if_zenith_los_m": dimm_r0_if_zenith_to_los,
        "r0_wfs_over_dimm_direct": ratio_direct,
        "r0_wfs_over_dimm_if_zenith_los": ratio_if_zenith,
        "dimm_reference_independently_verified": False,
        "quality_flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", type=Path, required=True,
        help="Completed PSF-R smoke output folders",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [analyse_case(p) for p in args.cases]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "P1-A read-only archive provenance; not RTC sign validation",
        "assumptions": [
            "WFS r0 already measures science line of sight",
            "DIMM reference is unknown: direct and conditional zenith-corrected comparison shown",
            "A seven-layer and a one-layer run of n0120 are not independent acquisitions",
            "No science image fitting and no calibration of AO r0 to DIMM",
        ],
        "results": results,
    }
    (output/"p1_r0_baseline_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output/"p1_r0_baseline_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        rows = [dict(v, quality_flags=";".join(v["quality_flags"]))
                for v in results]
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for report in results:
        print(
            f"[P1 {report['case']}] "
            f"r0_los={report['r0_wfs_los_m']:.6g}m, "
            f"L0={report['L0_wfs_fit_m']:.6g}m, "
            f"DIMM/WFS alternatives="
            f"{report['r0_wfs_over_dimm_direct']}/"
            f"{report['r0_wfs_over_dimm_if_zenith_los']}, "
            f"flags={','.join(report['quality_flags']) or 'none'}",
            flush=True,
        )
    print(f"[P1-A] Wrote {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
