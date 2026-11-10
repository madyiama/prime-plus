"""Summarize Keck PSF-R runs with physical provenance, without fitting science.

Usage:
    uv run python scripts/keck_campaign_audit.py \
        --results results/benchmark_v1 --output results/benchmark_v1/audit.csv

Each subdirectory should contain metadata.json and the emitted atmosphere,
pupil and MASS-DIMM diagnostics. Older baseline runs are reported as
provenance-incomplete rather than silently upgraded to the new conventions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "case", "ao_mode", "code_revision", "tracked_physics_code_dirty",
    "reference_wavelength_um", "tt_covariance_mode", "tt_kernel_convention",
    "tt_phase_structure_multiplier_vs_noll", "ho_covariance_policy",
    "atmosphere_layers", "atmosphere_profile_source", "mass_dimm_profile_age_min",
    "dimm_seeing_arcsec", "telemetry_seeing_arcsec", "seeing_telemetry_over_dimm",
    "r0_tel_m_500nm", "r0_dimm_m_500nm", "r0_tel_over_dimm",
    "L0_fit_m", "L0_on_fit_bound", "r0_fit_success",
    "pseudo_ol_delay_frames", "pseudo_ol_frames_used",
    "pseudo_ol_sign_verified",
    "pupil_angle_nirc2_deg", "pupil_angle_wfs_deg", "sky_pa_deg",
    "pupil_mask_zero_verified", "ncpa_enabled", "filter_profile_source",
    "quality_flags",
]


def _read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_case(run: Path):
    meta = _read_json(run / "metadata.json")
    atm = _read_json(run / "atmosphere_reconstruction_audit.json")
    pupil = _read_json(run / "pupil_orientation_diagnostics.json")
    mass = _read_json(run / "massdimm_diagnostics.json")
    code = meta.get("code_revision") or {}
    filter_info = meta.get("filter_profile_diagnostics") or {}
    delay = atm.get("pseudo_open_loop") or {}
    flags = []
    if not all((atm, pupil, mass, code)):
        flags.append("MISSING_NEW_PROVENANCE")
    if code.get("tracked_physics_code_has_uncommitted_edits"):
        flags.append("DIRTY_SOURCE")
    if delay and not delay.get("sign_independently_verified"):
        flags.append("OL_SIGN_UNVERIFIED")
    if pupil and not pupil.get("mask_zero_offset_calibrated"):
        flags.append("PUPIL_MAP_ZERO_UNVERIFIED")
    if mass.get("dimm_source") != "measured_dimm":
        flags.append("DIMM_FALLBACK")
    if mass.get("mass_source") not in (
        "measured_mass_and_mass_profile", "mass_profile_integral"
    ):
        flags.append("MASS_PROFILE_FALLBACK")
    if atm.get("r0_fit_L0_at_upper_bound"):
        flags.append("L0_FIT_UPPER_BOUND")
    if atm and not atm.get("r0_fit_success", True):
        flags.append("R0_FIT_FAILED")
    if (meta.get("tt_kernel_convention") == "legacy_unscaled_opd"):
        flags.append("TT_LEGACY_KERNEL_EXPLORATORY")
    if (meta.get("addNCPA") is False):
        flags.append("NCPA_NOT_MODELED")

    seeing_tel = atm.get("seeing_tel_arcsec")
    seeing_dimm = atm.get("seeing_dimm_arcsec")
    rel = (seeing_tel/seeing_dimm
           if isinstance(seeing_tel, (int,float))
           and isinstance(seeing_dimm, (int,float)) and seeing_dimm > 0
           else None)
    if rel is not None and (rel > 1.5 or rel < 2.0/3.0):
        flags.append("WFS_DIMM_SEEING_DISCREPANCY")

    r0_dimm = (0.976 * 500e-9 * 180*3600/3.141592653589793 / seeing_dimm
               if isinstance(seeing_dimm, (int,float)) and seeing_dimm > 0
               else None)

    return {
        "case": run.name,
        "ao_mode": meta.get("ao_mode"),
        "code_revision": code.get("git_revision"),
        "tracked_physics_code_dirty": code.get("tracked_physics_code_has_uncommitted_edits"),
        "reference_wavelength_um": (
            1e6 * meta["psf_reference_wavelength_m"]
            if "psf_reference_wavelength_m" in meta else None
        ),
        "tt_covariance_mode": meta.get("tt_jitter_covariance_method"),
        "tt_kernel_convention": meta.get("tt_kernel_convention"),
        "tt_phase_structure_multiplier_vs_noll": meta.get(
            "tt_kernel_phase_structure_multiplier_vs_noll"
        ),
        "ho_covariance_policy": meta.get("ho_covariance_policy"),
        "atmosphere_layers": atm.get("atmosphere_layer_count"),
        "atmosphere_profile_source": mass.get("mass_source"),
        "mass_dimm_profile_age_min": mass.get("mass_profile_age_minutes"),
        "dimm_seeing_arcsec": seeing_dimm,
        "telemetry_seeing_arcsec": seeing_tel,
        "seeing_telemetry_over_dimm": rel,
        "r0_tel_m_500nm": atm.get("r0_tel_m_at_atmosphere_wavelength"),
        "r0_dimm_m_500nm": r0_dimm,
        "r0_tel_over_dimm": atm.get("r0_tel_over_dimm"),
        "L0_fit_m": atm.get("fitted_L0_m"),
        "L0_on_fit_bound": atm.get("r0_fit_L0_at_upper_bound"),
        "r0_fit_success": atm.get("r0_fit_success"),
        "pseudo_ol_delay_frames": delay.get("delay_frames"),
        "pseudo_ol_frames_used": delay.get("frames_used"),
        "pseudo_ol_sign_verified": delay.get("sign_independently_verified"),
        "pupil_angle_nirc2_deg": pupil.get("nirc2_pupil_angle_deg"),
        "pupil_angle_wfs_deg": pupil.get("wfs_pupil_angle_deg"),
        "sky_pa_deg": pupil.get("sky_position_angle_deg"),
        "pupil_mask_zero_verified": pupil.get("mask_zero_offset_calibrated"),
        "ncpa_enabled": meta.get("addNCPA"),
        "filter_profile_source": filter_info.get("source"),
        "quality_flags": "|".join(flags),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folders = sorted(set(p.parent for p in args.results.rglob("metadata.json")))
    if not folders:
        parser.error(f"No metadata.json found in {args.results}")
    rows = [summarize_case(folder) for folder in folders]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Keck PSF-R audit] {len(rows)} runs -> {args.output}")
    for row in rows:
        print(row["case"], row["ao_mode"], row["quality_flags"])


if __name__ == "__main__":
    main()
