"""Conservative, reproducible Keck STRAP TT method selection.

This policy layer changes NO noise estimator, loop transfer, covariance
algebra, telescope geometry or optical transfer function. It exposes the
already implemented physically normalized TT options and makes the provisional
reduced-jitter baseline reproducible for later end-to-end simulations.

Profiles are explicit presets, not fitted/tuned to a science image. When no
profile is requested, existing TOML fields remain authoritative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from p3.telemetry.ttNoise import TT_METHODS


TT_COVARIANCE_METHODS = ("legacy_psfr", "spectral_closed_loop")
TT_MIRROR_MODELS = ("unity", "ragland_2016_keck_ii")
PROFILES = {
    "reduced_jitter": {
        "noise_method": "van_dam",
        "noise_band": (0.65, 0.95),
        "noise_welch": 512,
        "covariance_method": "legacy_psfr",
        "mirror_model": "unity",
        "jitter_welch": 512,
    },
    "spectral_comparison": {
        "noise_method": "van_dam",
        "noise_band": (0.65, 0.95),
        "noise_welch": 512,
        "covariance_method": "spectral_closed_loop",
        "mirror_model": "unity",
        "jitter_welch": 512,
    },
    "gendron_comparison": {
        "noise_method": "gendron",
        "noise_band": (0.65, 0.95),
        "noise_welch": 512,
        "covariance_method": "legacy_psfr",
        "mirror_model": "unity",
        "jitter_welch": 512,
    },
}
DEFAULT_REDUCED_JITTER_PROFILE = "reduced_jitter"


@dataclass(frozen=True)
class TTSelection:
    noise_method: str
    noise_band: tuple[float, float]
    noise_welch: int
    covariance_method: str
    mirror_model: str
    jitter_welch: int
    source: str
    profile: str | None
    instrument_calibrated: bool = False

    def report(self):
        return {
            "noise_method": self.noise_method,
            "noise_band_fraction_nyquist": list(self.noise_band),
            "noise_welch_nperseg": self.noise_welch,
            "covariance_method": self.covariance_method,
            "mirror_model": self.mirror_model,
            "jitter_welch_nperseg": self.jitter_welch,
            "selection_source": self.source,
            "profile": self.profile,
            "science_psf_used_for_method_selection": False,
            "instrument_calibrated": self.instrument_calibrated,
            "apd_noise_model_enabled": False,
            "noise_method_physically_validated": False,
            "loop_transfer_independently_calibrated_for_keck_2020": False,
            "method_status": (
                "provisional_legacy_noise_subtraction"
                if self.covariance_method == "legacy_psfr"
                else "exploratory_spectral_physical"
            ),
            "recommended_for_framework_default": (
                self.noise_method == "van_dam"
                and self.covariance_method == "legacy_psfr"
                and self.mirror_model == "unity"
                and self.noise_band == (0.65, 0.95)
                and self.noise_welch == 512
            ),
        }


def resolve_tt_selection(full_config, *, profile=None, overrides=None):
    """Resolve TOML, optional named profile, then explicit command-line fields.

    Precedence:
      explicit CLI individual override > explicit named profile > TOML values.
    A profile is used *only when requested*, never silently overrides a
    user-modified TOML. Defaults from TOML are the established P4-A reference.
    """
    if not isinstance(full_config, dict):
        raise ValueError("TT configuration must be a TOML mapping")
    if profile is not None and profile not in PROFILES:
        raise ValueError(
            f"Unknown TT profile {profile!r}; expected {tuple(PROFILES)}"
        )
    cfg_n = full_config.get("tip_tilt_noise", {})
    cfg_j = full_config.get("tip_tilt_jitter", {})
    if not isinstance(cfg_n, dict) or not isinstance(cfg_j, dict):
        raise ValueError("tip_tilt_noise / tip_tilt_jitter must be TOML tables")
    if profile is not None:
        raw = dict(PROFILES[profile])
        source = f"named_profile:{profile}"
    else:
        raw = {
            "noise_method": cfg_n.get("method", "van_dam"),
            "noise_band": cfg_n.get("high_frequency_band", (0.65, 0.95)),
            "noise_welch": cfg_n.get("welch_nperseg", 512),
            "covariance_method": cfg_j.get("method", "legacy_psfr"),
            "mirror_model": cfg_j.get("mirror_model", "unity"),
            "jitter_welch": cfg_j.get(
                "welch_nperseg", cfg_n.get("welch_nperseg", 512)
            ),
        }
        source = "TOML_or_legacy_defaults"
    explicit_keys = set()
    for key, val in (overrides or {}).items():
        if key not in raw:
            raise ValueError(f"Unknown TT override {key!r}")
        if val is not None:
            raw[key] = val
            explicit_keys.add(key)
    if explicit_keys:
        source += "+explicit_CLI:" + ",".join(sorted(explicit_keys))

    noise = str(raw["noise_method"])
    covariance = str(raw["covariance_method"])
    mirror = str(raw["mirror_model"])
    if noise not in TT_METHODS:
        raise ValueError(f"Unsupported STRAP noise estimator {noise!r}: {TT_METHODS}")
    if covariance not in TT_COVARIANCE_METHODS:
        raise ValueError(f"Unsupported TT covariance method {covariance!r}")
    if mirror not in TT_MIRROR_MODELS:
        raise ValueError(f"Unsupported Keck TT mirror transfer {mirror!r}")
    # In the historical covariance path P3 always uses a unity TTM,
    # irrespective of the supplied mirror model. Do not silently ignore
    # a requested alternative transfer and label the result as Ragland.
    if covariance == "legacy_psfr" and mirror != "unity":
        raise ValueError(
            "legacy_psfr does not use the mirror transfer; select unity, "
            "or spectral_closed_loop to compare Ragland 2016."
        )
    band = raw["noise_band"]
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        raise ValueError("TT van Dam fit band must have exactly two numbers")
    band = tuple(float(x) for x in band)
    if not (all(math.isfinite(x) for x in band)
            and 0 < band[0] < band[1] < 1):
        raise ValueError("TT van Dam fit band must satisfy 0 < lo < hi < 1")
    for key in ("noise_welch", "jitter_welch"):
        x = raw[key]
        if isinstance(x, bool) or int(x) != x or x < 128:
            raise ValueError(f"{key} must be an integer >= 128")
        raw[key] = int(x)
    return TTSelection(
        noise_method=noise, noise_band=band,
        noise_welch=raw["noise_welch"],
        covariance_method=covariance, mirror_model=mirror,
        jitter_welch=raw["jitter_welch"], source=source, profile=profile,
    )
