"""Structured, unit-labelled PSF-R initialization timing diagnostics."""
from __future__ import annotations

import math


FOURIER_METRICS = (
    "t_init", "t_initFreq", "t_atmo", "t_powerSpectrumDensity",
    "t_reconstructor", "t_finalReconstructor", "t_tomo",
    "t_opt", "t_controller", "t_aliasingPSD",
    "t_focalAnisoplanatism", "t_spatioTemporalPSD",
    "t_noisePSD", "t_errorBreakDown",
)


def _timing(value):
    """Discard absent, invalid or negative timing values."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0 else None


def psfr_timing_report(model):
    """Return measured milliseconds, never a guessed time decomposition.

    'stages_ms' are distinct measured top-level stage durations, except
    'psfr_init_total' (the inclusive overall duration).

    'nested_frequency_ms' is a subset of the frequency-domain duration;
    in Keck SLAO, 'anisoplanatism' includes focal/angular/anisokinetic
    calculations. The Fourier model's own timers can also overlap one
    another. DO NOT sum nested durations with the top-level stages.
    """
    raw = getattr(model, "timing_breakdown_ms", {})
    stages = {
        str(name): t
        for name, value in raw.items()
        if (t := _timing(value)) is not None
    }
    freq = getattr(model, "freq", None)
    freq_nested = {}
    for public_name, attr in (
        ("anisoplanatism", "tani"),
        ("frequency_grid", "tfreq"),
        ("static_otf_setup", "totf"),
    ):
        value = _timing(getattr(freq, attr, None))
        if value is not None:
            freq_nested[public_name] = value
    fourier = getattr(model, "fao", None)
    fourier_metrics = {}
    for attr in FOURIER_METRICS:
        value = _timing(getattr(fourier, attr, None))
        if value is not None:
            fourier_metrics[attr] = value
    ordered = sorted(
        ((name, value) for name, value in stages.items()
         if name not in ("psfr_init_total", "frequency_anisoplanatism_nested")),
        key=lambda row: row[1], reverse=True,
    )
    return {
        "units": "milliseconds",
        "stages_ms": stages,
        "frequency_nested_ms": freq_nested,
        "fourier_model_internal_ms": fourier_metrics,
        "largest_disjoint_stages": [
            {"stage": name, "elapsed_ms": value} for name, value in ordered
        ],
        "method": "wall clock (perf_counter) for separate stages; nested timers vary",
        "scope": "psfR __init__, including the initial PSF in its error breakdown",
        "warning": "Nested timers overlap parent stages; do not sum nested values.",
    }
