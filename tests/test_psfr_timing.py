"""Timing diagnostic contracts: synthetic objects only."""
from types import SimpleNamespace as NS

from p3.psfr.timing import psfr_timing_report


def test_nested_anisoplanatism_is_not_counted_as_an_independent_stage():
    model = NS(
        timing_breakdown_ms={
            "ao_system": 4.,
            "frequency_domain": 150.,
            "fourier_model": 6.,
            "ho_sf": 0.2,
            "error_budget_including_forward": 0.5,
            "psfr_init_total": 161.,
        },
        freq=NS(tani=147., tfreq=1., totf=0.),
        fao=NS(t_init=6., t_tomo=0.4, t_aliasingPSD=0.3),
    )
    report=psfr_timing_report(model)
    assert report["units"]=="milliseconds"
    assert report["frequency_nested_ms"]["anisoplanatism"]==147.
    assert report["stages_ms"]["frequency_domain"]==150.
    assert report["fourier_model_internal_ms"]["t_tomo"]==0.4
    assert report["largest_disjoint_stages"][0]=={
        "stage":"frequency_domain","elapsed_ms":150.
    }
    assert all(v["stage"]!="psfr_init_total"
               for v in report["largest_disjoint_stages"])
    assert "do not sum nested" in report["warning"]


def test_missing_optional_timers_are_safe_and_invalid_values_excluded():
    model=NS(
        timing_breakdown_ms={
            "psfr_init_total":1.,
            "good":0.,
            "invalid":float("nan"),
            "negative":-2.,
            "text":"not a number",
        }
    )
    result=psfr_timing_report(model)
    assert result["stages_ms"]=={"psfr_init_total":1., "good":0.}
    assert result["frequency_nested_ms"]=={}
    assert result["fourier_model_internal_ms"]=={}
