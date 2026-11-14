"""Reproducible Keck V1 cutout/template fitting handoff without AO recomputation."""
import json
import numpy as np
import pytest

from scripts import export_psfr_template as exporter
from p3.psfr.template import load_psfr_template
from p3.psfr.stellar import render_stellar_scene,fit_stellar_scene


def test_pixel_scale_is_derived_from_reconstructed_case(tmp_path,monkeypatch):
    case=tmp_path/"n0120"
    case.mkdir()
    raw=np.zeros((32,32,3))
    raw[16,16,:]=[2,3,5]
    np.save(case/"telemetry_only_psf.npy",raw)
    (case/"metadata.json").write_text(json.dumps({
        "psf_spectral_model":(
            "Single reference-wavelength PSF shared across the three "
            "science passband bins with different throughput weights"
        ),
        "camera_pixel_scale_mas":9.942,
        "psf_reference_wavelength_m":1.95e-6,
        "ao_mode":"LGS",
    }))
    called=[]
    monkeypatch.setattr(exporter,"validate_run",lambda path:(
        called.append(path) or {"status":"passed_numerical_artifact_gates"}
    ))
    saved=exporter.export_keck_case(case)
    t=load_psfr_template(saved)
    assert called==[case]
    assert t.pixel_scale_mas==pytest.approx(9.942)
    assert t.provenance["spectral_propagation"]==(
        "reference_wavelength_psf_reweighted_not_chromatic"
    )
    # No AO physics: test only that the science image/template share grid.
    observed=render_stellar_scene(
        t,positions_xy_px=[[16.3,15.8]],fluxes=[1000.],background=3.5
    )
    result=fit_stellar_scene(
        observed,t,initial_positions_xy_px=[[16,16]],
        initial_fluxes=[900],initial_background=3.,
    )
    assert result.converged
    np.testing.assert_allclose(result.positions_xy_px,[[16.3,15.8]],atol=1e-2)
    np.testing.assert_allclose(result.fluxes,[1000],rtol=1e-3)
    assert result.background==pytest.approx(3.5,abs=.01)


def test_old_case_without_scale_requires_explicit_calibration(tmp_path,monkeypatch):
    case=tmp_path/"old"
    case.mkdir()
    raw=np.zeros((16,16))
    raw[8,8]=1
    np.save(case/"telemetry_only_psf.npy",raw)
    (case/"metadata.json").write_text(json.dumps({
        "psf_spectral_model":"Single reference-wavelength PSF"
    }))
    monkeypatch.setattr(exporter,"validate_run",lambda path:{
        "status":"passed_numerical_artifact_gates"
    })
    with pytest.raises(ValueError,match="No camera pixel scale"):
        exporter.export_keck_case(case)
