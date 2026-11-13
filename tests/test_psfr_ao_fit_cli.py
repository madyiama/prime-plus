"""CLI handoff: AO fitting uses only previously generated .npz/.npy."""
import json
import sys

import numpy as np

from p3.psfr.ao_basis import render_ao_psf,save_ao_basis
from p3.psfr.stellar import render_stellar_scene
from p3.psfr.template import save_psfr_template,template_from_psfr
from scripts import fit_psfr_ao
from p3.psfr.ao_basis import FrozenAOBasis,SCHEMA
from dataclasses import replace


def make_basis(n=64):
    yy,xx=np.indices((n,n))
    c=n//2
    r2=(xx-c)**2+(yy-c)**2
    psf=np.exp(-r2/(2*2.5**2))+1e-4
    otf=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    otf=otf/otf[c,c]
    sf=1-np.exp(-r2/(2*9**2))
    basis=FrozenAOBasis(
        dphi_ao=.5*sf,dphi_tt=.22*sf,
        dphi_fit=.055*sf,dphi_alias=.025*sf,
        dphi_ani=np.empty((0,)),layer_weights=np.array([1.]),
        static_pixel_otf=otf,diffraction_otf=np.ones((n,n)),
        reference_psf=np.ones((n//2,n//2))/(n//2)**2,
        r0_m=.3,atmosphere_wavelength_m=500e-9,
        reference_wavelength_m=2.055e-6,k_ref=2,n_pixels=n//2,
        camera_fov=n//2,is_anisoplanatic=False,
        provenance={"schema":SCHEMA},
    )
    return replace(basis,reference_psf=render_ao_psf(basis))


def test_hybrid_ao_cli_reuses_basis_without_telemetry_or_psfr_init(
    tmp_path,monkeypatch,
):
    case=tmp_path/"n0120"
    case.mkdir()
    (case/"stellar_fit").mkdir()
    basis=make_basis()
    save_ao_basis(basis,case/"ao_fit_basis.npz")
    template=template_from_psfr(basis.reference_psf,pixel_scale_mas=9.942)
    save_psfr_template(template,case/"psfr_fit_template.npz")
    altered=template_from_psfr(
        render_ao_psf(basis,r0_m=.27),pixel_scale_mas=9.942,
    )
    observed=render_stellar_scene(
        altered,positions_xy_px=[[15.25,16.35]],
        fluxes=[10000.],background=4.,
    )
    np.save(case/"observed_nirc2_cutout.npy",observed)
    report={
        "stars":[{
            "x_column_px":15.25,"y_row_px":16.35,
            "flux_detector_counts_on_unshifted_template_support":10000.,
        }],
        "background_detector_counts_per_pixel":4.,
    }
    (case/"stellar_fit"/"fit_report.json").write_text(json.dumps(report))
    np.save(case/"stellar_fit"/"valid_pixels.npy",
            np.ones(observed.shape,dtype=bool))
    monkeypatch.setattr(sys,"argv",[
        "fit_psfr_ao.py","--case",str(case),"--n-stars","1",
        "--ao-params","r0","--alternations","1",
    ])
    assert fit_psfr_ao.main()==0
    out=case/"hybrid_ao_fit"
    result=json.loads((out/"ao_fit_report.json").read_text())
    assert result["n_sources"]==1
    assert result["numeric_nominal_PSF_parity"]["nominal_parity_passed"]
    assert result["best_AO"]["gho"]==1.
    assert result["best_AO"]["gtt"]==1.
    assert result["RMS_ratio"]<1.
    assert (out/"ao_fit_comparison.png").exists()
    assert np.load(out/"hybrid_model.npy").shape==observed.shape
