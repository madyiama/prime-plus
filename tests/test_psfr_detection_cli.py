"""CLI regression: detect from frozen .npz/.npy artifacts, then fit stars."""
import json
import sys

import numpy as np

from p3.psfr.stellar import render_stellar_scene
from p3.psfr.template import template_from_psfr,save_psfr_template
from scripts import fit_psfr_stars


def test_cli_detect_then_fit_rejects_isolated_hot_pixel(tmp_path,monkeypatch):
    n=56
    yy,xx=np.indices((n,n))
    kernel=np.exp(-((xx-28)**2+(yy-28)**2)/(2*2.5**2))
    template=template_from_psfr(kernel,pixel_scale_mas=9.942)
    case=tmp_path/"n0120"
    case.mkdir()
    save_psfr_template(template,case/"psfr_fit_template.npz")
    rng=np.random.default_rng(31)
    image=render_stellar_scene(
        template,positions_xy_px=[[23.2,29.4]],
        fluxes=[4500.],background=10.
    )
    image+=rng.normal(0,1,size=(n,n))
    image[7,47]+=8000
    np.save(case/"observed_nirc2_cutout.npy",image)

    monkeypatch.setattr(sys,"argv",[
        "fit_psfr_stars.py","--case",str(case),
        "--max-sources","3","--detect-only",
    ])
    assert fit_psfr_stars.main()==0
    output=case/"stellar_fit"
    catalog=json.loads((output/"source_detection.json").read_text())
    assert catalog["n_detected"]==1
    assert catalog["n_rejected"]>=1
    assert (output/"source_detection.png").is_file()
    saved_mask=np.load(output/"source_detection_valid_pixels.npy")
    assert saved_mask.shape==(n,n)
    assert not saved_mask[7,47]

    monkeypatch.setattr(sys,"argv",[
        "fit_psfr_stars.py","--case",str(case),"--max-sources","3",
    ])
    assert fit_psfr_stars.main()==0
    report=json.loads((output/"fit_report.json").read_text())
    assert report["n_stars"]==1
    assert report["source_detection_method"]=="psf_prior_morphological"
    assert report["source_detection_rejected_pixels"]>=1
    assert abs(report["stars"][0]["x_column_px"]-23.2)<.3
    assert abs(report["stars"][0]["y_row_px"]-29.4)<.3
    assert (output/"fit_comparison.png").is_file()
    assert (output/"residual.npy").is_file()
    # No telemetry load, FITS/SAV, or AO reinitialization required.
