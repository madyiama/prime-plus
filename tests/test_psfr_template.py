"""PSF-R fitting-template contracts, independent of real Keck archives."""
import numpy as np
import pytest

from p3.psfr.template import (
    SCHEMA, KECK_V1_SPECTRAL_STATUS,
    template_from_psfr, save_psfr_template, load_psfr_template,
)


def synthetic(n=16):
    psf=np.zeros((n,n,3))
    psf[n//2,n//2,:]=[2,3,5]
    return psf


def test_template_preserves_original_spectral_fractions_and_is_independent(tmp_path):
    predicted=synthetic()
    initial=predicted.copy()
    template=template_from_psfr(
        predicted,pixel_scale_mas=9.942,
        provenance={"observation":"synthetic"},
    )
    np.testing.assert_array_equal(predicted,initial)
    np.testing.assert_allclose(template.spectral_fractions,[.2,.3,.5])
    assert template.image.sum()==pytest.approx(1)
    assert template.cube.sum()==pytest.approx(1)
    assert template.provenance["spectral_propagation"]==KECK_V1_SPECTRAL_STATUS
    assert template.provenance["absolute_strap_calibration"] is False
    assert template.provenance["science_image_psf_fit_performed"] is False
    assert template.provenance["schema"]==SCHEMA
    assert template.provenance["observation"]=="synthetic"
    saved=save_psfr_template(template,tmp_path/"template")
    assert saved.name=="template.npz"
    restored=load_psfr_template(saved)
    np.testing.assert_array_equal(restored.image,template.image)
    np.testing.assert_array_equal(restored.cube,template.cube)
    np.testing.assert_array_equal(
        restored.spectral_fractions,template.spectral_fractions
    )
    assert restored.pixel_scale_mas==pytest.approx(9.942)
    assert restored.provenance==template.provenance


def test_source_index_is_required_for_multiple_sources():
    two=np.stack((synthetic(),np.roll(synthetic(),1,axis=0)),axis=2)
    with pytest.raises(ValueError,match="source_index"):
        template_from_psfr(two,pixel_scale_mas=9.942)
    template=template_from_psfr(
        two,pixel_scale_mas=9.942,source_index=1
    )
    np.testing.assert_allclose(template.image.sum(),1)
    assert template.provenance["source_index"]==1


@pytest.mark.parametrize("pixels",[0.,-1.,np.nan,np.inf])
def test_invalid_pixel_scale_fails(pixels):
    with pytest.raises(ValueError,match="pixel scale"):
        template_from_psfr(synthetic(),pixel_scale_mas=pixels)


def test_nonphysical_psf_not_silently_repaired():
    psf=synthetic()
    psf[1,1,0]=-1e-13
    with pytest.raises(ValueError,match="negative"):
        template_from_psfr(psf,pixel_scale_mas=9.942)
    psf=synthetic()
    psf[1,1,0]=np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        template_from_psfr(psf,pixel_scale_mas=9.942)


def test_loader_rejects_modified_flux_and_schema(tmp_path):
    template=template_from_psfr(synthetic(),pixel_scale_mas=9.942)
    good=save_psfr_template(template,tmp_path/"good.npz")
    assert load_psfr_template(good).image.sum()==pytest.approx(1)
    with np.load(good,allow_pickle=False) as archive:
        elements={k:archive[k] for k in archive.files}
    bad=tmp_path/"altered.npz"
    elements["image"]=elements["image"]*.5
    np.savez_compressed(bad,**elements)
    with pytest.raises(ValueError,match="unit flux"):
        load_psfr_template(bad)
    wrong=tmp_path/"missing.npz"
    np.savez_compressed(wrong,image=np.ones((4,4)))
    with pytest.raises(ValueError,match="schema"):
        load_psfr_template(wrong)


def test_2d_template_has_one_photon_channel():
    inp=np.zeros((8,8))
    inp[4,4]=3
    template=template_from_psfr(inp,pixel_scale_mas=12.)
    assert template.cube.shape==(8,8,1)
    np.testing.assert_array_equal(template.spectral_fractions,[1.])
