"""Frozen Keck V1 AO basis: reweight prepared SF kernels without rebuilding AO.

This is NOT an independently calibrated AO model. It uses the same
reference-wavelength SF and FourierUtils.sf_3D_to_psf_3D as nominal P3.
Importantly: construction on a real telemetry run is optional, and
acceptance requires an explicit nominal-image parity gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .products import normalize_psf
from .structure import assemble_phase_structure_function


SCHEMA = "p3.psfr.frozen_ao_basis.v1"


@dataclass(frozen=True)
class FrozenAOBasis:
    dphi_ao: np.ndarray
    dphi_tt: np.ndarray
    dphi_fit: np.ndarray
    dphi_alias: np.ndarray
    dphi_ani: np.ndarray
    layer_weights: np.ndarray
    static_pixel_otf: np.ndarray
    diffraction_otf: np.ndarray
    reference_psf: np.ndarray
    r0_m: float
    atmosphere_wavelength_m: float
    reference_wavelength_m: float
    k_ref: int
    n_pixels: int
    camera_fov: int
    is_anisoplanatic: bool
    provenance: dict


def _checked(basis):
    arrays=(basis.dphi_ao,basis.dphi_tt,basis.dphi_fit,
            basis.dphi_alias,basis.static_pixel_otf,
            basis.diffraction_otf)
    n=basis.dphi_ao.shape[0]
    if (n<8 or any(a.shape!=(n,n) for a in arrays)
            or not all(np.isfinite(a).all() for a in arrays)):
        raise ValueError("AO SF and optical OTF arrays must be finite [nOtf,nOtf]")
    if (basis.is_anisoplanatic and
            (basis.dphi_ani.ndim!=4 or basis.dphi_ani.shape[0]!=1
             or basis.dphi_ani.shape[2:]!=(n,n)
             or basis.layer_weights.shape!=(basis.dphi_ani.shape[1],))):
        raise ValueError("AO anisoplanatism must be [1,layer,nOtf,nOtf]")
    if basis.is_anisoplanatic and not np.isfinite(basis.dphi_ani).all():
        raise ValueError("Nonfinite AO anisoplanatism kernel")
    if (not isinstance(basis.k_ref,int) or basis.k_ref<1
            or n!=basis.k_ref*basis.n_pixels
            or not (1<=basis.n_pixels<=basis.camera_fov)
            or basis.reference_psf.shape!=(basis.n_pixels,basis.n_pixels)
            or not np.isfinite(basis.reference_psf).all()
            or np.any(basis.reference_psf<0)
            or not np.isclose(basis.reference_psf.sum(),1,rtol=1e-12,atol=1e-12)):
        raise ValueError("AO basis detector sampling or reference PSF is invalid")
    for name in ("r0_m","reference_wavelength_m","atmosphere_wavelength_m"):
        value=float(getattr(basis,name))
        if not np.isfinite(value) or value<=0:
            raise ValueError("AO basis has invalid "+name)
    if basis.provenance.get("schema")!=SCHEMA:
        raise ValueError("AO basis schema mismatch")


def frozen_basis_from_model(model, prediction, *, provenance=None):
    """Snapshot the P3 kernels once after its Keck V1 reconstruction.

    Fail loudly for more than one guide/source direction: a single
    field-invariant template cannot model spatially varying multi-source
    anisoplanatism, and we must not silently choose a source.
    """
    if int(model.ao.src.nSrc)!=1:
        raise ValueError("Frozen AO fitting currently supports one reference direction")
    forward=normalize_psf(prediction)
    if not np.isclose(forward.image.sum(),1.):
        raise ValueError("Invalid reference PSF")
    f=model.freq
    n=int(f.nOtf)
    k=int(f.kRef_)
    info=dict(provenance or {})
    info.update({
        "schema":SCHEMA,
        "physics":"Keck_V1_reference_wavelength",
        "calibrated_tiptilt_gain":False,
        "science_image_fit_performed":False,
        "nominal_gho":1.0,
        "nominal_gtt":1.0,
    })
    basis=FrozenAOBasis(
        dphi_ao=np.asarray(model.dphi_ao,dtype=np.float64).copy(),
        dphi_tt=np.asarray(model.dphi_tt,dtype=np.float64).copy(),
        dphi_fit=np.asarray(model.dphi_fit,dtype=np.float64).copy(),
        dphi_alias=np.asarray(model.dphi_alias,dtype=np.float64).copy(),
        dphi_ani=(np.asarray(model.dphi_ani,dtype=np.float64).copy()
                  if f.isAniso else np.empty((0,),dtype=np.float64)),
        layer_weights=np.asarray(model.ao.atm.weights,dtype=np.float64).reshape(-1).copy(),
        static_pixel_otf=np.asarray(f.otfNCPA*model.otfPixel,dtype=np.complex128).copy(),
        diffraction_otf=np.asarray(f.otfDL,dtype=np.float64).copy(),
        reference_psf=forward.image.copy(),
        r0_m=float(model.ao.atm.r0),
        atmosphere_wavelength_m=float(model.ao.atm.wvl),
        reference_wavelength_m=float(f.wvlRef),
        k_ref=k,
        n_pixels=int(forward.image.shape[0]),
        camera_fov=int(model.ao.cam.fovInPix),
        is_anisoplanatic=bool(f.isAniso),
        provenance=info,
    )
    _checked(basis)
    return basis


def render_ao_psf(basis, *, r0_m=None, gho=1., gtt=1.):
    """Compute unit-flux monochromatic detector PSF by legacy P3 math.

    No AO model is instantiated; the Fourier renderer is called with
    read-only data stubs. As a matter of scientific policy, the
    *nominal* image MUST first match the original Keck V1 PSF in
    validate_ao_basis_parity, before any parameter optimization.
    """
    _checked(basis)
    r0=basis.r0_m if r0_m is None else float(r0_m)
    if not np.isfinite(r0) or r0<=0:
        raise ValueError("r0 must be positive")
    if not np.isfinite(gho) or gho<0 or not np.isfinite(gtt) or gtt<0:
        raise ValueError("gho and gtt must be nonnegative finite values")
    assembly=assemble_phase_structure_function(
        r0_m=r0,gains=np.array([gho,gtt]),
        dphi_ao=basis.dphi_ao,dphi_tt=basis.dphi_tt,
        dphi_fit=basis.dphi_fit,dphi_alias=basis.dphi_alias,
        n_sources=1,
        anisoplanatism_kernel=(basis.dphi_ani if basis.is_anisoplanatic else None),
        layer_weights=(basis.layer_weights if basis.is_anisoplanatic else None),
        atmosphere_wavelength_m=(basis.atmosphere_wavelength_m
                                  if basis.is_anisoplanatic else None),
        science_wavelength_m=(basis.reference_wavelength_m
                              if basis.is_anisoplanatic else None),
    )
    # Same P3 Fourier routine, pixel integration and detector binning.
    # No stellar or additional jitter terms; TT is already in dphi_tt.
    from p3.aoSystem.FourierUtils import sf_3D_to_psf_3D
    freq=SimpleNamespace(
        nOtf=basis.dphi_ao.shape[0],kRef_=basis.k_ref,nWvl=1,
        otfNCPA=basis.static_pixel_otf,
        otfDL=basis.diffraction_otf,
        wvlRef=basis.reference_wavelength_m,
    )
    ao=SimpleNamespace(
        dtype=np.float64,
        src=SimpleNamespace(nSrc=1),
        cam=SimpleNamespace(fovInPix=basis.camera_fov),
    )
    rendered,_=sf_3D_to_psf_3D(
        assembly.dphi_rad2,freq,ao,nPix=basis.n_pixels,
        otfPixel=1.,
    )
    img=normalize_psf(np.squeeze(np.asarray(rendered))).image
    if img.shape!=basis.reference_psf.shape:
        raise ValueError("AO renderer changed detector image geometry")
    return img


def validate_ao_basis_parity(basis, *, rtol=1e-12, atol=1e-12):
    """Mandatory nominal numerical parity; fail before fitting on mismatch."""
    nominal=render_ao_psf(basis)
    diff=float(np.max(np.abs(nominal-basis.reference_psf)))
    relative=float(
        np.linalg.norm(nominal-basis.reference_psf) /
        np.linalg.norm(basis.reference_psf)
    )
    if not np.allclose(nominal,basis.reference_psf,rtol=rtol,atol=atol):
        raise ValueError(
            "AO frozen basis fails Keck V1 nominal PSF numerical parity: "
            f"max_abs={diff:.6e}, relative_l2={relative:.6e}; "
            "do not optimize against a different forward renderer"
        )
    return {"max_absolute_difference":diff,
            "relative_l2_difference":relative,
            "rtol":rtol,"atol":atol,
            "nominal_parity_passed":True}


def save_ao_basis(basis, path):
    _checked(basis)
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        target,
        dphi_ao=basis.dphi_ao,dphi_tt=basis.dphi_tt,
        dphi_fit=basis.dphi_fit,dphi_alias=basis.dphi_alias,
        dphi_ani=basis.dphi_ani,layer_weights=basis.layer_weights,
        static_pixel_otf=basis.static_pixel_otf,
        diffraction_otf=basis.diffraction_otf,
        reference_psf=basis.reference_psf,
        r0_m=np.asarray(basis.r0_m),
        atmosphere_wavelength_m=np.asarray(basis.atmosphere_wavelength_m),
        reference_wavelength_m=np.asarray(basis.reference_wavelength_m),
        k_ref=np.asarray(basis.k_ref),
        n_pixels=np.asarray(basis.n_pixels),
        camera_fov=np.asarray(basis.camera_fov),
        is_anisoplanatic=np.asarray(basis.is_anisoplanatic),
        provenance_json=np.asarray(json.dumps(basis.provenance,allow_nan=False)),
    )
    return target if target.suffix==".npz" else Path(str(target)+".npz")


def load_ao_basis(path):
    """Load without pickle and require full schema (no silent defaults)."""
    from dataclasses import fields
    with np.load(path,allow_pickle=False) as raw:
        names={f.name for f in fields(FrozenAOBasis)}
        fields_on_disk=(names-{"provenance"})|{"provenance_json"}
        if set(raw.files)!=fields_on_disk:
            raise ValueError("AO archive missing expected fields")
        kw={name:np.array(raw[name],copy=True)
            for name in ("dphi_ao","dphi_tt","dphi_fit","dphi_alias",
                         "dphi_ani","layer_weights","static_pixel_otf",
                         "diffraction_otf","reference_psf")}
        for name in ("r0_m","atmosphere_wavelength_m","reference_wavelength_m"):
            kw[name]=float(raw[name])
        for name in ("k_ref","n_pixels","camera_fov"):
            kw[name]=int(raw[name])
        kw["is_anisoplanatic"]=bool(raw["is_anisoplanatic"])
        kw["provenance"]=json.loads(str(raw["provenance_json"].item()))
    basis=FrozenAOBasis(**kw)
    _checked(basis)
    return basis
