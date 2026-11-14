"""Constrained hybrid AO + stellar fitting with precomputed Keck V1 kernels.

No HO/TT estimation or telescope initialization happens in this module.
The original telemetry PSF is a strict numerical baseline. Atmospheric
and HO coefficients have Gaussian telemetry-centred priors. TT fitting
requires explicit opt-in because STRAP gain is not independently calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .ao_basis import FrozenAOBasis, render_ao_psf, validate_ao_basis_parity
from .stellar import fit_stellar_scene, render_stellar_scene, StellarFitResult
from .template import PSFTemplate


PARAMETERS=("r0","gho","gtt")
PRIOR_SIGMAS={"r0":.15,"gho":.15,"gtt":.07}
BOUNDS={"r0":(.75,1.25),"gho":(.70,1.30),"gtt":(.85,1.15)}


@dataclass(frozen=True)
class HybridAOFit:
    ao_parameters: dict
    active_parameters: tuple[str,...]
    prior_sigmas: dict
    baseline_stellar: StellarFitResult
    stellar: StellarFitResult
    baseline_model: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    nominal_parity: dict
    optimizer_status: tuple[dict,...]
    noise_source: str
    empirical_noise_scale: float | None
    initial_rms: float
    final_rms: float
    jacobian_condition: float | None
    warnings: tuple[str,...]


def _template(psf, reference):
    img=np.asarray(psf,dtype=float)
    return PSFTemplate(
        image=img,cube=img[:,:,None],spectral_fractions=np.array([1.]),
        pixel_scale_mas=reference.pixel_scale_mas,
        provenance=reference.provenance,
    )


def _mad_scale(values):
    centre=float(np.median(values))
    scatter=float(1.4826*np.median(np.abs(values-centre)))
    return scatter if np.isfinite(scatter) and scatter>0 else 1e-5


def fit_hybrid_ao(
    observed, basis: FrozenAOBasis, template: PSFTemplate,
    *, initial_positions_xy_px, initial_fluxes=None,
    initial_background=None, valid_mask=None, sigma=None,
    active_parameters=("r0",), allow_exploratory_gtt=False,
    prior_sigmas=None, n_alternations=2,
    max_nfev=35, stellar_max_nfev=150,
    max_position_offset_px=3.,
):
    """Alternate bounded AO and flux/astrometry/background fits.

    r0 is in metres at the saved atmosphere wavelength. HO/TT are
    dimensionless scaling factors of their existing phase SF (not
    independently estimated optical sensor gains).
    """
    parity=validate_ao_basis_parity(basis)
    data=np.asarray(observed,dtype=float)
    if data.shape!=basis.reference_psf.shape:
        raise ValueError("Science image and AO basis detector shapes differ")
    if template.image.shape!=data.shape or not np.allclose(
        template.image,basis.reference_psf,rtol=1e-12,atol=1e-12
    ):
        raise ValueError("Fitting template must equal nominal AO basis PSF")
    active=tuple(active_parameters)
    if not active or len(set(active))!=len(active) or any(
        p not in PARAMETERS for p in active
    ):
        raise ValueError("Choose distinct r0, gho, gtt AO parameters")
    if "gtt" in active and not allow_exploratory_gtt:
        raise ValueError("gtt requires explicit exploratory STRAP opt-in")
    if (not isinstance(n_alternations,int) or n_alternations<1
        or not isinstance(max_nfev,int) or max_nfev<1
        or not isinstance(stellar_max_nfev,int) or stellar_max_nfev<1):
        raise ValueError("Optimization iteration limits must be positive")
    priors=PRIOR_SIGMAS.copy()
    if prior_sigmas is not None:
        priors.update(prior_sigmas)
    if any(not np.isfinite(priors[p]) or priors[p]<=0 for p in active):
        raise ValueError("All active Gaussian priors must be finite and positive")
    starts=np.asarray(initial_positions_xy_px,dtype=float)
    if starts.ndim!=2 or starts.shape[1]!=2 or starts.shape[0]<1:
        raise ValueError("Provide at least one [x,y] stellar position")

    valid=np.isfinite(data)
    if valid_mask is not None:
        v=np.asarray(valid_mask)
        if v.shape!=data.shape or v.dtype.kind!="b":
            raise ValueError("valid_mask must be a boolean science image")
        valid &= v
    if sigma is not None:
        sigma=np.asarray(sigma,dtype=float)
        if sigma.shape!=data.shape:
            raise ValueError("sigma must match detector image")
        valid &= np.isfinite(sigma)&(sigma>0)
    if valid.sum()<8+4*len(starts):
        raise ValueError("Insufficient valid pixels for the hybrid fit")

    fixed={"r0":float(basis.r0_m),"gho":1.,"gtt":1.}
    lower,upper=[],[]
    for p in active:
        lo,hi=BOUNDS[p]
        if p=="r0":
            lo*=basis.r0_m
            hi*=basis.r0_m
        lower.append(lo)
        upper.append(hi)
    lower=np.asarray(lower)
    upper=np.asarray(upper)
    current=np.asarray([fixed[p] for p in active],dtype=float)

    def values(x):
        result=fixed.copy()
        result.update(zip(active,map(float,x)))
        return result

    def psf_for(x):
        p=values(x)
        return render_ao_psf(
            basis,r0_m=p["r0"],gho=p["gho"],gtt=p["gtt"]
        )

    def image_for(x,star):
        return render_stellar_scene(
            _template(psf_for(x),template),
            positions_xy_px=star.positions_xy_px,
            fluxes=star.fluxes,background=star.background,
        )

    stellar_kw=dict(
        valid_mask=valid,sigma=sigma,
        max_position_offset_px=max_position_offset_px,
        max_nfev=stellar_max_nfev,
    )
    baseline=fit_stellar_scene(
        data,template,initial_positions_xy_px=starts,
        initial_fluxes=initial_fluxes,
        initial_background=initial_background,**stellar_kw,
    )
    if not baseline.converged:
        raise RuntimeError("Baseline stellar-only fit did not converge")
    if sigma is None:
        empirical=_mad_scale(baseline.residual[valid])
        weights=np.full(data.shape,1/empirical)
        noise_source="empirical_residual_MAD_not_calibrated"
    else:
        empirical=None
        weights=np.zeros(data.shape)
        weights[valid]=1/sigma[valid]
        noise_source="user_supplied_sigma"

    def prior_residual(x):
        values_out=[]
        for name,value in zip(active,x):
            p=(np.log(value/basis.r0_m) if name=="r0" else value-1.)
            values_out.append(p/priors[name])
        return np.asarray(values_out)

    stellar=baseline
    iterations=[]
    for i in range(n_alternations):
        star_fixed=stellar
        def cost(x):
            model=image_for(x,star_fixed)
            return np.r_[((model-data)*weights)[valid],prior_residual(x)]
        sol=least_squares(
            cost,current,bounds=(lower,upper),
            max_nfev=max_nfev,method="trf",
            x_scale="jac",ftol=1e-9,xtol=1e-9,gtol=1e-9,
        )
        if not sol.success:
            raise RuntimeError("AO optimizer stopped without convergence: "+str(sol.message))
        current=sol.x.copy()
        stellar=fit_stellar_scene(
            data,_template(psf_for(current),template),
            initial_positions_xy_px=star_fixed.positions_xy_px,
            initial_fluxes=star_fixed.fluxes,
            initial_background=star_fixed.background,**stellar_kw,
        )
        if not stellar.converged:
            raise RuntimeError("Stellar refit did not converge after AO update")
        iterations.append({
            "alternation":i+1,"nfev":int(sol.nfev),
            "cost_with_priors":float(sol.cost),
            "rms":float(stellar.rms),
            "ao":values(current),
        })

    final=stellar.model.copy()
    residual=data-final
    # Diagnostic: shape-only Jacobian at fixed stellar parameters,
    # excluding prior regularisation so degeneracy is not concealed.
    gradients=[]
    for j,p in enumerate(active):
        step=1e-3*(basis.r0_m if p=="r0" else 1.)
        lo=current.copy()
        hi=current.copy()
        lo[j]=max(lower[j],current[j]-step)
        hi[j]=min(upper[j],current[j]+step)
        if hi[j]<=lo[j]:
            gradients.append(np.zeros(int(valid.sum())))
        else:
            gradients.append(
                (((image_for(hi,stellar)-image_for(lo,stellar))*weights)[valid]/
                 (hi[j]-lo[j]))
            )
    condition=None
    warnings=[]
    if len(active)>1:
        jac=np.stack(gradients,axis=1)
        norm=np.linalg.norm(jac,axis=0)
        if (norm<=0).any():
            condition=float("inf")
        else:
            singular=np.linalg.svd(jac/norm[None,:],compute_uv=False)
            condition=float(singular[0]/singular[-1]) if singular[-1]>0 else float("inf")
        if condition>100:
            warnings.append("AO shape Jacobian nearly degenerate; joint parameter fit ambiguous")
    for j,p in enumerate(active):
        if (abs(current[j]-lower[j])<1e-6*(upper[j]-lower[j])
                or abs(current[j]-upper[j])<1e-6*(upper[j]-lower[j])):
            warnings.append(p+" at fit boundary; result is not calibrated")
    if "gtt" in active:
        warnings.append("Exploratory TT scaling is NOT independent STRAP calibration")
    initial_rms=float(np.sqrt(np.mean(baseline.residual[valid]**2)))
    final_rms=float(np.sqrt(np.mean(residual[valid]**2)))
    return HybridAOFit(
        ao_parameters=values(current),active_parameters=active,
        prior_sigmas={p:float(priors[p]) for p in active},
        baseline_stellar=baseline,stellar=stellar,
        baseline_model=baseline.model.copy(),model=final,
        residual=residual.copy(),nominal_parity=parity,
        optimizer_status=tuple(iterations),
        noise_source=noise_source,empirical_noise_scale=empirical,
        initial_rms=initial_rms,final_rms=final_rms,
        jacobian_condition=condition,warnings=tuple(warnings),
    )
