"""Compare actual Keck full-complex and real-FFT factorized Vii outputs.

Identical FITS/SAV, pupil orientation and geometric-cache state are
required. Acceptance gate is full spectral science PSF at 1e-12;
intermediate HO SF strict parity is reported independently, never hidden.
No cProfile runs may be compared for absolute speed measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_keck_vii_factorized import _read, _compare


def _science_moments(psf):
    """Flux, peak and flux-weighted centroid per wavelength (pixel units)."""
    if psf.ndim!=3 or psf.size<1:
        raise ValueError("Expected a [y,x,wavelength] Keck science PSF cube")
    y,x=np.indices(psf.shape[:2],dtype=float)
    total=np.sum(psf,axis=(0,1))
    if np.any(total<=0):
        raise ValueError("Keck PSF flux must be positive")
    return {
        "integrated_flux":total.tolist(),
        "peak_flux":np.max(psf,axis=(0,1)).tolist(),
        "centroid_x_pixels":(
            np.sum(psf*x[:,:,None],axis=(0,1))/total
        ).tolist(),
        "centroid_y_pixels":(
            np.sum(psf*y[:,:,None],axis=(0,1))/total
        ).tolist(),
    }


def compare(complex_folder,real_folder,*,rtol=1e-12,atol=1e-12):
    m0,p0,psf0,ho0=_read(Path(complex_folder),"factorized")
    m1,p1,psf1,ho1=_read(Path(real_folder),"factorized_real")
    same=(
        "input_fits","input_sav",
        "p3_pupil_rotation_hypothesis","p3_pupil_resampling",
        "p3_pupil_parity","focal_anisoplanatism_algorithm",
        "focal_cache_configured",
    )
    changed=[k for k in same if m0.get(k)!=m1.get(k)]
    if changed:
        raise ValueError("Different Keck science or pupil settings: "+", ".join(changed))
    if p0.get("cprofile_enabled") or p1.get("cprofile_enabled"):
        raise ValueError("Disable cProfile for valid timing comparisons")
    cache0=p0.get("focal_geometric_kernel",{})
    cache1=p1.get("focal_geometric_kernel",{})
    if (cache0.get("cache_status")!=cache1.get("cache_status")
            or cache0.get("cache_key")!=cache1.get("cache_key")):
        raise ValueError("Cold/warm focal cache mismatch biases FFT speedup")
    full=_compare(psf0,psf1,rtol=rtol,atol=atol)
    ho=_compare(ho0,ho1,rtol=rtol,atol=atol)
    if psf0.ndim!=3 or psf1.ndim!=3:
        raise ValueError("Science PSFs must be wavelength cubes [y,x,wavelength]")
    moments0=_science_moments(psf0)
    moments1=_science_moments(psf1)
    differences={
        k:(np.asarray(moments1[k])-np.asarray(moments0[k])).tolist()
        for k in moments0
    }
    stats0=p0["psfr_stage_timings_ms"]
    stats1=p1["psfr_stage_timings_ms"]
    t0=float(stats0["ho_sf"])*1e-3
    t1=float(stats1["ho_sf"])*1e-3
    init0=float(stats0["psfr_init_total"])*1e-3
    init1=float(stats1["psfr_init_total"])*1e-3
    return {
        "kind":"real_Keck_factorized_real_fft_vs_factorized_complex",
        "rtol":rtol,"atol":atol,
        "science_psf_accepted":bool(full["parity_passed"]),
        "ho_sf_strict_parity_passed":bool(ho["parity_passed"]),
        "ho_sf_numerical_warning":(
            None if ho["parity_passed"] else
            "Intermediate HO SF exceeds strict 1e-12; retain diagnostics. "
            "This does not override a successful final science PSF parity gate."
        ),
        "science_psf":full,
        "ho_structure_function":ho,
        "science_psf_observables_complex":moments0,
        "science_psf_observables_real":moments1,
        "science_psf_observable_differences":differences,
        "ho_vii_real_fft_diagnostics":p1.get("ho_vii_kernel",{}),
        "focal_cache_status":cache0.get("cache_status"),
        "ho_sf_seconds":{"factorized_complex":t0,"factorized_real":t1},
        "ho_sf_speedup":t0/t1,
        "psfr_init_seconds":{"factorized_complex":init0,"factorized_real":init1},
        "psfr_init_speedup":init0/init1,
        "caution":"One acquisition only. Validate pupil rotations and repeat "
                  "timings before changing default reconstruction settings.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--complex",type=Path,required=True)
    p.add_argument("--real",type=Path,required=True)
    p.add_argument("--output",type=Path,
                   default=Path("results/compare_keck_vii_real_fft.json"))
    args=p.parse_args()
    report=compare(args.complex,args.real)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps(report,indent=2,allow_nan=False))
    if not report["science_psf_accepted"]:
        raise SystemExit(
            "FAIL: final science PSF parity to 1e-12. "
            "Keep factorized_real experimental and disabled."
        )
    if not report["ho_sf_strict_parity_passed"]:
        print("WARNING: intermediate HO SF strict parity failed; "
              "final science PSF passed. See the separate numerical report.")


if __name__=="__main__":
    main()
