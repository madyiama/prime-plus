"""Audit-only vectorization of the Fourier core of P3's 4-D PSF."""
import numpy as np
import pytest

from scripts.bench_psfr_otf4d_loops import (
    benchmark,otf4d_reference,otf4d_vectorized,
)


@pytest.mark.parametrize("n_sources,n_wvl",[(1,1),(1,3),(2,3),(3,6)])
def test_vectorized_fourier_4d_matches_per_wavelength_loop(n_sources,n_wvl):
    n=16
    rng=np.random.default_rng(310+n_sources+n_wvl)
    sf=np.abs(rng.normal(size=(n,n,n_sources)))
    stat=rng.uniform(.02,1.,size=(n,n,n_wvl))
    phasor=np.exp(1j*rng.normal(
        scale=.5,size=(n,n,n_sources,n_wvl)
    ))
    waves=np.linspace(1.6e-6,2.3e-6,n_wvl)
    baseline=otf4d_reference(sf,stat,phasor,waves)
    candidate=otf4d_vectorized(sf,stat,phasor,waves)
    assert baseline.shape==(n,n,n_sources,n_wvl)
    np.testing.assert_allclose(candidate,baseline,rtol=1e-12,atol=1e-12)


def test_4d_vectorization_benchmark_measures_time_ram_and_parity():
    result=benchmark(n=24,n_src=1,n_wvl=3,trials=2)
    assert result["parity_passed"]
    assert result["shape"]==[24,24,1,3]
    assert result["results"]["wavelength_loop"]["median_s"]>0
    assert result["results"]["all_wavelengths"]["median_s"]>0
    assert result["input_wavelength_cube_bytes"]>0
    assert result["results"]["wavelength_loop"]["tracemalloc_peak_bytes"]>0
    assert "getStaticOTF" in result["not_included"]


def test_4d_vectorization_rejects_invalid_geometry():
    n=8
    sf=np.zeros((n,n,1))
    stat=np.ones((n,n,3))
    phasor=np.ones((n,n,1,3),dtype=complex)
    wave=np.array([1.9e-6,2.e-6,2.1e-6])
    np.testing.assert_array_equal(
        otf4d_reference(sf,stat,phasor,wave),
        otf4d_vectorized(sf,stat,phasor,wave),
    )
    with pytest.raises(ValueError,match="Invalid"):
        otf4d_vectorized(sf,stat,phasor,wave[:2])
    with pytest.raises(ValueError,match="Invalid"):
        otf4d_vectorized(sf,stat,phasor,np.array([1.,0.,3.]))
