"""Single-worker Vii real-vs-complex benchmark and diagnostics gate."""
import numpy as np

from scripts.bench_psfr_vii_real_fft import benchmark


def test_small_synthetic_real_fft_benchmark_has_science_parity_and_ram_audit():
    report=benchmark(n_pupil=12,n_modes=5,trials=2)
    assert report["all_internal_parity_passed"]
    assert report["fft_workers"]==1
    assert report["gpu"] is False
    assert report["results"]["factorized"]["median_seconds"]>0
    assert report["results"]["factorized_real"]["median_seconds"]>0
    assert report["results"]["factorized_real"]["python_tracemalloc_peak_bytes"]>0
    assert report["results"]["factorized_real"]["vii_diagnostics"]["workers"]==1
    assert report["results"]["factorized_real"]["vii_diagnostics"]["fft_backend"]=="numpy"
    assert np.isfinite(report["speedup_real_vs_complex"])
