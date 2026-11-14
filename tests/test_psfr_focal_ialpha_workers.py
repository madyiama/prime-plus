"""PR47: CPU worker parallelism is elementwise, opt-in and deterministic."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import numpy as np
import pytest

import p3.aoSystem.anisoplanatismModel as aniso
from p3.aoSystem.focalFast import parallel_ialpha, direct_focal_quadrant_map


def case(l0=25.):
    tel = NS(D=10.5)
    atm = NS(nL=3, heights=np.array([0., 4500., 12000.]), L0=l0)
    src = NS(nSrc=2, direction=np.array([
        [1.3e-5, -2.2e-5], [-0.6e-5, 1.1e-5]
    ]))
    gs = NS(height=np.array([90000.]),
            direction=np.array([[0.], [0.]]))
    return tel, atm, src, gs


def evaluate(workers, *, l0=25., nact=4, filter_=1, cache_dir=None):
    tel, atm, src, gs = case(l0)
    diag = {}
    sf = aniso.focal_angular_function(
        tel, atm, src, gs, nact, 40, 2.,
        covariance_algorithm="quadrant", focal_workers=workers,
        h_filt=filter_, focal_cache_dir=cache_dir,
        focal_diagnostics=diag,
    )
    return sf, diag


@pytest.mark.parametrize("workers", [2, 4])
def test_ialpha_parallel_exact_identity_with_mixed_broadcast(workers):
    x = np.linspace(-3., 3., 240)[:, None]
    y = np.linspace(-2., 2., 240)[None, :]
    reference = parallel_ialpha(x, y, workers=1)
    diag = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        test = parallel_ialpha(
            x, y, executor=pool, workers=workers,
            min_chunk_elements=1024, diagnostics=diag)
    # Same scipy.special.kv ufunc at each identical float64 input.
    np.testing.assert_array_equal(test, reference)
    assert diag["ialpha_parallel_calls"] == 1


@pytest.mark.parametrize("workers", [2, 4])
@pytest.mark.parametrize("l0", [25., 40., np.inf])
def test_complete_focal_sf_is_worker_independent(workers, l0):
    # nact=8 ensures >= 4*4096 radial elements for four-way chunking.
    serial, sd = evaluate(1, l0=l0, nact=8)
    parallel, pd = evaluate(workers, l0=l0, nact=8)
    np.testing.assert_allclose(parallel, serial, rtol=1e-12, atol=1e-12)
    assert pd["ialpha_workers_requested"] == workers
    assert pd["quadrant_direct"] is True
    assert pd["full_covariance_matrix_allocated"] is False
    if np.isposinf(l0):
        assert pd["ialpha_workers_effective"] == 1
        assert pd["ialpha_parallel_calls"] == 0
    else:
        assert pd["ialpha_workers_effective"] == workers
        assert pd["ialpha_parallel_calls"] > 0
        assert pd["ialpha_serial_calls"] >= 0


@pytest.mark.parametrize("workers", [2, 4])
def test_full_direct_map_same_float64_output_with_threads(workers):
    pupil = np.linspace(-5.25, 5.25, 31)
    opts = dict(
        cone_fraction=.21,
        theta_shift_x_m=.7,
        theta_shift_y_m=-.3,
        finite_outer_scale_m=25.,
    )
    ref = direct_focal_quadrant_map(pupil, **opts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        got = direct_focal_quadrant_map(
            pupil, workers=workers, executor=pool, **opts)
    np.testing.assert_array_equal(got, ref)


def test_small_ialpha_arrays_use_sequential_fallback():
    x = np.linspace(-.1, .1, 8)[:, None]
    y = np.linspace(-.3, .3, 8)[None, :]
    diag = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = parallel_ialpha(x, y, workers=4, executor=pool,
                             diagnostics=diag)
    np.testing.assert_array_equal(got, parallel_ialpha(x, y))
    assert diag["ialpha_serial_calls"] == 1
    assert diag.get("ialpha_parallel_calls", 0) == 0


@pytest.mark.parametrize("method", ["legacy", "symmetry"])
def test_no_workers_for_unchanged_reference_paths(method):
    tel, atm, src, gs = case()
    with pytest.raises(ValueError, match="only with focal quadrant"):
        aniso.focal_angular_function(
            tel, atm, src, gs, 2, 32, 2.,
            covariance_algorithm=method, focal_workers=2)


@pytest.mark.parametrize("invalid", [0, 3, 5, -1, True, 2.0, "4"])
def test_worker_count_strict_validation(invalid):
    tel, atm, src, gs = case()
    with pytest.raises(ValueError, match="must be 1, 2 or 4"):
        aniso.focal_angular_function(
            tel, atm, src, gs, 2, 32, 2.,
            covariance_algorithm="quadrant", focal_workers=invalid)


def test_non_scalar_filter_fallback_is_same_as_symmetry():
    tel, atm, src, gs = case()
    h = np.eye(49)*.9
    expected = aniso.focal_angular_function(
        tel, atm, src, gs, 3, 40, 2., h_filt=h,
        covariance_algorithm="symmetry")
    report = {}
    actual = aniso.focal_angular_function(
        tel, atm, src, gs, 3, 40, 2., h_filt=h,
        covariance_algorithm="quadrant", focal_workers=4,
        focal_diagnostics=report)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-11)
    assert report["quadrant_filtered_fallback"] is True
    assert report["ialpha_workers_effective"] == 1
    assert report["ialpha_parallel_calls"] == 0


def test_cache_key_same_for_worker_counts_and_no_compute_on_hit(tmp_path):
    initial, d1 = evaluate(1, nact=3, cache_dir=tmp_path)
    replay, d2 = evaluate(4, nact=3, cache_dir=tmp_path)
    np.testing.assert_array_equal(replay, initial)
    assert d1["cache_status"] == "miss"
    assert d2["cache_status"] == "hit"
    assert d1["cache_key"] == d2["cache_key"]
    assert d2["ialpha_parallel_calls"] == 0
    assert d2["ialpha_workers_effective"] == 0


def test_no_threadpool_for_exact_infinite_outer_scale(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No Ialpha or worker pool for L0=infinity")
    monkeypatch.setattr(aniso, "ThreadPoolExecutor", forbidden)
    got, report = evaluate(4, l0=np.inf, nact=3)
    assert np.isfinite(got).all()
    assert report["ialpha_workers_effective"] == 1
    assert report["ialpha_parallel_calls"] == 0
