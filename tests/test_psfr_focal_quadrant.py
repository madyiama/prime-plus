"""PR46: direct negative-lag Flicker quadrant vs full symmetry reference.

The original orientation is deliberately asymmetric in x/y: never infer
that pupil pair indices are interchangeable without checking the exact
vector_to_separation convention.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

import p3.aoSystem.anisoplanatismModel as aniso
from p3.aoSystem.focalFast import direct_focal_quadrant_map


def scene(l0=25.0, *, nsrc=2):
    tel = NS(D=10.5)
    atm = NS(nL=3, heights=np.array([0., 4500., 12000.]), L0=l0)
    directions = np.array([[1.0e-5, -1.5e-5], [-2.0e-5, 1.1e-5]])
    src = NS(nSrc=nsrc, direction=directions[:, :nsrc])
    gs = NS(height=np.array([90000.]),
            direction=np.array([[0.], [0.]]))
    return tel, atm, src, gs


def calc(tel, atm, src, gs, algorithm, *, nact=3, n_otf=40,
         grid="legacy", output="otf", filt=1, cache=None):
    diag = {}
    data = aniso.focal_angular_function(
        tel, atm, src, gs, np.array([nact]), n_otf, 2.,
        covariance_algorithm=algorithm, pupil_grid_mode=grid,
        output_grid=output, h_filt=filt, focal_cache_dir=cache,
        focal_diagnostics=diag,
    )
    return data, diag


@pytest.mark.parametrize("l0", [25., 40., np.inf])
@pytest.mark.parametrize("grid", ["legacy", "physical_diameter"])
@pytest.mark.parametrize("output", ["native", "otf"])
def test_flicker_quadrant_matches_exact_full_symmetry(l0, grid, output):
    tel, atm, src, gs = scene(l0)
    reference, _ = calc(tel, atm, src, gs, "symmetry",
                        grid=grid, output=output)
    fast, diag = calc(tel, atm, src, gs, "quadrant",
                      grid=grid, output=output)
    np.testing.assert_allclose(fast, reference, rtol=1e-12, atol=2e-11)
    assert diag["quadrant_direct"] is True
    assert diag["full_covariance_matrix_allocated"] is False
    assert diag["n_computed_layers"] == 2
    assert diag["grouping_ms"] == 0


@pytest.mark.parametrize("thx,thy", [(0., 0.), (.35, -.21), (-1.2, .6)])
@pytest.mark.parametrize("n", [3, 5, 9])
def test_direct_unweighted_native_quadrant_matches_legacy_and_symmetry(
    thx, thy, n,
):
    u = np.linspace(-5.25, 5.25, n)
    direct = direct_focal_quadrant_map(
        u, cone_fraction=.23, theta_shift_x_m=thx,
        theta_shift_y_m=thy, finite_outer_scale_m=25.,
    )
    tel, atm, src, gs = scene()
    # Compare equivalent full geometric computation at the same axis,
    # with n=2*nActu+1 where n is odd.
    tel.D = 10.5
    atm.nL = 2
    atm.heights = np.array([0., 90000.*.23])
    src.nSrc = 1
    src.direction = np.array([[thx/(90000.*.23)], [thy/(90000.*.23)]])
    full, _ = calc(tel, atm, src, gs, "symmetry",
                   nact=(n-1)//2, output="native",
                   grid="physical_diameter")
    fast, _ = calc(tel, atm, src, gs, "quadrant",
                   nact=(n-1)//2, output="native",
                   grid="physical_diameter")
    scale = 0.12184*.06*(2*np.pi)**2*25.**(5./3.)
    np.testing.assert_allclose(direct*scale, full[0, 1],
                               rtol=1e-12, atol=2e-11)
    np.testing.assert_allclose(fast, full, rtol=1e-12, atol=2e-11)


def test_scalar_quadrant_does_not_call_full_matrix_path(monkeypatch):
    tel, atm, src, gs = scene()
    def fail(*args, **kwargs):
        raise AssertionError("Allocated full symmetry path")
    monkeypatch.setattr(aniso, "symmetric_focal_layer", fail)
    monkeypatch.setattr(aniso, "grouped_matrix_to_map", fail)
    result, diag = calc(tel, atm, src, gs, "quadrant")
    assert np.isfinite(result).all()
    assert diag["quadrant_direct"] is True


def test_non_scalar_filter_strictly_falls_back_to_full_symmetry():
    tel, atm, src, gs = scene()
    n = 7
    filt = np.eye(n*n)*.9
    full, _ = calc(tel, atm, src, gs, "symmetry", filt=filt)
    fast, diag = calc(tel, atm, src, gs, "quadrant", filt=filt)
    np.testing.assert_allclose(fast, full, rtol=1e-12, atol=2e-11)
    assert diag["quadrant_direct"] is False
    assert diag["quadrant_filtered_fallback"] is True
    assert diag["full_covariance_matrix_allocated"] is True


def test_quadrant_cache_is_method_specific_and_rotation_safe(tmp_path):
    tel, atm, src, gs = scene()
    full, d0 = calc(tel, atm, src, gs, "symmetry", cache=tmp_path)
    quad, d1 = calc(tel, atm, src, gs, "quadrant", cache=tmp_path)
    np.testing.assert_allclose(quad, full, rtol=1e-12, atol=2e-11)
    assert d0["cache_status"] == d1["cache_status"] == "miss"
    assert d0["cache_key"] != d1["cache_key"]
    again, d2 = calc(tel, atm, src, gs, "quadrant", cache=tmp_path)
    assert d2["cache_status"] == "hit"
    np.testing.assert_array_equal(quad, again)
    tel.pupilAngle = 90.
    rotated, d3 = calc(tel, atm, src, gs, "quadrant", cache=tmp_path)
    assert d3["cache_status"] == "hit"
    np.testing.assert_array_equal(rotated, quad)
    src.direction[0, 0] += 1e-7
    changed, d4 = calc(tel, atm, src, gs, "quadrant", cache=tmp_path)
    assert d4["cache_status"] == "miss"
    assert d4["cache_key"] != d3["cache_key"]
    assert np.isfinite(changed).all()


def test_quadrant_no_cache_when_filtered(tmp_path):
    tel, atm, src, gs = scene()
    _, diag = calc(tel, atm, src, gs, "quadrant",
                   filt=np.eye(49), cache=tmp_path)
    assert diag["cache_status"] == "bypass_filtered"
    assert diag["quadrant_filtered_fallback"] is True
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("u", [np.array([0.]), np.array([0., 1., 3.]),
                               np.array([0., np.nan, 2.])])
def test_invalid_pupil_quadrant_input_rejected(u):
    with pytest.raises(ValueError):
        direct_focal_quadrant_map(
            u, cone_fraction=.2,
            theta_shift_x_m=0., theta_shift_y_m=0.,
            finite_outer_scale_m=25.,
        )


def test_other_focal_algorithms_remain_opt_in():
    tel, atm, src, gs = scene(nsrc=1)
    reference, _ = calc(tel, atm, src, gs, "legacy", nact=2)
    full, _ = calc(tel, atm, src, gs, "symmetry", nact=2)
    quadrant, _ = calc(tel, atm, src, gs, "quadrant", nact=2)
    np.testing.assert_allclose(full, reference, rtol=1e-12, atol=2e-11)
    np.testing.assert_allclose(quadrant, reference, rtol=1e-12, atol=2e-11)
