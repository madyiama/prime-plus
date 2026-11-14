"""Public optimized PSF-R defaults and explicit reference compatibility.

These checks keep the Keck PSF Lab deployment from accidentally falling back
to a slow numerical method or relaxing physical covariance / TT conventions.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from p3.psfr.psfR import psfR
from p3.aoSystem.frequencyDomain import frequencyDomain

ROOT = Path(__file__).resolve().parents[1]


def _cli_arg_default(filename: str, option: str):
    """Read the parser's actual literal default without running FITS/SAV I/O."""
    tree = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
    candidates = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == option
    ]
    assert len(candidates) == 1
    defaults = [kw.value for kw in candidates[0].keywords if kw.arg == "default"]
    assert len(defaults) == 1
    return ast.literal_eval(defaults[0])


@pytest.mark.parametrize("cls", [psfR, frequencyDomain])
def test_public_psfr_constructors_enable_exact_fast_focal_by_default(cls):
    params = inspect.signature(cls.__init__).parameters
    assert params["focal_algorithm"].default == "quadrant"
    # None resolves to 4 only with quadrant, 1 for legacy/symmetry.
    assert params["focal_workers"].default is None
    assert params["focal_cache_dir"].default is None


def test_ho_default_and_physical_covariance_contract_unchanged():
    params = inspect.signature(psfR.__init__).parameters
    assert params["ho_vii_method"].default == "factorized_real"
    assert params["ho_covariance_policy"].default == "strict"
    assert params["tt_covariance_mode"].default == "legacy_psfr"
    assert params["tt_mirror_model"].default == "unity"
    assert params["tt_kernel_convention"].default == "noll_angular"


@pytest.mark.parametrize("filename", [
    "keck_n0013_smoke.py",
    "keck_psfr_v1_e2e.py",
])
def test_keck_cli_defaults_match_public_psfr_constructor(filename):
    assert _cli_arg_default(filename, "--focal-algorithm") == "quadrant"
    assert _cli_arg_default(filename, "--ho-vii-method") == "factorized_real"
    # Lazy workers sentinel lets --focal-algorithm legacy select one worker
    # without forcing an additional CLI flag.
    assert _cli_arg_default(filename, "--focal-workers") is None
