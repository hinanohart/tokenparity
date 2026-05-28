"""Tests for the self-implemented Kraskov-1 MI estimator.

Analytic ground truths used:
  - Joint bivariate Gaussian with correlation ρ:
      I(X; Y) = -0.5 * log(1 - ρ²)  [nats]
  - Two independent Gaussians:
      I(X; Y) = 0
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tokenparity._estimators.kraskov import mi

RNG = np.random.default_rng(12345)
N = 2000
RHO = 0.5
ANALYTIC_MI = -0.5 * math.log(1 - RHO**2)  # ≈ 0.1438 nats


def _correlated_gaussian(rho: float, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) from a bivariate standard normal with correlation rho."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rho * x + math.sqrt(1 - rho**2) * rng.standard_normal(n)
    return x.reshape(-1, 1), y.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Correlated Gaussian: MI should be close to analytic value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 5, 7])
def test_correlated_gaussian_close_to_analytic(k: int) -> None:
    """KSG-1 estimate of I(X;Y) for ρ=0.5 Gaussian must be within ±0.05 nats."""
    x, y = _correlated_gaussian(RHO, N, seed=k)
    estimated = mi(x, y, k=k)
    assert abs(estimated - ANALYTIC_MI) < 0.05, (
        f"k={k}: estimated={estimated:.4f}, analytic={ANALYTIC_MI:.4f}, "
        f"diff={abs(estimated - ANALYTIC_MI):.4f}"
    )


# ---------------------------------------------------------------------------
# Independent Gaussian: MI should be close to 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 5, 7])
def test_independent_gaussian_near_zero(k: int) -> None:
    """I(X;Y) for independent Gaussians must be within ±0.05 nats of 0."""
    rng = np.random.default_rng(999 + k)
    x = rng.standard_normal((N, 1))
    y = rng.standard_normal((N, 1))
    estimated = mi(x, y, k=k)
    assert estimated < 0.05, (
        f"k={k}: independent Gaussians, estimated MI={estimated:.4f} (expected ≈ 0)"
    )


# ---------------------------------------------------------------------------
# Output must be non-negative (clipped)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 5, 7])
def test_output_is_non_negative(k: int) -> None:
    """mi() must never return a negative value."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((200, 1))
    y = rng.standard_normal((200, 1))
    result = mi(x, y, k=k)
    assert result >= 0.0, f"mi() returned negative value {result} for k={k}"


# ---------------------------------------------------------------------------
# Higher correlation → higher MI
# ---------------------------------------------------------------------------


def test_higher_rho_gives_higher_mi() -> None:
    """MI should increase monotonically with correlation strength."""
    mi_low = mi(*_correlated_gaussian(0.2, N, seed=1), k=5)
    mi_high = mi(*_correlated_gaussian(0.8, N, seed=1), k=5)
    assert mi_high > mi_low, f"Expected MI(ρ=0.8) > MI(ρ=0.2); got {mi_high:.4f} vs {mi_low:.4f}"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_mismatched_samples_raises() -> None:
    """Different sample counts must raise ValueError."""
    x = np.ones((100, 1))
    y = np.ones((200, 1))
    with pytest.raises(ValueError, match="same number of samples"):
        mi(x, y, k=3)


def test_k_too_large_raises() -> None:
    """k >= N must raise ValueError."""
    x = np.ones((10, 1))
    y = np.ones((10, 1))
    with pytest.raises(ValueError, match="less than N"):
        mi(x, y, k=10)


# ---------------------------------------------------------------------------
# 1-D input (no explicit reshape)
# ---------------------------------------------------------------------------


def test_1d_input_accepted() -> None:
    """1-D arrays should be accepted and produce a valid result."""
    rng = np.random.default_rng(77)
    x = rng.standard_normal(300)
    y = rng.standard_normal(300)
    result = mi(x, y, k=3)
    assert isinstance(result, float)
    assert result >= 0.0
