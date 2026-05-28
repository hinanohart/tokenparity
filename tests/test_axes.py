"""Tests for the three evaluation axes (A, B, C)."""

from __future__ import annotations

import numpy as np
import pytest

from tokenparity.axes.ib import information_bottleneck
from tokenparity.axes.reconstruction import (
    average_rank,
    normalize_within_domain,
    reconstruction_error,
)
from tokenparity.axes.transfer import aggregate_transferability, transferability

# ---------------------------------------------------------------------------
# Axis A: Downstream Transferability
# ---------------------------------------------------------------------------


def test_transferability_basic() -> None:
    """transferability(0.8, 0.5) must return 0.3."""
    result = transferability(0.8, 0.5)
    assert abs(result - 0.3) < 1e-9, f"Expected 0.3, got {result}"


def test_transferability_negative_allowed() -> None:
    """Frozen features may outperform native; negative gap is valid."""
    assert transferability(0.5, 0.8) == pytest.approx(-0.3)


def test_transferability_zero_gap() -> None:
    assert transferability(0.7, 0.7) == pytest.approx(0.0)


def test_aggregate_transferability_median() -> None:
    scores = [(0.9, 0.6), (0.8, 0.5), (0.7, 0.4)]  # all gaps = 0.3
    result = aggregate_transferability(scores)
    assert result["median"] == pytest.approx(0.3)
    assert len(result["per_sample"]) == 3


def test_aggregate_transferability_empty() -> None:
    result = aggregate_transferability([])
    assert result["per_sample"] == []
    assert result["median"] != result["median"]  # nan check


# ---------------------------------------------------------------------------
# Axis B: Information Bottleneck
# ---------------------------------------------------------------------------


def test_ib_rank_only_for_high_dim_z() -> None:
    """Z with >64 dims must set rank_only=True and suppress absolute values."""
    rng = np.random.default_rng(0)
    n = 50
    x = rng.standard_normal((n, 4))
    z = rng.standard_normal((n, 65))  # >64 dims
    y = rng.standard_normal((n, 2))

    result = information_bottleneck(x, z, y)
    assert result["rank_only"] is True
    assert result["ib_median"] is None
    assert result["ci_low"] is None
    assert result["ci_high"] is None
    for k, v in result["ib_per_k"].items():
        assert v != v, f"Expected nan for k={k}, got {v}"  # nan != nan


def test_ib_low_dim_returns_values() -> None:
    """Low-dim Z (<= 64) must return non-None median and CI."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal((n, 2))
    z = x + 0.1 * rng.standard_normal((n, 2))
    y = x[:, :1] + 0.3 * rng.standard_normal((n, 1))

    result = information_bottleneck(x, z, y)
    assert result["rank_only"] is False
    assert result["ib_median"] is not None
    assert result["ci_low"] is not None
    assert result["ci_high"] is not None
    assert result["ci_low"] <= result["ci_high"]


def test_ib_exactly_64_dim_not_rank_only() -> None:
    """Z with exactly 64 dims must NOT trigger rank_only."""
    rng = np.random.default_rng(2)
    n = 100
    x = rng.standard_normal((n, 4))
    z = rng.standard_normal((n, 64))
    y = rng.standard_normal((n, 2))
    result = information_bottleneck(x, z, y)
    assert result["rank_only"] is False


# ---------------------------------------------------------------------------
# Axis C: Reconstruction
# ---------------------------------------------------------------------------


def test_reconstruction_error_mse() -> None:
    """MSE of identical arrays must be 0."""
    arr = np.ones((10, 5))
    assert reconstruction_error(arr, arr) == pytest.approx(0.0)


def test_reconstruction_error_known_value() -> None:
    orig = np.array([0.0, 0.0, 0.0, 0.0])
    recon = np.array([1.0, 1.0, 1.0, 1.0])
    assert reconstruction_error(orig, recon) == pytest.approx(1.0)


def test_reconstruction_error_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        reconstruction_error(np.ones((3, 4)), np.ones((4, 3)))


def test_normalize_within_domain_mean_and_std() -> None:
    """Normalized values must have mean ≈ 0 and std ≈ 1."""
    errors = {"tok_a": 1.0, "tok_b": 2.0, "tok_c": 3.0, "tok_d": 4.0}
    normalized = normalize_within_domain(errors)
    values = list(normalized.values())
    assert abs(np.mean(values)) < 1e-9
    assert abs(np.std(values) - 1.0) < 1e-9


def test_normalize_within_domain_empty() -> None:
    assert normalize_within_domain({}) == {}


def test_normalize_within_domain_all_equal() -> None:
    """All-equal errors → all-zero z-scores (std=0 guard)."""
    errors = {"a": 5.0, "b": 5.0, "c": 5.0}
    normalized = normalize_within_domain(errors)
    for v in normalized.values():
        assert v == pytest.approx(0.0)


def test_average_rank_basic() -> None:
    """Rank averaging across two domains."""
    per_domain = {
        "protein": {"tok_a": 0.1, "tok_b": 0.5},
        "video": {"tok_a": 0.8, "tok_b": 0.2},
    }
    ranks = average_rank(per_domain)
    # protein: tok_a=rank1, tok_b=rank2
    # video:   tok_b=rank1, tok_a=rank2
    # tok_a avg = (1+2)/2 = 1.5; tok_b avg = (2+1)/2 = 1.5
    assert ranks["tok_a"] == pytest.approx(1.5)
    assert ranks["tok_b"] == pytest.approx(1.5)


def test_average_rank_single_domain() -> None:
    per_domain = {"robot": {"tok_a": 2.0, "tok_b": 1.0}}
    ranks = average_rank(per_domain)
    assert ranks["tok_b"] < ranks["tok_a"]


def test_average_rank_empty() -> None:
    assert average_rank({}) == {}
