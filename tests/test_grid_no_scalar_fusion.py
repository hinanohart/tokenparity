"""Dedicated tests confirming scalar fusion is absent from GridResult and Engine."""

from __future__ import annotations

import numpy as np
import pytest

from tokenparity.engine import Engine, GridResult

# Exhaustive banned name list — CI-enforced mirror of architecture.md
_BANNED_METHODS = [
    "total_score",
    "aggregate",
    "combined_score",
    "overall_score",
    "scalar_score",
    "final_score",
    "fused_score",
    "score",
    "merge_axes",
    "collapse",
]


@pytest.mark.parametrize("method_name", _BANNED_METHODS)
def test_grid_result_does_not_have_banned_method(method_name: str) -> None:
    """GridResult must not expose any scalar-fusion method."""
    assert not hasattr(GridResult, method_name), (
        f"GridResult.{method_name} found — scalar fusion is machine-banned"
    )


@pytest.mark.parametrize("method_name", _BANNED_METHODS)
def test_engine_does_not_have_banned_method(method_name: str) -> None:
    """Engine must not expose any scalar-fusion method."""
    assert not hasattr(Engine, method_name), (
        f"Engine.{method_name} found — scalar fusion is machine-banned"
    )


def test_grid_result_axes_always_three() -> None:
    """GridResult.axes must always be exactly ['A', 'B', 'C']."""
    gr = GridResult(
        domains=["protein"],
        tokenizers=["tok"],
        axes=["A", "B", "C"],
        values=np.zeros((1, 1, 3)),
    )
    assert gr.axes == ["A", "B", "C"]


def test_grid_result_values_shape_matches_dims() -> None:
    """Values shape must be (n_domains, n_tokenizers, 3)."""
    n_d, n_t = 2, 3
    gr = GridResult(
        domains=["protein", "video"],
        tokenizers=["a", "b", "c"],
        values=np.zeros((n_d, n_t, 3)),
    )
    assert gr.values.shape == (n_d, n_t, 3)


def test_to_radar_shape_correct() -> None:
    """to_radar must return a dict per tokenizer with exactly 3 axis keys."""
    gr = GridResult(
        domains=["protein"],
        tokenizers=["tok_a", "tok_b"],
        axes=["A", "B", "C"],
        values=np.full((1, 2, 3), 0.5),
    )
    engine = Engine()
    radar = engine.to_radar(gr)
    assert set(radar.keys()) == {"tok_a", "tok_b"}
    for v in radar.values():
        assert set(v.keys()) == {"A", "B", "C"}


def test_to_pareto_front_subset_of_tokenizers() -> None:
    """Pareto front must be a subset of registered tokenizer names."""
    gr = GridResult(
        domains=["protein"],
        tokenizers=["tok_a", "tok_b"],
        axes=["A", "B", "C"],
        values=np.array([[[0.1, 0.5, 0.2], [0.3, 0.4, 0.1]]]),
    )
    engine = Engine()
    front = engine.to_pareto_front(gr)
    assert set(front).issubset({"tok_a", "tok_b"})


def test_to_pareto_front_all_nan_returns_all() -> None:
    """With all-NaN values, all tokenizers should be non-dominated."""
    gr = GridResult(
        domains=["protein"],
        tokenizers=["tok_a", "tok_b"],
        axes=["A", "B", "C"],
        values=np.full((1, 2, 3), float("nan")),
    )
    engine = Engine()
    front = engine.to_pareto_front(gr)
    # With no finite comparisons, none is dominated
    assert set(front) == {"tok_a", "tok_b"}
