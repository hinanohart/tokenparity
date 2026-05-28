"""Tests for the evaluation engine."""

from __future__ import annotations

import numpy as np
import pytest

from tokenparity.core.types import Domain, Sample, Tokens
from tokenparity.engine import Engine, GridResult

# ---------------------------------------------------------------------------
# Mock adapters for testing
# ---------------------------------------------------------------------------


class _MockAdapter:
    """Minimal mock adapter satisfying TokenizerAdapter protocol."""

    def __init__(self, name: str, domain: Domain) -> None:
        self.name = name
        self.domain = domain
        self.license = "Apache-2.0"
        self.is_synthetic = True

    def encode(self, x: Sample) -> Tokens:
        payload = np.asarray(x.payload, dtype=float).flatten()
        ids = (np.abs(payload[:8]) * 10).astype(np.int64) % 32
        return Tokens(domain=self.domain, ids=ids, vocab_size=32, latents=payload[:8])

    def decode(self, tokens: Tokens) -> Sample:
        # Approximate reconstruction: map ids back to [0,1] range
        recon = tokens.ids.astype(float) / 32.0
        return Sample(domain=self.domain, payload=recon)

    def features(self, tokens: Tokens) -> np.ndarray:
        if tokens.latents is not None:
            return tokens.latents.reshape(1, -1)
        return tokens.ids.reshape(1, -1).astype(float)

    def native_score(self, sample: Sample) -> float:
        return 0.8


class _MockAdapterB(_MockAdapter):
    def native_score(self, sample: Sample) -> float:
        return 0.6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_adapters() -> tuple[_MockAdapter, _MockAdapterB]:
    return _MockAdapter("tok_a", "protein"), _MockAdapterB("tok_b", "protein")


@pytest.fixture()
def protein_samples() -> list[Sample]:
    rng = np.random.default_rng(0)
    return [Sample(domain="protein", payload=rng.standard_normal((8,))) for _ in range(5)]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_adapter(two_adapters: tuple) -> None:
    engine = Engine()
    a, b = two_adapters
    engine.register_adapter(a)
    engine.register_adapter(b)
    assert "tok_a" in engine._adapters
    assert "tok_b" in engine._adapters


# ---------------------------------------------------------------------------
# eval_grid shape
# ---------------------------------------------------------------------------


def test_eval_grid_shape(
    two_adapters: tuple,
    protein_samples: list[Sample],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """eval_grid must return values with shape (n_domains, n_tokenizers, 3)."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    engine = Engine()
    a, b = two_adapters
    engine.register_adapter(a)
    engine.register_adapter(b)

    result = engine.eval_grid({"protein": protein_samples})  # type: ignore[arg-type]

    assert isinstance(result, GridResult)
    assert result.values.shape == (1, 2, 3)  # 1 domain, 2 tokenizers, 3 axes
    assert result.domains == ["protein"]
    assert set(result.tokenizers) == {"tok_a", "tok_b"}
    assert result.axes == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Scalar fusion is BANNED
# ---------------------------------------------------------------------------


def test_grid_result_has_no_total_score() -> None:
    """GridResult must NOT have a total_score method."""
    assert not hasattr(GridResult, "total_score"), (
        "GridResult.total_score exists — scalar fusion is machine-banned"
    )


def test_grid_result_has_no_aggregate() -> None:
    """GridResult must NOT have an aggregate method."""
    assert not hasattr(GridResult, "aggregate"), (
        "GridResult.aggregate exists — scalar fusion is machine-banned"
    )


def test_grid_result_has_no_combined_score() -> None:
    """GridResult must NOT have a combined_score method."""
    assert not hasattr(GridResult, "combined_score"), (
        "GridResult.combined_score exists — scalar fusion is machine-banned"
    )


def test_grid_result_scalar_fusion_attributes_absent() -> None:
    """Check a broader set of scalar-fusion attribute names."""
    banned_names = [
        "total_score",
        "aggregate",
        "combined_score",
        "overall_score",
        "scalar_score",
        "final_score",
        "fused_score",
    ]
    for name in banned_names:
        assert not hasattr(GridResult, name), (
            f"GridResult.{name} exists — scalar fusion is machine-banned"
        )


# ---------------------------------------------------------------------------
# to_radar and to_pareto_front
# ---------------------------------------------------------------------------


def test_to_radar_returns_per_tokenizer_per_axis(
    two_adapters: tuple,
    protein_samples: list[Sample],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    engine = Engine()
    a, b = two_adapters
    engine.register_adapter(a)
    engine.register_adapter(b)

    result = engine.eval_grid({"protein": protein_samples})  # type: ignore[arg-type]
    radar = engine.to_radar(result)

    assert "tok_a" in radar
    assert "tok_b" in radar
    for tok_vals in radar.values():
        assert set(tok_vals.keys()) == {"A", "B", "C"}


def test_to_pareto_front_returns_list(
    two_adapters: tuple,
    protein_samples: list[Sample],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    engine = Engine()
    a, b = two_adapters
    engine.register_adapter(a)
    engine.register_adapter(b)

    result = engine.eval_grid({"protein": protein_samples})  # type: ignore[arg-type]
    front = engine.to_pareto_front(result)

    assert isinstance(front, list)
    # At least one tokenizer on the Pareto front
    assert len(front) >= 1
    # All names on the front must be valid tokenizer names
    assert all(name in ["tok_a", "tok_b"] for name in front)


# ---------------------------------------------------------------------------
# Honest gate integration
# ---------------------------------------------------------------------------


def test_engine_raises_on_real_data_env(
    two_adapters: tuple,
    protein_samples: list[Sample],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine must raise HonestGateError when KINETOKEN_REAL_DATA=1 without allow_real."""
    from tokenparity.safety.honest_gate import HonestGateError

    monkeypatch.setenv("KINETOKEN_REAL_DATA", "1")
    engine = Engine()  # allow_real=False by default
    a, _ = two_adapters
    engine.register_adapter(a)

    with pytest.raises(HonestGateError):
        engine.eval_grid({"protein": protein_samples})  # type: ignore[arg-type]
