"""Tests for core IR types (Sample, Tokens, TokenizerAdapter)."""

from __future__ import annotations

import numpy as np
import pytest

from tokenparity.core.types import Sample, TokenizerAdapter, Tokens

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def test_sample_creation() -> None:
    s = Sample(domain="protein", payload=np.ones((5, 3)))
    assert s.domain == "protein"
    assert np.array_equal(s.payload, np.ones((5, 3)))


def test_sample_meta_default_empty() -> None:
    s = Sample(domain="video", payload=np.zeros((4,)))
    assert s.meta == {}


def test_sample_meta_custom() -> None:
    s = Sample(domain="robot", payload=np.zeros(3), meta={"src": "test"})
    assert s.meta["src"] == "test"


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_tokens_valid_creation() -> None:
    ids = np.array([0, 1, 2], dtype=np.int64)
    t = Tokens(domain="protein", ids=ids, vocab_size=10)
    assert t.ids.shape == (3,)
    assert t.vocab_size == 10


def test_tokens_non_integer_dtype_raises() -> None:
    ids = np.array([0.5, 1.5], dtype=float)
    with pytest.raises(ValueError, match="integer dtype"):
        Tokens(domain="protein", ids=ids, vocab_size=10)


def test_tokens_non_1d_raises() -> None:
    ids = np.array([[0, 1], [2, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match="1-D"):
        Tokens(domain="protein", ids=ids, vocab_size=10)


def test_tokens_out_of_range_max_raises() -> None:
    ids = np.array([0, 5, 10], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range"):
        Tokens(domain="protein", ids=ids, vocab_size=5)


def test_tokens_negative_id_raises() -> None:
    ids = np.array([-1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range"):
        Tokens(domain="protein", ids=ids, vocab_size=5)


def test_tokens_empty_ids_valid() -> None:
    """Empty ids array should be valid (no range check)."""
    ids = np.array([], dtype=np.int64)
    t = Tokens(domain="protein", ids=ids, vocab_size=10)
    assert t.ids.size == 0


def test_tokens_with_latents() -> None:
    ids = np.array([0, 1], dtype=np.int64)
    latents = np.ones((2, 4))
    t = Tokens(domain="video", ids=ids, vocab_size=5, latents=latents)
    assert t.latents is not None
    assert t.latents.shape == (2, 4)


def test_tokens_latents_none_by_default() -> None:
    ids = np.array([0], dtype=np.int64)
    t = Tokens(domain="robot", ids=ids, vocab_size=10)
    assert t.latents is None


# ---------------------------------------------------------------------------
# TokenizerAdapter protocol runtime check
# ---------------------------------------------------------------------------


def test_tokenizer_adapter_is_protocol() -> None:
    """TokenizerAdapter must be a runtime-checkable Protocol."""
    from tokenparity.adapters.protein_mock import MockFoldTokenAdapter

    adapter = MockFoldTokenAdapter()
    assert isinstance(adapter, TokenizerAdapter)


def test_non_adapter_fails_protocol_check() -> None:
    """An object missing protocol methods should not satisfy the check."""

    class NotAnAdapter:
        pass

    assert not isinstance(NotAnAdapter(), TokenizerAdapter)
