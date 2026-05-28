"""Tests for the honest gate (safety/honest_gate.py)."""

from __future__ import annotations

import pytest

from tokenparity.safety.honest_gate import (
    HonestGateError,
    assert_synthetic_only,
    is_synthetic_mode,
)


def test_gate_open_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No error when KINETOKEN_REAL_DATA is not set."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    assert_synthetic_only()  # must not raise


def test_gate_open_when_env_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """No error when KINETOKEN_REAL_DATA=0."""
    monkeypatch.setenv("KINETOKEN_REAL_DATA", "0")
    assert_synthetic_only()  # must not raise


def test_gate_raises_when_env_one_no_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    """HonestGateError raised when KINETOKEN_REAL_DATA=1 without allow_real."""
    monkeypatch.setenv("KINETOKEN_REAL_DATA", "1")
    with pytest.raises(HonestGateError, match="allow_real=False"):
        assert_synthetic_only()


def test_gate_open_when_env_one_with_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    """No error when KINETOKEN_REAL_DATA=1 AND allow_real=True."""
    monkeypatch.setenv("KINETOKEN_REAL_DATA", "1")
    assert_synthetic_only(allow_real=True)  # must not raise


def test_is_synthetic_mode_true_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    assert is_synthetic_mode() is True


def test_is_synthetic_mode_false_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KINETOKEN_REAL_DATA", "1")
    assert is_synthetic_mode() is False


def test_allow_real_without_env_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing allow_real=True when env is unset is a silent no-op."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    assert_synthetic_only(allow_real=True)  # must not raise


def test_honest_gate_error_is_runtime_error() -> None:
    """HonestGateError must be a RuntimeError subclass."""
    assert issubclass(HonestGateError, RuntimeError)
