"""Tests for the honest gate (safety/honest_gate.py)."""

from __future__ import annotations

import pytest

from tokenparity.safety.honest_gate import (
    HonestGateError,
    HonestGateMisconfigWarning,
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


def test_allow_real_without_env_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing allow_real=True without env=1 must surface a misconfig warning.

    a2 hardening: previous a1 silently no-op'd this misconfiguration; the
    two-factor gate stays closed (synthetic-only is safe), but the caller is
    now warned so the mistake doesn't hide.
    """
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    with pytest.warns(HonestGateMisconfigWarning, match="KINETOKEN_REAL_DATA=1 is not set"):
        assert_synthetic_only(allow_real=True)


def test_no_warning_when_both_factors_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both env=1 and allow_real=True are set, no warning fires."""
    import warnings as _w

    monkeypatch.setenv("KINETOKEN_REAL_DATA", "1")
    with _w.catch_warnings():
        _w.simplefilter("error", HonestGateMisconfigWarning)
        assert_synthetic_only(allow_real=True)  # must not warn or raise


def test_no_warning_when_neither_factor_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default synthetic mode (env unset, allow_real=False) must not warn."""
    import warnings as _w

    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    with _w.catch_warnings():
        _w.simplefilter("error", HonestGateMisconfigWarning)
        assert_synthetic_only(allow_real=False)  # must not warn or raise


def test_honest_gate_error_is_runtime_error() -> None:
    """HonestGateError must be a RuntimeError subclass."""
    assert issubclass(HonestGateError, RuntimeError)


def test_misconfig_warning_is_user_warning() -> None:
    """HonestGateMisconfigWarning must be a UserWarning subclass."""
    assert issubclass(HonestGateMisconfigWarning, UserWarning)
