"""Additional safety and honest_gate coverage tests."""

from __future__ import annotations

import os

import pytest

from tokenparity.safety import HonestGateError, assert_synthetic_only, is_synthetic_mode
from tokenparity.safety.honest_gate import HonestGateError as DirectHonestGateError


def test_package_exports_honest_gate_error() -> None:
    """safety package must export HonestGateError."""
    assert HonestGateError is DirectHonestGateError


def test_is_synthetic_mode_various_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only '1' triggers real-data mode; other values stay synthetic."""
    for val in ("0", "false", "yes", "true", "2", ""):
        monkeypatch.setenv("KINETOKEN_REAL_DATA", val)
        assert is_synthetic_mode() is True, (
            f"Expected synthetic mode for KINETOKEN_REAL_DATA={val!r}"
        )


def test_assert_synthetic_only_with_env_zero_no_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KINETOKEN_REAL_DATA", "0")
    assert_synthetic_only()  # must not raise


def test_honest_gate_error_message_contains_allow_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Error message should guide the user toward the correct fix."""
    monkeypatch.setenv("KINETOKEN_REAL_DATA", "1")
    with pytest.raises(HonestGateError) as exc_info:
        assert_synthetic_only()
    assert "allow_real" in str(exc_info.value)
    assert "KINETOKEN_REAL_DATA" in str(exc_info.value)


def test_honest_gate_error_message_contains_weightlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Error message should reference the license gate."""
    monkeypatch.setenv("KINETOKEN_REAL_DATA", "1")
    with pytest.raises(HonestGateError) as exc_info:
        assert_synthetic_only()
    assert "weightlock" in str(exc_info.value)


def test_env_var_name_is_kinetoken_real_data() -> None:
    """Verify the module uses the correct env var name (no typos)."""
    import inspect

    import tokenparity.safety.honest_gate as mod

    source = inspect.getsource(mod)
    assert "KINETOKEN_REAL_DATA" in source


def test_is_synthetic_mode_default_without_env() -> None:
    """Without the env var set, should default to synthetic mode."""
    # Remove if present; os.environ.pop is safe
    env_backup = os.environ.pop("KINETOKEN_REAL_DATA", None)
    try:
        assert is_synthetic_mode() is True
    finally:
        if env_backup is not None:
            os.environ["KINETOKEN_REAL_DATA"] = env_backup
