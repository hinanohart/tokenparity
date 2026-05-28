"""Honest gate: prevents real-data evaluation without explicit opt-in.

The environment variable ``KINETOKEN_REAL_DATA`` controls whether real
(non-synthetic) data may be processed.  By default the gate is closed;
synthetic-only mode is the safe default.

Usage::

    from tokenparity.safety.honest_gate import assert_synthetic_only

    # Raises HonestGateError if KINETOKEN_REAL_DATA=1 without allow_real:
    assert_synthetic_only()

    # To allow real data (user must set the env var AND pass the flag):
    assert_synthetic_only(allow_real=True)
"""

from __future__ import annotations

import os
import warnings


class HonestGateMisconfigWarning(UserWarning):
    """Warn the caller that ``allow_real=True`` was set without ``KINETOKEN_REAL_DATA=1``.

    The two-factor gate stays closed, but the caller almost certainly intended
    to open it.  Silently swallowing this mismatch hides the misconfiguration.
    """


class HonestGateError(RuntimeError):
    """Raised when real-data evaluation is attempted without explicit opt-in.

    This is a hard safety stop, not a warning.  The caller must:
    1. Set ``KINETOKEN_REAL_DATA=1`` in the environment.
    2. Pass ``allow_real=True`` to :func:`assert_synthetic_only`.

    Both conditions are required simultaneously to prevent accidental real-data
    evaluation from misconfigured environments.
    """


def is_synthetic_mode() -> bool:
    """Return True if the environment is in synthetic-only mode.

    Returns
    -------
    bool
        ``True``  — ``KINETOKEN_REAL_DATA`` is unset or not ``"1"``.
        ``False`` — ``KINETOKEN_REAL_DATA=1`` is set (real-data mode active).
    """
    return os.environ.get("KINETOKEN_REAL_DATA", "0") != "1"


def assert_synthetic_only(*, allow_real: bool = False) -> None:
    """Assert that the current run is synthetic-only, or that real data was explicitly approved.

    Parameters
    ----------
    allow_real:
        If ``True``, the gate is open regardless of the environment variable.
        The caller is responsible for ensuring real data has passed the license
        gate (``hinanohart/weightlock``) before setting this flag.
        Default: ``False`` (gate closed).

    Raises
    ------
    HonestGateError
        If ``KINETOKEN_REAL_DATA=1`` is set **and** ``allow_real=False``.
        This is the hard-stop path for accidental real-data evaluation.

    Warns
    -----
    HonestGateMisconfigWarning
        If ``allow_real=True`` is passed but ``KINETOKEN_REAL_DATA=1`` is not set.
        The gate stays closed (synthetic-only is the safe default), but the
        mismatch almost certainly means the caller misconfigured the run.
        Surfacing this avoids the silent no-op the previous version returned.

    Notes
    -----
    The two-factor requirement prevents a code change alone from bypassing
    the gate.  Both the environment variable AND the keyword argument must be
    set in the same run for real-data evaluation to proceed.
    """
    real_data_env = os.environ.get("KINETOKEN_REAL_DATA", "0") == "1"

    if real_data_env and not allow_real:
        raise HonestGateError(
            "KINETOKEN_REAL_DATA=1 is set but allow_real=False. "
            "To process real data you must explicitly pass allow_real=True "
            "after ensuring cached tokens have passed the license gate "
            "(hinanohart/weightlock). "
            "This is a safety stop, not a warning."
        )

    if allow_real and not real_data_env:
        warnings.warn(
            "allow_real=True was passed but KINETOKEN_REAL_DATA=1 is not set. "
            "The gate stays closed (synthetic-only). To process real data, "
            "set KINETOKEN_REAL_DATA=1 in the environment as well.",
            HonestGateMisconfigWarning,
            stacklevel=2,
        )
