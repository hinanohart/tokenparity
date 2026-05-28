"""Safety layer for tokenparity.

Provides the honest gate that prevents accidental evaluation of real data
without explicit user consent.
"""

from tokenparity.safety.honest_gate import HonestGateError, assert_synthetic_only, is_synthetic_mode

__all__ = ["HonestGateError", "assert_synthetic_only", "is_synthetic_mode"]
