"""Property-based tests using hypothesis.

Properties verified:
1. Kraskov-1 MI is invariant under linear bijection (scaling + offset).
2. normalize_within_domain is invariant under global scale of all errors.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tokenparity._estimators.kraskov import mi
from tokenparity.axes.reconstruction import normalize_within_domain

# ---------------------------------------------------------------------------
# Property 1: Kraskov-1 invariant under linear bijection on X
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=30000)
@given(
    scale=st.floats(min_value=0.5, max_value=5.0),
    offset=st.floats(min_value=-2.0, max_value=2.0),
)
def test_kraskov_invariant_under_linear_transform(scale: float, offset: float) -> None:
    """I(aX + b; Y) ≈ I(X; Y) for invertible linear transform."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.standard_normal((n, 1))
    y = 0.6 * x + 0.4 * rng.standard_normal((n, 1))

    mi_original = mi(x, y, k=5)
    mi_transformed = mi(scale * x + offset, y, k=5)

    # MI is invariant under invertible linear transforms (scale ≠ 0).
    # KSG-1 estimator variance grows with scale, so tolerance is 0.15 nats.
    assert abs(mi_original - mi_transformed) < 0.15, (
        f"MI changed more than 0.15 nats under linear transform "
        f"(scale={scale:.2f}, offset={offset:.2f}): "
        f"{mi_original:.4f} vs {mi_transformed:.4f}"
    )


# ---------------------------------------------------------------------------
# Property 2: normalize_within_domain invariant under global scale
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=5000)
@given(
    base_errors=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=10),
        elements=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    ),
    scale=st.floats(min_value=0.1, max_value=100.0),
)
def test_normalize_within_domain_invariant_under_scale(
    base_errors: np.ndarray, scale: float
) -> None:
    """Z-score normalisation must be invariant under global scaling of all errors.

    Requires that the original array has non-zero std (so z-scores are defined).
    All-equal inputs hit the std=0 guard and return zeros regardless of scale.
    """
    from hypothesis import assume

    # Skip degenerate case where all errors are identical (std=0 guard path)
    assume(float(np.std(base_errors)) > 1e-10)

    tokenizer_names = [f"tok_{i}" for i in range(len(base_errors))]
    errors_original = dict(zip(tokenizer_names, base_errors.tolist()))
    errors_scaled = dict(zip(tokenizer_names, (base_errors * scale).tolist()))

    normed_original = normalize_within_domain(errors_original)
    normed_scaled = normalize_within_domain(errors_scaled)

    for name in tokenizer_names:
        diff = abs(normed_original[name] - normed_scaled[name])
        assert diff < 1e-4, (
            f"normalize_within_domain changed under scale={scale:.2f} "
            f"for {name}: {normed_original[name]:.6f} vs {normed_scaled[name]:.6f}"
        )


# ---------------------------------------------------------------------------
# Property 3: MI output is always non-negative
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=30000)
@given(
    k=st.integers(min_value=3, max_value=7),
    n=st.integers(min_value=50, max_value=200),
)
def test_mi_always_non_negative(k: int, n: int) -> None:
    """mi() must return a non-negative float for any valid inputs."""
    rng = np.random.default_rng(k * 1000 + n)
    x = rng.standard_normal((n, 1))
    y = rng.standard_normal((n, 1))
    result = mi(x, y, k=k)
    assert result >= 0.0, f"mi() returned {result} < 0 for k={k}, n={n}"


# ---------------------------------------------------------------------------
# Property 4: reconstruction_error symmetric around origin
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=5000)
@given(
    arr=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=50),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
)
def test_reconstruction_error_zero_for_identical(arr: np.ndarray) -> None:
    """MSE of identical arrays must always be 0."""
    from tokenparity.axes.reconstruction import reconstruction_error

    assert reconstruction_error(arr, arr) == pytest.approx(0.0, abs=1e-10)
