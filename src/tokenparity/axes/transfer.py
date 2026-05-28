"""Axis A: Downstream Transferability.

Definition:
    T(d, t) = score_native(d) - score_frozen_features(d, t)

A positive value means the frozen tokenizer (t) loses downstream utility
relative to the native pipeline.  Higher T → larger gap → worse for t.

Cross-domain comparison is valid only by rank correlation; never sum T values
across domains to produce a single "total score".
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def transferability(native_score: float, frozen_score: float) -> float:
    """Compute the transferability gap for a single (domain, tokenizer) cell.

    Parameters
    ----------
    native_score:
        Downstream task score obtained with the natively-trained pipeline.
        Higher is better (e.g. accuracy ∈ [0, 1]).
    frozen_score:
        Downstream task score when the tokenizer features are frozen and only
        a linear head is trained on top.

    Returns
    -------
    float
        T = native_score - frozen_score.
        Positive → gap exists (frozen hurts).
        Zero → frozen-feature quality matches native.
        Negative → frozen features outperform (rare, but allowed; not clipped).
    """
    return native_score - frozen_score


def aggregate_transferability(
    scores: Sequence[tuple[float, float]],
) -> dict[str, object]:
    """Aggregate transferability over multiple (native, frozen) pairs.

    Parameters
    ----------
    scores:
        Iterable of (native_score, frozen_score) tuples, one per sample.

    Returns
    -------
    dict with keys:
        ``per_sample`` — list of per-sample T values (float)
        ``median``     — median T across samples (float)

    Notes
    -----
    The median is more robust than the mean for heavy-tailed score distributions.
    No scalar fusion across domains is performed here; callers must not sum medians
    from different domains.
    """
    if not scores:
        return {"per_sample": [], "median": float("nan")}

    per_sample = [transferability(n, f) for n, f in scores]
    med = statistics.median(per_sample)
    return {"per_sample": per_sample, "median": med}
