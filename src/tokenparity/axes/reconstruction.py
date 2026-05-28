"""Axis C: Per-Domain Normalized Reconstruction.

Definition:
    R(d, t) = z_score_within_d(reconstruction_error(d, t))

where reconstruction_error(d, t) = MSE(original, reconstructed).

Key constraint: cross-domain absolute summation of reconstruction errors is
REFUSED because MSE scales are incommensurable across modalities.
Cross-domain comparison is valid only via average rank.

Public API
----------
reconstruction_error(orig, recon) -> float
    MSE between two arrays of the same shape.

normalize_within_domain(errors: dict[str, float]) -> dict[str, float]
    Z-score normalise a dict of {tokenizer_name: mse} within a single domain.

average_rank(per_domain: dict[str, dict[str, float]]) -> dict[str, float]
    Rank tokenizers within each domain (lower MSE → lower rank number), then
    average ranks across domains.  This is the ONLY cross-domain aggregation
    allowed by this axis.
"""

from __future__ import annotations

import numpy as np


def reconstruction_error(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute MSE between *orig* and *recon*.

    Parameters
    ----------
    orig:
        Original signal array.  Any shape accepted.
    recon:
        Reconstructed signal array.  Must have the same shape as *orig*.

    Returns
    -------
    float
        Mean squared error.

    Raises
    ------
    ValueError
        If *orig* and *recon* have different shapes.
    """
    orig_arr = np.asarray(orig, dtype=float)
    recon_arr = np.asarray(recon, dtype=float)
    if orig_arr.shape != recon_arr.shape:
        raise ValueError(
            f"orig and recon must have the same shape; got {orig_arr.shape} vs {recon_arr.shape}"
        )
    return float(np.mean((orig_arr - recon_arr) ** 2))


def normalize_within_domain(errors: dict[str, float]) -> dict[str, float]:
    """Z-score normalise reconstruction errors within a single domain.

    Parameters
    ----------
    errors:
        Mapping of tokenizer name → MSE value.  All values must come from the
        same domain.  Never pass values from different domains here.

    Returns
    -------
    dict[str, float]
        Mapping of tokenizer name → z-score.  If all errors are identical
        (std == 0), returns a dict of zeros (avoids division by zero).

    Notes
    -----
    This function performs **within-domain** normalisation only.  Calling it
    with values from different domains would produce meaningless z-scores.
    Cross-domain comparison must use :func:`average_rank` instead.
    """
    if not errors:
        return {}

    values = np.array(list(errors.values()), dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))

    if std == 0.0:
        return {name: 0.0 for name in errors}

    return {name: (v - mean) / std for name, v in errors.items()}


def average_rank(per_domain: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute average rank of tokenizers across domains.

    Within each domain, tokenizers are ranked by ascending MSE (rank 1 = lowest
    error = best).  The ranks are then averaged across domains for each tokenizer.
    A tokenizer absent from a domain is excluded from that domain's ranking but
    included in others.

    Parameters
    ----------
    per_domain:
        Mapping of domain_name → {tokenizer_name: mse}.

    Returns
    -------
    dict[str, float]
        Mapping of tokenizer_name → average rank across domains that include it.
        Lower average rank = better cross-domain reconstruction performance (by rank).

    Notes
    -----
    This is the ONLY cross-domain aggregation sanctioned by Axis C.  Do not use
    raw MSE values or z-scores for cross-domain comparison.
    """
    if not per_domain:
        return {}

    # Collect all tokenizer names
    all_names: set[str] = set()
    for domain_errors in per_domain.values():
        all_names.update(domain_errors.keys())

    rank_sums: dict[str, float] = {name: 0.0 for name in all_names}
    rank_counts: dict[str, int] = {name: 0 for name in all_names}

    for domain_errors in per_domain.values():
        if not domain_errors:
            continue
        # Sort by MSE ascending; rank starts at 1
        sorted_names = sorted(domain_errors, key=lambda n: domain_errors[n])
        for rank_idx, name in enumerate(sorted_names, start=1):
            rank_sums[name] += rank_idx
            rank_counts[name] += 1

    return {
        name: rank_sums[name] / rank_counts[name] for name in all_names if rank_counts[name] > 0
    }
