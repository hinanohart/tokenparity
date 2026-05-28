"""Axis B: Information Bottleneck.

Definition:
    IB(d, t) = I(X; Z_t) - β · I(Z_t; Y)

where Z_t is the latent representation from tokenizer t, X is the input,
and Y is the downstream label / target.

Estimation uses the self-implemented Kraskov-1 (KSG-1) estimator.
NPEET (GPL) is not imported.

High-dimensionality guard: if Z_t has more than 64 dimensions, the KSG-1
estimator produces unreliable absolute estimates (curse of dimensionality).
In that case, rank_only=True is returned and no absolute IB value is given.

Sensitivity is always reported across k ∈ {3, 5, 7}.  A 95 % bootstrap CI
is provided via a lightweight percentile bootstrap (n_boot=200).
"""

from __future__ import annotations

import numpy as np

from tokenparity._estimators.kraskov import mi

_DEFAULT_K_VALUES = (3, 5, 7)
_DIM_LIMIT = 64
_N_BOOT = 200
_BOOT_SEED = 42


def information_bottleneck(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    k_values: tuple[int, ...] = _DEFAULT_K_VALUES,
    beta: float = 1.0,
) -> dict[str, object]:
    """Estimate IB(d, t) = I(X; Z) - β · I(Z; Y) via KSG-1.

    Parameters
    ----------
    x:
        Input representation, shape ``(N, d_x)`` or ``(N,)``.
    z:
        Tokenizer latent / feature representation, shape ``(N, d_z)`` or ``(N,)``.
    y:
        Target / label representation, shape ``(N, d_y)`` or ``(N,)``.
    k_values:
        Tuple of k values for KSG-1 sensitivity check.  Default ``(3, 5, 7)``.
    beta:
        Trade-off weight.  ``beta=1.0`` (default) balances compression vs. fidelity.

    Returns
    -------
    dict with keys:
        ``ib_per_k``  — dict mapping each k → IB estimate (float)
        ``ib_median`` — median IB across k values (float); ``None`` if rank_only
        ``ci_low``    — 2.5th-percentile of bootstrap distribution; ``None`` if rank_only
        ``ci_high``   — 97.5th-percentile; ``None`` if rank_only
        ``rank_only`` — True if z.shape[-1] > 64; absolute values are suppressed

    Notes
    -----
    When ``rank_only=True``, the ``ib_per_k`` dict is populated with ``nan``,
    ``ib_median`` / ``ci_low`` / ``ci_high`` are ``None``.
    Callers must not treat ``nan`` values as 0.
    """
    z_arr = np.atleast_2d(z).T if np.asarray(z).ndim == 1 else np.asarray(z, dtype=float)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(-1, 1)

    rank_only = z_arr.shape[-1] > _DIM_LIMIT

    if rank_only:
        return {
            "ib_per_k": {k: float("nan") for k in k_values},
            "ib_median": None,
            "ci_low": None,
            "ci_high": None,
            "rank_only": True,
        }

    x_arr = np.atleast_2d(x).T if np.asarray(x).ndim == 1 else np.asarray(x, dtype=float)
    y_arr = np.atleast_2d(y).T if np.asarray(y).ndim == 1 else np.asarray(y, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    ib_per_k: dict[int, float] = {}
    for k in k_values:
        i_xz = mi(x_arr, z_arr, k=k)
        i_zy = mi(z_arr, y_arr, k=k)
        ib_per_k[k] = i_xz - beta * i_zy

    median_val = float(np.median(list(ib_per_k.values())))

    # Percentile bootstrap CI using k=median k (middle value)
    n = x_arr.shape[0]
    mid_k = sorted(k_values)[len(k_values) // 2]
    rng = np.random.default_rng(_BOOT_SEED)
    boot_vals: list[float] = []
    for _ in range(_N_BOOT):
        idx = rng.integers(0, n, size=n)
        i_xz_b = mi(x_arr[idx], z_arr[idx], k=mid_k)
        i_zy_b = mi(z_arr[idx], y_arr[idx], k=mid_k)
        boot_vals.append(i_xz_b - beta * i_zy_b)

    ci_low = float(np.percentile(boot_vals, 2.5))
    ci_high = float(np.percentile(boot_vals, 97.5))

    return {
        "ib_per_k": ib_per_k,
        "ib_median": median_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "rank_only": False,
    }
