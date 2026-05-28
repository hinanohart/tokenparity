"""Kraskov-1 mutual information estimator.

Apache-2.0 reimplementation from the 2004 paper:
  Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
  Estimating mutual information. Physical Review E, 69(6), 066138.
  https://doi.org/10.1103/PhysRevE.69.066138

This module is NPEET-free and GPL-free. No code was copied from NPEET.
The formula used is Algorithm 1 (KSG estimator 1):

  I(X; Y) = ψ(k) + ψ(N) - <ψ(n_x + 1) + ψ(n_y + 1)>

where:
  - ψ is the digamma function
  - k is the number of nearest neighbours (default 3; also use 5, 7)
  - N is the number of samples
  - n_x(i), n_y(i) are the number of points in the marginal balls of radius ε(i)/2
    where ε(i) = 2 * Chebyshev distance to the k-th neighbour in the joint space
  - The marginal counts use STRICT inequality (< ε/2), hence +1 is applied after

Notes on the strict vs ≤ convention:
  KSG-1 uses strict inequality in the marginals.  We implement this with
  `query_ball_point(..., eps=0)` (the scipy default) which returns points
  STRICTLY within the radius.  The +1 absorbs the point itself.

Typical usage::

    import numpy as np
    from tokenparity._estimators.kraskov import mi

    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 2))
    y = x + 0.5 * rng.standard_normal((500, 2))
    print(mi(x, y, k=5))  # positive float
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def mi(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> float:
    """Estimate I(X; Y) via the Kraskov-1 (KSG-1) estimator.

    Parameters
    ----------
    x:
        Shape ``(N, d_x)`` or ``(N,)`` for 1-D.  Float64 recommended.
    y:
        Shape ``(N, d_y)`` or ``(N,)``.  Must share the first axis with *x*.
    k:
        Number of nearest neighbours.  The paper recommends k ∈ {3, 5, 7}.
        Larger k reduces variance at the cost of bias for multi-modal densities.

    Returns
    -------
    float
        Estimated mutual information in nats (natural logarithm).  Clipped to
        ``≥ 0.0`` because finite-sample noise can produce small negative estimates
        that are artefacts of the estimator, not true negative MI.

    Raises
    ------
    ValueError
        If *x* and *y* have different sample counts, or *k* ≥ N.

    References
    ----------
    Kraskov, Stögbauer, Grassberger (2004), Physical Review E 69, 066138.
    Apache-2.0 reimplementation — NPEET (GPL) not used.
    """
    x = np.atleast_2d(x).T if x.ndim == 1 else np.asarray(x, dtype=float)
    y = np.atleast_2d(y).T if y.ndim == 1 else np.asarray(y, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"x and y must have the same number of samples; got {n} vs {y.shape[0]}")
    if k >= n:
        raise ValueError(f"k={k} must be less than N={n}")

    # Joint space: concatenate and use Chebyshev (L-inf) metric for KSG-1
    z = np.hstack([x, y])

    tree_z = cKDTree(z)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # For each point, find the distance to the k-th neighbour in joint space
    # p=np.inf is Chebyshev metric.  k+1 because the point itself is included.
    distances, _ = tree_z.query(z, k=k + 1, p=np.inf, workers=1)
    # distances[:, 0] == 0 (self), distances[:, k] == ε(i)/2 in KSG notation
    eps_half = distances[:, k]  # shape (N,)

    # Count marginal neighbours strictly within radius ε(i)/2
    # query_ball_point returns indices of points with dist < r (strict < by default)
    # We subtract 1 to exclude the query point itself.
    nx = np.array(
        [len(tree_x.query_ball_point(x[i], r=eps_half[i], p=np.inf)) - 1 for i in range(n)],
        dtype=float,
    )
    ny = np.array(
        [len(tree_y.query_ball_point(y[i], r=eps_half[i], p=np.inf)) - 1 for i in range(n)],
        dtype=float,
    )

    # KSG-1 formula:  I = ψ(k) + ψ(N) - <ψ(n_x+1) + ψ(n_y+1)>
    result: float = float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))

    # Clip negative artefacts from finite-sample noise
    return max(result, 0.0)
