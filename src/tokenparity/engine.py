"""Evaluation engine for tokenparity.

The engine schedules the 3×3 grid (3 domains × 3 axes) and collects results
into a :class:`GridResult`.

Design constraints
------------------
- Scalar fusion is BANNED.  :class:`GridResult` has no ``.total_score()``,
  ``.aggregate()``, or any method that collapses the 9-cell grid into a scalar.
- ``to_radar()`` returns per-tokenizer per-axis values (no cross-axis merge).
- ``to_pareto_front()`` returns Pareto-dominant tokenizer names per domain.
- Real-data evaluation requires ``KINETOKEN_REAL_DATA=1`` AND ``allow_real=True``
  to be passed to :meth:`Engine.eval_grid`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from tokenparity.axes.ib import information_bottleneck
from tokenparity.axes.reconstruction import (
    reconstruction_error,
)
from tokenparity.axes.transfer import transferability
from tokenparity.core.types import Domain, Sample, TokenizerAdapter
from tokenparity.safety.honest_gate import assert_synthetic_only

logger = logging.getLogger(__name__)

Axis = Literal["A", "B", "C"]


@dataclass
class GridResult:
    """Result of a full 3×3 grid evaluation.

    Attributes
    ----------
    domains:
        List of domain names evaluated.
    tokenizers:
        List of tokenizer names evaluated.
    axes:
        Always ``["A", "B", "C"]``.
    values:
        3-D array of shape ``(n_domains, n_tokenizers, 3)`` where the last
        dimension indexes [A, B, C].  ``nan`` for missing cells.
    ci_bands:
        Optional dict mapping ``(domain, tokenizer, axis)`` to ``(ci_low, ci_high)``.
        Only populated for axis B.
    raw_cells:
        Full per-cell dicts returned by :meth:`Engine.eval_cell`, keyed by
        ``(domain, tokenizer, axis)``.
    """

    domains: list[str]
    tokenizers: list[str]
    axes: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_bands: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)
    raw_cells: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Scalar fusion is MACHINE-BANNED here.  Do NOT add total_score(),
    # aggregate(), combined_score(), or any method returning a single float
    # that conflates multiple axes.
    # ------------------------------------------------------------------


class Engine:
    """Scheduler for the tokenparity 9-cell evaluation grid.

    Parameters
    ----------
    allow_real:
        Pass ``True`` only when real cached tokens have been license-gated.
        Requires ``KINETOKEN_REAL_DATA=1`` in the environment.  Default: ``False``.

    Example
    -------
    >>> from tokenparity.engine import Engine
    >>> engine = Engine()
    >>> engine.register_adapter(my_adapter)
    >>> result = engine.eval_grid(samples_per_domain={"protein": [sample1]})
    """

    def __init__(self, *, allow_real: bool = False) -> None:
        self._adapters: dict[str, TokenizerAdapter] = {}
        self._allow_real = allow_real

    # ------------------------------------------------------------------
    # Adapter registry
    # ------------------------------------------------------------------

    def register_adapter(self, adapter: TokenizerAdapter) -> None:
        """Register a tokenizer adapter.

        Parameters
        ----------
        adapter:
            Must satisfy :class:`~tokenparity.core.types.TokenizerAdapter`.
        """
        self._adapters[adapter.name] = adapter
        logger.debug("Registered adapter %r (domain=%s)", adapter.name, adapter.domain)

    # ------------------------------------------------------------------
    # Single cell
    # ------------------------------------------------------------------

    def eval_cell(
        self,
        domain: Domain,
        tokenizer_name: str,
        axis: Axis,
        sample_iter: Any,
    ) -> dict[str, Any]:
        """Evaluate one (domain, tokenizer, axis) cell.

        Parameters
        ----------
        domain:
            Domain identifier.
        tokenizer_name:
            Name of a registered adapter.
        axis:
            One of ``"A"``, ``"B"``, ``"C"``.
        sample_iter:
            Iterable of :class:`~tokenparity.core.types.Sample` objects.

        Returns
        -------
        dict
            Cell-specific result dict.  Always contains ``"axis"``, ``"domain"``,
            ``"tokenizer"``.  Axis-specific keys follow each axis's convention.
        """
        assert_synthetic_only(allow_real=self._allow_real)

        adapter = self._adapters[tokenizer_name]
        samples: list[Sample] = list(sample_iter)

        if axis == "A":
            return self._eval_axis_a(domain, adapter, samples)
        elif axis == "B":
            return self._eval_axis_b(domain, adapter, samples)
        elif axis == "C":
            return self._eval_axis_c(domain, adapter, samples)
        else:
            raise ValueError(f"Unknown axis {axis!r}; must be 'A', 'B', or 'C'")

    def _eval_axis_a(
        self,
        domain: Domain,
        adapter: TokenizerAdapter,
        samples: list[Sample],
    ) -> dict[str, Any]:
        gaps: list[float] = []
        for sample in samples:
            native = adapter.native_score(sample)
            tokens = adapter.encode(sample)
            features = adapter.features(tokens)
            # Frozen proxy: use the mean feature norm as a stand-in for a
            # linear probe's performance (synthetic-only; real probe deferred).
            frozen = float(1.0 / (1.0 + np.linalg.norm(features.mean(axis=0))))
            gaps.append(transferability(native, frozen))
        median_gap = float(np.median(gaps)) if gaps else float("nan")
        return {
            "axis": "A",
            "domain": domain,
            "tokenizer": adapter.name,
            "per_sample": gaps,
            "median": median_gap,
        }

    def _eval_axis_b(
        self,
        domain: Domain,
        adapter: TokenizerAdapter,
        samples: list[Sample],
    ) -> dict[str, Any]:
        if not samples:
            return {
                "axis": "B",
                "domain": domain,
                "tokenizer": adapter.name,
                "ib_median": float("nan"),
                "rank_only": True,
            }

        x_list, z_list, y_list = [], [], []
        for sample in samples:
            tokens = adapter.encode(sample)
            feats = adapter.features(tokens)
            x_list.append(np.asarray(sample.payload, dtype=float).flatten())
            z_list.append(feats.flatten())
            # Synthetic: use native_score as scalar "Y" label
            y_list.append([adapter.native_score(sample)])

        x = np.array(x_list)
        z = np.array(z_list)
        y = np.array(y_list)

        result = information_bottleneck(x, z, y)
        return {"axis": "B", "domain": domain, "tokenizer": adapter.name, **result}

    def _eval_axis_c(
        self,
        domain: Domain,
        adapter: TokenizerAdapter,
        samples: list[Sample],
    ) -> dict[str, Any]:
        errors: list[float] = []
        for sample in samples:
            tokens = adapter.encode(sample)
            recon_sample = adapter.decode(tokens)
            orig = np.asarray(sample.payload, dtype=float).flatten()
            recon = np.asarray(recon_sample.payload, dtype=float).flatten()
            # Align lengths: compare the shorter prefix (mock decoders may
            # return a different number of elements due to approximate inversion)
            n = min(len(orig), len(recon))
            errors.append(reconstruction_error(orig[:n], recon[:n]))
        median_mse = float(np.median(errors)) if errors else float("nan")
        return {
            "axis": "C",
            "domain": domain,
            "tokenizer": adapter.name,
            "per_sample_mse": errors,
            "median_mse": median_mse,
        }

    # ------------------------------------------------------------------
    # Full grid
    # ------------------------------------------------------------------

    def eval_grid(
        self,
        samples_per_domain: dict[Domain, list[Sample]],
    ) -> GridResult:
        """Evaluate the full grid across all registered adapters.

        Parameters
        ----------
        samples_per_domain:
            Mapping of domain → list of samples.  Only domains present here
            are evaluated.

        Returns
        -------
        GridResult
            The full 3-D result array.  No scalar fusion.
        """
        assert_synthetic_only(allow_real=self._allow_real)

        domains = list(samples_per_domain.keys())
        tokenizers = list(self._adapters.keys())
        axes: list[Axis] = ["A", "B", "C"]

        n_d, n_t, n_a = len(domains), len(tokenizers), len(axes)
        values = np.full((n_d, n_t, n_a), float("nan"))
        ci_bands: dict[tuple[str, str, str], tuple[float, float]] = {}
        raw_cells: dict[tuple[str, str, str], dict[str, Any]] = {}

        for di, domain in enumerate(domains):
            samples = samples_per_domain[domain]
            for ti, tok_name in enumerate(tokenizers):
                adapter = self._adapters[tok_name]
                if adapter.domain != domain:
                    logger.debug(
                        "Skipping adapter %r (domain=%s) for domain %s",
                        tok_name,
                        adapter.domain,
                        domain,
                    )
                    continue
                for ai, axis in enumerate(axes):
                    try:
                        cell = self.eval_cell(domain, tok_name, axis, samples)
                        raw_cells[(domain, tok_name, axis)] = cell

                        if axis == "A":
                            values[di, ti, ai] = cell.get("median", float("nan"))
                        elif axis == "B":
                            ib_val = cell.get("ib_median")
                            values[di, ti, ai] = float("nan") if ib_val is None else ib_val
                            if cell.get("ci_low") is not None:
                                ci_bands[(domain, tok_name, axis)] = (
                                    cell["ci_low"],
                                    cell["ci_high"],
                                )
                        elif axis == "C":
                            values[di, ti, ai] = cell.get("median_mse", float("nan"))
                    except Exception:
                        logger.exception(
                            "Cell eval failed: domain=%s tok=%s axis=%s",
                            domain,
                            tok_name,
                            axis,
                        )

        return GridResult(
            domains=list(domains),
            tokenizers=tokenizers,
            axes=list(axes),
            values=values,
            ci_bands=ci_bands,
            raw_cells=raw_cells,
        )

    # ------------------------------------------------------------------
    # Views (no scalar fusion)
    # ------------------------------------------------------------------

    def to_radar(self, grid_result: GridResult) -> dict[str, dict[str, float]]:
        """Return per-tokenizer per-axis values (no cross-axis aggregation).

        Parameters
        ----------
        grid_result:
            Output of :meth:`eval_grid`.

        Returns
        -------
        dict
            ``{tokenizer_name: {"A": float, "B": float, "C": float}}``
            Values are medians across domains (or nan if all cells are nan).
        """
        result: dict[str, dict[str, float]] = {}
        for ti, tok_name in enumerate(grid_result.tokenizers):
            axis_vals: dict[str, float] = {}
            for ai, axis in enumerate(grid_result.axes):
                col = grid_result.values[:, ti, ai]
                finite = col[np.isfinite(col)]
                axis_vals[axis] = float(np.median(finite)) if len(finite) > 0 else float("nan")
            result[tok_name] = axis_vals
        return result

    def to_pareto_front(self, grid_result: GridResult) -> list[str]:
        """Return tokenizer names on the Pareto front (per-domain, all axes).

        A tokenizer is Pareto-dominated if another tokenizer is at least as
        good on ALL axes in ALL domains with strict improvement on at least one.
        Lower values are better for axes A and C; higher values are better for B.

        Parameters
        ----------
        grid_result:
            Output of :meth:`eval_grid`.

        Returns
        -------
        list[str]
            Tokenizer names on the Pareto front (non-dominated set).
        """
        tokenizers = grid_result.tokenizers
        n_t = len(tokenizers)

        # Build a per-tokenizer objective vector (lower = better for all axes
        # after sign-flip of axis B).
        # Shape: (n_tokenizers, n_domains * n_axes)
        objectives = np.full((n_t, len(grid_result.domains) * len(grid_result.axes)), float("nan"))
        for ti in range(n_t):
            idx = 0
            for di in range(len(grid_result.domains)):
                for ai, axis in enumerate(grid_result.axes):
                    val = grid_result.values[di, ti, ai]
                    # Axis B: higher is better → negate for Pareto minimisation
                    objectives[ti, idx] = -val if axis == "B" else val
                    idx += 1

        dominated = [False] * n_t
        for i in range(n_t):
            for j in range(n_t):
                if i == j:
                    continue
                obj_i = objectives[i]
                obj_j = objectives[j]
                # j dominates i if j ≤ i on all finite objectives AND strict < on ≥1
                finite_mask = np.isfinite(obj_i) & np.isfinite(obj_j)
                if not np.any(finite_mask):
                    continue
                if np.all(obj_j[finite_mask] <= obj_i[finite_mask]) and np.any(
                    obj_j[finite_mask] < obj_i[finite_mask]
                ):
                    dominated[i] = True
                    break

        return [tok for tok, dom in zip(tokenizers, dominated, strict=True) if not dom]
