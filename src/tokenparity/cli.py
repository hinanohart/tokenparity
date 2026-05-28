"""CLI entry point for tokenparity.

Subcommands
-----------
eval     Evaluate a single (domain, tokenizer, axis) cell using synthetic data.
grid     Run the full 3×3 synthetic grid and save results to JSON.
version  Print the installed version string.

Usage examples::

    tokenparity version
    tokenparity eval --domain protein --tokenizer mock --axis A
    tokenparity grid --synthetic --output bench_results/synthetic_v0.1.0a1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _get_version() -> str:
    from tokenparity import __version__

    return __version__


def _make_synthetic_samples(domain: str, n: int = 5, seed: int = 0) -> list[Any]:
    """Generate synthetic samples for the given domain."""
    from tokenparity.core.types import Sample

    rng = np.random.default_rng(seed)

    if domain == "protein":
        return [Sample(domain="protein", payload=rng.standard_normal((20, 3))) for _ in range(n)]
    elif domain == "video":
        return [
            Sample(domain="video", payload=rng.standard_normal((4, 16, 16, 3))) for _ in range(n)
        ]
    elif domain == "robot":
        return [Sample(domain="robot", payload=rng.standard_normal((10, 6))) for _ in range(n)]
    else:
        raise ValueError(f"Unknown domain: {domain!r}")


def _load_mock_adapters(domains: list[str]) -> list[Any]:
    """Load mock adapters for the requested domains."""
    adapters = []
    for domain in domains:
        if domain == "protein":
            from tokenparity.adapters.protein_mock import MockFoldTokenAdapter

            adapters.append(MockFoldTokenAdapter())
        elif domain == "video":
            from tokenparity.adapters.video_mock import MockVideoGridAdapter

            adapters.append(MockVideoGridAdapter())
        elif domain == "robot":
            from tokenparity.adapters.robot_mock import MockLeRobotActionAdapter

            adapters.append(MockLeRobotActionAdapter())
    return adapters


def _numpy_to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _numpy_to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return str(obj)
    return obj


def cmd_version(args: argparse.Namespace) -> int:
    """Print the installed version."""
    print(f"tokenparity {_get_version()}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a single (domain, tokenizer, axis) cell."""
    from tokenparity.engine import Engine
    from tokenparity.safety.honest_gate import assert_synthetic_only

    assert_synthetic_only()

    domain = args.domain
    tok_name = args.tokenizer
    axes = ["A", "B", "C"] if args.axis == "all" else [args.axis.upper()]

    adapters = _load_mock_adapters([domain])
    if not adapters:
        print(f"No adapter found for domain {domain!r}", file=sys.stderr)
        return 1

    engine = Engine()
    for adapter in adapters:
        engine.register_adapter(adapter)

    samples = _make_synthetic_samples(domain, n=args.n_samples)

    # If tokenizer name is "mock", use the first registered adapter
    if tok_name == "mock":
        tok_name = list(engine._adapters.keys())[0]

    results = {}
    for axis in axes:
        try:
            cell = engine.eval_cell(domain, tok_name, axis, samples)  # type: ignore[arg-type]
            results[axis] = _numpy_to_serialisable(cell)
        except Exception as exc:
            print(f"Error evaluating axis {axis}: {exc}", file=sys.stderr)
            return 1

    print(json.dumps(results, indent=2))
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    """Run the full 3×3 synthetic grid and save to JSON."""
    from tokenparity.core.types import Domain
    from tokenparity.engine import Engine
    from tokenparity.safety.honest_gate import assert_synthetic_only

    if args.synthetic:
        assert_synthetic_only()
    else:
        # Non-synthetic path requires real data env var; gate handled in Engine
        pass

    domains: list[Domain] = ["protein", "video", "robot"]
    adapters = _load_mock_adapters(domains)  # type: ignore[arg-type]

    engine = Engine()
    for adapter in adapters:
        engine.register_adapter(adapter)

    samples_per_domain = {
        domain: _make_synthetic_samples(domain, n=args.n_samples, seed=42) for domain in domains
    }

    grid_result = engine.eval_grid(samples_per_domain)  # type: ignore[arg-type]

    # Build serialisable output
    output: dict[str, Any] = {
        "version": _get_version(),
        "synthetic": True,
        "domains": grid_result.domains,
        "tokenizers": grid_result.tokenizers,
        "axes": grid_result.axes,
        "values": _numpy_to_serialisable(grid_result.values),
        "radar": _numpy_to_serialisable(engine.to_radar(grid_result)),
        "pareto_front": engine.to_pareto_front(grid_result),
        "raw_cells": {
            f"{d}|{t}|{a}": _numpy_to_serialisable(v)
            for (d, t, a), v in grid_result.raw_cells.items()
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Grid results written to {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tokenparity",
        description="GPU-free cross-modality tokenizer parity harness.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # version
    sub.add_parser("version", help="Print the installed version.")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a single cell.")
    p_eval.add_argument("--domain", required=True, choices=["protein", "video", "robot"])
    p_eval.add_argument("--tokenizer", required=True, help='Tokenizer name, or "mock".')
    p_eval.add_argument(
        "--axis",
        default="all",
        choices=["A", "B", "C", "all"],
        help="Axis to evaluate (default: all).",
    )
    p_eval.add_argument("--n-samples", type=int, default=10, dest="n_samples")

    # grid
    p_grid = sub.add_parser("grid", help="Run the full 3×3 synthetic grid.")
    p_grid.add_argument("--synthetic", action="store_true", default=True)
    p_grid.add_argument(
        "--output",
        default="bench_results/synthetic_v0.1.0a1.json",
        help="Output JSON path.",
    )
    p_grid.add_argument("--n-samples", type=int, default=10, dest="n_samples")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=os.environ.get("TOKENPARITY_LOG_LEVEL", "WARNING"))
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        return cmd_version(args)
    elif args.command == "eval":
        return cmd_eval(args)
    elif args.command == "grid":
        return cmd_grid(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
