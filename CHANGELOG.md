# Changelog

All notable changes to tokenparity are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0a1] - 2026-05-29 (pre-release)

### Added

- **3-axis evaluation framework**: Downstream Transferability (A), Information
  Bottleneck (B), Per-Domain Normalized Reconstruction (C).
- **Self-implemented Kraskov-1 (KSG-1) MI estimator** (`_estimators/kraskov.py`).
  Apache-2.0; NPEET (GPL) is not a dependency.
- **3 analytical-mock adapters**: protein structure (MockFoldTokenAdapter),
  video frames (MockVideoGridAdapter), robot action sequences
  (MockLeRobotActionAdapter).  All are synthetic-only with analytic ±1%
  golden tests.
- **Evaluation engine** (`engine.py`): `Engine.eval_grid()` returns a
  `GridResult` (3D array, no scalar fusion).  `to_radar()` and
  `to_pareto_front()` as the only cross-tokenizer views.
- **Honest gate** (`safety/honest_gate.py`): `KINETOKEN_REAL_DATA=1` + 
  `allow_real=True` required for real data; default is synthetic-only.
- **CLI** (`cli.py`): `eval`, `grid`, `version` subcommands (argparse).
- **CI 3-layer**: lint (ruff + mypy) / unit (138 tests, 91% coverage) /
  license audit (GPL grep rc=1).
- **Scalar fusion ban**: `GridResult` has no `total_score()` or `aggregate()`
  — machine-checked by `tests/test_grid_no_scalar_fusion.py`.
- **Reproducible bench results**: `bench_results/synthetic_v0.1.0a1.json`
  (SHA-256 pinned in README; deterministic seed=42).
- Docs: `ARCHITECTURE.md`, `adapter_guide.md`, `LIMITATIONS.md`,
  `DUPLICATION_DECLARATIONS.md`.

### Design constraints

- CPU-only core (no torch in the core import path).
- Synthetic + cached-token only in this version.  Live model inference is
  deferred to v0.1.1.
- 3 modalities only.  Text and audio extras are deferred to v0.1.1+.
- Scalar fusion is refused by design; the output is a typed 3×3 grid.

### Known limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

[0.1.0a1]: https://github.com/hinanohart/tokenparity/releases/tag/v0.1.0a1
