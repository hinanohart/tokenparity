# Changelog

All notable changes to tokenparity are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0a2] - 2026-05-29 (pre-release, hotfix)

Post-release 3-agent independent audit (GitHub-state / compact-fidelity /
code-sophistication) found 2 MAJOR correctness bugs in a1, plus several
hardening opportunities.  No yank — a1 stays available — but a2 is the
recommended pre-alpha.

### Fixed

- **MAJOR**: `engine.py:286` Information-Bottleneck cell with legitimate
  `ib_median == 0.0` was silently coerced to `NaN` by a truthy-OR fallback
  (`cell.get("ib_median") or float("nan")`).  Replaced with explicit
  `None` check.  A zero-MI estimate now survives into `GridResult.values`.
- **MAJOR**: `scripts/verify_step.py:60` raised `AttributeError` on the
  fallback path because `match.group(1)` was called after the explicit
  `if not match` branch.  Wrapped in `else` so the fallback path returns
  cleanly.

### Hardened

- `safety/honest_gate.py`: passing `allow_real=True` without
  `KINETOKEN_REAL_DATA=1` now emits a new `HonestGateMisconfigWarning`
  instead of silently no-op'ing.  The two-factor gate still stays closed
  (synthetic-only is the safe default).  Docstring updated.
- `.github/workflows/readme-grep.yml`: banned-phrase list expanded with
  `永久` and `完璧` so that the regex catches the synonym pair the
  internal style guide tracks.
- `bench_results/synthetic_v0.1.0a1.json` regenerated under strict JSON
  semantics: bare `NaN` literals replaced with the `"nan"` string and
  `json.dump(..., allow_nan=False)` enforced.  As a side-effect of the
  M1 fix above, legitimate IB=0.0 cells now serialise correctly.
- `cli.py:_numpy_to_serialisable`: recurse through `ndarray.tolist()`
  output so element-level NaN floats get the string conversion (a1
  forgot that step and shipped non-strict JSON).
- New SHA-256 for `bench_results/synthetic_v0.1.0a1.json`:
  `71434df87c853b08f04e1b532a665ccd33ab832a9eca5560ec8f133ba646ed51`
  (a1 was `0cd8eae081eccabcb3920400ddad8485563f7f5fb60f327e86c0cf91f867e7fd`).

### Added

- `tests/test_readme_grep_fires.py`: 15-case negative fixture that asserts
  the workflow's banned-phrase regex actually fires on each phrase
  (`grep -E -i` rc=0) and does NOT fire on clean text (rc=1).
  BRE/ERE drift now breaks a unit test instead of silently disabling the
  README gate.
- 3 new tests for the `HonestGateMisconfigWarning` warning path and the
  no-warning branches.

### CI / repo hygiene

- `.github/dependabot.yml`: `github-actions` ecosystem also pinned with
  `open-pull-requests-limit: 0 + ignore: version-update:semver-major`.
  Stops the noisy "actions/checkout v4→v6, actions/setup-python v5→v6"
  PRs (#1, #2) that flooded the PR list with red checks.  Those two
  PRs have been closed.
- Stale `release_url` in `.tokenparity-progress.json` corrected from
  `hinanohart/keep-empty` to `hinanohart/tokenparity`.

### Not changed

- API surface, scalar-fusion ban, two-factor honest gate semantics,
  3×3 grid shape, branch-protection contexts, license, version-tag
  scheme — all stable.

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

[0.1.0a2]: https://github.com/hinanohart/tokenparity/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/hinanohart/tokenparity/releases/tag/v0.1.0a1
