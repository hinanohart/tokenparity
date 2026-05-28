# tokenparity Architecture

## Overview

tokenparity measures whether discrete tokenizers from three orthogonal modalities
score equivalently on three honest axes.  The result is a **3×3 grid** (3 domains ×
3 axes = 9 cells), not a scalar.  Scalar fusion is machine-banned in CI.

---

## 1. Intermediate Representation (IR)

All domain-specific data flows through two dataclasses defined in
`src/tokenparity/core/types.py`:

```python
@dataclass(frozen=True)
class Sample:
    domain: Domain          # "protein" | "video" | "robot"
    payload: Any            # domain-native array (e.g. (L,3) for protein coords)
    meta: dict[str, Any]

@dataclass(frozen=True)
class Tokens:
    domain: Domain
    ids: np.ndarray         # 1-D integer array, values in [0, vocab_size)
    vocab_size: int
    latents: np.ndarray | None   # optional continuous side-channel for axis B
    meta: dict[str, Any]
```

The IR is intentionally domain-agnostic.  Per-domain semantics live in the adapter,
not in the IR.

---

## 2. Adapter Protocol

Every tokenizer exposes a uniform interface:

```python
class TokenizerAdapter(Protocol):
    domain: Literal["protein", "video", "robot"]
    name: str
    license: str          # SPDX identifier
    is_synthetic: bool

    def encode(self, x: Sample) -> Tokens: ...
    def decode(self, tokens: Tokens) -> Sample: ...     # axis C
    def features(self, tokens: Tokens) -> np.ndarray:  # axis A (2-D array)
    def native_score(self, sample: Sample) -> float:   # axis A baseline
```

See [`docs/adapter_guide.md`](adapter_guide.md) for implementation instructions.

---

## 3. The Three Honest Axes

### Axis A — Downstream Transferability

```
T(d, t) = score_native(d) − score_frozen_features(d, t)
```

- Higher T → larger gap → frozen tokenizer loses more downstream utility.
- Cross-domain comparison: **rank correlation only**.  Never sum T values across
  domains.
- Implementation: `src/tokenparity/axes/transfer.py`

### Axis B — Information Bottleneck

```
IB(d, t) = I(X; Z_t) − β · I(Z_t; Y)
```

- `Z_t` = adapter latent, `X` = input, `Y` = downstream label.
- Estimated with the self-implemented **Kraskov-1 (KSG-1) estimator**
  (`src/tokenparity/_estimators/kraskov.py`).
  Apache-2.0; NPEET (GPL) is not used.
- Sensitivity: reported across k ∈ {3, 5, 7} with a 95 % bootstrap CI.
- **High-dim guard**: if `Z_t.shape[-1] > 64`, the estimator is unreliable.
  `rank_only=True` is returned; no absolute value is claimed.
- Implementation: `src/tokenparity/axes/ib.py`

### Axis C — Per-Domain Normalized Reconstruction

```
R(d, t) = z_score_within_d(reconstruction_error(d, t))
```

- `reconstruction_error = MSE(original, decoded)`.
- Z-scoring is **within-domain only** (MSE scales are incommensurable across
  modalities).
- Cross-domain view: **average rank** only — rank within each domain, then average.
- Implementation: `src/tokenparity/axes/reconstruction.py`

---

## 4. Scalar Fusion Ban

The output of `Engine.eval_grid()` is a `GridResult` with shape
`(n_domains, n_tokenizers, 3)`.

`GridResult` intentionally has **no** `total_score()`, `aggregate()`,
`combined_score()`, or any method that collapses the 9-cell grid into a scalar.
This is checked by:
- Static: `tests/test_grid_no_scalar_fusion.py` — parametrised over 10 banned names.
- CI grep: `readme-grep.yml` checks for overclaim language in README.

The canonical views are:
- **Radar chart**: `engine.to_radar(grid_result)` — per-tokenizer per-axis values.
- **Pareto front**: `engine.to_pareto_front(grid_result)` — non-dominated set.

---

## 5. Honest Gate

`KINETOKEN_REAL_DATA=1` must be set **and** `allow_real=True` passed to
`Engine(allow_real=True)` before any real data is processed.

Default behaviour: synthetic-only.  Violation raises `HonestGateError`.

---

## 6. CPU-Only Core

Core dependencies: `numpy>=1.26`, `scipy>=1.11`.
No `torch` import anywhere in the core path.  CI enforces:

```bash
python -c "import sys; import tokenparity; assert 'torch' not in sys.modules"
```

---

## 7. Repo Layout

```
src/tokenparity/
├── core/types.py                  # Sample, Tokens, TokenizerAdapter Protocol
├── _estimators/kraskov.py         # self-impl Kraskov-1 (Apache-2.0, NPEET-free)
├── axes/transfer.py               # Axis A
├── axes/ib.py                     # Axis B
├── axes/reconstruction.py         # Axis C
├── adapters/{protein,video,robot}_mock.py  # analytical-mock adapters
├── engine.py                      # Engine, GridResult (no scalar fusion)
├── safety/honest_gate.py          # KINETOKEN_REAL_DATA env lock
└── cli.py                         # argparse CLI (eval / grid / version)
```
