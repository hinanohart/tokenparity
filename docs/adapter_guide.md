# Writing a New TokenizerAdapter

This guide explains how to implement a `TokenizerAdapter` for a new tokenizer
and register it with the tokenparity engine.

---

## 1. Implement the Protocol

Create a class that satisfies `tokenparity.core.types.TokenizerAdapter`:

```python
# src/tokenparity/adapters/my_adapter.py
import numpy as np
from tokenparity.core.types import Domain, Sample, Tokens

class MyTokenizerAdapter:
    domain: Domain = "protein"   # or "video" / "robot"
    name: str = "my_tokenizer"
    license: str = "Apache-2.0"  # SPDX identifier — must be non-GPL
    is_synthetic: bool = False   # True only for analytical-mock adapters

    def encode(self, x: Sample) -> Tokens:
        # Convert sample.payload to token ids
        ids = ...  # 1-D numpy integer array, values in [0, vocab_size)
        latents = ...  # optional continuous features for axis B (or None)
        return Tokens(domain=self.domain, ids=ids, vocab_size=1024, latents=latents)

    def decode(self, tokens: Tokens) -> Sample:
        # Approximate inversion for axis C (MSE computation)
        recon_payload = ...
        return Sample(domain=self.domain, payload=recon_payload)

    def features(self, tokens: Tokens) -> np.ndarray:
        # Continuous feature representation for axis A (2-D array required)
        # Shape: (1, feature_dim) or (n_tokens, feature_dim)
        return tokens.latents.reshape(1, -1) if tokens.latents is not None \
               else tokens.ids.reshape(1, -1).astype(float)

    def native_score(self, sample: Sample) -> float:
        # Baseline score for axis A (without frozen features)
        # Must return a float in [0, 1]
        return 0.8
```

---

## 2. License Gate

tokenparity CI enforces `pip-licenses | grep GPL` returns rc=1.  Your adapter
must not import GPL dependencies.  If you need a non-commercial dependency
(e.g. ESM-3, Cosmos), isolate it in an optional extra and gate it behind:

```toml
# pyproject.toml
[project.optional-dependencies]
my-extras = ["esm @ git+..."]   # CC-BY-NC-4.0 — user opt-in
```

---

## 3. Real Data Gate

If your adapter reads real (non-synthetic) data, you must check the honest gate:

```python
from tokenparity.safety.honest_gate import assert_synthetic_only

def encode(self, x: Sample) -> Tokens:
    assert_synthetic_only(allow_real=self._allow_real)
    ...
```

The user must set `KINETOKEN_REAL_DATA=1` **and** pass `allow_real=True` to
`Engine(allow_real=True)`.

---

## 4. Register and Run

```python
from tokenparity.engine import Engine
from my_adapter import MyTokenizerAdapter

engine = Engine()
engine.register_adapter(MyTokenizerAdapter())

from tokenparity.core.types import Sample
import numpy as np

samples = [Sample(domain="protein", payload=np.random.randn(50, 3)) for _ in range(20)]
result = engine.eval_grid({"protein": samples})

radar = engine.to_radar(result)
print(radar)
# {"my_tokenizer": {"A": 0.12, "B": 0.34, "C": -0.05}}
```

---

## 5. Writing Tests

Every adapter must pass the trinity test:

```python
def test_encode_valid_ids(adapter, sample):
    tokens = adapter.encode(sample)
    assert tokens.ids.ndim == 1
    assert int(tokens.ids.min()) >= 0
    assert int(tokens.ids.max()) < tokens.vocab_size

def test_features_2d(adapter, sample):
    tokens = adapter.encode(sample)
    feats = adapter.features(tokens)
    assert feats.ndim == 2

def test_native_score_in_range(adapter, sample):
    score = adapter.native_score(sample)
    assert 0.0 <= score <= 1.0
```

If `is_synthetic=True`, you must also provide an analytic golden test:

```python
def test_canonical_input_score_near_expected(adapter):
    canonical_payload = ...  # analytically perfect input
    sample = Sample(domain=adapter.domain, payload=canonical_payload)
    score = adapter.native_score(sample)
    assert abs(score - expected) < 0.01  # ±1% tolerance
```

---

## 6. Axis B High-Dim Guard

If your tokenizer produces latents with more than 64 dimensions, axis B will
automatically return `rank_only=True` and suppress absolute IB values.  This is
by design (KSG-1 estimator is unreliable in high dimensions).  Reduce latent
dimensionality via PCA before axis B if you want absolute estimates.
