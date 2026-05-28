# tokenparity v0.1.0a1 Limitations

This file is **machine-grepped at CI time**. The literal strings below are load-bearing.

## Synthetic-only declaration

**tokenparity v0.1.0a1 ships analytical-mock adapters only.** No real protein folding model, no real video codec, no real LeRobot policy is invoked. All scores are computed against synthetic inputs whose statistics are known in closed form. Tests assert each axis returns the analytical expected value within ±1%.

## Cached-token-only declaration

Live model inference is **disabled** in v0.1.0a1. Adapters that would require a GPU or network call refuse to run unless the user sets `KINETOKEN_REAL_DATA=1`. CI sets `KINETOKEN_REAL_DATA=0` and asserts the env-lock fires.

## IB dimensionality caveat

Axis B (information bottleneck) uses a self-implemented Kraskov-1 estimator. The estimator's variance grows with dimensionality. **We refuse to report IB for `features.shape[-1] > 64`** without a `--allow-high-dim` flag, and we always emit CI bands across `k ∈ {3, 5, 7}`. If those bands overlap zero, the result is reported as "rank-only" — absolute IB claims are withdrawn.

## Three modalities is intentionally small

Calling tokenparity a "complete multimodal tokenizer harness" is overclaim. We cover **three modalities** in v0.1.0a1 — protein structure, video, robotics — chosen because they have no obvious shared geometry, which makes the parity question non-trivial. Text and audio land in v0.1.1 as extras. MD trajectory lands separately if at all.

## What tokenparity does NOT do

- Does not train tokenizers.
- Does not provide a single scalar "best tokenizer" ranking. Scalar fusion is **machine-banned**.
- Does not gate hallucinations. `dreamcheck` (the world-model dual-tokenizer disagreement gate referenced in our R14 search notes) lives in a separate future repo.
- Does not provide a reward / fitness function. The 3×3 grid is meant to be read, not optimised against.

## Prior-art positioning (load-bearing)

See [`DUPLICATION_DECLARATIONS.md`](DUPLICATION_DECLARATIONS.md) for the literal 11-entry table.

In particular:

- **InfoTok (ICLR 2026 Oral, arxiv 2512.16975)** designs an information-theoretic adaptive *video* tokenizer. tokenparity's IB axis is a measurement applied across *three* modalities; InfoTok is a candidate-under-test for the video slot, not a competing harness.
- **StructTokenBench (arxiv 2503.00089)** is a protein-only benchmark using different sub-axes; tokenparity's protein axis is one of three.
- **LMMS-Eval, Beyond Text Compression, Ordered Action Tokenization** address downstream LLM behaviour / text-only tokenizers / new robot tokenizer designs respectively — all orthogonal to tokenparity's typed-grid eval.

## What would falsify tokenparity's premise

1. If the IB CI bands always overlap zero across `k ∈ {3,5,7}`, axis B is reported as rank-only and absolute claims are withdrawn. (Pivot F1 in the bootstrap protocol.)
2. If the video mock adapter scores all candidate tokenizers identically on axis A, the video slot pivots to static image patches and the cross-modality claim weakens to "2.5 modality". (Pivot F2.)
3. If post-release audit finds that a hinanohart sibling (e.g. tracecal) already covers ≥70% of the robot axis, the robot slot is dropped or replaced. (Pivot F3.)
