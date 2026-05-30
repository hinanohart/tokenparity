# tokenparity

> **CLAIM (machine-enforced)**: A GPU-free harness that measures whether discrete tokenizers from three orthogonal modalities (protein structure / video / robotics) score equivalently on three honest axes — downstream transferability, information bottleneck, and per-domain normalized reconstruction. Not a leaderboard. Not a survey. Synthetic + cached-token only in v0.1.0a1.

[![CI](https://github.com/hinanohart/tokenparity/actions/workflows/ci-core.yml/badge.svg)](https://github.com/hinanohart/tokenparity/actions/workflows/ci-core.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: pre-alpha](https://img.shields.io/badge/Status-pre--alpha-orange.svg)](#status)

**Status**: pre-alpha (`v0.1.0a1`). No live model inference, no real datasets, no leaderboard. Reads pre-computed tokens or runs analytical synthetic adapters.

---

## What it is

tokenparity is **not** a tokenizer. tokenparity *measures* tokenizers, the way a calibrated voltmeter measures voltage:

- **3 modalities** that share no obvious geometry — protein 3-D structure, video frames, robotics action sequences.
- **3 honest axes** that each survive Goodhart pressure on their own:
  - **A. Downstream Transferability**: `T(d, t) = score_native(d) − score_frozen_features(d, t)`. Lower is better; cross-domain comparable only by rank.
  - **B. Information Bottleneck**: `IB(d, t) = I(X; Z_t) − β · I(Z_t; Y)` via a self-implemented Kraskov-1 estimator (MIT, NPEET-free). `k ∈ {3, 5, 7}` sensitivity reported with CI bands.
  - **C. Per-Domain Normalized Reconstruction**: `R(d, t) = z_score_within_d(reconstruction_error(d, t))`. Cross-domain compared only by average rank.

The output is a 3×3 grid (9 cells) rendered as a radar chart plus a Pareto front. **Scalar fusion is machine-banned in CI.**

## What it is not

- Not a leaderboard. There is no "best" tokenizer.
- Not a survey of existing tokenizers. The shipped adapters are analytical-mock for v0.1.0a1.
- Not text or audio. Those are deferred to v0.1.1+.
- Not a hallucination gate. `dreamcheck` (the world-model dual-tokenizer disagreement gate) lives in a separate future repo, not here.
- Not a reward / fitness function. tokenparity returns a typed grid, not a number you can `argmax`.

## Differentiation

This repo intentionally does **not** overlap with the following sibling work. See [`docs/DUPLICATION_DECLARATIONS.md`](docs/DUPLICATION_DECLARATIONS.md) for the literal 13-entry list (8 hinanohart-internal + 5 external prior-art):

| Project | tokenparity vs. it |
|---|---|
| [transduce](https://github.com/hinanohart/transduce) | scores **hidden-state probes** inside one model; tokenparity scores **tokenizers themselves** across 3 modalities |
| [gavagai](https://github.com/hinanohart/gavagai) | translates between **SAE feature dictionaries** within one LLM; tokenparity compares **tokens from different modalities** |
| [oraclematch](https://github.com/hinanohart/oraclematch) | calibrated fitness penalty on **molecular search candidates**; tokenparity ranks **tokenizers** |
| [tracecal](https://github.com/hinanohart/tracecal) | physics gate on **LeRobot dataset samples**; tokenparity measures **tokenizers using cached LeRobot tokens** |
| [mlipgauge](https://github.com/hinanohart/mlipgauge) | runtime guardrail for **MLIP MD predictions**; tokenparity's MD axis is deferred to v0.1.1 |
| [speclattice](https://github.com/hinanohart/speclattice) | **cross-vocabulary SD acceptance-length** between two LLM tokenizers (text); tokenparity is **cross-modality**, not cross-vocab |
| [polyphonic-eval](https://github.com/hinanohart/polyphonic-eval) | typed disagreement across **multiple LLM judges on one output**; tokenparity returns typed disagreement across **tokenizers on a 3-modality 3-axis grid** (inherits the "no scalar fusion" rule, operates one layer below) |
| [saelet](https://github.com/hinanohart/saelet) | evaluates **SAE probes** on a single LLM family's residuals (single-modality text); tokenparity evaluates **tokenizers** across three orthogonal modalities |
| [InfoTok](https://arxiv.org/abs/2512.16975) (ICLR 2026) | **designs** an IB-optimised video tokenizer; tokenparity **evaluates** arbitrary tokenizers on IB as one of three axes — InfoTok would be a candidate-under-test, not a competitor |
| [StructTokenBench](https://arxiv.org/abs/2503.00089) | protein-only; tokenparity is cross-modality (protein × video × robot) |
| [LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | evaluates **downstream multimodal LLM behaviour**; tokenparity scores the tokenizer artefact itself |
| [Beyond Text Compression](https://arxiv.org/abs/2506.03101) | text tokenizers, scalar; tokenparity refuses scalar fusion |
| [Ordered Action Tokenization](https://ordered-action-tokenization.github.io/) | a new robot tokenizer **design**; tokenparity is an eval harness for designs like it |

## Quickstart (synthetic-only, CPU)

```bash
pip install tokenparity
tokenparity version
tokenparity eval --domain protein --tokenizer mock --axis A
tokenparity grid --synthetic --output bench_results/synthetic_v0.1.0a1.json
```

The CLI refuses to compute "real" scores unless you set `KINETOKEN_REAL_DATA=1` and provide cached tokens that pass the license gate (`hinanohart/weightlock`).

## Reproducible bench results

`bench_results/synthetic_v0.1.0a1.json` is generated deterministically (seed=42, n=10 samples per domain).

SHA-256: `71434df87c853b08f04e1b532a665ccd33ab832a9eca5560ec8f133ba646ed51` (v0.1.0a2; a1 was `0cd8eae081eccabcb3920400ddad8485563f7f5fb60f327e86c0cf91f867e7fd` — strict-JSON change in a2)

To regenerate and verify: `tokenparity grid --synthetic --output bench_results/synthetic_v0.1.0a1.json`

## Why "tokenparity"

Mosaicraft (Hungarian + Oklab perceptual placement) proves *visual* equivalence without collapsing the domains it bridges. tokenparity does the same for tokenizers: it proves typed equivalence (or disagreement) across modalities without collapsing them into a leaderboard scalar. The honest answer to "which tokenizer is best?" is sometimes "you are asking the wrong question, here is the typed disagreement."

## Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). Briefly:

- v0.1.0a1 ships **analytical-mock adapters only**. Real adapters land in v0.1.1.
- No live model inference. Cached tokens only.
- IB axis is sensitive to dimensionality; we report `k ∈ {3,5,7}` CI bands and refuse claims past dim 64.
- 3 modalities is intentionally small. Calling tokenparity a "full harness" before v0.1.1 (which adds text + audio extras) is overclaim.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

The Kraskov-1 estimator in `src/tokenparity/_estimators/kraskov.py` is an MIT reimplementation from the 2004 paper. **NPEET (GPL) is not a dependency.** CI enforces `pip-licenses | grep GPL` returns rc=1.

## Citation

```bibtex
@software{tokenparity_2026,
  title  = {tokenparity: GPU-free cross-modality tokenizer parity harness},
  author = {hinanohart},
  year   = {2026},
  url    = {https://github.com/hinanohart/tokenparity},
  version = {0.1.0a1}
}
```
