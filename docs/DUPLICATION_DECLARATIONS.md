# Org & Prior-Art Duplication Declarations

R14 mandatory: literal differentiation vs every hinanohart repo with motif overlap ≥30%, plus external prior-art directly addressed by tokenparity's CLAIM line.

## Internal (hinanohart) overlap

### 1. transduce (80% motif overlap)
- transduce: "Transduction probes for foundation models — five hook-based observation metrics plus a causal intervention probe, with bootstrap CIs and HF text/image/audio adapters."
- **tokenparity ≠ transduce**: tokenparity scores **discrete tokenizers themselves** on three honest axes across three modalities (protein structure / video / robotics). transduce probes hidden-state hooks inside a *single* foundation model. No tokenizer-level metric, no cross-modality grid, no synthetic-only honest gate.

### 2. gavagai (70% motif overlap)
- gavagai: "Quantify translation indeterminacy between SAE feature dictionaries (Quine × Mechanistic Interpretability)"
- **tokenparity ≠ gavagai**: gavagai compares **SAE feature dictionaries** within the same LLM family. tokenparity compares **token IDs / latents from different modalities** (protein vs video vs robot). gavagai = single-modality vocab translation; tokenparity = cross-modality parity.

### 3. oraclematch (70% motif overlap)
- oraclematch: "Cross-paradigm oracle-disagreement (DL co-folding × classical docking) as a calibrated fitness penalty for MAP-Elites molecular search."
- **tokenparity ≠ oraclematch**: oraclematch ranks **molecular search candidates** by oracle disagreement. tokenparity ranks **tokenizers**, not search candidates. Different consumer (MAP-Elites vs eval harness), different output (penalty score vs radar+Pareto), different domain (drug discovery vs cross-modality token grid).

### 4. tracecal (40% motif overlap, robotics axis)
- tracecal: "Conformal-calibrated, URDF physics-gated validity & abstention auditing for LeRobot robot-learning datasets"
- **tokenparity ≠ tracecal**: tracecal gates **dataset samples** (physics validity, abstention). tokenparity's robot axis evaluates **tokenizer encode/decode quality** on cached LeRobot tokens. Orthogonal layers: dataset filter vs tokenizer eval.

### 5. mlipgauge (30% motif overlap, MD axis future-deferred)
- mlipgauge: "Uncertainty-driven runtime guardrail for MLIP molecular dynamics: a fail-closed physics-validity gate over trajectory windows."
- **tokenparity ≠ mlipgauge**: mlipgauge is a runtime guardrail for **MD predictions** at execution time. tokenparity's deferred MD axis would evaluate **tokenizers for MD trajectories**. v0.1.0a1 ships protein/video/robot only; MD deferred to v0.1.1.

### 6. speclattice (30% motif overlap, "cross-vocabulary" surface keyword)
- speclattice: "Cross-vocabulary speculative decoding: a CPU-verifiable reference implementation and acceptance-length (tau) measurement harness."
- **tokenparity ≠ speclattice**: speclattice measures **draft-target token alignment for SD throughput** between two LLM tokenizers (same modality, text). tokenparity measures **token-level information geometry across modalities** (protein/video/robot). Same surface keyword "cross-vocabulary", entirely different denotation.

## External prior-art

### A. InfoTok (ICLR 2026 Oral, arxiv 2512.16975)
- InfoTok is an **information-theoretic adaptive video tokenizer** with a router + ELBO objective, claiming 2.3× compression on video.
- **tokenparity ≠ InfoTok**: InfoTok *designs* a single-modality (video) tokenizer optimised for IB; tokenparity *evaluates* arbitrary tokenizers across three modalities on a 3-axis grid. tokenparity's IB axis is one of three honest measurements, not a training objective. InfoTok is a candidate-under-test, not a competitor; if benchmarked, it would slot in as a video adapter implementation.

### B. StructTokenBench (arxiv 2503.00089)
- StructTokenBench evaluates protein structure tokenizers on local substructure quality.
- **tokenparity ≠ StructTokenBench**: single-modality (protein) vs cross-modality (3-domain). tokenparity's protein axis is one of three; StructTokenBench is protein-only and uses different sub-axes.

### C. LMMS-Eval v0.7 (Feb 2026)
- LMMS-Eval evaluates **downstream multimodal LLM tasks** (VQA, captioning, etc.).
- **tokenparity ≠ LMMS-Eval**: LMMS-Eval scores end-to-end LLM behaviour; tokenparity scores the tokenizer artefact itself. Different layer of the stack.

### D. "Beyond Text Compression" (arxiv 2506.03101)
- Evaluates **text tokenizers** across scales (compression as proxy for downstream).
- **tokenparity ≠ Beyond Text Compression**: text-only vs cross-modality; tokenparity refuses scalar fusion (Goodhart guard), so even on text-only it would not reduce to one number.

### E. Ordered Action Tokenization (RSS 2026)
- A new **robot action tokenizer design** (autoregressive prefix-decodable).
- **tokenparity ≠ OAT**: tokenparity is an eval harness; OAT is a candidate-under-test for the robotics axis.

## How this file is enforced

- `scripts/verify_step.py S6` greps `docs/DUPLICATION_DECLARATIONS.md` for the literal string of every entry above (6 internal + 5 external = 11 entries).
- README §"Differentiation" links to this file with a SHA256 pinned commit.
- CI fails closed if any declaration string is altered without bumping the SHA256 pin.
