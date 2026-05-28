"""tokenparity: GPU-free cross-modality tokenizer parity harness.

Three modalities (protein structure / video / robotics) × three honest axes
(downstream transferability / information bottleneck / per-domain normalized
reconstruction). Apache-2.0, synthetic + cached-token only in v0.1.0a1.
"""

__version__ = "0.1.0a1"

from tokenparity.core.types import Sample, TokenizerAdapter, Tokens

__all__ = ["__version__", "Sample", "Tokens", "TokenizerAdapter"]
