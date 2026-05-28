"""Core types for tokenparity.

Sample / Tokens IR is intentionally domain-agnostic; per-domain semantics
live in the adapter implementation, not in the IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

Domain = Literal["protein", "video", "robot"]


@dataclass(frozen=True)
class Sample:
    """A domain-agnostic input sample.

    `payload` carries the modality-native object (numpy array of coordinates
    for protein, frame tensor for video, action sequence for robot). The
    adapter is responsible for interpreting it.
    """

    domain: Domain
    payload: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Tokens:
    """A discrete token sequence produced by an adapter."""

    domain: Domain
    ids: np.ndarray  # 1-D int array
    vocab_size: int
    latents: np.ndarray | None = None  # optional continuous side-channel for axis B
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ids.ndim != 1:
            raise ValueError(f"Tokens.ids must be 1-D, got shape {self.ids.shape}")
        if not np.issubdtype(self.ids.dtype, np.integer):
            raise ValueError(f"Tokens.ids must be integer dtype, got {self.ids.dtype}")
        if self.ids.size > 0 and (self.ids.min() < 0 or self.ids.max() >= self.vocab_size):
            raise ValueError(
                f"Tokens.ids out of range: min={self.ids.min()}, max={self.ids.max()}, "
                f"vocab_size={self.vocab_size}"
            )


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Protocol every tokenizer adapter must satisfy.

    Synthetic adapters (`is_synthetic=True`) are the only thing CI exercises.
    Real adapters require `KINETOKEN_REAL_DATA=1` to be set.
    """

    domain: Domain
    name: str
    license: str  # SPDX identifier
    is_synthetic: bool

    def encode(self, x: Sample) -> Tokens: ...

    def decode(self, tokens: Tokens) -> Sample: ...

    def features(self, tokens: Tokens) -> np.ndarray:
        """Continuous feature representation for axis A (downstream transfer)."""
        ...

    def native_score(self, sample: Sample) -> float:
        """Baseline score on the *native* (non-frozen) pipeline for axis A."""
        ...
