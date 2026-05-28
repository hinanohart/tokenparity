"""Mock LeRobot action adapter for the robotics domain.

Payload convention: numpy array of shape (T, D_action) — T timesteps,
D_action action dimensions (e.g. joint angles + gripper).

Tokenization: per-timestep action vectors are uniformly binned per dimension,
then mapped to a flat integer token via a mixed-radix encoding.
vocab_size = 128 (4 bins per dim × up to D_action dims folded into 128 codes).

native_score: 1 / (1 + ||payload - canonical_trajectory||)
  canonical_trajectory is a sine wave over [0, 2π].
  Perfect sine-wave action → score ≈ 1.0.

License: Apache-2.0.  No GPL dependencies.
"""

from __future__ import annotations

import numpy as np

from tokenparity.core.types import Domain, Sample, Tokens

VOCAB_SIZE = 128
_N_BINS = 4  # bins per action dimension
_BIN_EDGES = np.linspace(-1.0, 1.0, _N_BINS + 1)
_BIN_CENTROIDS = (_BIN_EDGES[:-1] + _BIN_EDGES[1:]) / 2.0  # shape (_N_BINS,)


def _canonical_sine_trajectory(T: int, D: int) -> np.ndarray:
    """Generate a canonical sine-wave trajectory of shape (T, D).

    Each dimension d oscillates as sin(2π * t / T + d * π / D).
    """
    t = np.linspace(0, 2 * np.pi, T)
    d = np.arange(D)
    phase = d * np.pi / max(D, 1)
    return np.sin(t[:, None] + phase[None, :])  # (T, D)


def _encode_timestep(action: np.ndarray) -> int:
    """Encode a single D-dimensional action vector to a token id in [0, 128).

    Steps:
    1. Clip to [-1, 1].
    2. Bin each dimension to 0..3.
    3. Hash the bin tuple to [0, 128) via a modular sum.
    """
    clipped = np.clip(action, -1.0, 1.0)
    bins = np.digitize(clipped, _BIN_EDGES[1:-1]).astype(np.int64)  # 0..3 per dim
    bins = np.clip(bins, 0, _N_BINS - 1)
    # Mixed-radix fold: sum weighted bins mod vocab_size
    weights = np.array([_N_BINS**i for i in range(len(bins))], dtype=np.int64)
    code = int(np.sum(bins * weights) % VOCAB_SIZE)
    return code


def _decode_timestep(token_id: int, D: int) -> np.ndarray:
    """Approximate decode: reconstruct bin centroids from token id.

    Since the encoding is many-to-one (mixed-radix mod), the decode is
    approximate.  We distribute the token id across D dims via modular
    arithmetic to recover approximate bin indices.
    """
    centroids = np.zeros(D, dtype=float)
    remaining = token_id
    for d in range(D):
        bin_idx = remaining % _N_BINS
        centroids[d] = _BIN_CENTROIDS[bin_idx]
        remaining = remaining // _N_BINS
    return centroids


class MockLeRobotActionAdapter:
    """Analytical-mock LeRobot action sequence tokenizer.

    Tokenizes per-timestep action vectors via uniform binning per dimension.

    Parameters
    ----------
    pad_to:
        Pad or truncate the action sequence to this length before encoding.
        Default None (no padding).
    """

    domain: Domain = "robot"
    name: str = "mock_lerobot"
    license: str = "Apache-2.0"
    is_synthetic: bool = True

    def __init__(self, pad_to: int | None = None) -> None:
        self.pad_to = pad_to

    def encode(self, x: Sample) -> Tokens:
        """Encode a robot action sequence to token ids.

        Parameters
        ----------
        x:
            Sample with payload shape ``(T, D_action)``.

        Returns
        -------
        Tokens
            ids: per-timestep token ids, shape ``(T,)``
            latents: padded action sequence, shape ``(1, T*D)``
            vocab_size: 128
        """
        actions = np.asarray(x.payload, dtype=float)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        T, D = actions.shape
        if self.pad_to is not None:
            if T < self.pad_to:
                pad = np.zeros((self.pad_to - T, D), dtype=float)
                actions = np.vstack([actions, pad])
            else:
                actions = actions[: self.pad_to]
            T = actions.shape[0]

        ids = np.array([_encode_timestep(actions[t]) for t in range(T)], dtype=np.int64)
        latents = actions.flatten()

        return Tokens(
            domain="robot",
            ids=ids,
            vocab_size=VOCAB_SIZE,
            latents=latents,
        )

    def decode(self, tokens: Tokens) -> Sample:
        """Approximate decode: reconstruct action sequence from token ids.

        The decoded payload has shape ``(T,)`` (flattened approximate actions).
        Full D-dimensional reconstruction requires knowing D, which is not
        stored in Tokens; we use D=1 as a conservative default for mock purposes.
        """
        # Reconstruct with D=1 (conservative; sufficient for round-trip shape test)
        recon = np.array([_decode_timestep(int(tok_id), D=1)[0] for tok_id in tokens.ids])
        return Sample(domain="robot", payload=recon)

    def features(self, tokens: Tokens) -> np.ndarray:
        """Return the flattened action sequence latents, shape ``(1, T*D)``."""
        if tokens.latents is not None:
            return tokens.latents.reshape(1, -1)
        return tokens.ids.reshape(1, -1).astype(float)

    def native_score(self, sample: Sample) -> float:
        """Score based on distance to a canonical sine-wave trajectory.

        Score = 1 / (1 + ||payload - canonical_trajectory||_F)
        Perfect sine wave → score ≈ 1.0.
        """
        actions = np.asarray(sample.payload, dtype=float)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        T, D = actions.shape
        canonical = _canonical_sine_trajectory(T, D)
        dist = float(np.linalg.norm(actions - canonical))
        return float(1.0 / (1.0 + dist))
