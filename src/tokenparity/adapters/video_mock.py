"""Mock video grid adapter for the video domain.

Payload convention: numpy array of shape (T, H, W, C) — T frames, H×W
spatial resolution, C channels (typically 3 for RGB).

Tokenization: 4×4 spatial patches are L2-quantised to a synthetic codebook
of vocab_size=512 codewords.  The codebook is deterministically generated
from a seeded RNG and is fixed across all instances.

native_score: 1 - var(mean_per_frame) / (1 + var(mean_per_frame))
  Spatially-uniform video (low temporal variance) → score ≈ 1.
  High-variance / flickering video → score near 0.

License: Apache-2.0.  No GPL dependencies.
"""

from __future__ import annotations

import numpy as np

from tokenparity.core.types import Domain, Sample, Tokens

VOCAB_SIZE = 512
_PATCH_SIZE = 4
_CODEBOOK_SEED = 12345
_CODEBOOK_DIM = _PATCH_SIZE * _PATCH_SIZE * 3  # 48


def _build_codebook(vocab_size: int, dim: int, seed: int) -> np.ndarray:
    """Build a fixed synthetic codebook of shape (vocab_size, dim)."""
    rng = np.random.default_rng(seed)
    cb = rng.standard_normal((vocab_size, dim)).astype(np.float32)
    # L2-normalise rows for stable quantisation
    norms = np.linalg.norm(cb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return cb / norms


_CODEBOOK: np.ndarray = _build_codebook(VOCAB_SIZE, _CODEBOOK_DIM, _CODEBOOK_SEED)


def _extract_patches(frame: np.ndarray, patch_size: int = _PATCH_SIZE) -> np.ndarray:
    """Extract non-overlapping patches from a single frame (H, W, C).

    Returns shape (n_patches, patch_size*patch_size*C).
    """
    H, W, C = frame.shape
    pH = H // patch_size
    pW = W // patch_size
    patches = frame[: pH * patch_size, : pW * patch_size, :]
    patches = patches.reshape(pH, patch_size, pW, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * C)
    return patches.astype(np.float32)  # type: ignore[no-any-return]


def _quantise_patches(patches: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Assign each patch to the nearest codebook entry via L2 distance.

    Parameters
    ----------
    patches:
        Shape ``(n_patches, dim)``.
    codebook:
        Shape ``(vocab_size, dim)``.

    Returns
    -------
    np.ndarray
        Integer ids, shape ``(n_patches,)``.
    """
    # Trim or pad patches to match codebook dim
    dim = codebook.shape[1]
    p = patches[:, :dim]
    if p.shape[1] < dim:
        pad = np.zeros((p.shape[0], dim - p.shape[1]), dtype=np.float32)
        p = np.hstack([p, pad])

    # L2-normalise patches
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    p_normed = p / norms

    # Nearest neighbour via dot product (since both are unit vectors)
    similarities = p_normed @ codebook.T  # (n_patches, vocab_size)
    return np.argmax(similarities, axis=1).astype(np.int64)  # type: ignore[no-any-return]


class MockVideoGridAdapter:
    """Analytical-mock video grid tokenizer.

    Tokenizes video frames via 4×4 patch L2 quantisation.
    The codebook is fixed and deterministic (seeded).

    Parameters
    ----------
    patch_size:
        Spatial patch size.  Default 4.
    """

    domain: Domain = "video"
    name: str = "mock_videogrid"
    license: str = "Apache-2.0"
    is_synthetic: bool = True

    def __init__(self, patch_size: int = _PATCH_SIZE) -> None:
        self.patch_size = patch_size

    def encode(self, x: Sample) -> Tokens:
        """Encode a video array to token ids.

        Parameters
        ----------
        x:
            Sample with payload shape ``(T, H, W, C)``.

        Returns
        -------
        Tokens
            ids: flattened patch token ids across all frames
            latents: mean-pooled patch features across all frames, shape ``(1, dim)``
            vocab_size: 512
        """
        video = np.asarray(x.payload, dtype=np.float32)
        if video.ndim == 3:  # (H, W, C) → (1, H, W, C)
            video = video[np.newaxis]

        all_ids: list[np.ndarray] = []
        all_patch_vecs: list[np.ndarray] = []

        for t in range(video.shape[0]):
            frame = video[t]
            patches = _extract_patches(frame, self.patch_size)
            ids = _quantise_patches(patches, _CODEBOOK)
            all_ids.append(ids)
            all_patch_vecs.append(patches)

        combined_ids = np.concatenate(all_ids)
        combined_patches = np.concatenate(all_patch_vecs, axis=0)
        mean_features = combined_patches.mean(axis=0, keepdims=True)

        return Tokens(
            domain="video",
            ids=combined_ids,
            vocab_size=VOCAB_SIZE,
            latents=mean_features,
        )

    def decode(self, tokens: Tokens) -> Sample:
        """Approximate decode: map token ids back to codebook vectors.

        Returns a 1-D array of concatenated codebook entries.
        Shape: (n_tokens * codebook_dim,).
        """
        recon = _CODEBOOK[tokens.ids].flatten()
        return Sample(domain="video", payload=recon)

    def features(self, tokens: Tokens) -> np.ndarray:
        """Return mean-pooled patch features, shape ``(1, dim)``."""
        if tokens.latents is not None:
            return tokens.latents.reshape(1, -1)
        return _CODEBOOK[tokens.ids].mean(axis=0, keepdims=True)  # type: ignore[no-any-return]

    def native_score(self, sample: Sample) -> float:
        """Score based on temporal variance of spatial mean.

        Score = 1 - var(mean_per_frame) / (1 + var(mean_per_frame))
        Uniform/stable video → score close to 1.
        High-variance video → score near 0.5.
        """
        video = np.asarray(sample.payload, dtype=float)
        if video.ndim == 3:
            video = video[np.newaxis]
        # Mean across spatial dims (H, W, C) for each frame
        frame_means = video.mean(axis=(1, 2, 3))  # shape (T,)
        var = float(np.var(frame_means))
        return float(1.0 - var / (1.0 + var))
