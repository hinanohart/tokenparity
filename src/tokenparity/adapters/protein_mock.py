"""Mock FoldToken adapter for the protein-structure domain.

This is an analytical-mock adapter; no live model inference occurs.
Payload convention: numpy array of shape (L, 3) representing Cα coordinates
in Ångströms, where L is the sequence length.

Tokenization: pairwise Chebyshev-normalised distances between residues are
binned into a fixed grid of vocab_size=256 codes.

native_score: 1.0 - clip(rmsd_to_canonical_helix, 0, 1)
  A perfect alpha-helix → score ≈ 1.0 (analytically verifiable ±1%).

License: Apache-2.0.  No GPL dependencies.
"""

from __future__ import annotations

import numpy as np

from tokenparity.core.types import Domain, Sample, Tokens

# Helix parameters (ideal α-helix geometry)
_HELIX_RISE_PER_RESIDUE = 1.5  # Å
_HELIX_RADIUS = 2.3  # Å
_HELIX_ROTATION_DEG = 100.0  # degrees per residue

VOCAB_SIZE = 256
_BIN_EDGES = np.linspace(0.0, 1.0, VOCAB_SIZE + 1)


def _canonical_helix(n_residues: int) -> np.ndarray:
    """Generate ideal alpha-helix Cα coordinates (shape: (n, 3))."""
    angles = np.deg2rad(_HELIX_ROTATION_DEG) * np.arange(n_residues)
    x = _HELIX_RADIUS * np.cos(angles)
    y = _HELIX_RADIUS * np.sin(angles)
    z = _HELIX_RISE_PER_RESIDUE * np.arange(n_residues)
    return np.stack([x, y, z], axis=1)


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Kabsch-free RMSD after centering (no rotation alignment for mock)."""
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    return float(np.sqrt(np.mean((a_c - b_c) ** 2)))


def _coords_to_token_ids(coords: np.ndarray) -> np.ndarray:
    """Encode (L, 3) coordinates to token ids via pairwise Chebyshev binning.

    Steps:
    1. Compute pairwise Chebyshev distances (L×L matrix).
    2. Flatten upper triangle.
    3. Normalise to [0, 1] by dividing by the 99th percentile.
    4. Bin into vocab_size=256 bins.
    Returns 1-D int array of ids.
    """
    L = coords.shape[0]
    if L < 2:
        return np.array([0], dtype=np.int64)

    # Pairwise Chebyshev distances
    diff = coords[:, None, :] - coords[None, :, :]  # (L, L, 3)
    cheb = np.max(np.abs(diff), axis=-1)  # (L, L)

    # Upper triangle indices
    triu_i, triu_j = np.triu_indices(L, k=1)
    distances = cheb[triu_i, triu_j]

    # Normalise
    p99 = np.percentile(distances, 99) if distances.size > 0 else 1.0
    if p99 == 0.0:
        p99 = 1.0
    normalised = np.clip(distances / p99, 0.0, 1.0)

    # Bin to integer codes
    ids = np.digitize(normalised, _BIN_EDGES[1:-1]).astype(np.int64)
    ids = np.clip(ids, 0, VOCAB_SIZE - 1)
    return ids  # type: ignore[no-any-return]


class MockFoldTokenAdapter:
    """Analytical-mock protein structure tokenizer.

    Encodes Cα coordinate arrays via pairwise Chebyshev-distance binning.
    Decodes by mapping bin centroids back to a flat coordinate array (shape
    is not preserved; only the token round-trip is tested).

    Parameters
    ----------
    seed:
        RNG seed for any stochastic components (none currently).
    """

    domain: Domain = "protein"
    name: str = "mock_foldtoken"
    license: str = "Apache-2.0"
    is_synthetic: bool = True

    def encode(self, x: Sample) -> Tokens:
        """Encode protein coordinates to token ids.

        Parameters
        ----------
        x:
            Sample with payload of shape ``(L, 3)``.

        Returns
        -------
        Tokens
            ids: pairwise-distance token ids, shape ``(n_pairs,)``
            latents: flattened raw coordinates, shape ``(L*3,)``
            vocab_size: 256
        """
        coords = np.asarray(x.payload, dtype=float)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 3)
        ids = _coords_to_token_ids(coords)
        latents = coords.flatten()
        return Tokens(
            domain="protein",
            ids=ids,
            vocab_size=VOCAB_SIZE,
            latents=latents,
        )

    def decode(self, tokens: Tokens) -> Sample:
        """Approximate decode: map bin centroids back to a flat coordinate array.

        Note: The decoded shape is ``(n_pairs,)`` (flattened pairwise distances),
        not the original ``(L, 3)``.  This is expected for a mock adapter; full
        inversion is not analytically possible from pairwise distances alone.
        """
        centroids = (_BIN_EDGES[tokens.ids] + _BIN_EDGES[tokens.ids + 1]) / 2.0
        return Sample(domain="protein", payload=centroids)

    def features(self, tokens: Tokens) -> np.ndarray:
        """Return the raw coordinate latents as a 2-D feature array.

        Returns shape ``(1, L*3)`` where L is the sequence length.
        """
        if tokens.latents is not None:
            return tokens.latents.reshape(1, -1)
        return tokens.ids.reshape(1, -1).astype(float)

    def native_score(self, sample: Sample) -> float:
        """Score how close the structure is to a canonical alpha-helix.

        Returns
        -------
        float
            ``1.0 - clip(rmsd_to_canonical_helix, 0, 1)``
            Perfect helix → 1.0; random coil → near 0.
        """
        coords = np.asarray(sample.payload, dtype=float)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 3)
        n = coords.shape[0]
        helix = _canonical_helix(n)
        rmsd = _rmsd(coords, helix)
        return float(1.0 - np.clip(rmsd, 0.0, 1.0))
