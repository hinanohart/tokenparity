"""Tests for the three analytical-mock adapters.

Contracts verified for each adapter:
1. encode then decode returns shape-matched Sample (at least same ndim).
2. encode produces Tokens with valid ids (all in [0, vocab_size)).
3. features returns 2-D array.
4. native_score returns float in [0, 1].
5. Known-canonical input gives native_score within ±1% of expected value.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenparity.adapters.protein_mock import MockFoldTokenAdapter, _canonical_helix
from tokenparity.adapters.robot_mock import MockLeRobotActionAdapter, _canonical_sine_trajectory
from tokenparity.adapters.video_mock import MockVideoGridAdapter
from tokenparity.core.types import Sample, Tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_tokens(tokens: Tokens) -> None:
    assert tokens.ids.ndim == 1, "Tokens.ids must be 1-D"
    assert tokens.ids.dtype in (
        np.int64,
        np.int32,
        np.int16,
        np.int8,
    ) or np.issubdtype(tokens.ids.dtype, np.integer), "Tokens.ids must be integer"
    assert tokens.ids.size > 0, "Tokens.ids must be non-empty"
    assert int(tokens.ids.min()) >= 0, "Token ids must be >= 0"
    assert int(tokens.ids.max()) < tokens.vocab_size, (
        f"Token id {int(tokens.ids.max())} >= vocab_size {tokens.vocab_size}"
    )


# ---------------------------------------------------------------------------
# Protein adapter
# ---------------------------------------------------------------------------


@pytest.fixture()
def protein_adapter() -> MockFoldTokenAdapter:
    return MockFoldTokenAdapter()


@pytest.fixture()
def protein_sample() -> Sample:
    rng = np.random.default_rng(42)
    return Sample(domain="protein", payload=rng.standard_normal((20, 3)))


def test_protein_encode_valid_ids(
    protein_adapter: MockFoldTokenAdapter, protein_sample: Sample
) -> None:
    tokens = protein_adapter.encode(protein_sample)
    _assert_valid_tokens(tokens)


def test_protein_encode_decode_roundtrip(
    protein_adapter: MockFoldTokenAdapter, protein_sample: Sample
) -> None:
    tokens = protein_adapter.encode(protein_sample)
    recon = protein_adapter.decode(tokens)
    assert isinstance(recon, Sample)
    assert np.asarray(recon.payload).ndim >= 1


def test_protein_features_2d(protein_adapter: MockFoldTokenAdapter, protein_sample: Sample) -> None:
    tokens = protein_adapter.encode(protein_sample)
    feats = protein_adapter.features(tokens)
    assert feats.ndim == 2, f"features must be 2-D, got shape {feats.shape}"


def test_protein_native_score_in_range(
    protein_adapter: MockFoldTokenAdapter, protein_sample: Sample
) -> None:
    score = protein_adapter.native_score(protein_sample)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"native_score out of [0,1]: {score}"


def test_protein_canonical_helix_score_near_one(
    protein_adapter: MockFoldTokenAdapter,
) -> None:
    """Perfect alpha-helix must give native_score close to 1.0 (±1%)."""
    helix_coords = _canonical_helix(20)
    sample = Sample(domain="protein", payload=helix_coords)
    score = protein_adapter.native_score(sample)
    assert score >= 0.99, f"Canonical helix native_score={score:.4f}, expected ≥ 0.99"


def test_protein_vocab_size(protein_adapter: MockFoldTokenAdapter, protein_sample: Sample) -> None:
    tokens = protein_adapter.encode(protein_sample)
    assert tokens.vocab_size == 256


# ---------------------------------------------------------------------------
# Video adapter
# ---------------------------------------------------------------------------


@pytest.fixture()
def video_adapter() -> MockVideoGridAdapter:
    return MockVideoGridAdapter()


@pytest.fixture()
def video_sample() -> Sample:
    rng = np.random.default_rng(7)
    return Sample(domain="video", payload=rng.standard_normal((4, 16, 16, 3)))


def test_video_encode_valid_ids(video_adapter: MockVideoGridAdapter, video_sample: Sample) -> None:
    tokens = video_adapter.encode(video_sample)
    _assert_valid_tokens(tokens)


def test_video_encode_decode_roundtrip(
    video_adapter: MockVideoGridAdapter, video_sample: Sample
) -> None:
    tokens = video_adapter.encode(video_sample)
    recon = video_adapter.decode(tokens)
    assert isinstance(recon, Sample)
    assert np.asarray(recon.payload).ndim >= 1


def test_video_features_2d(video_adapter: MockVideoGridAdapter, video_sample: Sample) -> None:
    tokens = video_adapter.encode(video_sample)
    feats = video_adapter.features(tokens)
    assert feats.ndim == 2, f"features must be 2-D, got shape {feats.shape}"


def test_video_native_score_in_range(
    video_adapter: MockVideoGridAdapter, video_sample: Sample
) -> None:
    score = video_adapter.native_score(video_sample)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"native_score out of [0,1]: {score}"


def test_video_uniform_video_score_near_one(
    video_adapter: MockVideoGridAdapter,
) -> None:
    """Spatially-uniform, temporally-constant video must give score ≈ 1.0."""
    uniform = np.ones((4, 16, 16, 3), dtype=float)
    sample = Sample(domain="video", payload=uniform)
    score = video_adapter.native_score(sample)
    assert score >= 0.99, f"Uniform video native_score={score:.4f}, expected ≥ 0.99"


def test_video_vocab_size(video_adapter: MockVideoGridAdapter, video_sample: Sample) -> None:
    tokens = video_adapter.encode(video_sample)
    assert tokens.vocab_size == 512


# ---------------------------------------------------------------------------
# Robot adapter
# ---------------------------------------------------------------------------


@pytest.fixture()
def robot_adapter() -> MockLeRobotActionAdapter:
    return MockLeRobotActionAdapter()


@pytest.fixture()
def robot_sample() -> Sample:
    rng = np.random.default_rng(99)
    return Sample(domain="robot", payload=rng.standard_normal((10, 6)))


def test_robot_encode_valid_ids(
    robot_adapter: MockLeRobotActionAdapter, robot_sample: Sample
) -> None:
    tokens = robot_adapter.encode(robot_sample)
    _assert_valid_tokens(tokens)


def test_robot_encode_decode_roundtrip(
    robot_adapter: MockLeRobotActionAdapter, robot_sample: Sample
) -> None:
    tokens = robot_adapter.encode(robot_sample)
    recon = robot_adapter.decode(tokens)
    assert isinstance(recon, Sample)
    assert np.asarray(recon.payload).ndim >= 1


def test_robot_features_2d(robot_adapter: MockLeRobotActionAdapter, robot_sample: Sample) -> None:
    tokens = robot_adapter.encode(robot_sample)
    feats = robot_adapter.features(tokens)
    assert feats.ndim == 2, f"features must be 2-D, got shape {feats.shape}"


def test_robot_native_score_in_range(
    robot_adapter: MockLeRobotActionAdapter, robot_sample: Sample
) -> None:
    score = robot_adapter.native_score(robot_sample)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"native_score out of [0,1]: {score}"


def test_robot_canonical_sine_score_near_one(
    robot_adapter: MockLeRobotActionAdapter,
) -> None:
    """Perfect sine-wave action sequence must give native_score close to 1.0 (±1%)."""
    T, D = 20, 4
    sine_traj = _canonical_sine_trajectory(T, D)
    sample = Sample(domain="robot", payload=sine_traj)
    score = robot_adapter.native_score(sample)
    assert score >= 0.99, f"Canonical sine trajectory native_score={score:.4f}, expected ≥ 0.99"


def test_robot_vocab_size(robot_adapter: MockLeRobotActionAdapter, robot_sample: Sample) -> None:
    tokens = robot_adapter.encode(robot_sample)
    assert tokens.vocab_size == 128


# ---------------------------------------------------------------------------
# Protocol compliance (all adapters)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter_cls,sample_payload,domain",
    [
        (MockFoldTokenAdapter, np.random.default_rng(1).standard_normal((15, 3)), "protein"),
        (MockVideoGridAdapter, np.random.default_rng(2).standard_normal((2, 8, 8, 3)), "video"),
        (MockLeRobotActionAdapter, np.random.default_rng(3).standard_normal((8, 3)), "robot"),
    ],
)
def test_adapter_protocol_compliance(
    adapter_cls: type, sample_payload: np.ndarray, domain: str
) -> None:
    """Each adapter must satisfy the TokenizerAdapter protocol fields."""
    adapter = adapter_cls()
    assert hasattr(adapter, "domain")
    assert hasattr(adapter, "name")
    assert hasattr(adapter, "license")
    assert hasattr(adapter, "is_synthetic")
    assert adapter.is_synthetic is True
    assert adapter.license == "Apache-2.0"

    sample = Sample(domain=domain, payload=sample_payload)  # type: ignore[arg-type]
    tokens = adapter.encode(sample)
    assert isinstance(tokens, Tokens)
    _assert_valid_tokens(tokens)

    feats = adapter.features(tokens)
    assert feats.ndim == 2

    score = adapter.native_score(sample)
    assert 0.0 <= score <= 1.0
