"""Integration tests for CLI subcommands (eval + grid).

These tests exercise the actual adapter loading and engine paths via
the CLI entry point, covering the code paths missed by unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenparity.cli import build_parser, cmd_eval, cmd_grid, main


@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path / "out.json"


# ---------------------------------------------------------------------------
# cmd_eval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["protein", "video", "robot"])
def test_cmd_eval_axis_a(
    domain: str, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eval --axis A must succeed for all domains and print JSON."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    args = build_parser().parse_args(
        ["eval", "--domain", domain, "--tokenizer", "mock", "--axis", "A", "--n-samples", "3"]
    )
    rc = cmd_eval(args)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "A" in data


@pytest.mark.parametrize("domain", ["protein", "video", "robot"])
def test_cmd_eval_axis_c(
    domain: str, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eval --axis C must succeed for all domains."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    args = build_parser().parse_args(
        ["eval", "--domain", domain, "--tokenizer", "mock", "--axis", "C", "--n-samples", "3"]
    )
    rc = cmd_eval(args)
    assert rc == 0


def test_cmd_eval_all_axes(capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """eval --axis all must return A, B, C keys. Needs >=8 samples for kNN k=7."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    args = build_parser().parse_args(
        ["eval", "--domain", "protein", "--tokenizer", "mock", "--axis", "all", "--n-samples", "20"]
    )
    rc = cmd_eval(args)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert set(data.keys()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# cmd_grid
# ---------------------------------------------------------------------------


def test_cmd_grid_writes_json(tmp_output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grid command must write a JSON file with expected keys."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    args = build_parser().parse_args(
        ["grid", "--synthetic", "--output", str(tmp_output), "--n-samples", "3"]
    )
    rc = cmd_grid(args)
    assert rc == 0
    assert tmp_output.exists()
    data = json.loads(tmp_output.read_text())
    assert "domains" in data
    assert "tokenizers" in data
    assert "axes" in data
    assert "values" in data
    assert "radar" in data
    assert "pareto_front" in data


def test_cmd_grid_domains_correct(tmp_output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Grid output must include all 3 domains."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    args = build_parser().parse_args(
        ["grid", "--synthetic", "--output", str(tmp_output), "--n-samples", "3"]
    )
    cmd_grid(args)
    data = json.loads(tmp_output.read_text())
    assert set(data["domains"]) == {"protein", "video", "robot"}


def test_cmd_grid_axes_correct(tmp_output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Grid output must include all 3 axes."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    args = build_parser().parse_args(
        ["grid", "--synthetic", "--output", str(tmp_output), "--n-samples", "3"]
    )
    cmd_grid(args)
    data = json.loads(tmp_output.read_text())
    assert data["axes"] == ["A", "B", "C"]


def test_cmd_grid_no_scalar_total(tmp_output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Grid JSON output must not contain a 'total_score' key."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    args = build_parser().parse_args(
        ["grid", "--synthetic", "--output", str(tmp_output), "--n-samples", "3"]
    )
    cmd_grid(args)
    data = json.loads(tmp_output.read_text())
    assert "total_score" not in data


def test_cmd_grid_version_field(tmp_output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Grid JSON must include the version field."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    args = build_parser().parse_args(
        ["grid", "--synthetic", "--output", str(tmp_output), "--n-samples", "3"]
    )
    cmd_grid(args)
    data = json.loads(tmp_output.read_text())
    from tokenparity import __version__

    assert data["version"] == __version__


def test_main_eval_end_to_end(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() with eval args must exit 0 and produce valid JSON."""
    monkeypatch.delenv("KINETOKEN_REAL_DATA", raising=False)
    rc = main(
        ["eval", "--domain", "robot", "--tokenizer", "mock", "--axis", "A", "--n-samples", "2"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["A"]["axis"] == "A"
