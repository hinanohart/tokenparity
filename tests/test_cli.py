"""Tests for the CLI (cli.py)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tokenparity import __version__
from tokenparity.cli import build_parser, cmd_version, main


def test_version_subcommand_output(capsys: pytest.CaptureFixture) -> None:
    """'tokenparity version' must print 'tokenparity <version>'."""
    args = build_parser().parse_args(["version"])
    rc = cmd_version(args)
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == f"tokenparity {__version__}"


def test_version_via_main(capsys: pytest.CaptureFixture) -> None:
    """main(['version']) must print version and return 0."""
    rc = main(["version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert __version__ in captured.out


def test_version_subprocess() -> None:
    """'tokenparity version' via subprocess must output '0.1.0a1'."""
    result = subprocess.run(
        [sys.executable, "-m", "tokenparity.cli", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "0.1.0a1" in result.stdout


def test_no_subcommand_returns_nonzero(capsys: pytest.CaptureFixture) -> None:
    """No subcommand → non-zero exit code."""
    rc = main([])
    assert rc != 0


def test_unknown_subcommand_exits_nonzero() -> None:
    """Unknown subcommand must exit non-zero."""
    with pytest.raises(SystemExit) as exc_info:
        main(["nonexistent"])
    assert exc_info.value.code != 0


def test_parse_eval_args() -> None:
    """eval subcommand parser accepts domain/tokenizer/axis."""
    parser = build_parser()
    args = parser.parse_args(["eval", "--domain", "protein", "--tokenizer", "mock", "--axis", "A"])
    assert args.domain == "protein"
    assert args.tokenizer == "mock"
    assert args.axis == "A"


def test_parse_grid_args() -> None:
    """grid subcommand parser accepts --output."""
    parser = build_parser()
    args = parser.parse_args(["grid", "--synthetic", "--output", "/tmp/out.json"])
    assert args.output == "/tmp/out.json"
    assert args.synthetic is True


def test_version_string_matches_package() -> None:
    """CLI version must match tokenparity.__version__."""
    assert __version__ == "0.1.0a1"
