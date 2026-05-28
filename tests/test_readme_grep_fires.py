"""Negative fixture: confirm the README banned-phrase regex actually fires.

This is the BRE/ERE regression guard requested by the a1 audit.  We assert
that the literal banned regex (mirrored from ``.github/workflows/readme-grep.yml``)
matches each banned phrase when given as input.  Failure means a future edit
of the workflow has silently turned the gate into dead code (e.g. switching
``grep -E`` to default BRE while the alternation uses ``|`` literals).

The regex string here MUST stay byte-identical to the workflow's ``banned=``
literal; the test reads the workflow file and compares.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "readme-grep.yml"

EXPECTED_BANNED = (
    "revolutionary|world-first|sota|state-of-the-art|完全|完璧|永続|永久|forever|never fails"
)


def _extract_banned_from_workflow() -> str:
    """Read the workflow file and return the literal banned regex string."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    match = re.search(r"banned='([^']+)'", text)
    assert match is not None, (
        f"Workflow {WORKFLOW_PATH} no longer contains a 'banned=...' literal; "
        "the regex sync between code and CI has broken."
    )
    return match.group(1)


def test_workflow_banned_regex_matches_test_constant() -> None:
    """Source-of-truth check: workflow banned= must equal EXPECTED_BANNED.

    Keeps the banned list in sync between CI yaml and this fixture.
    """
    actual = _extract_banned_from_workflow()
    assert actual == EXPECTED_BANNED, (
        "Workflow banned regex drifted from test constant.\n"
        f"  workflow: {actual!r}\n"
        f"  expected: {EXPECTED_BANNED!r}\n"
        "Update EXPECTED_BANNED in this file or fix the workflow."
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "revolutionary",
        "world-first",
        "sota",
        "state-of-the-art",
        "完全",
        "完璧",
        "永続",
        "永久",
        "forever",
        "never fails",
        # case-insensitivity (grep -E -i)
        "Revolutionary",
        "SOTA",
        "FOREVER",
    ],
)
def test_grep_rc_zero_on_banned_phrase(tmp_path: Path, phrase: str) -> None:
    """For each banned phrase, ``grep -E -i`` returns rc=0 (match).

    This is the negative test: if the gate were dead (e.g. BRE/ERE confusion),
    rc would be 1 and CI would silently pass on overclaim README updates.
    """
    fixture = tmp_path / "README_BAD.md"
    fixture.write_text(f"# tokenparity\n\nThis is a {phrase} release.\n", encoding="utf-8")

    result = subprocess.run(
        ["grep", "-E", "-i", EXPECTED_BANNED, str(fixture)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"grep -E -i FAILED to match banned phrase {phrase!r} "
        f"(rc={result.returncode}, stdout={result.stdout!r}). "
        "The README honest-marketing gate has become dead code."
    )


def test_grep_rc_one_on_clean_text(tmp_path: Path) -> None:
    """A clean README returns rc=1 (no match) — confirms the gate doesn't false-fire."""
    fixture = tmp_path / "README_OK.md"
    fixture.write_text(
        "# tokenparity\n\nA GPU-free harness that measures discrete tokenizers.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["grep", "-E", "-i", EXPECTED_BANNED, str(fixture)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        f"grep matched on clean text (rc={result.returncode}, stdout={result.stdout!r}). "
        "The README gate is false-positive."
    )
