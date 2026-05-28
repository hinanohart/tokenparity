"""Phase verification script for tokenparity CI.

Usage::

    python scripts/verify_step.py S5   # assert ≥60 tests passed
    python scripts/verify_step.py S6   # README sha256 matches bench_results JSON
    python scripts/verify_step.py S8   # vacuous test detector

Exit codes: 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# S5: assert pytest passed with ≥60 tests
# ---------------------------------------------------------------------------


def verify_s5() -> bool:
    """Run pytest and assert ≥60 tests passed."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests"),
            "--no-header",
            "--tb=no",
            "--no-cov",
            "-v",  # verbose: ensures "X passed" appears in output
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    output = result.stdout + result.stderr
    print(output)

    # Parse "X passed" from pytest verbose output
    match = re.search(r"(\d+) passed", output)
    if not match:
        # Fallback: count "PASSED" lines
        passed = output.count(" PASSED")
        if passed == 0:
            print("FAIL [S5]: Could not parse pytest output for passed count.")
            return False
    else:
        passed = int(match.group(1))

    passed = int(match.group(1))
    if passed < 60:
        print(f"FAIL [S5]: {passed} tests passed, need ≥60.")
        return False

    print(f"PASS [S5]: {passed} tests passed (≥60).")
    return True


# ---------------------------------------------------------------------------
# S6: README sha256 matches bench_results/synthetic_v0.1.0a1.json
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_s6() -> bool:
    """Check that README contains the sha256 of bench_results/synthetic_v0.1.0a1.json."""
    bench_path = ROOT / "bench_results" / "synthetic_v0.1.0a1.json"
    readme_path = ROOT / "README.md"

    if not bench_path.exists():
        print(f"FAIL [S6]: {bench_path} does not exist.")
        return False

    if not readme_path.exists():
        print(f"FAIL [S6]: {readme_path} does not exist.")
        return False

    actual_sha = _sha256_of_file(bench_path)
    readme_text = readme_path.read_text()

    if actual_sha not in readme_text:
        print(f"FAIL [S6]: SHA256 {actual_sha} not found in README.md.")
        print(f"  bench_results file: {bench_path}")
        return False

    print(f"PASS [S6]: README contains correct sha256 {actual_sha[:16]}...")
    return True


# ---------------------------------------------------------------------------
# S8: vacuous test detector
# ---------------------------------------------------------------------------


def _is_vacuous_body(body: list[ast.stmt]) -> bool:
    """Return True if a function body consists only of pass or `assert True`."""
    real_stmts = [s for s in body if not isinstance(s, (ast.Pass, ast.Expr))]

    # Check for `assert True` only bodies
    non_trivial = False
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            # Bare string literal (docstring) — skip
            continue
        if isinstance(stmt, ast.Assert):
            # `assert True` is vacuous
            if isinstance(stmt.test, ast.Constant) and stmt.test.value is True:
                continue
            # Any other assert is meaningful
            non_trivial = True
        else:
            non_trivial = True

    return not non_trivial


def verify_s8() -> bool:
    """Scan tests/ for vacuous test functions (body is only pass or assert True)."""
    tests_dir = ROOT / "tests"
    vacuous: list[str] = []

    for test_file in sorted(tests_dir.glob("test_*.py")):
        try:
            tree = ast.parse(test_file.read_text())
        except SyntaxError as exc:
            print(f"FAIL [S8]: SyntaxError in {test_file}: {exc}")
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if _is_vacuous_body(node.body):
                    vacuous.append(f"{test_file.name}:{node.lineno}:{node.name}")

    if vacuous:
        print(f"FAIL [S8]: {len(vacuous)} vacuous test(s) detected:")
        for v in vacuous:
            print(f"  {v}")
        return False

    print(f"PASS [S8]: No vacuous tests found in {tests_dir}.")
    return True


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_VERIFIERS = {
    "S5": verify_s5,
    "S6": verify_s6,
    "S8": verify_s8,
}


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) != 2 or sys.argv[1] not in _VERIFIERS:
        print(f"Usage: python scripts/verify_step.py [{' | '.join(_VERIFIERS)}]")
        return 1

    step = sys.argv[1]
    ok = _VERIFIERS[step]()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
