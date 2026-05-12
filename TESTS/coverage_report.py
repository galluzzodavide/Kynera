"""
coverage_report.py
==================
Runs the Kynera test suite and produces an HTML coverage report.

Usage:
    python coverage_report.py

Output:
    htmlcov/index.html   — open in any browser to inspect line-by-line coverage

Requirements:
    pip install pytest pytest-cov coverage
"""

import subprocess
import sys
import os

REPORT_DIR = "htmlcov"

def main():
    print("=" * 60)
    print("Kynera — Test suite + Coverage report")
    print("=" * 60)

    # 1. Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "test.py",
        "-v",
        "--tb=short",
        f"--cov=kynera",
        "--cov-branch",                      # branch coverage (not just line)
        f"--cov-report=html:{REPORT_DIR}",   # HTML report
        "--cov-report=term-missing",         # also print summary to terminal
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    # 2. Summary
    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("All tests passed.")
    else:
        print(f"Some tests failed (exit code {result.returncode}).")

    report_path = os.path.abspath(os.path.join(REPORT_DIR, "index.html"))
    if os.path.exists(report_path):
        print(f"\n HTML coverage report: {report_path}")
        print("   Open it in your browser to inspect line-by-line coverage.")
    else:
        print("\n  HTML report not generated — check pytest-cov is installed:")
        print("   pip install pytest pytest-cov coverage")

    print("=" * 60)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
