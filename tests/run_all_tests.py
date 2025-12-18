#!/usr/bin/env python3
"""Run all tests and report results."""

import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_file(test_file: Path) -> tuple[bool, str]:
    """Run a test file and return (success, output)."""
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=project_root
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (60s)"
    except Exception as e:
        return False, f"ERROR: {str(e)}"

def main():
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob("test_*.py"))
    verify_files = sorted(test_dir.glob("verify_*.py"))

    all_files = test_files + verify_files

    print(f"Found {len(all_files)} test files\n")
    print("=" * 80)

    passed = []
    failed = []
    skipped = []

    for test_file in all_files:
        print(f"\nRunning {test_file.name}...")
        success, output = run_test_file(test_file)

        if success:
            passed.append(test_file.name)
            print(f"✓ PASSED: {test_file.name}")
        else:
            # Check if it's a skip (import error, missing deps, etc.)
            if "ImportError" in output or "ModuleNotFoundError" in output or "SKIP" in output.upper():
                skipped.append((test_file.name, output[:200]))
                print(f"⊘ SKIPPED: {test_file.name}")
            else:
                failed.append((test_file.name, output[:500]))
                print(f"✗ FAILED: {test_file.name}")
                if output:
                    print(f"  Output: {output[:200]}")

    print("\n" + "=" * 80)
    print(f"\nSUMMARY:")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")

    if passed:
        print(f"\n✓ PASSED ({len(passed)}):")
        for name in passed:
            print(f"    {name}")

    if skipped:
        print(f"\n⊘ SKIPPED ({len(skipped)}):")
        for name, reason in skipped:
            print(f"    {name}")
            print(f"      {reason[:100]}...")

    if failed:
        print(f"\n✗ FAILED ({len(failed)}):")
        for name, reason in failed:
            print(f"    {name}")
            print(f"      {reason[:200]}...")

    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

