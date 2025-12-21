#!/usr/bin/env python3
"""Simple test runner for integration tests.

This script validates that tests can be imported and have correct structure.
It doesn't require all dependencies to be installed.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_test_imports():
    """Check that test modules can be imported."""
    errors = []
    warnings = []

    # Check if pytest is available
    try:
        import pytest
        pytest_available = True
    except ImportError:
        pytest_available = False
        warnings.append("pytest not installed - some checks will be skipped")

    try:
        # Try to import without pytest dependency
        if pytest_available:
            import tests.integration.test_linguistic_quality
            print("✓ test_linguistic_quality.py imports successfully")
        else:
            # Check syntax only
            import py_compile
            py_compile.compile('tests/integration/test_linguistic_quality.py', doraise=True)
            print("✓ test_linguistic_quality.py syntax is valid (pytest not available)")
    except Exception as e:
        if "pytest" in str(e).lower():
            warnings.append("test_linguistic_quality.py requires pytest")
            print("⚠ test_linguistic_quality.py requires pytest (not installed)")
        else:
            errors.append(f"test_linguistic_quality.py: {e}")
            print(f"✗ test_linguistic_quality.py: {e}")

    try:
        if pytest_available:
            import tests.integration.test_structural_integrity
            print("✓ test_structural_integrity.py imports successfully")
        else:
            import py_compile
            py_compile.compile('tests/integration/test_structural_integrity.py', doraise=True)
            print("✓ test_structural_integrity.py syntax is valid (pytest not available)")
    except Exception as e:
        if "pytest" in str(e).lower():
            warnings.append("test_structural_integrity.py requires pytest")
            print("⚠ test_structural_integrity.py requires pytest (not installed)")
        else:
            errors.append(f"test_structural_integrity.py: {e}")
            print(f"✗ test_structural_integrity.py: {e}")

    try:
        if pytest_available:
            import tests.integration.test_narrative_flow
            print("✓ test_narrative_flow.py imports successfully")
        else:
            import py_compile
            py_compile.compile('tests/integration/test_narrative_flow.py', doraise=True)
            print("✓ test_narrative_flow.py syntax is valid (pytest not available)")
    except Exception as e:
        if "pytest" in str(e).lower():
            warnings.append("test_narrative_flow.py requires pytest")
            print("⚠ test_narrative_flow.py requires pytest (not installed)")
        else:
            errors.append(f"test_narrative_flow.py: {e}")
            print(f"✗ test_narrative_flow.py: {e}")

    try:
        import tests.metrics.track_llm_calls
        print("✓ track_llm_calls.py imports successfully")
    except Exception as e:
        errors.append(f"track_llm_calls.py: {e}")
        print(f"✗ track_llm_calls.py: {e}")

    return len(errors) == 0, errors


def check_test_structure():
    """Check that test files have correct structure."""
    errors = []

    # Check if pytest is available
    try:
        import pytest
        pytest_available = True
    except ImportError:
        pytest_available = False
        print("⚠ pytest not available - skipping structure checks")
        return True, []  # Skip if pytest not available

    # Check linguistic quality tests
    try:
        from tests.integration import test_linguistic_quality
        classes = [
            'TestAntiStutterZipperMerge',
            'TestActionEchoDetection',
            'TestGroundingValidation',
            'TestPerspectiveLock'
        ]
        for cls_name in classes:
            if hasattr(test_linguistic_quality, cls_name):
                print(f"✓ Found test class: {cls_name}")
            else:
                errors.append(f"Missing test class: {cls_name}")
                print(f"✗ Missing test class: {cls_name}")
    except Exception as e:
        errors.append(f"Error checking test_linguistic_quality structure: {e}")
        print(f"✗ Error checking test_linguistic_quality structure: {e}")

    # Check structural integrity tests
    try:
        from tests.integration import test_structural_integrity
        classes = [
            'TestImpossibleConstraint',
            'TestEmptySlot',
            'TestSledgehammerConvergence'
        ]
        for cls_name in classes:
            if hasattr(test_structural_integrity, cls_name):
                print(f"✓ Found test class: {cls_name}")
            else:
                errors.append(f"Missing test class: {cls_name}")
                print(f"✗ Missing test class: {cls_name}")
    except Exception as e:
        errors.append(f"Error checking test_structural_integrity structure: {e}")
        print(f"✗ Error checking test_structural_integrity structure: {e}")

    return len(errors) == 0, errors


def check_verify_perspective():
    """Check that verify_perspective method exists."""
    try:
        # Try to import, but handle missing dependencies gracefully
        try:
            from src.generator.translator import StyleTranslator
        except ImportError as e:
            if "requests" in str(e) or "spacy" in str(e):
                print("⚠ verify_perspective check skipped (dependencies not installed)")
                return True, []  # Not an error, just missing deps
            raise

        if hasattr(StyleTranslator, 'verify_perspective'):
            print("✓ verify_perspective method exists in StyleTranslator")
            return True, []
        else:
            return False, ["verify_perspective method not found in StyleTranslator"]
    except Exception as e:
        if "requests" in str(e) or "spacy" in str(e):
            print("⚠ verify_perspective check skipped (dependencies not installed)")
            return True, []
        return False, [f"Error checking verify_perspective: {e}"]


if __name__ == "__main__":
    print("=" * 60)
    print("Integration Tests Validation")
    print("=" * 60)
    print()

    all_passed = True
    all_errors = []

    print("1. Checking test imports...")
    passed, errors = check_test_imports()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    print("2. Checking test structure...")
    passed, errors = check_test_structure()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    print("3. Checking verify_perspective method...")
    passed, errors = check_verify_perspective()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    print("=" * 60)
    if all_passed:
        print("✓ All checks passed!")
        sys.exit(0)
    else:
        print("✗ Some checks failed:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)

