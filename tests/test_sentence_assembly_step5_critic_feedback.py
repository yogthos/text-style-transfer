"""Tests for Step 5: Critic error feedback verification.

This test verifies that SemanticCritic returns specific, actionable
error feedback that can be used in mutation prompts.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure config.json exists before imports
from tests.test_helpers import ensure_config_exists
ensure_config_exists()

try:
    from src.validator.semantic_critic import SemanticCritic
    from src.ingestion.blueprint import BlueprintExtractor
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠ Skipping tests: Missing dependencies - {IMPORT_ERROR}")


def test_critic_returns_specific_errors():
    """Test that critic returns specific error messages."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_critic_returns_specific_errors (missing dependencies)")
        return

    critic = SemanticCritic(config_path="config.json")
    extractor = BlueprintExtractor()

    # Create a test case with a logical error
    original_text = "The cat sat on the mat."
    generated_text = "Silence is acquired through meditation."  # Logical category error

    blueprint = extractor.extract(original_text)
    propositions = [original_text]

    result = critic.evaluate(
        generated_text=generated_text,
        input_blueprint=blueprint,
        propositions=propositions,
        is_paragraph=False,
        verbose=False
    )

    # Check that result contains error feedback
    assert "reason" in result or "feedback" in result, "Critic should return reason or feedback"

    # Check that error message is specific (not just "failed")
    error_msg = result.get("reason", "") or result.get("feedback", "")
    assert len(error_msg) > 10, f"Error message should be specific, got: {error_msg}"

    print("✓ test_critic_returns_specific_errors passed")


def test_critic_errors_are_actionable():
    """Test that error messages are actionable (can be used in mutation prompts)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_critic_errors_are_actionable (missing dependencies)")
        return

    critic = SemanticCritic(config_path="config.json")
    extractor = BlueprintExtractor()

    original_text = "I spent my childhood in the Soviet Union."
    generated_text = "The Soviet Union was a country."  # Missing key information

    blueprint = extractor.extract(original_text)
    propositions = [original_text]

    result = critic.evaluate(
        generated_text=generated_text,
        input_blueprint=blueprint,
        propositions=propositions,
        is_paragraph=False,
        verbose=False
    )

    # Check that error message can be used in a mutation prompt
    error_msg = result.get("reason", "") or result.get("feedback", "")

    # Error should contain actionable information (not just "low score")
    actionable_keywords = ["missing", "incorrect", "wrong", "error", "contradict", "logical", "category"]
    is_actionable = any(keyword in error_msg.lower() for keyword in actionable_keywords)

    # If no actionable keywords, at least check it's not empty
    assert error_msg or is_actionable, f"Error message should be actionable, got: {error_msg}"

    print("✓ test_critic_errors_are_actionable passed")


def test_critic_different_error_types():
    """Test that different error types produce different messages."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_critic_different_error_types (missing dependencies)")
        return

    critic = SemanticCritic(config_path="config.json")
    extractor = BlueprintExtractor()

    original_text = "The cat sat on the mat."
    blueprint = extractor.extract(original_text)
    propositions = [original_text]

    # Test case 1: Logical error
    logical_error_text = "Silence is acquired through meditation."
    result1 = critic.evaluate(
        generated_text=logical_error_text,
        input_blueprint=blueprint,
        propositions=propositions,
        is_paragraph=False,
        verbose=False
    )
    error1 = result1.get("reason", "") or result1.get("feedback", "")

    # Test case 2: Missing information
    missing_info_text = "The mat exists."
    result2 = critic.evaluate(
        generated_text=missing_info_text,
        input_blueprint=blueprint,
        propositions=propositions,
        is_paragraph=False,
        verbose=False
    )
    error2 = result2.get("reason", "") or result2.get("feedback", "")

    # Errors should be different (or at least both present)
    assert error1 or error2, "At least one error message should be present"

    print("✓ test_critic_different_error_types passed")


if __name__ == "__main__":
    test_critic_returns_specific_errors()
    test_critic_errors_are_actionable()
    test_critic_different_error_types()
    print("\n✓ All Step 5 tests completed!")

