"""Golden Paragraph Regression Tests.

Compares new generations against manually verified "golden" outputs using
semantic similarity to detect quality drift.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.validator.semantic_critic import SemanticCritic
from tests.test_helpers import ensure_config_exists
from tests.mocks.mock_llm_provider import get_mock_llm_provider


def load_golden_examples() -> List[Dict]:
    """Load all golden examples from the examples directory."""
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        return []

    golden_examples = []
    for json_file in examples_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                example = json.load(f)
                example['_file_path'] = str(json_file)
                golden_examples.append(example)
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Failed to parse {json_file}: {e}")
            continue

    return golden_examples


def calculate_semantic_drift(
    old_output: str,
    new_output: str,
    expected_similarity: float,
    threshold: float = 0.15
) -> tuple[float, float, bool]:
    """Calculate semantic drift between old and new outputs.

    Args:
        old_output: Original golden output
        new_output: Newly generated output
        expected_similarity: Expected similarity score (0.0-1.0)
        threshold: Maximum allowed relative drift (default 0.15 = 15%)

    Returns:
        Tuple of (actual_similarity, drift_percentage, passes_threshold)
    """
    critic = SemanticCritic()

    # Calculate actual similarity
    actual_similarity = critic._calculate_semantic_similarity(old_output, new_output)

    # Calculate relative drift: (expected - actual) / expected
    if expected_similarity > 0:
        drift = (expected_similarity - actual_similarity) / expected_similarity
    else:
        drift = 1.0 if actual_similarity < 0.5 else 0.0

    # Pass if drift is within threshold
    passes = drift <= threshold

    return actual_similarity, drift, passes


@pytest.fixture(scope="module")
def golden_examples():
    """Load golden examples once per test session."""
    return load_golden_examples()


@pytest.fixture(scope="function")
def mock_translator():
    """Create a translator with mocked LLM provider."""
    from src.generator.translator import StyleTranslator
    from unittest.mock import patch

    ensure_config_exists()
    translator = StyleTranslator(config_path="config.json")

    # Replace LLM provider with mock
    mock_provider = get_mock_llm_provider()
    translator.llm_provider = mock_provider

    return translator


def test_golden_regression_suite(golden_examples, mock_translator):
    """Run regression tests for all golden examples."""
    if not golden_examples:
        pytest.skip("No golden examples found. Create examples in tests/golden_set/examples/")

    failures = []

    for example in golden_examples:
        example_id = example.get("id", "unknown")
        input_text = example.get("input_text", "")
        expected_output = example.get("expected_output", "")
        expected_similarity = example.get("expected_similarity", 0.95)
        author_name = example.get("author_name", "Unknown")

        if not input_text or not expected_output:
            print(f"⚠ Skipping {example_id}: Missing input_text or expected_output")
            continue

        # Generate new output using mocked translator
        try:
            new_output, _, _ = mock_translator.translate_paragraph_statistical(
                paragraph=input_text,
                author_name=author_name,
                verbose=False
            )
        except Exception as e:
            failures.append(f"{example_id}: Generation failed - {e}")
            continue

        # Calculate semantic drift
        actual_sim, drift, passes = calculate_semantic_drift(
            expected_output,
            new_output,
            expected_similarity,
            threshold=0.15
        )

        if not passes:
            failures.append(
                f"{example_id}: Drift {drift:.1%} exceeds 15% threshold "
                f"(similarity: {actual_sim:.3f} vs expected: {expected_similarity:.3f})"
            )

    if failures:
        failure_msg = "\n".join(failures)
        pytest.fail(f"Golden regression failures:\n{failure_msg}")


@pytest.mark.parametrize("example", load_golden_examples())
def test_individual_golden_example(example, mock_translator):
    """Test individual golden example (parametrized)."""
    example_id = example.get("id", "unknown")
    input_text = example.get("input_text", "")
    expected_output = example.get("expected_output", "")
    expected_similarity = example.get("expected_similarity", 0.95)
    author_name = example.get("author_name", "Unknown")

    if not input_text or not expected_output:
        pytest.skip(f"{example_id}: Missing input_text or expected_output")

    # Generate new output
    new_output, _, _ = mock_translator.translate_paragraph_statistical(
        paragraph=input_text,
        author_name=author_name,
        verbose=False
    )

    # Calculate semantic drift
    actual_sim, drift, passes = calculate_semantic_drift(
        expected_output,
        new_output,
        expected_similarity,
        threshold=0.15
    )

    assert passes, (
        f"Golden example {example_id} failed: "
        f"Drift {drift:.1%} exceeds 15% threshold "
        f"(similarity: {actual_sim:.3f} vs expected: {expected_similarity:.3f})"
    )


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])

