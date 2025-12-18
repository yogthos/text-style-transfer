"""Tests for rhetorical classification."""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.rhetoric import RhetoricalType, RhetoricalClassifier


def test_heuristic_classification_definition():
    """Test heuristic classification for definitions."""
    classifier = RhetoricalClassifier()

    # DEFINITION tests
    assert classifier.classify_heuristic("A revolution is a process.") == RhetoricalType.DEFINITION
    assert classifier.classify_heuristic("This means something important.") == RhetoricalType.DEFINITION
    assert classifier.classify_heuristic("It is defined as a process.") == RhetoricalType.DEFINITION

    print("✓ test_heuristic_classification_definition passed")


def test_heuristic_classification_argument():
    """Test heuristic classification for arguments."""
    classifier = RhetoricalClassifier()

    # ARGUMENT tests
    assert classifier.classify_heuristic("Therefore, we must act.") == RhetoricalType.ARGUMENT
    assert classifier.classify_heuristic("Thus, the conclusion follows.") == RhetoricalType.ARGUMENT
    assert classifier.classify_heuristic("Because of this, we proceed.") == RhetoricalType.ARGUMENT
    assert classifier.classify_heuristic("Since it is true, we continue.") == RhetoricalType.ARGUMENT

    print("✓ test_heuristic_classification_argument passed")


def test_heuristic_classification_observation():
    """Test heuristic classification for observations."""
    classifier = RhetoricalClassifier()

    # OBSERVATION tests (default)
    assert classifier.classify_heuristic("The sun rises in the east.") == RhetoricalType.OBSERVATION
    assert classifier.classify_heuristic("The cat sat on the mat.") == RhetoricalType.OBSERVATION
    assert classifier.classify_heuristic("Human experience reinforces the rule.") == RhetoricalType.OBSERVATION

    print("✓ test_heuristic_classification_observation passed")


def test_heuristic_classification_imperative():
    """Test heuristic classification for imperatives."""
    classifier = RhetoricalClassifier()

    # IMPERATIVE tests
    assert classifier.classify_heuristic("Study the problem carefully.") == RhetoricalType.IMPERATIVE
    assert classifier.classify_heuristic("Learn from your mistakes.") == RhetoricalType.IMPERATIVE
    assert classifier.classify_heuristic("Consider the implications.") == RhetoricalType.IMPERATIVE
    assert classifier.classify_heuristic("Remember this lesson.") == RhetoricalType.IMPERATIVE

    print("✓ test_heuristic_classification_imperative passed")


def test_llm_classification():
    """Test LLM classification with mocked LLM."""
    classifier = RhetoricalClassifier()

    # Mock LLM provider
    mock_llm = Mock()

    # Test DEFINITION
    mock_llm.call.return_value = "DEFINITION"
    result = classifier.classify_llm("A revolution is a process.", mock_llm)
    assert result == RhetoricalType.DEFINITION

    # Test ARGUMENT
    mock_llm.call.return_value = "ARGUMENT"
    result = classifier.classify_llm("Therefore, we proceed.", mock_llm)
    assert result == RhetoricalType.ARGUMENT

    # Test fallback on error
    mock_llm.call.side_effect = Exception("LLM error")
    result = classifier.classify_llm("The sun rises.", mock_llm)
    assert result == RhetoricalType.OBSERVATION  # Should fall back to heuristic

    print("✓ test_llm_classification passed")


def test_empty_text():
    """Test classification of empty text."""
    classifier = RhetoricalClassifier()

    assert classifier.classify_heuristic("") == RhetoricalType.UNKNOWN
    assert classifier.classify_heuristic("   ") == RhetoricalType.UNKNOWN

    print("✓ test_empty_text passed")


if __name__ == "__main__":
    test_heuristic_classification_definition()
    test_heuristic_classification_argument()
    test_heuristic_classification_observation()
    test_heuristic_classification_imperative()
    test_llm_classification()
    test_empty_text()
    print("\n✓ All rhetoric tests completed!")

