"""Narrative flow integration tests.

Parametrized tests for perspective locking.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.generator.translator import StyleTranslator
from tests.test_helpers import ensure_config_exists

# Ensure config exists
ensure_config_exists()


@pytest.fixture
def translator():
    """Create translator instance for tests."""
    ensure_config_exists()
    return StyleTranslator(config_path="config.json")


@pytest.mark.parametrize("input_text, expected_perspective", [
    ("I went to the store.", "first_person_singular"),
    ("He walked home.", "third_person"),
    ("We are here.", "first_person_plural"),
    ("She drove to work.", "third_person"),
    ("They arrived together.", "third_person"),
])
def test_perspective_locking(translator, input_text, expected_perspective):
    """Test that perspective is locked correctly for various inputs."""
    # For this test, we verify that the translator can detect and verify perspective
    # In a real scenario, we would generate text and verify it matches

    # First, verify detection works
    detected = translator._detect_input_perspective(input_text)

    # Normalize to match expected format
    normalized = translator._normalize_perspective(detected)

    # Check that detection matches expected (allowing for some flexibility)
    if expected_perspective == "first_person_singular":
        assert normalized == "first_person_singular", \
            f"Expected first_person_singular, got {normalized}"
    elif expected_perspective == "first_person_plural":
        assert normalized == "first_person_plural", \
            f"Expected first_person_plural, got {normalized}"
    elif expected_perspective == "third_person":
        assert normalized == "third_person", \
            f"Expected third_person, got {normalized}"


@pytest.mark.parametrize("generated_text, expected_perspective, should_pass", [
    ("I went to the store. I bought food.", "first_person_singular", True),
    ("I went to the store. He bought food.", "first_person_singular", False),
    ("He walked home. She drove to work.", "third_person", True),
    ("He walked home. I drove to work.", "third_person", False),
    ("We are here. We are together.", "first_person_plural", True),
    ("We are here. They are together.", "first_person_plural", False),
])
def test_verify_perspective(translator, generated_text, expected_perspective, should_pass):
    """Test perspective verification with various text samples."""
    result = translator.verify_perspective(generated_text, expected_perspective)

    assert result == should_pass, \
        f"Perspective verification failed: text='{generated_text}', " \
        f"expected={expected_perspective}, should_pass={should_pass}, got={result}"


def test_perspective_detection_edge_cases(translator):
    """Test perspective detection with edge cases."""
    # Empty text
    result = translator._detect_input_perspective("")
    assert result == "third_person", "Empty text should default to third_person"

    # Text with no pronouns
    result = translator._detect_input_perspective("The door opened.")
    assert result == "third_person", "Text without pronouns should default to third_person"

    # Mixed perspectives (should detect dominant)
    result = translator._detect_input_perspective("I went to the store. He was there.")
    # Should detect first person as dominant
    assert result in ["first_person_singular", "third_person"], \
        "Mixed perspectives should detect dominant one"

