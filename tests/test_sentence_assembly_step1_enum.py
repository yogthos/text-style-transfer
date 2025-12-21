"""Tests for Step 1: NARRATIVE enum addition.

This test verifies that RhetoricalType.NARRATIVE exists and can be used
without causing crashes in existing code.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.rhetoric import RhetoricalType, RhetoricalClassifier


def test_narrative_enum_exists():
    """Test that RhetoricalType.NARRATIVE exists and equals 'NARRATIVE'."""
    assert hasattr(RhetoricalType, 'NARRATIVE'), "RhetoricalType.NARRATIVE should exist"
    assert RhetoricalType.NARRATIVE.value == "NARRATIVE", "NARRATIVE value should be 'NARRATIVE'"
    print("✓ test_narrative_enum_exists passed")


def test_narrative_can_be_used():
    """Test that NARRATIVE can be used in code without crashes."""
    # Test direct access
    narrative_type = RhetoricalType.NARRATIVE
    assert narrative_type is not None

    # Test comparison
    assert narrative_type == RhetoricalType.NARRATIVE
    assert narrative_type.value == "NARRATIVE"

    # Test in list iteration
    all_types = list(RhetoricalType)
    assert RhetoricalType.NARRATIVE in all_types

    print("✓ test_narrative_can_be_used passed")


def test_narrative_in_classification_logic():
    """Test that NARRATIVE can be used in classification logic."""
    classifier = RhetoricalClassifier()

    # Test that we can check for NARRATIVE type
    # (The classifier may not return NARRATIVE, but we should be able to check for it)
    test_type = RhetoricalType.NARRATIVE
    assert test_type in RhetoricalType

    # Test that existing code that references NARRATIVE won't crash
    # This simulates the code in translator.py that uses mode_map
    mode_map = {
        "NARRATIVE": RhetoricalType.NARRATIVE,
        "ARGUMENTATIVE": RhetoricalType.ARGUMENT,
        "DESCRIPTIVE": RhetoricalType.OBSERVATION
    }

    # This should not raise AttributeError
    narrative_type = mode_map.get("NARRATIVE", RhetoricalType.OBSERVATION)
    assert narrative_type == RhetoricalType.NARRATIVE

    print("✓ test_narrative_in_classification_logic passed")


def test_narrative_llm_classification():
    """Test that LLM classification can return NARRATIVE."""
    # Test that NARRATIVE is in the enum and can be matched
    # (We don't need to actually call LLM, just verify the enum works)
    response_upper = "NARRATIVE"

    # Simulate the logic in classify_llm
    for rtype in RhetoricalType:
        if rtype.value in response_upper:
            if rtype == RhetoricalType.NARRATIVE:
                assert rtype.value == "NARRATIVE"
                print("✓ test_narrative_llm_classification passed")
                return

    # If we get here, NARRATIVE wasn't found (shouldn't happen)
    assert False, "NARRATIVE should be found in enum iteration"


if __name__ == "__main__":
    test_narrative_enum_exists()
    test_narrative_can_be_used()
    test_narrative_in_classification_logic()
    test_narrative_llm_classification()
    print("\n✓ All Step 1 tests passed!")

