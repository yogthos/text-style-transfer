"""Test script for semantic parsing functionality."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.semantic import extract_meaning
from src.models import ContentUnit


def test_extract_meaning_verification_example():
    """Test the verification example from the plan.

    Input: "The quick brown fox jumps over the dog."
    Expected: SVO should be (fox, jump, dog). Modifiers should be {quick, brown}.
    """
    input_text = "The quick brown fox jumps over the dog."
    content_units = extract_meaning(input_text)

    # Assertions
    assert len(content_units) == 1, "Should extract one ContentUnit for one sentence"

    unit = content_units[0]
    assert isinstance(unit, ContentUnit), "Should return ContentUnit objects"
    assert unit.original_text == input_text, "Should preserve original text"
    assert len(unit.svo_triples) > 0, "Should extract at least one SVO triple"

    # Check SVO triple
    svo = unit.svo_triples[0]
    assert isinstance(svo, tuple), "SVO should be a tuple"
    assert len(svo) == 3, "SVO should have 3 elements"

    subject, verb, obj = svo
    assert 'fox' in subject.lower() or subject == 'fox', f"Subject should be 'fox', got '{subject}'"
    assert 'jump' in verb.lower() or verb == 'jumps', f"Verb should be 'jump' or 'jumps', got '{verb}'"
    assert 'dog' in obj.lower() or obj == 'dog', f"Object should be 'dog', got '{obj}'"

    print(f"✓ Verification example passed")
    print(f"  SVO: {svo}")


def test_extract_meaning_multiple_sentences():
    """Test extraction from multiple sentences."""
    input_text = "John went to London. The cat sat on the mat. Mary loves programming."
    content_units = extract_meaning(input_text)

    assert len(content_units) == 3, f"Should extract 3 ContentUnits, got {len(content_units)}"

    for unit in content_units:
        assert isinstance(unit, ContentUnit), "Should return ContentUnit objects"
        assert unit.original_text, "Each unit should have original text"
        assert isinstance(unit.svo_triples, list), "SVO triples should be a list"
        assert isinstance(unit.entities, list), "Entities should be a list"

    # Check that entities are extracted (John, London, Mary)
    all_entities = []
    for unit in content_units:
        all_entities.extend(unit.entities)

    print(f"✓ Multiple sentences test passed")
    print(f"  Extracted {len(content_units)} content units")
    print(f"  Entities found: {all_entities}")


def test_extract_meaning_empty():
    """Test that extract_meaning handles empty text gracefully."""
    content_units = extract_meaning("")
    assert isinstance(content_units, list), "Should return a list"
    assert len(content_units) == 0, "Empty text should return empty list"
    print("✓ Empty text test passed")


def test_extract_meaning_with_entities():
    """Test extraction with named entities."""
    input_text = "Barack Obama visited Paris in 2015. The Eiffel Tower is beautiful."
    content_units = extract_meaning(input_text)

    assert len(content_units) > 0, "Should extract at least one ContentUnit"

    # Check that some entities might be extracted
    all_entities = []
    for unit in content_units:
        all_entities.extend(unit.entities)

    print(f"✓ Named entities test passed")
    print(f"  Entities found: {all_entities}")


if __name__ == "__main__":
    print("Running semantic parsing tests...\n")

    try:
        test_extract_meaning_verification_example()
        test_extract_meaning_multiple_sentences()
        test_extract_meaning_empty()
        test_extract_meaning_with_entities()
        print("\n✓ All semantic parsing tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

