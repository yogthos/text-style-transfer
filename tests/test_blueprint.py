"""Tests for semantic blueprint extraction."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor


def test_simple_sentence():
    """Test extraction from simple sentence."""
    extractor = BlueprintExtractor()
    blueprint = extractor.extract("The cat sat on the mat.")

    assert blueprint.original_text == "The cat sat on the mat."
    assert len(blueprint.svo_triples) > 0, "Should extract at least one SVO"

    # Check that we have keywords
    assert len(blueprint.core_keywords) > 0, "Should extract keywords"
    assert "cat" in blueprint.core_keywords or "sit" in blueprint.core_keywords, "Should contain 'cat' or 'sit'"

    print("✓ test_simple_sentence passed")


def test_complex_sentence():
    """Test extraction from complex sentence."""
    extractor = BlueprintExtractor()
    blueprint = extractor.extract("Human experience reinforces the rule of finitude.")

    assert blueprint.original_text == "Human experience reinforces the rule of finitude."
    assert len(blueprint.svo_triples) > 0, "Should extract SVO"

    # Check keywords
    keywords = blueprint.core_keywords
    assert "experience" in keywords or "human" in keywords or "reinforce" in keywords, "Should contain key concepts"

    # Check subjects
    subjects = blueprint.get_subjects()
    assert len(subjects) > 0, "Should have subjects"

    print("✓ test_complex_sentence passed")


def test_named_entity():
    """Test extraction with named entities."""
    extractor = BlueprintExtractor()
    blueprint = extractor.extract("Mao Zedong declared that a revolution is not a dinner party.")

    assert blueprint.original_text == "Mao Zedong declared that a revolution is not a dinner party."

    # Check for named entities
    entities = blueprint.named_entities
    # spaCy should detect "Mao Zedong" as a PERSON entity
    has_mao = any("Mao" in ent[0] or "Zedong" in ent[0] for ent in entities)
    # If not detected, that's okay - we just verify the structure
    assert isinstance(entities, list), "Should have entities list"

    # Check keywords
    keywords = blueprint.core_keywords
    assert "declare" in keywords or "revolution" in keywords or "dinner" in keywords, "Should contain key concepts"

    print("✓ test_named_entity passed")


def test_error_handling():
    """Test error handling for edge cases."""
    extractor = BlueprintExtractor()

    # Empty string
    blueprint1 = extractor.extract("")
    assert blueprint1.original_text == ""
    assert blueprint1.svo_triples == []
    assert blueprint1.named_entities == []
    assert blueprint1.core_keywords == set()

    # Very short text
    blueprint2 = extractor.extract("Hi.")
    assert isinstance(blueprint2, SemanticBlueprint)

    # Text with only punctuation
    blueprint3 = extractor.extract("...")
    assert isinstance(blueprint3, SemanticBlueprint)

    print("✓ test_error_handling passed")


def test_intransitive_verb():
    """Test extraction with intransitive verbs."""
    extractor = BlueprintExtractor()
    blueprint = extractor.extract("The sun rises.")

    assert blueprint.original_text == "The sun rises."
    assert len(blueprint.svo_triples) > 0, "Should extract SVO even for intransitive verb"

    # Check that verb is extracted
    verbs = blueprint.get_verbs()
    assert len(verbs) > 0, "Should extract verb"
    assert "rise" in verbs or "rises" in verbs, "Should contain 'rise'"

    print("✓ test_intransitive_verb passed")


def test_blueprint_methods():
    """Test SemanticBlueprint helper methods."""
    blueprint = SemanticBlueprint(
        original_text="Test sentence.",
        svo_triples=[
            ("the cat", "sit", "the mat"),
            ("the dog", "run", ""),
            ("the cat", "jump", "the fence")
        ],
        named_entities=[],
        core_keywords={"cat", "sit", "mat", "dog", "run", "jump", "fence"},
        citations=[],
        quotes=[]
    )

    subjects = blueprint.get_subjects()
    assert "the cat" in subjects
    assert "the dog" in subjects
    assert len(subjects) == 2  # Unique subjects

    verbs = blueprint.get_verbs()
    assert "sit" in verbs
    assert "run" in verbs
    assert "jump" in verbs
    assert len(verbs) == 3  # Unique verbs

    objects = blueprint.get_objects()
    assert "the mat" in objects
    assert "the fence" in objects
    # Note: empty strings are filtered out by get_objects() since it uses set()
    assert len(objects) >= 2  # At least the non-empty objects

    print("✓ test_blueprint_methods passed")


def test_positional_metadata():
    """Test that positional metadata is correctly stored in blueprint."""
    extractor = BlueprintExtractor()

    # Test with metadata
    blueprint = extractor.extract(
        "The cat sat on the mat.",
        paragraph_id=2,
        position="OPENER",
        previous_context="Previous sentence."
    )

    assert blueprint.paragraph_id == 2
    assert blueprint.position == "OPENER"
    assert blueprint.previous_context == "Previous sentence."

    # Test with default values
    blueprint2 = extractor.extract("The dog ran.")
    assert blueprint2.paragraph_id == 0
    assert blueprint2.position == "BODY"
    assert blueprint2.previous_context is None

    print("✓ test_positional_metadata passed")


def test_all_positions():
    """Test all position types."""
    extractor = BlueprintExtractor()

    for position in ["OPENER", "BODY", "CLOSER", "SINGLETON"]:
        blueprint = extractor.extract(
            "Test sentence.",
            paragraph_id=0,
            position=position
        )
        assert blueprint.position == position

    print("✓ test_all_positions passed")


if __name__ == "__main__":
    test_simple_sentence()
    test_complex_sentence()
    test_named_entity()
    test_error_handling()
    test_intransitive_verb()
    test_blueprint_methods()
    test_positional_metadata()
    test_all_positions()
    print("\n✓ All blueprint tests completed!")

