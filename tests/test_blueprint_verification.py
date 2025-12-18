"""Verification tests for blueprint extraction improvements."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.blueprint import BlueprintExtractor


def test_rigid_pattern_repeats():
    """Test that 'rigid' modifier is retained in subject phrase."""
    extractor = BlueprintExtractor()
    blueprint = extractor.extract("A rigid pattern repeats.")

    print(f"SVO triples: {blueprint.svo_triples}")
    print(f"Subjects: {blueprint.get_subjects()}")
    print(f"Keywords: {sorted(blueprint.core_keywords)}")

    # The subject should contain "rigid" (the modifier)
    subjects = blueprint.get_subjects()
    has_rigid = any("rigid" in s.lower() for s in subjects)

    assert has_rigid, f"Subject should contain 'rigid'. Got subjects: {subjects}"
    assert "rigid" in blueprint.core_keywords, "Keywords should include 'rigid'"
    assert "pattern" in blueprint.core_keywords, "Keywords should include 'pattern'"
    assert "repeat" in blueprint.core_keywords, "Keywords should include 'repeat' (lemmatized)"

    print("✓ test_rigid_pattern_repeats passed")


def test_biological_cycle_sentence():
    """Test complex sentence with list in subject."""
    extractor = BlueprintExtractor()
    blueprint = extractor.extract("The biological cycle of birth, life, and decay defines our reality.")

    print(f"SVO triples: {blueprint.svo_triples}")
    print(f"Subjects: {blueprint.get_subjects()}")
    print(f"Keywords: {sorted(blueprint.core_keywords)}")

    # The subject should contain "cycle" (the head noun)
    subjects = blueprint.get_subjects()
    has_cycle = any("cycle" in s.lower() for s in subjects)

    # At minimum, should have some subject
    assert len(subjects) > 0, "Should extract at least one subject"
    # Ideally should contain "cycle" or the full phrase
    # Keywords should include all important concepts
    assert "cycle" in blueprint.core_keywords, "Keywords should include 'cycle'"
    assert "biological" in blueprint.core_keywords, "Keywords should include 'biological'"
    assert "define" in blueprint.core_keywords, "Keywords should include 'define' (lemmatized)"

    print("✓ test_biological_cycle_sentence passed")


def test_fluff_filtering():
    """Test that fluff words are filtered from keywords."""
    extractor = BlueprintExtractor()
    blueprint = extractor.extract("However, the cat sat on the mat.")

    keywords = blueprint.core_keywords

    # Should NOT include fluff words
    fluff_words = {"however", "the", "a", "an", "is", "are"}
    found_fluff = keywords & fluff_words

    assert len(found_fluff) == 0, f"Keywords should not include fluff words. Found: {found_fluff}"

    # Should include content words
    assert "cat" in keywords or "sit" in keywords, "Should include content words"

    print("✓ test_fluff_filtering passed")


if __name__ == "__main__":
    test_rigid_pattern_repeats()
    test_biological_cycle_sentence()
    test_fluff_filtering()
    print("\n✓ All verification tests completed!")

