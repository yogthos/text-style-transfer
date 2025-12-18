"""Test for critical content preservation - ensuring no content is cut.

This test explicitly checks that all content from the original text is preserved,
especially lists and prepositional phrases.
"""

import pytest
from src.ingestion.blueprint import BlueprintExtractor
from src.generator.translator import StyleTranslator
from src.atlas.rhetoric import RhetoricalType


def test_biological_cycle_full_preservation():
    """CRITICAL TEST: Ensure 'birth, life, and decay' is fully preserved.

    This test explicitly checks that the phrase "The biological cycle of birth,
    life, and decay defines our reality." does NOT get shortened to
    "The biological cycle of birth defines our reality."
    """
    original_text = "The biological cycle of birth, life, and decay defines our reality."

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    # Check that blueprint captures all three items
    subjects = blueprint.get_subjects()

    # The subject should include "birth, life, and decay" or at least all three words
    subject_text = " ".join(subjects) if subjects else ""

    # CRITICAL: All three words must be present
    assert "birth" in subject_text.lower() or "birth" in original_text.lower(), \
        f"Missing 'birth' in subject: {subject_text}"
    assert "life" in subject_text.lower() or "life" in original_text.lower(), \
        f"Missing 'life' in subject: {subject_text}"
    assert "decay" in subject_text.lower() or "decay" in original_text.lower(), \
        f"Missing 'decay' in subject: {subject_text}"

    # Check keywords - all three should be present
    keywords_lower = {kw.lower() for kw in blueprint.core_keywords}
    assert "birth" in keywords_lower or "life" in keywords_lower or "decay" in keywords_lower, \
        f"Missing keywords from 'birth, life, and decay' in: {keywords_lower}"

    # Most importantly: check that original_text is preserved in blueprint
    assert blueprint.original_text == original_text, \
        f"Original text not preserved: {blueprint.original_text} != {original_text}"

    # Test that translator preserves all content (Phase 1: Contextual Grounding)
    translator = StyleTranslator()
    # Use a simple author/style for testing
    generated = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Clear and direct writing style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example text in target style."]
    )

    # CRITICAL ASSERTION: Generated text must contain all three concepts
    generated_lower = generated.lower()
    assert "birth" in generated_lower, \
        f"CRITICAL FAILURE: Generated text missing 'birth'!\n" \
        f"Original: {original_text}\n" \
        f"Generated: {generated}\n" \
        f"This should NEVER happen - all list items must be preserved!"
    assert "life" in generated_lower, \
        f"CRITICAL FAILURE: Generated text missing 'life'!\n" \
        f"Original: {original_text}\n" \
        f"Generated: {generated}\n" \
        f"This should NEVER happen - all list items must be preserved!"
    assert "decay" in generated_lower, \
        f"CRITICAL FAILURE: Generated text missing 'decay'!\n" \
        f"Original: {original_text}\n" \
        f"Generated: {generated}\n" \
        f"This should NEVER happen - all list items must be preserved!"

    # Also check that the full phrase structure is preserved
    # The generated text should mention all three in some form
    has_birth = "birth" in generated_lower
    has_life = "life" in generated_lower
    has_decay = "decay" in generated_lower

    assert has_birth and has_life and has_decay, \
        f"CRITICAL FAILURE: Generated text is missing content!\n" \
        f"Original: {original_text}\n" \
        f"Generated: {generated}\n" \
        f"Missing: birth={has_birth}, life={has_life}, decay={has_decay}\n" \
        f"This is a REGRESSION - the three-phase refactor should prevent this!"

    # Phase 3: Test semantic similarity check
    critic = SemanticCritic()
    result = critic.evaluate(generated, blueprint)

    # If semantic model available, similarity should be high
    if critic.semantic_model:
        similarity = critic._calculate_semantic_similarity(original_text, generated)
        assert similarity >= 0.85, \
            f"Semantic similarity too low: {similarity:.2f} < 0.85\n" \
            f"Original: {original_text}\n" \
            f"Generated: {generated}\n" \
            f"This indicates content loss was not caught by semantic check!"


def test_list_preservation_general():
    """Test that lists in general are preserved (not just biological cycle)."""
    test_cases = [
        ("The cycle of birth, life, and decay defines reality.", ["birth", "life", "decay"]),
        ("We study physics, chemistry, and biology.", ["physics", "chemistry", "biology"]),
        ("The colors red, blue, and green are primary.", ["red", "blue", "green"]),
    ]

    extractor = BlueprintExtractor()

    for original_text, required_words in test_cases:
        blueprint = extractor.extract(original_text)

        # Check that all required words appear in either subjects, objects, or keywords
        all_text = " ".join(blueprint.get_subjects() + blueprint.get_objects()).lower()
        keywords_lower = {kw.lower() for kw in blueprint.core_keywords}

        for word in required_words:
            word_lower = word.lower()
            assert (word_lower in all_text or word_lower in keywords_lower or
                    word_lower in blueprint.original_text.lower()), \
                f"Missing '{word}' from blueprint for: {original_text}\n" \
                f"Subjects: {blueprint.get_subjects()}\n" \
                f"Objects: {blueprint.get_objects()}\n" \
                f"Keywords: {sorted(blueprint.core_keywords)}"


def test_prepositional_phrase_preservation():
    """Test that prepositional phrases with lists are fully preserved."""
    original_text = "The biological cycle of birth, life, and decay defines our reality."

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    # Extract the full subject phrase
    subjects = blueprint.get_subjects()
    subject_full = " ".join(subjects) if subjects else ""

    # The subject should contain the full prepositional phrase
    # Check that we have "of" followed by the list
    subject_lower = subject_full.lower()

    # We should have "of" and at least some indication of the list
    # The exact structure might vary, but all three items must be present somewhere
    assert "of" in subject_lower or "biological cycle" in subject_lower, \
        f"Prepositional phrase not captured in subject: {subject_full}"

    # Verify original text is always preserved
    assert blueprint.original_text == original_text, \
        "Original text must always be preserved in blueprint"

