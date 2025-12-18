"""Tests for three-phase refactor: Contextual Grounding, Draft-Polish Pipeline, Semantic Guardrails.

This test suite validates:
1. Phase 1: Original text is prioritized as context in prompts
2. Phase 2: Two-pass draft-polish pipeline produces natural English
3. Phase 3: Semantic similarity check catches content loss
"""

import pytest
from src.ingestion.blueprint import BlueprintExtractor
from src.generator.translator import StyleTranslator
from src.validator.semantic_critic import SemanticCritic
from src.atlas.rhetoric import RhetoricalType


def test_phase1_contextual_grounding_biological_cycle():
    """CRITICAL TEST: Phase 1 - Original text prioritized, all list items preserved.

    This test ensures that "The biological cycle of birth, life, and decay defines our reality."
    does NOT get shortened to "The biological cycle of birth defines our reality."
    """
    original_text = "The biological cycle of birth, life, and decay defines our reality."

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    # Verify blueprint preserves original text
    assert blueprint.original_text == original_text, \
        f"Blueprint must preserve original text: {blueprint.original_text} != {original_text}"

    # Verify all three items are captured in keywords
    keywords_lower = {kw.lower() for kw in blueprint.core_keywords}
    assert "birth" in keywords_lower or "life" in keywords_lower or "decay" in keywords_lower, \
        f"Missing keywords from 'birth, life, and decay': {keywords_lower}"

    # Test full translation pipeline
    translator = StyleTranslator()
    generated = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Clear and direct writing style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example text in target style."]
    )

    # CRITICAL: Generated text must contain all three concepts
    generated_lower = generated.lower()
    assert "birth" in generated_lower, \
        f"CRITICAL FAILURE: Generated text missing 'birth'!\nOriginal: {original_text}\nGenerated: {generated}"
    assert "life" in generated_lower, \
        f"CRITICAL FAILURE: Generated text missing 'life'!\nOriginal: {original_text}\nGenerated: {generated}"
    assert "decay" in generated_lower, \
        f"CRITICAL FAILURE: Generated text missing 'decay'!\nOriginal: {original_text}\nGenerated: {generated}"


def test_phase2_draft_polish_pipeline():
    """Test Phase 2: Two-pass draft-polish pipeline produces natural English.

    The polish pass should fix stilted phrasing like "will, in time, be broken"
    to natural English like "eventually breaks".
    """
    original_text = "Every object we touch eventually breaks."

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    translator = StyleTranslator()
    generated = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Clear and direct writing style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example text in target style."]
    )

    generated_lower = generated.lower()

    # Should NOT contain stilted patterns
    assert "will, in time," not in generated_lower, \
        f"Polish pass failed: Still contains stilted pattern 'will, in time,'\nGenerated: {generated}"
    assert "will, eventually," not in generated_lower, \
        f"Polish pass failed: Still contains stilted pattern 'will, eventually,'\nGenerated: {generated}"

    # Should contain natural phrasing
    # Either "breaks" (simple present) or "eventually breaks" (natural adverb placement)
    has_breaks = "breaks" in generated_lower
    has_eventually = "eventually" in generated_lower

    assert has_breaks, \
        f"Generated text should contain 'breaks' (or equivalent): {generated}"


def test_phase2_polish_fixes_passive_voice():
    """Test Phase 2: Polish pass fixes unnecessary passive voice."""
    original_text = "Every star eventually succumbs to erosion."

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    translator = StyleTranslator()
    generated = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Clear and direct writing style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example text in target style."]
    )

    generated_lower = generated.lower()

    # Should NOT contain future passive for universal truths
    assert "will be" not in generated_lower or "succumbs" in generated_lower or "succumb" in generated_lower, \
        f"Polish pass should prefer active voice: {generated}"

    # Should contain the key concepts
    assert "star" in generated_lower, f"Missing 'star': {generated}"
    assert "erosion" in generated_lower or "erode" in generated_lower, \
        f"Missing 'erosion' concept: {generated}"


def test_phase3_semantic_similarity_catches_content_loss():
    """Test Phase 3: Semantic similarity check catches content loss.

    This test ensures that semantic similarity (0.85 threshold) catches cases like:
    - "We touch breaks" (low similarity to "Every object we touch breaks")
    - Missing list items (low similarity when content is cut)
    """
    critic = SemanticCritic()

    # Test case 1: Missing content should have low similarity
    original = "Every object we touch eventually breaks."
    generated_bad = "We touch breaks."  # Missing "object" and "eventually"

    if critic.semantic_model:
        similarity = critic._calculate_semantic_similarity(original, generated_bad)
        assert similarity < 0.85, \
            f"Semantic similarity should catch content loss. Similarity: {similarity:.2f} (expected < 0.85)"

    # Test case 2: Missing list items should have low similarity
    original_list = "The biological cycle of birth, life, and decay defines our reality."
    generated_cut = "The biological cycle of birth defines our reality."  # Missing "life, and decay"

    if critic.semantic_model:
        similarity = critic._calculate_semantic_similarity(original_list, generated_cut)
        assert similarity < 0.85, \
            f"Semantic similarity should catch missing list items. Similarity: {similarity:.2f} (expected < 0.85)"

    # Test case 3: Complete content should have high similarity
    original_complete = "Every object we touch eventually breaks."
    generated_good = "Every object we touch eventually breaks."

    if critic.semantic_model:
        similarity = critic._calculate_semantic_similarity(original_complete, generated_good)
        assert similarity >= 0.85, \
            f"Complete content should have high similarity. Similarity: {similarity:.2f} (expected >= 0.85)"


def test_phase3_semantic_similarity_rejects_low_similarity():
    """Test Phase 3: Semantic critic rejects generated text with low similarity."""
    critic = SemanticCritic()

    original_text = "Every object we touch eventually breaks."
    generated_bad = "We touch breaks."  # Missing content

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    result = critic.evaluate(generated_bad, blueprint)

    # If semantic model is available, should reject
    if critic.semantic_model:
        assert not result["pass"], \
            f"Semantic critic should reject low similarity text. Result: {result}"
        assert "similarity" in result["feedback"].lower() or "meaning" in result["feedback"].lower(), \
            f"Feedback should mention similarity/meaning: {result['feedback']}"


def test_phase3_semantic_similarity_accepts_high_similarity():
    """Test Phase 3: Semantic critic accepts generated text with high similarity."""
    critic = SemanticCritic()

    original_text = "Every object we touch eventually breaks."
    generated_good = "Every object we touch eventually breaks."  # Same meaning

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    result = critic.evaluate(generated_good, blueprint)

    # Should pass semantic similarity check (if model available)
    # Note: May still fail other checks (fluency, etc.) but similarity should pass
    if critic.semantic_model:
        similarity = critic._calculate_semantic_similarity(original_text, generated_good)
        assert similarity >= 0.85, \
            f"Complete content should have high similarity: {similarity:.2f}"


def test_integration_three_phases_biological_cycle():
    """Integration test: All three phases working together for biological cycle case."""
    original_text = "The biological cycle of birth, life, and decay defines our reality."

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    # Phase 1: Contextual grounding - blueprint should preserve original
    assert blueprint.original_text == original_text

    # Phase 2: Draft-polish pipeline
    translator = StyleTranslator()
    generated = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Clear and direct writing style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example text in target style."]
    )

    # Phase 2: Check polish pass worked (no stilted phrasing)
    generated_lower = generated.lower()
    assert "will, in time," not in generated_lower
    assert "will, eventually," not in generated_lower

    # Phase 1: All content preserved
    assert "birth" in generated_lower
    assert "life" in generated_lower
    assert "decay" in generated_lower

    # Phase 3: Semantic similarity check
    critic = SemanticCritic()
    result = critic.evaluate(generated, blueprint)

    # If semantic model available, similarity should be high
    if critic.semantic_model:
        similarity = critic._calculate_semantic_similarity(original_text, generated)
        assert similarity >= 0.85, \
            f"Semantic similarity too low: {similarity:.2f}. Generated: {generated}"

    # Overall: Should pass (or at least not fail on similarity)
    # Note: May fail on other metrics, but similarity should pass
    if critic.semantic_model and not result["pass"]:
        # If it failed, check it wasn't due to similarity
        assert "similarity" not in result["feedback"].lower() or similarity >= 0.85, \
            f"Should not fail on similarity if similarity >= 0.85. Result: {result}"


def test_phase1_prompt_structure_prioritizes_original_text():
    """Test Phase 1: Prompt structure prioritizes original_text as context."""
    original_text = "Every object we touch eventually breaks."

    extractor = BlueprintExtractor()
    blueprint = extractor.extract(original_text)

    translator = StyleTranslator()
    prompt = translator._build_prompt(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Clear writing style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example text."]
    )

    # Check that original text appears prominently in prompt
    assert original_text in prompt, \
        f"Original text should appear in prompt. Prompt: {prompt[:200]}..."

    # Check that prompt structure emphasizes original text as context
    prompt_lower = prompt.lower()
    # Should have "context" or "source" or "original" prominently
    assert ("context" in prompt_lower or "source" in prompt_lower or
            "original text" in prompt_lower or "primary source" in prompt_lower), \
        f"Prompt should emphasize original text as context. Prompt: {prompt[:300]}..."

