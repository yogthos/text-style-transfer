"""Tests for semantic critic punctuation normalization.

This test ensures that the Semantic Critic correctly handles punctuation differences
(e.g., "toolset" vs toolset) and doesn't give false 0.000 scores.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.blueprint import SemanticBlueprint
from src.validator.semantic_critic import SemanticCritic


def test_punctuation_normalization():
    """Test that keywords with quotes match keywords without quotes in recall check."""
    critic = SemanticCritic()

    # Input blueprint with keywords that have quotes (as they might appear in original text)
    input_blueprint = SemanticBlueprint(
        original_text='Dialectical Materialism is a practical "toolset" for analyzing reality.',
        svo_triples=[("Dialectical Materialism", "is", "toolset")],
        named_entities=[("Dialectical Materialism", "CONCEPT")],
        core_keywords={'"toolset"', "practical", "analyzing", "reality"},
        citations=[],
        quotes=[]
    )

    # Generated blueprint with keywords WITHOUT quotes (as they appear in generated text)
    from src.ingestion.blueprint import BlueprintExtractor
    extractor = BlueprintExtractor()
    generated_blueprint = extractor.extract("Dialectical Materialism is a practical toolset for analyzing reality and understanding dynamic systems.")

    # Test the recall check directly (this is where normalization happens)
    recall_score, feedback = critic._check_recall(generated_blueprint, input_blueprint)

    # The recall should be > 0 because normalization should match "toolset" with toolset
    assert recall_score > 0.0, \
        f"Recall should be > 0.0, got {recall_score}. Keywords with quotes should match keywords without quotes. Feedback: {feedback}"

    # Should have at least 50% recall (at least 2 of 4 keywords should match)
    assert recall_score >= 0.5, \
        f"Recall should be >= 0.5 (at least half the keywords should match), got {recall_score}"

    print(f"✓ test_punctuation_normalization passed (recall: {recall_score:.2f})")


def test_normalize_token_method():
    """Test the _normalize_token method directly."""
    critic = SemanticCritic()

    # Test cases: input -> expected normalized output
    # Note: Aggressive normalization removes spaces, so multi-word becomes single word
    test_cases = [
        ('"toolset"', 'toolset'),
        ("'toolset'", 'toolset'),
        ('toolset', 'toolset'),
        ('"Diamat"', 'diamat'),
        ('Diamat', 'diamat'),
        ('"dialectical materialism"', 'dialecticalmaterialism'),  # Spaces removed
        ('dialectical materialism', 'dialecticalmaterialism'),  # Spaces removed
        ('"word",', 'word'),
        ('word.', 'word'),
        ('WORD', 'word'),
        ('tool-set', 'toolset'),  # Hyphens removed
    ]

    for input_token, expected_normalized in test_cases:
        normalized = critic._normalize_token(input_token)
        assert normalized == expected_normalized, \
            f"Expected '{input_token}' to normalize to '{expected_normalized}', got '{normalized}'"

    print("✓ test_normalize_token_method passed")


def test_keyword_matching_with_punctuation():
    """Test that keyword sets with different punctuation match correctly."""
    critic = SemanticCritic()

    # Input keywords with various punctuation
    input_keywords = {'"toolset"', '"Diamat"', 'dialectical', 'materialism', 'practical'}

    # Generated keywords without punctuation
    generated_keywords = {'toolset', 'diamat', 'dialectical', 'materialism', 'practical'}

    # Normalize both sets
    input_norm = {critic._normalize_token(k) for k in input_keywords}
    generated_norm = {critic._normalize_token(k) for k in generated_keywords}

    # They should match after normalization
    assert input_norm == generated_norm, \
        f"Normalized sets should match. Input norm: {input_norm}, Generated norm: {generated_norm}"

    # Calculate overlap
    overlap = input_norm.intersection(generated_norm)
    assert len(overlap) == len(input_norm), \
        f"All keywords should match after normalization. Overlap: {overlap}, Input: {input_norm}"

    print("✓ test_keyword_matching_with_punctuation passed")


def test_real_world_scenario():
    """Test a real-world scenario: input with quotes, output without quotes."""
    critic = SemanticCritic()

    # Real-world input blueprint (keywords might have quotes from original text)
    input_blueprint = SemanticBlueprint(
        original_text='Many people hear the term "Dialectical Materialism" (or "Diamat") and assume they are stepping into a realm of strange mysticism.',
        svo_triples=[
            ("people", "hear", "term"),
            ("people", "assume", "realm")
        ],
        named_entities=[("Dialectical Materialism", "CONCEPT"), ("Diamat", "CONCEPT")],
        core_keywords={'"Diamat"', "people", "realm", "mysticism"},
        citations=[],
        quotes=[('"Dialectical Materialism"', 0), ('"Diamat"', 0)]
    )

    # Generated blueprint with keywords WITHOUT quotes
    from src.ingestion.blueprint import BlueprintExtractor
    extractor = BlueprintExtractor()
    generated_blueprint = extractor.extract("Many people hear the term Dialectical Materialism, or Diamat, and assume they are stepping into a realm of strange mysticism, but it is actually a practical method.")

    # Test the recall check directly (this is where normalization happens)
    recall_score, feedback = critic._check_recall(generated_blueprint, input_blueprint)

    # The recall should be > 0 because normalization should match "Diamat" with Diamat
    assert recall_score > 0.0, \
        f"Recall should be > 0.0, got {recall_score}. Keywords with quotes should match keywords without quotes. Feedback: {feedback}"

    # Should have at least 50% recall (at least 2 of 4 keywords should match)
    assert recall_score >= 0.5, \
        f"Recall should be >= 0.5 (at least half the keywords should match), got {recall_score}"

    print(f"✓ test_real_world_scenario passed (recall: {recall_score:.2f})")


def test_full_evaluation_with_punctuation():
    """Test the full evaluate() method in paragraph mode to ensure punctuation differences don't cause 0.000 scores.

    This is the critical test: it verifies that when input has keywords with quotes
    (e.g., "toolset", "Diamat") and generated text has the same keywords without quotes,
    the critic correctly matches them and gives a good score (not 0.000).

    Uses paragraph mode to focus on keyword matching rather than other validation checks.
    """
    critic = SemanticCritic()

    # Input blueprint with keywords that have quotes in core_keywords
    # This simulates the case where keywords are extracted with quotes
    input_blueprint = SemanticBlueprint(
        original_text='It is not a realm of strange mysticism or impenetrable jargon, but a practical "toolset". Many people hear the term "Diamat" and assume they are stepping into a realm of strange mysticism.',
        svo_triples=[
            ("it", "is", "toolset"),
            ("people", "hear", "term"),
            ("people", "assume", "realm")
        ],
        named_entities=[("Diamat", "CONCEPT")],
        core_keywords={'"toolset"', '"Diamat"', "people", "realm", "mysticism", "practical"},
        citations=[],
        quotes=[]  # No quotes to avoid quote validation
    )

    # Generated text with the same keywords but WITHOUT quotes
    # This is high-quality text that should score well
    generated_text = "It is not a realm of strange mysticism or impenetrable jargon, but a practical toolset. Many people hear the term Diamat and assume they are stepping into a realm of strange mysticism."

    # Propositions that match the input (for paragraph mode)
    propositions = [
        "It is not a realm of strange mysticism or impenetrable jargon, but a practical toolset.",
        "Many people hear the term Diamat and assume they are stepping into a realm of strange mysticism."
    ]

    # Evaluate using paragraph mode (this uses the keyword matching we fixed)
    result = critic.evaluate(
        generated_text,
        input_blueprint,
        propositions=propositions,
        is_paragraph=True,
        verbose=False
    )

    # Assertions
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "score" in result, "Result should contain 'score'"
    assert "recall_score" in result, "Result should contain 'recall_score'"

    # CRITICAL: The score should NOT be 0.000
    # The normalization should ensure "toolset" matches toolset and "Diamat" matches Diamat
    assert result["score"] > 0.0, \
        f"Score should be > 0.0, got {result['score']}. " \
        f"This indicates normalization is failing. " \
        f"Recall: {result.get('recall_score', 'N/A')}, " \
        f"Feedback: {result.get('feedback', 'N/A')[:100]}"

    # The recall should be > 0 because keywords with quotes should match keywords without quotes
    assert result["recall_score"] > 0.0, \
        f"Recall should be > 0.0, got {result['recall_score']}. " \
        f"Keywords with quotes should match keywords without quotes. " \
        f"Feedback: {result.get('feedback', 'N/A')[:100]}"

    # Should have at least 50% recall (at least 3 of 6 keywords should match)
    assert result["recall_score"] >= 0.5, \
        f"Recall should be >= 0.5 (at least half the keywords should match), got {result['recall_score']}"

    print(f"✓ test_full_evaluation_with_punctuation passed (score: {result['score']:.3f}, recall: {result['recall_score']:.3f})")


def test_false_negative_prevention():
    """Test that the critic doesn't give false 0.000 scores due to punctuation.

    This test specifically targets the bug where valid text gets 0.000 because
    of punctuation mismatches (e.g., "toolset" vs toolset).

    Uses paragraph mode to focus on keyword matching normalization.
    """
    critic = SemanticCritic()

    # Input with quoted keywords in core_keywords (realistic scenario)
    # Keywords extracted from text might have quotes, but we're testing keyword matching, not quote preservation
    input_blueprint = SemanticBlueprint(
        original_text='Dialectical Materialism is a practical "toolset" for analyzing reality. The term "Diamat" refers to this method.',
        svo_triples=[
            ("Dialectical Materialism", "is", "toolset"),
            ("term", "refers", "method")
        ],
        named_entities=[("Dialectical Materialism", "CONCEPT"), ("Diamat", "CONCEPT")],
        core_keywords={'"toolset"', '"Diamat"', "practical", "analyzing", "reality"},
        citations=[],
        quotes=[]  # No quotes to avoid quote validation
    )

    # Generated text: semantically excellent, but keywords don't have quotes
    generated_text = "It is not a realm of strange mysticism or impenetrable jargon, but a practical toolset for analyzing reality. The term Diamat refers to this method."

    # Propositions for paragraph mode
    propositions = [
        "It is not a realm of strange mysticism or impenetrable jargon, but a practical toolset for analyzing reality.",
        "The term Diamat refers to this method."
    ]

    # Evaluate in paragraph mode (uses keyword matching we fixed)
    result = critic.evaluate(
        generated_text,
        input_blueprint,
        propositions=propositions,
        is_paragraph=True,
        verbose=False
    )

    # The key assertion: score should NOT be 0.000
    # This was the bug: valid text was getting 0.000 due to punctuation mismatches
    assert result["score"] > 0.0, \
        f"CRITICAL BUG: Score is 0.000 for valid text! " \
        f"This indicates normalization is not working. " \
        f"Recall: {result.get('recall_score', 'N/A')}, " \
        f"Score: {result.get('score', 'N/A')}, " \
        f"Feedback: {result.get('feedback', 'N/A')[:150]}"

    # Recall should be good (most keywords should match after normalization)
    assert result["recall_score"] > 0.0, \
        f"CRITICAL BUG: Recall is 0.000! " \
        f"Keywords with quotes should match keywords without quotes. " \
        f"Score: {result.get('score', 'N/A')}, " \
        f"Feedback: {result.get('feedback', 'N/A')[:150]}"

    # Should have at least 60% recall (at least 3 of 5 keywords should match)
    assert result["recall_score"] >= 0.6, \
        f"Recall should be >= 0.6 (most keywords should match), got {result['recall_score']}"

    print(f"✓ test_false_negative_prevention passed (score: {result['score']:.3f}, recall: {result['recall_score']:.3f})")


if __name__ == "__main__":
    test_punctuation_normalization()
    test_normalize_token_method()
    test_keyword_matching_with_punctuation()
    test_real_world_scenario()
    test_full_evaluation_with_punctuation()
    test_false_negative_prevention()
    print("\n✅ All punctuation normalization tests passed!")

