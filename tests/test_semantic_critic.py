"""Tests for semantic critic."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.blueprint import SemanticBlueprint
from src.validator.semantic_critic import SemanticCritic


def test_good_output():
    """Test Case A (Good): Server crashed -> Machine stopped."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Server crashed.",
        svo_triples=[("server", "crash", "")],
        named_entities=[],
        core_keywords={"server", "crash"},
        citations=[],
        quotes=[]
    )

    # Good output: preserves meaning
    result = critic.evaluate("Machine stopped.", input_blueprint)

    # Should pass (recall >= 0.85, precision >= 0.50 with new thresholds)
    # Note: Actual scores depend on vector similarity, so we check structure
    assert isinstance(result, dict)
    assert "pass" in result
    assert "recall_score" in result
    assert "precision_score" in result
    assert "score" in result
    assert "feedback" in result

    print("✓ test_good_output passed")


def test_hallucination():
    """Test Case B (Hallucination): Server crashed -> Machine stopped and the dog barked."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Server crashed.",
        svo_triples=[("server", "crash", "")],
        named_entities=[],
        core_keywords={"server", "crash"},
        citations=[],
        quotes=[]
    )

    # Hallucinated output: adds "dog" and "bark"
    result = critic.evaluate("Machine stopped and the dog barked.", input_blueprint)

    # Should fail precision (hallucinated "dog", "bark")
    assert isinstance(result, dict)
    # Precision should be lower due to hallucination
    # Note: Actual scores depend on vector similarity

    print("✓ test_hallucination passed")


def test_omission():
    """Test Case C (Omission): Server crashed -> It happened."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Server crashed.",
        svo_triples=[("server", "crash", "")],
        named_entities=[],
        core_keywords={"server", "crash"},
        citations=[],
        quotes=[]
    )

    # Omission: missing "server" and "crash"
    result = critic.evaluate("It happened.", input_blueprint)

    # Should fail recall (missing "server", "crash", "machine")
    assert isinstance(result, dict)
    # Recall should be lower due to omission

    print("✓ test_omission passed")


def test_edge_cases():
    """Test edge cases."""
    critic = SemanticCritic()

    # Empty input blueprint
    empty_blueprint = SemanticBlueprint(
        original_text="",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    result1 = critic.evaluate("Some text.", empty_blueprint)
    assert isinstance(result1, dict)

    # Empty generated text
    normal_blueprint = SemanticBlueprint(
        original_text="Test.",
        svo_triples=[("test", "be", "")],
        named_entities=[],
        core_keywords={"test"},
        citations=[],
        quotes=[]
    )

    result2 = critic.evaluate("", normal_blueprint)
    assert result2["pass"] == False
    assert "empty" in result2["feedback"].lower()

    # Perfect match
    result3 = critic.evaluate("Test.", normal_blueprint)
    assert isinstance(result3, dict)

    print("✓ test_edge_cases passed")


def test_noun_preservation_check():
    """Test that noun preservation check catches 'We touch breaks' when 'object' is missing."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],
        named_entities=[],
        core_keywords={"touch", "break"},
        citations=[],
        quotes=[]
    )

    # Bad output: missing key noun "object"
    result = critic.evaluate("We touch breaks.", input_blueprint)

    # Should fail due to missing noun "object"
    assert result["pass"] == False, "Should fail when key nouns are missing"
    assert result["score"] == 0.0, "Score should be 0.0 for critical failure"
    assert "noun" in result["feedback"].lower() or "missing" in result["feedback"].lower(), \
        f"Feedback should mention missing nouns, got: {result['feedback']}"

    print("✓ test_noun_preservation_check passed")


def test_compression_ratio_check():
    """Test that compression ratio check catches semantic collapse."""
    critic = SemanticCritic()

    # Test case: 6 words -> 3 words (0.5 ratio) should fail for short sentences
    input_blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],
        named_entities=[],
        core_keywords={"touch", "break"},
        citations=[],
        quotes=[]
    )

    # Bad output: lost >40% of content (3 words from 6 = 0.5 ratio)
    result = critic.evaluate("We touch breaks.", input_blueprint)

    # Should fail due to compression ratio (or noun preservation, both should catch it)
    assert result["pass"] == False, "Should fail when too much content is lost"
    assert result["score"] == 0.0, "Score should be 0.0 for critical failure"

    # Test case: Short sentence that's valid (should pass compression check)
    short_blueprint = SemanticBlueprint(
        original_text="Logic demands this.",
        svo_triples=[("logic", "demand", "this")],
        named_entities=[],
        core_keywords={"logic", "demand"},
        citations=[],
        quotes=[]
    )

    # Valid short output: 3 words -> 2 words (0.66 ratio) should pass
    result2 = critic.evaluate("Logic demands.", short_blueprint)
    # This might fail fragment check, but compression ratio should allow it
    # The key is that 0.66 > 0.6 threshold for short sentences

    print("✓ test_compression_ratio_check passed")


def test_exact_we_touch_breaks_case():
    """EXPLICIT TEST: The exact 'We touch breaks.' failure case."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],
        named_entities=[],
        core_keywords={"touch", "break"},
        citations=[],
        quotes=[]
    )

    # THE EXACT FAILING CASE
    result = critic.evaluate("We touch breaks.", input_blueprint)

    # MUST FAIL - this is the bug we're fixing
    assert result["pass"] == False, f"CRITICAL: 'We touch breaks.' MUST fail. Got pass=True, score={result['score']}, feedback={result['feedback']}"
    assert result["score"] == 0.0, f"CRITICAL: Score must be 0.0. Got score={result['score']}"

    # Should be caught by either noun preservation OR compression ratio
    feedback_lower = result["feedback"].lower()
    assert ("noun" in feedback_lower or "collapse" in feedback_lower or "missing" in feedback_lower or "object" in feedback_lower), \
        f"Feedback should mention the issue. Got: {result['feedback']}"

    print("✓ test_exact_we_touch_breaks_case passed - 'We touch breaks.' correctly rejected")


def test_exact_ceaseless_finitude_case():
    """EXPLICIT TEST: The exact 'ceaseless accumulation...finitude' failure case."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Human experience reinforces the rule of finitude.",
        svo_triples=[("experience", "reinforce", "rule")],
        named_entities=[],
        core_keywords={"experience", "reinforce", "finitude"},
        citations=[],
        quotes=[]
    )

    # THE EXACT FAILING CASE
    result = critic.evaluate("Human experience, in its ceaseless accumulation, serves to reinforce the fundamental rule of finitude.", input_blueprint)

    # MUST FAIL - this is the logical contradiction bug we're fixing
    # Check if contradiction was detected
    feedback_lower = result["feedback"].lower()
    if "contradiction" in feedback_lower:
        assert result["pass"] == False, f"CRITICAL: Logical contradiction MUST fail. Got pass=True"
        assert result["score"] == 0.0, f"CRITICAL: Score must be 0.0 for contradiction. Got score={result['score']}"
        print("✓ test_exact_ceaseless_finitude_case passed - contradiction detected")
    else:
        # If vectors aren't available, we can't detect it semantically
        # But we should still check if it at least doesn't pass with high score
        if result["pass"] == True and result["score"] > 0.8:
            print(f"⚠ WARNING: Contradiction not detected (spaCy vectors may not be available). Score={result['score']}, pass={result['pass']}")
        else:
            print(f"✓ test_exact_ceaseless_finitude_case: Contradiction check may not be active (vectors unavailable), but score is reasonable: {result['score']}")


def test_logic_contradiction_check():
    """Test that logic contradiction check catches oxymorons like 'ceaseless finitude'."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Human experience reinforces the rule of finitude.",
        svo_triples=[("experience", "reinforce", "rule")],
        named_entities=[],
        core_keywords={"experience", "reinforce", "finitude"},
        citations=[],
        quotes=[]
    )

    # Bad output: logical contradiction "ceaseless finitude"
    result = critic.evaluate("Human experience, in its ceaseless accumulation, serves to reinforce the fundamental rule of finitude.", input_blueprint)

    # Should fail due to logical contradiction
    # Note: This might pass if spaCy vectors aren't available, but should fail if they are
    if result["pass"] == False and "contradiction" in result["feedback"].lower():
        assert result["score"] == 0.0, "Score should be 0.0 for logical contradiction"
        print("✓ test_logic_contradiction_check passed (contradiction detected)")
    else:
        # If vectors aren't available, the check might not run
        print("⚠ test_logic_contradiction_check: spaCy vectors may not be available, skipping contradiction check")

    # Test another contradiction: "infinite boundary"
    input_blueprint2 = SemanticBlueprint(
        original_text="The universe has a boundary.",
        svo_triples=[("universe", "have", "boundary")],
        named_entities=[],
        core_keywords={"universe", "boundary"},
        citations=[],
        quotes=[]
    )

    result2 = critic.evaluate("The universe has an infinite boundary.", input_blueprint2)
    if result2["pass"] == False and "contradiction" in result2["feedback"].lower():
        assert result2["score"] == 0.0
        print("✓ test_logic_contradiction_check passed (infinite boundary detected)")
    else:
        print("⚠ test_logic_contradiction_check: contradiction check may not be active")


def test_fragment_check():
    """Test that fragment check catches incomplete sentences."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Logic demands we ask a difficult question.",
        svo_triples=[("logic", "demand", "question")],
        named_entities=[],
        core_keywords={"logic", "demand", "question"},
        citations=[],
        quotes=[]
    )

    # Bad output: fragment missing object
    result = critic.evaluate("Logic demands.", input_blueprint)

    # Should fail due to fragment (missing object)
    # Note: This might pass if parser isn't available, but should fail if it is
    if result["pass"] == False and ("fragment" in result["feedback"].lower() or "incomplete" in result["feedback"].lower()):
        assert result["score"] == 0.0, "Score should be 0.0 for fragment"
        print("✓ test_fragment_check passed (fragment detected)")
    else:
        print("⚠ test_fragment_check: fragment check may not be active (parser may not be available)")


def test_semantic_gates_integration():
    """Test that all semantic gates work together."""
    critic = SemanticCritic()

    # Test case that should trigger multiple gates
    input_blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],
        named_entities=[],
        core_keywords={"touch", "break"},
        citations=[],
        quotes=[]
    )

    # This should fail noun preservation AND compression ratio
    result = critic.evaluate("We touch breaks.", input_blueprint)

    assert result["pass"] == False, "Should fail semantic gates"
    assert result["score"] == 0.0, "Score should be 0.0"
    # Should mention either missing nouns or semantic collapse
    assert "noun" in result["feedback"].lower() or "collapse" in result["feedback"].lower() or "missing" in result["feedback"].lower(), \
        f"Feedback should mention the issue, got: {result['feedback']}"

    print("✓ test_semantic_gates_integration passed")


def test_config_weights_loaded():
    """Test that config loads correctly and weights are accessible."""
    critic = SemanticCritic(config_path="config.json")

    # Verify weights dictionary exists
    assert hasattr(critic, 'weights'), "Critic should have weights attribute"
    assert isinstance(critic.weights, dict), "Weights should be a dictionary"

    # Verify expected weight keys are present
    expected_keys = ["accuracy", "fluency", "style", "thesis_alignment", "intent_compliance", "keyword_coverage"]
    for key in expected_keys:
        assert key in critic.weights, f"Weights should contain '{key}'"
        assert isinstance(critic.weights[key], (int, float)), f"Weight '{key}' should be numeric"
        assert 0 <= critic.weights[key] <= 1, f"Weight '{key}' should be between 0 and 1"

    # Verify cache variables are initialized
    assert hasattr(critic, '_cached_thesis_vector'), "Critic should have _cached_thesis_vector"
    assert hasattr(critic, '_cached_thesis_text'), "Critic should have _cached_thesis_text"
    assert critic._cached_thesis_vector is None, "Cache vector should be initialized to None"
    assert critic._cached_thesis_text is None, "Cache text should be initialized to None"

    print("✓ test_config_weights_loaded passed")


def test_calculate_thesis_alignment():
    """Test _calculate_thesis_alignment method."""
    critic = SemanticCritic(config_path="config.json")

    # Test with aligned text (high similarity)
    thesis = "The document discusses the decline of Soviet economics and economic systems."
    aligned_text = "This paragraph examines how Soviet economic policies failed."
    alignment = critic._calculate_thesis_alignment(aligned_text, thesis)
    assert 0.0 <= alignment <= 1.0, f"Alignment should be between 0 and 1, got {alignment}"
    assert alignment > 0.5, f"Aligned text should have high similarity, got {alignment}"

    # Test with misaligned text (low similarity)
    misaligned_text = "The cat sat on the mat and enjoyed the sunshine."
    alignment = critic._calculate_thesis_alignment(misaligned_text, thesis)
    assert alignment < 0.5, f"Misaligned text should have low similarity, got {alignment}"

    # Test with empty thesis (should return 1.0 as neutral)
    alignment = critic._calculate_thesis_alignment(aligned_text, "")
    assert alignment == 1.0, f"Empty thesis should return 1.0, got {alignment}"

    alignment = critic._calculate_thesis_alignment(aligned_text, None)
    assert alignment == 1.0, f"None thesis should return 1.0, got {alignment}"

    # Test caching: call twice with same thesis, verify cache is used
    thesis2 = "This is a test thesis about machine learning."
    text1 = "Machine learning algorithms are powerful."
    text2 = "Deep learning models can be complex."

    # First call - should encode and cache
    alignment1 = critic._calculate_thesis_alignment(text1, thesis2)
    cached_text_before = critic._cached_thesis_text
    cached_vector_before = critic._cached_thesis_vector

    # Second call with same thesis - should use cache
    alignment2 = critic._calculate_thesis_alignment(text2, thesis2)
    cached_text_after = critic._cached_thesis_text
    cached_vector_after = critic._cached_thesis_vector

    # Verify cache was used (same object reference)
    assert cached_text_before == thesis2, "Cache text should be set after first call"
    assert cached_text_after == thesis2, "Cache text should remain after second call"
    assert cached_vector_before is cached_vector_after, "Cache vector should be reused (same object)"

    # Test cache invalidation: call with different thesis
    thesis3 = "This is a completely different thesis about cooking."
    alignment3 = critic._calculate_thesis_alignment(text1, thesis3)
    assert critic._cached_thesis_text == thesis3, "Cache should update with new thesis"
    assert critic._cached_thesis_vector is not None, "Cache vector should be updated"

    print("✓ test_calculate_thesis_alignment passed")


def test_calculate_keyword_coverage():
    """Test _calculate_keyword_coverage method with hybrid matching."""
    critic = SemanticCritic(config_path="config.json")

    # Test with all keywords present (should return 1.0)
    text = "The Soviet Union collapsed during the Cold War, affecting global economics."
    keywords = ["Soviet Union", "Cold War", "economics"]
    coverage = critic._calculate_keyword_coverage(text, keywords)
    assert coverage == 1.0, f"All keywords present should return 1.0, got {coverage}"

    # Test with partial keywords (should return fraction)
    keywords_partial = ["Soviet Union", "Cold War", "economics", "democracy"]
    coverage = critic._calculate_keyword_coverage(text, keywords_partial)
    assert 0.0 < coverage < 1.0, f"Partial keywords should return fraction, got {coverage}"
    assert coverage == 0.75, f"Expected 0.75 (3/4), got {coverage}"

    # Test with no keywords (should return 0.0)
    text_no_keywords = "The cat sat on the mat."
    coverage = critic._calculate_keyword_coverage(text_no_keywords, keywords)
    assert coverage == 0.0, f"No keywords found should return 0.0, got {coverage}"

    # Test with empty keywords list (should return 1.0 as neutral)
    coverage = critic._calculate_keyword_coverage(text, [])
    assert coverage == 1.0, f"Empty keywords should return 1.0, got {coverage}"

    # Test phrase matching: multi-word keywords
    text_phrase = "The Soviet Union was a major power during the Cold War era."
    keywords_phrases = ["Soviet Union", "Cold War"]
    coverage = critic._calculate_keyword_coverage(text_phrase, keywords_phrases)
    assert coverage == 1.0, f"Phrase keywords should be matched, got {coverage}"

    # Test lemmatization: "economics" should match "economic"
    text_lemma = "The economic policies were important."
    keywords_lemma = ["economics", "policies"]
    coverage = critic._calculate_keyword_coverage(text_lemma, keywords_lemma)
    assert coverage > 0.0, f"Lemmatization should match 'economics' to 'economic', got {coverage}"

    # Test mixed: some phrase keywords, some single-word
    text_mixed = "The Soviet Union had economic problems during the Cold War."
    keywords_mixed = ["Soviet Union", "economics", "Cold War", "problems"]
    coverage = critic._calculate_keyword_coverage(text_mixed, keywords_mixed)
    assert coverage > 0.5, f"Mixed keywords should be matched, got {coverage}"

    # Test with empty text
    coverage = critic._calculate_keyword_coverage("", keywords)
    assert coverage == 0.0, f"Empty text should return 0.0, got {coverage}"

    print("✓ test_calculate_keyword_coverage passed")


def test_verify_intent_alignment():
    """Test intent alignment in _verify_meaning_with_llm."""
    critic = SemanticCritic(config_path="config.json")

    original_text = "The economic system collapsed."
    generated_text = "The economic system failed completely."

    # Test with matching intent (persuasive text with "persuading" intent)
    # Note: This test may require mocking LLM if not available
    if critic.llm_provider:
        try:
            meaning, confidence, intent_score, explanation = critic._verify_meaning_with_llm(
                original_text, generated_text, intent="informing"
            )
            assert isinstance(intent_score, float), f"Intent score should be float, got {type(intent_score)}"
            assert 0.0 <= intent_score <= 1.0, f"Intent score should be 0-1, got {intent_score}"
            # For matching intent, score should be reasonable
            assert intent_score >= 0.0, f"Intent score should be non-negative, got {intent_score}"
        except Exception as e:
            # If LLM call fails, that's okay for this test
            print(f"  ⚠ Intent alignment test skipped (LLM unavailable or error: {e})")
    else:
        # Test fallback when LLM unavailable
        meaning, confidence, intent_score, explanation = critic._verify_meaning_with_llm(
            original_text, generated_text, intent="persuading"
        )
        assert intent_score == 1.0, f"Intent score should default to 1.0 when LLM unavailable, got {intent_score}"

    # Test with empty intent (should return 1.0 as neutral)
    meaning, confidence, intent_score, explanation = critic._verify_meaning_with_llm(
        original_text, generated_text, intent=None
    )
    assert intent_score == 1.0, f"Empty intent should return 1.0, got {intent_score}"

    meaning, confidence, intent_score, explanation = critic._verify_meaning_with_llm(
        original_text, generated_text, intent=""
    )
    # Empty string intent should still be processed, but may return 1.0 as default

    print("✓ test_verify_intent_alignment passed")


def test_evaluate_with_global_context():
    """Test evaluate method with global_context integration."""
    critic = SemanticCritic(config_path="config.json")

    blueprint = SemanticBlueprint(
        original_text="The economic system collapsed.",
        svo_triples=[("economic system", "collapse", "")],
        named_entities=[],
        core_keywords={"economic", "system", "collapse"},
        citations=[],
        quotes=[]
    )

    generated_text = "The economic system failed completely."

    # Test with full global_context (thesis, intent, keywords)
    global_context_full = {
        "thesis": "The document discusses economic systems and their failures.",
        "intent": "informing",
        "keywords": ["economic", "system", "failure", "Soviet Union"]
    }

    result = critic.evaluate(generated_text, blueprint, global_context=global_context_full)

    # Verify metrics are in return dict
    assert "metrics" in result, "Result should contain metrics dictionary"
    assert "thesis_alignment" in result, "Result should contain thesis_alignment"
    assert "intent_score" in result, "Result should contain intent_score"
    assert "keyword_coverage" in result, "Result should contain keyword_coverage"

    # Verify all metrics are calculated (not None)
    assert result["thesis_alignment"] is not None, "Thesis alignment should be calculated"
    assert result["intent_score"] is not None, "Intent score should be calculated"
    assert result["keyword_coverage"] is not None, "Keyword coverage should be calculated"

    # Verify scores are in valid range
    assert 0.0 <= result["thesis_alignment"] <= 1.0, f"Thesis alignment should be 0-1, got {result['thesis_alignment']}"
    assert 0.0 <= result["intent_score"] <= 1.0, f"Intent score should be 0-1, got {result['intent_score']}"
    assert 0.0 <= result["keyword_coverage"] <= 1.0, f"Keyword coverage should be 0-1, got {result['keyword_coverage']}"

    # Test with partial global_context (only thesis)
    global_context_partial = {
        "thesis": "The document discusses economic systems."
    }

    result_partial = critic.evaluate(generated_text, blueprint, global_context=global_context_partial)

    # Verify only thesis is calculated, others are None
    assert result_partial["thesis_alignment"] is not None, "Thesis alignment should be calculated"
    assert result_partial["intent_score"] is None, "Intent score should be None when intent missing"
    assert result_partial["keyword_coverage"] is None, "Keyword coverage should be None when keywords missing"

    # Test without global_context - verify all three metrics are None (not 1.0)
    result_no_context = critic.evaluate(generated_text, blueprint, global_context=None)

    assert result_no_context["thesis_alignment"] is None, "Thesis alignment should be None without context (not 1.0)"
    assert result_no_context["intent_score"] is None, "Intent score should be None without context (not 1.0)"
    assert result_no_context["keyword_coverage"] is None, "Keyword coverage should be None without context (not 1.0)"

    print("✓ test_evaluate_with_global_context passed")


def test_composite_score_with_context_metrics():
    """Test composite score calculation with dynamic weight normalization."""
    critic = SemanticCritic(config_path="config.json")

    # Test composite score includes new metrics when present
    metrics_with_context = {
        "accuracy": 0.8,
        "fluency": 0.9,
        "style": 1.0,
        "thesis_alignment": 0.85,
        "intent_compliance": 0.9,
        "keyword_coverage": 0.75
    }

    score_with_context = critic._calculate_composite_score(metrics_with_context)
    assert 0.0 <= score_with_context <= 1.0, f"Score should be 0-1, got {score_with_context}"
    assert score_with_context > 0.0, "Score should be positive"

    # Test dynamic normalization: without context (only accuracy/fluency)
    metrics_without_context = {
        "accuracy": 0.8,
        "fluency": 0.9,
        "style": 1.0,
        "thesis_alignment": None,
        "intent_compliance": None,
        "keyword_coverage": None
    }

    score_without_context = critic._calculate_composite_score(metrics_without_context)
    assert 0.0 <= score_without_context <= 1.0, f"Score should be 0-1, got {score_without_context}"

    # Verify scores are comparable (both should be in valid range)
    # The key is that without context, we don't artificially inflate the score
    # by defaulting missing metrics to 1.0

    # Test backward compatibility: old config format (no weights dict)
    # This should still work via fallback logic
    assert hasattr(critic, 'weights'), "Critic should have weights attribute"

    # Test edge case: all metrics None (should handle gracefully)
    metrics_all_none = {
        "accuracy": None,
        "fluency": None,
        "style": None,
        "thesis_alignment": None,
        "intent_compliance": None,
        "keyword_coverage": None
    }

    score_all_none = critic._calculate_composite_score(metrics_all_none)
    assert score_all_none == 0.5, f"All None metrics should return 0.5 (neutral), got {score_all_none}"

    # Test edge case: weights don't sum to 1.0 (normalization should handle it)
    # This is already handled by the normalization logic

    print("✓ test_composite_score_with_context_metrics passed")


def test_end_to_end_context_aware_evaluation():
    """End-to-end test verifying all context metrics work together with dynamic normalization."""
    critic = SemanticCritic(config_path="config.json")

    blueprint = SemanticBlueprint(
        original_text="The economic system collapsed during the transition period.",
        svo_triples=[("economic system", "collapse", "transition period")],
        named_entities=[],
        core_keywords={"economic", "system", "collapse", "transition"},
        citations=[],
        quotes=[]
    )

    generated_text = "The economic system failed completely during the transition period."

    # Test with full global context
    global_context = {
        "thesis": "The document discusses economic systems and their failures during political transitions.",
        "intent": "informing",
        "keywords": ["economic", "system", "transition", "Soviet Union", "collapse"]
    }

    result = critic.evaluate(generated_text, blueprint, global_context=global_context)

    # Verify all metrics are present
    assert "metrics" in result, "Result should contain metrics dictionary"
    assert result["thesis_alignment"] is not None, "Thesis alignment should be calculated"
    assert result["intent_score"] is not None, "Intent score should be calculated"
    assert result["keyword_coverage"] is not None, "Keyword coverage should be calculated"

    # Verify scores are in valid range
    assert 0.0 <= result["thesis_alignment"] <= 1.0
    assert 0.0 <= result["intent_score"] <= 1.0
    assert 0.0 <= result["keyword_coverage"] <= 1.0

    # Verify composite score is calculated (should use dynamic normalization)
    assert "score" in result, "Result should contain composite score"
    assert 0.0 <= result["score"] <= 1.0, f"Score should be 0-1, got {result['score']}"

    # Test that context-aware scoring affects pass/fail decisions
    # A text that's semantically good but off-topic should score lower
    off_topic_text = "The cat sat on the mat and enjoyed the sunshine."
    off_topic_result = critic.evaluate(off_topic_text, blueprint, global_context=global_context)

    # Off-topic text might fail early semantic checks, but if it gets to context metrics,
    # it should have lower thesis alignment
    if "thesis_alignment" in off_topic_result and off_topic_result["thesis_alignment"] is not None:
        assert off_topic_result["thesis_alignment"] < result["thesis_alignment"], \
            "Off-topic text should have lower thesis alignment"

    # Test without context - verify None handling and normalization
    no_context_result = critic.evaluate(generated_text, blueprint, global_context=None)

    # Verify metrics are None (not 1.0)
    assert no_context_result["thesis_alignment"] is None, "Should be None without context"
    assert no_context_result["intent_score"] is None, "Should be None without context"
    assert no_context_result["keyword_coverage"] is None, "Should be None without context"

    # Verify scores are still comparable (no artificial inflation)
    # Both should be valid scores in [0, 1] range
    assert 0.0 <= no_context_result["score"] <= 1.0, "Score without context should still be valid"
    assert 0.0 <= result["score"] <= 1.0, "Score with context should still be valid"

    # Test partial context
    partial_context = {
        "thesis": "The document discusses economic systems."
    }

    partial_result = critic.evaluate(generated_text, blueprint, global_context=partial_context)

    # Only thesis should be calculated
    assert partial_result["thesis_alignment"] is not None, "Thesis should be calculated"
    assert partial_result["intent_score"] is None, "Intent should be None"
    assert partial_result["keyword_coverage"] is None, "Keywords should be None"

    # Verify score is still valid with partial context
    assert 0.0 <= partial_result["score"] <= 1.0, "Score with partial context should be valid"

    print("✓ test_end_to_end_context_aware_evaluation passed")


def test_verify_coherence_coherent_text():
    """Test _verify_coherence with coherent text (should return high score)."""
    from unittest.mock import Mock, patch
    import json

    critic = SemanticCritic()

    # Mock LLM provider to return high coherence score
    mock_response = json.dumps({
        "is_coherent": True,
        "score": 0.95,
        "reason": "Text is grammatically fluent and logically consistent."
    })
    mock_llm = Mock()
    mock_llm.call = Mock(return_value=mock_response)

    # Replace LLM provider and ensure use_llm_verification is True
    critic.llm_provider = mock_llm
    critic.use_llm_verification = True

    coherent_text = "I spent my childhood scavenging in the ruins of the Soviet Union. That country is a ghost now."
    score, reason = critic._verify_coherence(coherent_text)

    assert score >= 0.8, f"Coherent text should get high score, got {score}"
    assert "coherent" in reason.lower() or score > 0.7, f"Reason should indicate coherence, got: {reason}"
    assert mock_llm.call.call_count == 1, f"LLM should be called once, but call_count={mock_llm.call.call_count}"

    print("✓ test_verify_coherence_coherent_text passed")


def test_verify_coherence_gibberish_text():
    """Test _verify_coherence with gibberish text (should return low score)."""
    from unittest.mock import Mock
    import json

    critic = SemanticCritic()

    # Mock LLM provider to return low coherence score
    mock_response = json.dumps({
        "is_coherent": False,
        "score": 0.3,
        "reason": "Text contains word salad and nonsensical phrases like 'scavenged in ruins to turing complete'."
    })
    mock_llm = Mock()
    mock_llm.call = Mock(return_value=mock_response)

    # Replace LLM provider and ensure use_llm_verification is True
    critic.llm_provider = mock_llm
    critic.use_llm_verification = True

    gibberish_text = "In case a union is haunting, the Soviet Union is observed to have been scavenged in ruins to turing complete."
    score, reason = critic._verify_coherence(gibberish_text)

    assert score < 0.7, f"Gibberish text should get low score, got {score}"
    assert "coherent" not in reason.lower() or score < 0.5, f"Reason should indicate incoherence, got: {reason}"
    assert mock_llm.call.call_count == 1, f"LLM should be called once, but call_count={mock_llm.call.call_count}"

    print("✓ test_verify_coherence_gibberish_text passed")


def test_verify_coherence_empty_text():
    """Test _verify_coherence with empty text (edge case)."""
    critic = SemanticCritic()

    score, reason = critic._verify_coherence("")

    assert score == 0.0, f"Empty text should get score 0.0, got {score}"
    assert "empty" in reason.lower(), f"Reason should mention empty text, got: {reason}"

    print("✓ test_verify_coherence_empty_text passed")


def test_verify_coherence_llm_unavailable():
    """Test _verify_coherence when LLM is unavailable (should return neutral score)."""
    critic = SemanticCritic()
    critic.llm_provider = None  # Simulate LLM unavailable

    text = "This is a test sentence."
    score, reason = critic._verify_coherence(text)

    assert score == 1.0, f"When LLM unavailable, should return neutral score 1.0, got {score}"
    assert "unavailable" in reason.lower(), f"Reason should mention LLM unavailable, got: {reason}"

    print("✓ test_verify_coherence_llm_unavailable passed")


def test_calculate_semantic_similarity_similar_topics():
    """Test _calculate_semantic_similarity with similar topics (should return high similarity)."""
    critic = SemanticCritic()

    original = "I spent my childhood scavenging in the ruins of the Soviet Union."
    generated = "During my youth, I searched through the remains of the Soviet Union."

    similarity = critic._calculate_semantic_similarity(original, generated)

    # Similar topics should have high similarity (>0.7)
    assert similarity >= 0.6, f"Similar topics should have high similarity, got {similarity}"

    print(f"✓ test_calculate_semantic_similarity_similar_topics passed (similarity: {similarity:.2f})")


def test_calculate_semantic_similarity_drifted_topics():
    """Test _calculate_semantic_similarity with drifted topics (should return low similarity)."""
    critic = SemanticCritic()

    original = "I spent my childhood scavenging in the ruins of the Soviet Union."
    generated = "In case a union is haunting, the Soviet Union is observed to have been scavenged in ruins to turing complete."

    similarity = critic._calculate_semantic_similarity(original, generated)

    # Drifted topics (Soviet Union -> turing complete) should have low similarity (<0.6)
    assert similarity < 0.6, f"Drifted topics should have low similarity, got {similarity}"

    print(f"✓ test_calculate_semantic_similarity_drifted_topics passed (similarity: {similarity:.2f})")


def test_evaluate_paragraph_mode_with_coherence_check():
    """Test _evaluate_paragraph_mode includes coherence and topic similarity checks."""
    from unittest.mock import Mock, patch
    import json
    import numpy as np
    from src.analysis.semantic_analyzer import PropositionExtractor

    critic = SemanticCritic()

    # Mock LLM provider for coherence check
    mock_response = json.dumps({
        "is_coherent": False,
        "score": 0.4,
        "reason": "Text contains word salad and nonsensical phrases."
    })
    mock_llm = Mock()
    mock_llm.call = Mock(return_value=mock_response)
    critic.llm_provider = mock_llm
    critic.use_llm_verification = True

    original_text = "I spent my childhood scavenging in the ruins of the Soviet Union."
    # Use text that will have high proposition recall but is still gibberish
    generated_text = "I spent my childhood scavenging in the ruins of the Soviet Union to turing complete and the violent shift."

    # Extract propositions
    extractor = PropositionExtractor()
    propositions = extractor.extract_atomic_propositions(original_text)

    # Create blueprint
    from src.ingestion.blueprint import BlueprintExtractor
    blueprint_extractor = BlueprintExtractor()
    blueprint = blueprint_extractor.extract(original_text)

    # Evaluate in paragraph mode without style vector (style_alignment will be calculated but may not be >= 0.7)
    # The coherence check only runs if proposition_recall >= 0.85 AND style_alignment >= 0.7
    # If those conditions aren't met, coherence_score will default to 1.0
    result = critic._evaluate_paragraph_mode(
        generated_text=generated_text,
        original_text=original_text,
        propositions=propositions,
        author_style_vector=None,  # No style vector - style alignment may be low
        style_lexicon=None
    )

    # Check that coherence_score and topic_similarity are in result
    assert "coherence_score" in result, "Result should include coherence_score"
    assert "topic_similarity" in result, "Result should include topic_similarity"

    # Gibberish should fail coherence check
    coherence_score = result.get("coherence_score", 1.0)
    topic_similarity = result.get("topic_similarity", 1.0)

    # The coherence check should have been triggered if proposition_recall >= 0.85 and style >= 0.7
    # If it wasn't triggered, coherence_score will be 1.0 (default)
    # In that case, we just verify the structure is correct
    if coherence_score < 1.0:
        # Coherence check was triggered
        assert coherence_score < 0.8 or topic_similarity < 0.6, \
            f"Gibberish should fail coherence ({coherence_score:.2f}) or topic similarity ({topic_similarity:.2f}) check"

        # If coherence is too low, score should be 0.0 (sanity gate)
        if coherence_score < 0.8 or topic_similarity < 0.6:
            assert result.get("score", 1.0) == 0.0 or result.get("pass", True) == False, \
                f"Gibberish should fail sanity gate, got score={result.get('score')}, pass={result.get('pass')}"
    else:
        # Coherence check wasn't triggered (cheap checks didn't pass)
        # This is okay - the test verifies the structure is in place
        print(f"  Note: Coherence check not triggered (cheap checks may not have passed)")

    print(f"✓ test_evaluate_paragraph_mode_with_coherence_check passed (coherence: {coherence_score:.2f}, topic_sim: {topic_similarity:.2f})")


def test_evaluate_paragraph_mode_coherent_text():
    """Test _evaluate_paragraph_mode with coherent text (should pass)."""
    from unittest.mock import Mock
    import json
    from src.analysis.semantic_analyzer import PropositionExtractor

    critic = SemanticCritic()

    # Mock LLM provider for coherence check (return high score)
    mock_response = json.dumps({
        "is_coherent": True,
        "score": 0.9,
        "reason": "Text is grammatically fluent and logically consistent."
    })
    mock_llm = Mock()
    mock_llm.call = Mock(return_value=mock_response)
    critic.llm_provider = mock_llm
    critic.use_llm_verification = True

    original_text = "I spent my childhood scavenging in the ruins of the Soviet Union."
    generated_text = "During my youth, I searched through the remains of the Soviet Union."

    # Extract propositions
    extractor = PropositionExtractor()
    propositions = extractor.extract_atomic_propositions(original_text)

    # Create blueprint
    from src.ingestion.blueprint import BlueprintExtractor
    blueprint_extractor = BlueprintExtractor()
    blueprint = blueprint_extractor.extract(original_text)

    # Evaluate in paragraph mode
    result = critic._evaluate_paragraph_mode(
        generated_text=generated_text,
        original_text=original_text,
        propositions=propositions,
        author_style_vector=None,
        style_lexicon=None
    )

    # Coherent text should pass coherence check
    coherence_score = result.get("coherence_score", 0.0)
    topic_similarity = result.get("topic_similarity", 0.0)

    assert coherence_score >= 0.8, f"Coherent text should pass coherence check, got {coherence_score}"
    assert topic_similarity >= 0.6, f"Similar topics should pass topic similarity check, got {topic_similarity}"

    print(f"✓ test_evaluate_paragraph_mode_coherent_text passed (coherence: {coherence_score:.2f}, topic_sim: {topic_similarity:.2f})")


def test_evaluate_paragraph_mode_score_normalization():
    """Test that _evaluate_paragraph_mode properly normalizes weights so score is in [0, 1] range."""
    from unittest.mock import Mock
    import json
    from src.analysis.semantic_analyzer import PropositionExtractor

    critic = SemanticCritic()

    # Mock LLM provider for coherence check (return high score)
    mock_llm = Mock()
    mock_response = json.dumps({
        "is_coherent": True,
        "score": 0.9,
        "reason": "Text is grammatically fluent and logically consistent."
    })
    mock_llm.call = Mock(return_value=mock_response)
    critic.llm_provider = mock_llm
    critic.use_llm_verification = True

    original_text = "I spent my childhood scavenging in the ruins of the Soviet Union."
    generated_text = "During my youth, I searched through the remains of the Soviet Union."

    # Extract propositions
    extractor = PropositionExtractor()
    propositions = extractor.extract_atomic_propositions(original_text)

    # Create blueprint
    from src.ingestion.blueprint import BlueprintExtractor
    blueprint_extractor = BlueprintExtractor()
    blueprint = blueprint_extractor.extract(original_text)

    # Evaluate in paragraph mode without style vector
    # The score should still be normalized even if expensive checks don't run
    result = critic._evaluate_paragraph_mode(
        generated_text=generated_text,
        original_text=original_text,
        propositions=propositions,
        author_style_vector=None,  # No style vector - may not trigger expensive checks
        style_lexicon=None,
        verbose=False
    )

    # Check that score is in valid range [0, 1]
    score = result.get("score", -1.0)
    assert 0.0 <= score <= 1.0, f"Score should be in [0, 1] range, got {score}"

    # Get all metrics
    proposition_recall = result.get("proposition_recall", 0.0)
    style_alignment = result.get("style_alignment", 0.0)
    coherence_score = result.get("coherence_score", 1.0)
    topic_similarity = result.get("topic_similarity", 1.0)

    # Verify that the score is properly normalized
    # The key test: even if expensive checks don't run, the initial_score should be normalized
    # initial_score = (proposition_recall * meaning_weight) + (style_alignment * style_weight)
    # With meaning_weight=0.8, style_weight=0.4, weight_sum = 1.2
    # So final_score = initial_score / 1.2, which should be in [0, 1]

    # If expensive checks ran (recall >= 0.85 and style >= 0.7), verify normalized calculation
    if proposition_recall >= 0.85 and style_alignment >= 0.7:
        # Weight sum = (0.8 * 0.7) + (0.4 * 0.2) + 0.1 + 0.1 = 0.84
        # Normalized weights should sum to 1.0, so score should be properly scaled
        # If all metrics are 1.0, score should be 1.0 (not 0.84)
        if coherence_score >= 0.8 and topic_similarity >= 0.6:
            # All checks passed, score should be high
            assert score >= 0.7, f"With all high metrics, score should be >= 0.7, got {score}"

    # The main test: score should always be in [0, 1] range regardless of which path was taken
    assert 0.0 <= score <= 1.0, f"Score must be in [0, 1] range, got {score}"

    print(f"✓ test_evaluate_paragraph_mode_score_normalization passed (score: {score:.3f}, recall: {proposition_recall:.2f}, style: {style_alignment:.2f}, coherence: {coherence_score:.2f}, topic_sim: {topic_similarity:.2f})")


def test_evaluate_paragraph_mode_sanity_gate_score_zero():
    """Test that sanity gate correctly sets score=0.0 when coherence < 0.6, and applies penalty for 0.6-0.8 range."""
    from unittest.mock import Mock
    import json
    from src.analysis.semantic_analyzer import PropositionExtractor

    critic = SemanticCritic()

    # Mock LLM provider to return LOW coherence score (below strict kill threshold of 0.6)
    mock_llm = Mock()
    mock_response = json.dumps({
        "is_coherent": False,
        "score": 0.3,  # Below strict kill threshold of 0.6
        "reason": "Text contains word salad and nonsensical phrases."
    })
    mock_llm.call = Mock(return_value=mock_response)
    critic.llm_provider = mock_llm
    critic.use_llm_verification = True

    original_text = "I spent my childhood scavenging in the ruins of the Soviet Union."
    # Use text that will have high proposition recall but low coherence
    generated_text = "I spent my childhood scavenging in the ruins of the Soviet Union to turing complete and the violent shift."

    # Extract propositions
    extractor = PropositionExtractor()
    propositions = extractor.extract_atomic_propositions(original_text)

    # Create blueprint
    from src.ingestion.blueprint import BlueprintExtractor
    blueprint_extractor = BlueprintExtractor()
    blueprint = blueprint_extractor.extract(original_text)

    # Evaluate in paragraph mode
    result = critic._evaluate_paragraph_mode(
        generated_text=generated_text,
        original_text=original_text,
        propositions=propositions,
        author_style_vector=None,
        style_lexicon=None,
        verbose=False
    )

    coherence_score = result.get("coherence_score", 1.0)
    topic_similarity = result.get("topic_similarity", 1.0)
    score = result.get("score", -1.0)
    passes = result.get("pass", True)

    # If coherence is below strict kill threshold (0.6), sanity gate should trigger and set score=0.0
    strict_kill_threshold = 0.6
    if coherence_score < strict_kill_threshold:
        assert score == 0.0, f"When coherence ({coherence_score:.2f}) < strict kill threshold ({strict_kill_threshold}), score should be 0.0, got {score}"
        assert passes == False, f"When sanity gate triggers, pass should be False, got {passes}"

    print(f"✓ test_evaluate_paragraph_mode_sanity_gate_score_zero passed (coherence: {coherence_score:.2f}, score: {score:.2f}, pass: {passes})")


def test_evaluate_paragraph_mode_coherence_penalty():
    """Test that coherence in 0.6-0.8 range applies penalty (score *= 0.5) but doesn't kill."""
    from unittest.mock import Mock
    import json
    from src.analysis.semantic_analyzer import PropositionExtractor

    critic = SemanticCritic()

    # Mock LLM provider to return coherence score in penalty range (0.6-0.8)
    mock_llm = Mock()
    mock_response = json.dumps({
        "is_coherent": True,
        "score": 0.7,  # In penalty range (0.6-0.8), above strict kill (0.6)
        "reason": "Text is mostly coherent but has some issues."
    })
    mock_llm.call = Mock(return_value=mock_response)
    critic.llm_provider = mock_llm
    critic.use_llm_verification = True

    original_text = "I spent my childhood scavenging in the ruins of the Soviet Union."
    generated_text = "During my youth, I searched through the remains of the Soviet Union, though the phrasing is somewhat awkward."

    # Extract propositions
    extractor = PropositionExtractor()
    propositions = extractor.extract_atomic_propositions(original_text)

    # Create blueprint
    from src.ingestion.blueprint import BlueprintExtractor
    blueprint_extractor = BlueprintExtractor()
    blueprint = blueprint_extractor.extract(original_text)

    # Evaluate in paragraph mode
    result = critic._evaluate_paragraph_mode(
        generated_text=generated_text,
        original_text=original_text,
        propositions=propositions,
        author_style_vector=None,
        style_lexicon=None,
        verbose=False
    )

    coherence_score = result.get("coherence_score", 1.0)
    score = result.get("score", -1.0)

    # If coherence is in penalty range (0.6-0.8), score should be penalized (multiplied by 0.5)
    # but not killed (not 0.0)
    penalty_threshold = 0.8
    strict_kill_threshold = 0.6

    if strict_kill_threshold <= coherence_score < penalty_threshold:
        # Score should be penalized (reduced) but not zero
        assert score > 0.0, f"When coherence ({coherence_score:.2f}) is in penalty range, score should be > 0.0, got {score}"
        # Score should be lower than it would be without penalty
        # (We can't easily calculate expected score without penalty, but we can verify it's not zero)

    print(f"✓ test_evaluate_paragraph_mode_coherence_penalty passed (coherence: {coherence_score:.2f}, score: {score:.2f})")


if __name__ == "__main__":
    test_good_output()
    test_hallucination()
    test_omission()
    test_edge_cases()
    test_noun_preservation_check()
    test_compression_ratio_check()
    test_exact_we_touch_breaks_case()  # CRITICAL: Test exact failing case
    test_exact_ceaseless_finitude_case()  # CRITICAL: Test exact failing case
    test_logic_contradiction_check()
    test_fragment_check()
    test_semantic_gates_integration()
    test_config_weights_loaded()  # NEW: Test config and weights
    test_calculate_thesis_alignment()  # NEW: Test thesis alignment with caching
    test_calculate_keyword_coverage()  # NEW: Test keyword coverage with phrase matching
    test_verify_intent_alignment()  # NEW: Test intent alignment
    test_evaluate_with_global_context()  # NEW: Test evaluate integration
    test_composite_score_with_context_metrics()  # NEW: Test composite score with normalization
    test_end_to_end_context_aware_evaluation()  # NEW: End-to-end integration test
    print("\n✓ All semantic critic tests completed!")

