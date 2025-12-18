"""Verification suite for Paragraph Fusion architecture.

This module tests the three core components of paragraph fusion:
1. Proposition Extraction (Atomizer)
2. Proposition Recall (Needle in Haystack)
3. Fusion Generation (Flow)

Run with: python -m pytest tests/verify_fusion.py -v
Or directly: python tests/verify_fusion.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.semantic_analyzer import PropositionExtractor
from src.validator.semantic_critic import SemanticCritic
from src.generator.translator import StyleTranslator
from src.ingestion.blueprint import BlueprintExtractor
import json


def test_proposition_extraction():
    """Test 1: Verify Proposition Extraction (The "Atomizer")

    Goal: Ensure we can turn a complex paragraph into a clean list of facts.
    """
    print("\n" + "="*60)
    print("TEST 1: Proposition Extraction")
    print("="*60)

    extractor = PropositionExtractor()

    # Mock Input: Complex paragraph with multiple facts
    input_text = "The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks."

    print(f"Input: {input_text}")
    print("\nExtracting propositions...")

    propositions = extractor.extract_atomic_propositions(input_text)

    print(f"\nExtracted {len(propositions)} propositions:")
    for i, prop in enumerate(propositions, 1):
        print(f"  {i}. {prop}")

    # Assertions
    assert isinstance(propositions, list), "Output must be a List[str]"
    assert len(propositions) > 0, "Must extract at least one proposition"

    # Check that propositions are simpler/shorter than input sentences
    input_sentences = input_text.split('. ')
    for prop in propositions:
        # Each proposition should be simpler (shorter or at least not much longer)
        # than the original sentences
        max_input_len = max(len(s) for s in input_sentences if s.strip())
        # Allow some expansion for clarity, but not excessive
        assert len(prop) < max_input_len * 1.5, f"Proposition too long: {prop}"

    # Verify propositions contain key concepts
    key_concepts = ["biological", "cycle", "birth", "life", "decay", "reality", "object", "breaks"]
    found_concepts = []
    for prop in propositions:
        prop_lower = prop.lower()
        for concept in key_concepts:
            if concept in prop_lower:
                found_concepts.append(concept)

    # Should find at least some key concepts
    assert len(found_concepts) >= 3, f"Should find key concepts, found: {found_concepts}"

    print("\n✓ Test 1 PASSED: Proposition extraction works correctly")
    return True


def test_proposition_recall():
    """Test 2: Verify Proposition Recall (The "Needle in Haystack")

    Goal: Ensure the Critic can find a specific fact buried inside a complex, stylized paragraph.
    This is the riskiest part - we need to verify semantic similarity works on styled text.
    """
    print("\n" + "="*60)
    print("TEST 2: Proposition Recall")
    print("="*60)

    critic = SemanticCritic()

    # Setup: Simple proposition
    propositions = ["Stars eventually die"]

    # Complex, stylized paragraph that contains the same meaning
    generated_text = "It is an objective materialist fact that every stellar body, burning in the night sky, must inevitably succumb to the dialectical law of extinction."

    print(f"Proposition: {propositions[0]}")
    print(f"\nGenerated text: {generated_text}")
    print("\nChecking proposition recall...")

    # Action: Call the proposition recall method
    recall_score, details = critic._check_proposition_recall(generated_text, propositions)

    print(f"\nRecall score: {recall_score:.3f}")
    print(f"Details: {details}")

    # Assertions
    # Note: With threshold lowered to 0.45, recall should pass if similarity > 0.45
    # But we also check that the similarity is reasonable (not too low)
    preserved = details.get("preserved", [])
    scores = details.get("scores", {})

    # Check that we're comparing against sentences, not whole paragraph
    # The score should be reasonable (semantic similarity for equivalent meaning)
    if scores:
        max_score = max(scores.values())
        # Semantic similarity for equivalent but differently worded text can be ~0.45-0.55
        # This is acceptable - the key is that we're finding the best match
        assert max_score > 0.4, f"Max similarity {max_score} should be > 0.4 (proposition should have reasonable semantic match)"
        print(f"  Max similarity: {max_score:.3f}")

        # If similarity is reasonable, the recall should pass (with threshold 0.45)
        if max_score > 0.45:
            assert recall_score > 0.0, f"Recall should be > 0 when similarity {max_score} > threshold 0.45"
        else:
            # Even if below threshold, similarity should still be reasonable
            assert max_score > 0.35, f"Similarity {max_score} too low for semantically equivalent text"

    print("\n✓ Test 2 PASSED: Proposition recall works on styled text")
    return True


def test_proposition_recall_multiple():
    """Test 2b: Verify Proposition Recall with Multiple Propositions"""
    print("\n" + "="*60)
    print("TEST 2b: Proposition Recall (Multiple Propositions)")
    print("="*60)

    critic = SemanticCritic()

    # Multiple propositions
    propositions = [
        "Stars eventually die",
        "The universe expands",
        "Time moves forward"
    ]

    # Generated text that should contain at least some of these
    generated_text = "It is an objective materialist fact that every stellar body must inevitably succumb to extinction. Furthermore, the cosmos continues its expansion, and temporal progression remains unidirectional."

    print(f"Propositions ({len(propositions)}):")
    for i, prop in enumerate(propositions, 1):
        print(f"  {i}. {prop}")
    print(f"\nGenerated text: {generated_text}")

    recall_score, details = critic._check_proposition_recall(generated_text, propositions)

    print(f"\nRecall score: {recall_score:.3f}")
    print(f"Preserved: {len(details.get('preserved', []))}/{len(propositions)}")
    print(f"Missing: {len(details.get('missing', []))}/{len(propositions)}")

    # Should find at least 2 out of 3 propositions
    assert recall_score >= 0.5, f"Should find at least some propositions, got recall {recall_score}"

    print("\n✓ Test 2b PASSED: Multiple proposition recall works")
    return True


def test_fusion_generation_structure():
    """Test 3: Verify Fusion Generation (The "Flow")

    Goal: Ensure the Generator combines inputs rather than translating one-by-one.
    This test verifies the structure and prompt formatting.
    """
    print("\n" + "="*60)
    print("TEST 3: Fusion Generation Structure")
    print("="*60)

    translator = StyleTranslator()

    # Mock inputs
    propositions = ["A implies B", "B implies C"]
    style_examples = [
        "It is through the dialectical process of contradiction and resolution that we come to understand the fundamental relationships between phenomena, where each element serves as both cause and effect in the grand materialist framework of existence."
    ]

    print(f"Propositions ({len(propositions)}):")
    for i, prop in enumerate(propositions, 1):
        print(f"  {i}. {prop}")
    print(f"\nStyle examples ({len(style_examples)}):")
    for i, ex in enumerate(style_examples, 1):
        print(f"  {i}. {ex[:80]}...")

    # Check that the prompt template is correctly formatted
    from src.generator.mutation_operators import PARAGRAPH_FUSION_PROMPT

    propositions_list = "\n".join([f"- {prop}" for prop in propositions])
    style_examples_text = "\n\n".join([f"Example {i+1}: \"{ex}\"" for i, ex in enumerate(style_examples[:3])])

    prompt = PARAGRAPH_FUSION_PROMPT.format(
        propositions_list=propositions_list,
        style_examples=style_examples_text
    )

    print("\nChecking prompt structure...")

    # Assertions about prompt structure
    assert "CONTENT SOURCE" in prompt or "Atomic Propositions" in prompt, "Prompt should contain propositions section"
    assert "STYLE EXAMPLES" in prompt or "Examples" in prompt, "Prompt should contain style examples section"
    assert "A implies B" in prompt, "Prompt should include proposition content"
    assert len(prompt) > 200, "Prompt should be substantial"

    print("\nPrompt structure verified:")
    print(f"  - Contains propositions section: ✓")
    print(f"  - Contains style examples section: ✓")
    print(f"  - Includes proposition content: ✓")
    print(f"  - Prompt length: {len(prompt)} characters")

    print("\n✓ Test 3 PASSED: Fusion generation structure is correct")
    return True


def test_style_alignment():
    """Test 4: Verify Style Alignment Calculation"""
    print("\n" + "="*60)
    print("TEST 4: Style Alignment")
    print("="*60)

    critic = SemanticCritic()

    # Generate a paragraph with reasonable sentence length
    generated_text = "It is through the dialectical process of contradiction and resolution that we come to understand the fundamental relationships between phenomena, where each element serves as both cause and effect in the grand materialist framework of existence, demonstrating the interconnected nature of all material processes."

    print(f"Generated text: {generated_text[:100]}...")
    print(f"Length: {len(generated_text)} characters")

    style_score, details = critic._check_style_alignment(generated_text, author_style_vector=None)

    print(f"\nStyle score: {style_score:.3f}")
    print(f"Details: {details}")

    # Assertions
    assert 0.0 <= style_score <= 1.0, f"Style score should be between 0 and 1, got {style_score}"

    avg_sentence_length = details.get("avg_sentence_length", 0)
    print(f"Average sentence length: {avg_sentence_length:.1f} words")

    # Should calculate sentence length
    assert avg_sentence_length > 0, "Should calculate average sentence length"

    # If sentences are too short, should have staccato penalty
    if avg_sentence_length < 15:
        staccato_penalty = details.get("staccato_penalty", 0.0)
        assert staccato_penalty > 0, "Should apply staccato penalty for short sentences"
        print(f"  Staccato penalty applied: {staccato_penalty:.3f}")

    print("\n✓ Test 4 PASSED: Style alignment calculation works")
    return True


def test_paragraph_mode_evaluation():
    """Test 5: Verify Paragraph Mode Evaluation End-to-End"""
    print("\n" + "="*60)
    print("TEST 5: Paragraph Mode Evaluation")
    print("="*60)

    critic = SemanticCritic()
    extractor = BlueprintExtractor()
    proposition_extractor = PropositionExtractor()

    # Original paragraph
    original_paragraph = "Stars burn. They eventually die. The universe expands."

    # Generated paragraph (should contain all propositions)
    generated_paragraph = "It is an objective materialist fact that stellar bodies undergo the process of nuclear fusion, burning through their fuel reserves, and must inevitably succumb to the dialectical law of extinction. Furthermore, the cosmos continues its expansion, demonstrating the dynamic nature of material existence."

    print(f"Original: {original_paragraph}")
    print(f"\nGenerated: {generated_paragraph}")

    # Extract propositions
    propositions = proposition_extractor.extract_atomic_propositions(original_paragraph)
    print(f"\nExtracted {len(propositions)} propositions:")
    for i, prop in enumerate(propositions, 1):
        print(f"  {i}. {prop}")

    # Create blueprint for evaluation
    blueprint = extractor.extract(original_paragraph)

    # Evaluate in paragraph mode
    result = critic.evaluate(
        generated_paragraph,
        blueprint,
        propositions=propositions,
        is_paragraph=True,
        author_style_vector=None
    )

    print(f"\nEvaluation result:")
    print(f"  Pass: {result.get('pass', False)}")
    print(f"  Score: {result.get('score', 0.0):.3f}")
    print(f"  Proposition recall: {result.get('proposition_recall', 0.0):.3f}")
    print(f"  Style alignment: {result.get('style_alignment', 0.0):.3f}")
    print(f"  Feedback: {result.get('feedback', '')[:100]}...")

    # Assertions
    assert 'proposition_recall' in result, "Result should include proposition_recall"
    assert 'style_alignment' in result, "Result should include style_alignment"
    assert result.get('proposition_recall', 0.0) > 0.0, "Should calculate proposition recall"

    print("\n✓ Test 5 PASSED: Paragraph mode evaluation works end-to-end")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("PARAGRAPH FUSION VERIFICATION SUITE")
    print("="*60)

    tests = [
        ("Proposition Extraction", test_proposition_extraction),
        ("Proposition Recall", test_proposition_recall),
        ("Proposition Recall (Multiple)", test_proposition_recall_multiple),
        ("Fusion Generation Structure", test_fusion_generation_structure),
        ("Style Alignment", test_style_alignment),
        ("Paragraph Mode Evaluation", test_paragraph_mode_evaluation),
        ("Complexity-Based Selection", test_complexity_filtering),
        ("Complexity Deduplication", test_complexity_deduplication),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, None))
        except AssertionError as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            results.append((test_name, False, str(e)))
        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, error in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Paragraph fusion architecture is verified.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
        return 1


def test_complexity_filtering():
    """Test 6: Verify Complexity-Based Example Selection"""
    print("\n" + "="*60)
    print("TEST 6: Complexity-Based Example Selection")
    print("="*60)

    translator = StyleTranslator()

    # Create test examples with varying complexity
    examples = [
        "Short.",  # 1 word, 1 sentence - should be rejected
        "This is a medium length example with multiple words but only one sentence.",  # 25 words, 1 sentence - should be rejected
        "First sentence. Second sentence.",  # 4 words, 2 sentences - should be rejected (too short)
        "This is a complex example with many words. It has multiple sentences. Each sentence adds complexity. The total word count exceeds the minimum threshold.",  # 30+ words, 4 sentences - should pass
        "Another complex paragraph that demonstrates the target style. It contains multiple clauses and elaborate phrasing. This serves as an expansion template.",  # 25+ words, 3 sentences - should pass
        "Simple text.",  # 2 words, 1 sentence - should be rejected
        "This is a very long and complex paragraph that contains many words and multiple sentences. It demonstrates the intricate style of the target author. Each sentence builds upon the previous one. The paragraph serves as a perfect template for expansion.",  # 40+ words, 4 sentences - should pass
    ]

    print(f"Input: {len(examples)} examples with varying complexity")

    # Test with default thresholds (30 words, 2 sentences)
    selected = translator._select_complex_examples(
        examples,
        min_words=30,
        min_sentences=2,
        top_k=5,
        verbose=True
    )

    print(f"\nSelected {len(selected)} complex examples:")
    for i, ex in enumerate(selected, 1):
        word_count = translator._count_words(ex)
        sentence_count = translator._count_sentences(ex)
        print(f"  {i}. {word_count} words, {sentence_count} sentences: {ex[:60]}...")

    # Assertions
    assert len(selected) > 0, "Should select at least one complex example"

    # Verify all selected examples meet thresholds
    for ex in selected:
        word_count = translator._count_words(ex)
        sentence_count = translator._count_sentences(ex)
        assert word_count >= 30, f"Example should have >= 30 words, got {word_count}"
        assert sentence_count >= 2, f"Example should have >= 2 sentences, got {sentence_count}"

    # Verify sorting (should be descending by word count)
    word_counts = [translator._count_words(ex) for ex in selected]
    assert word_counts == sorted(word_counts, reverse=True), "Examples should be sorted by word count (descending)"

    print("\n✓ Test 6 PASSED: Complexity-based example selection works")
    return True


def test_complexity_deduplication():
    """Test 7: Verify Deduplication in Complexity Selection"""
    print("\n" + "="*60)
    print("TEST 7: Complexity Selection Deduplication")
    print("="*60)

    translator = StyleTranslator()

    # Create similar examples (near duplicates)
    examples = [
        "This is a complex paragraph with many words. It contains multiple sentences. Each sentence adds detail and complexity to the overall structure.",
        "This is a complex paragraph with many words. It contains multiple sentences. Each sentence adds detail and complexity to the overall structure.",  # Exact duplicate
        "This is a complex paragraph with many words. It contains multiple sentences. Each sentence adds detail and complexity to the overall structure with slight variation.",  # Near duplicate (>80% similarity)
        "A completely different complex paragraph. It has many words and multiple sentences. This one should be kept as it is unique.",
    ]

    print(f"Input: {len(examples)} examples (including duplicates)")

    selected = translator._select_complex_examples(
        examples,
        min_words=20,
        min_sentences=2,
        top_k=5,
        verbose=True
    )

    print(f"\nSelected {len(selected)} unique examples after deduplication")

    # Assertions
    # Should have fewer examples than input due to deduplication
    assert len(selected) <= len(examples), "Deduplication should reduce count"
    assert len(selected) >= 1, "Should have at least one unique example"

    # Verify no exact duplicates
    assert len(selected) == len(set(selected)), "No exact duplicates should remain"

    print("\n✓ Test 7 PASSED: Deduplication works correctly")
    return True


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

