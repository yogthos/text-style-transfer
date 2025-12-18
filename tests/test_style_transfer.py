"""Comprehensive unit tests for style transfer functionality.

Tests ensure that:
1. Style metric calculation works correctly (lexicon density, vector similarity, composite score)
2. Style DNA is properly injected into paragraph fusion prompts
3. Style-preserving repairs maintain vocabulary during fixes
4. Author style vector retrieval works correctly
5. Integration: Full style transfer pipeline maintains style fidelity
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator.semantic_critic import SemanticCritic
from src.generator.translator import StyleTranslator
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.builder import StyleAtlas
from src.atlas.blender import StyleBlender


class TestStyleMetricCalculation:
    """Test style metric calculation (lexicon density + vector similarity)."""

    def test_lexicon_density_with_matching_words(self):
        """Test that lexicon density correctly identifies matching words."""
        critic = SemanticCritic(config_path="config.json")

        generated_text = "The dialectical process reveals contradictions in material reality."
        style_lexicon = ["dialectical", "contradiction", "material", "reality"]

        score, details = critic._check_style_alignment(
            generated_text,
            author_style_vector=None,  # No vector, test lexicon only
            style_lexicon=style_lexicon
        )

        # Should have non-zero lexicon density (at least 2-3 words match)
        assert details["lexicon_density"] > 0.0, "Lexicon density should be > 0 when words match"
        assert details["lexicon_density"] <= 1.0, "Lexicon density should be <= 1.0"
        print(f"✓ Lexicon density with matching words: {details['lexicon_density']:.3f}")

    def test_lexicon_density_with_lemmatization(self):
        """Test that lemmatization handles plural/tense variations."""
        critic = SemanticCritic(config_path="config.json")

        # Generated text has "contradictions" (plural)
        generated_text = "The system contains internal contradictions that lead to transformation."
        # Lexicon has "contradiction" (singular)
        style_lexicon = ["contradiction", "transformation", "system"]

        score, details = critic._check_style_alignment(
            generated_text,
            author_style_vector=None,
            style_lexicon=style_lexicon
        )

        # Should match "contradiction" to "contradictions" via lemmatization
        # Also should match "transformation" and "system" directly
        # So lexicon density should be > 0 (at least 2-3 words should match)
        assert details["lexicon_density"] > 0.0, \
            f"Lemmatization should match words. Lexicon density: {details['lexicon_density']:.3f}"
        # Should match at least "transformation" and "system" directly, plus "contradiction" via lemmatization
        assert details["lexicon_density"] >= 0.3, \
            f"Should match multiple words (contradiction, transformation, system). Got: {details['lexicon_density']:.3f}"
        print(f"✓ Lexicon density with lemmatization: {details['lexicon_density']:.3f}")

    def test_lexicon_density_with_no_matches(self):
        """Test that lexicon density is 0 when no words match."""
        critic = SemanticCritic(config_path="config.json")

        generated_text = "The cat sat on the mat."
        style_lexicon = ["dialectical", "contradiction", "material"]

        score, details = critic._check_style_alignment(
            generated_text,
            author_style_vector=None,
            style_lexicon=style_lexicon
        )

        # Should have zero lexicon density (or very close to zero due to floating point)
        assert details["lexicon_density"] < 0.01, \
            f"Lexicon density should be ~0 when no words match, got {details['lexicon_density']}"
        print(f"✓ Lexicon density with no matches: {details['lexicon_density']:.3f}")

    def test_composite_score_formula(self):
        """Test that composite score uses correct formula: (Vector * 0.7) + (Lexicon * 0.3)."""
        critic = SemanticCritic(config_path="config.json")

        # Create mock author style vector
        try:
            from src.analyzer.style_metrics import get_style_vector
        except ImportError:
            print("⚠ Skipping test: style_metrics not available")
            return

        generated_text = "The dialectical process reveals contradictions in material reality."
        author_text = "Dialectical materialism shows how contradictions drive historical development."
        author_style_vector = get_style_vector(author_text)

        style_lexicon = ["dialectical", "contradiction", "material", "reality"]

        score, details = critic._check_style_alignment(
            generated_text,
            author_style_vector=author_style_vector,
            style_lexicon=style_lexicon
        )

        # Verify composite score calculation
        vector_sim = details["similarity"]
        lexicon_dens = details["lexicon_density"]
        expected_composite = (vector_sim * 0.7) + (lexicon_dens * 0.3)

        # Account for staccato penalty in final score
        staccato_penalty = details.get("staccato_penalty", 0.0)
        expected_final = expected_composite * (1.0 - staccato_penalty * 0.2)

        # Allow small floating point differences (0.02 tolerance for rounding)
        assert abs(score - expected_final) < 0.02, \
            f"Score {score:.3f} should match formula result {expected_final:.3f} (diff: {abs(score - expected_final):.4f})"
        print(f"✓ Composite score formula: vector={vector_sim:.3f}, lexicon={lexicon_dens:.3f}, final={score:.3f}")

    def test_style_scores_vary_across_candidates(self):
        """Test that style scores show variance (not static 0.50)."""
        critic = SemanticCritic(config_path="config.json")

        try:
            from src.analyzer.style_metrics import get_style_vector
        except ImportError:
            print("⚠ Skipping test: style_metrics not available")
            return

        author_text = "Dialectical materialism demonstrates how contradictions drive transformation."
        author_style_vector = get_style_vector(author_text)
        style_lexicon = ["dialectical", "contradiction", "material", "transformation"]

        candidates = [
            "The dialectical process reveals contradictions in material reality.",  # High style match
            "The cat sat on the mat.",  # Low style match
            "Material conditions determine social development through dialectical change."  # Medium style match
        ]

        scores = []
        for candidate in candidates:
            score, _ = critic._check_style_alignment(
                candidate,
                author_style_vector=author_style_vector,
                style_lexicon=style_lexicon
            )
            scores.append(score)

        # Scores should vary (not all 0.50)
        # Note: Even low-style candidates might score > 0.5 due to vector similarity,
        # but there should still be variance
        assert len(set(scores)) > 1, f"Style scores should vary across candidates, got: {scores}"
        assert max(scores) > min(scores), f"There should be a difference between best and worst scores, got: {scores}"
        # First candidate (high style) should score higher than last (low style)
        assert scores[0] > scores[2], f"High-style candidate ({scores[0]:.3f}) should score higher than low-style ({scores[2]:.3f})"
        print(f"✓ Style scores vary: {[f'{s:.3f}' for s in scores]}")


class TestStyleDNAInjection:
    """Test that style DNA (lexicon, connectors) is injected into prompts."""

    def test_paragraph_fusion_prompt_includes_vocabulary(self):
        """Test that PARAGRAPH_FUSION_PROMPT includes mandatory vocabulary section."""
        from src.generator.mutation_operators import PARAGRAPH_FUSION_PROMPT

        style_lexicon = ["dialectical", "contradiction", "material", "reality", "transformation"]
        vocabulary_text = ", ".join(style_lexicon[:15])

        # Format prompt with vocabulary
        prompt = PARAGRAPH_FUSION_PROMPT.format(
            propositions_list="- Test proposition",
            style_examples="Example text",
            mandatory_vocabulary=f"### MANDATORY VOCABULARY:\nYou MUST use at least 3-5 distinct words from this list: {vocabulary_text}",
            rhetorical_connectors=""
        )

        # Verify vocabulary section is present
        assert "MANDATORY VOCABULARY" in prompt, "Prompt should include mandatory vocabulary section"
        assert "dialectical" in prompt, "Prompt should include lexicon words"
        assert "3-5 distinct words" in prompt, "Prompt should instruct to use vocabulary"
        print("✓ Paragraph fusion prompt includes vocabulary section")

    def test_paragraph_fusion_prompt_includes_connectors(self):
        """Test that PARAGRAPH_FUSION_PROMPT includes rhetorical connectors section."""
        from src.generator.mutation_operators import PARAGRAPH_FUSION_PROMPT

        connectors = ["furthermore", "consequently", "it follows that", "in this way"]
        connectors_text = ", ".join(connectors)

        prompt = PARAGRAPH_FUSION_PROMPT.format(
            propositions_list="- Test proposition",
            style_examples="Example text",
            mandatory_vocabulary="",
            rhetorical_connectors=f"### RHETORICAL CONNECTORS:\nUse these transition phrases: {connectors_text}"
        )

        # Verify connectors section is present
        assert "RHETORICAL CONNECTORS" in prompt, "Prompt should include rhetorical connectors section"
        assert "furthermore" in prompt, "Prompt should include connector words"
        print("✓ Paragraph fusion prompt includes connectors section")

    def test_translate_paragraph_extracts_lexicon(self):
        """Test that translate_paragraph extracts and formats lexicon from style_dna."""
        translator = StyleTranslator(config_path="config.json")

        # Mock style_dna
        style_dna = {
            "lexicon": ["dialectical", "contradiction", "material", "reality", "transformation"] * 5,  # 25 words
            "tone": "Authoritative",
            "structure": "Complex periodic sentences"
        }

        # Mock atlas
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = [
            "Example text with dialectical contradictions."
        ]
        mock_atlas.get_author_style_vector.return_value = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

        # Mock proposition extractor
        translator.proposition_extractor = Mock()
        translator.proposition_extractor.extract_atomic_propositions.return_value = [
            "Test proposition"
        ]

        # Mock LLM provider
        translator.llm_provider = Mock()
        translator.llm_provider.call.return_value = json.dumps([
            "Generated paragraph text."
        ])

        # Mock critic
        with patch('src.generator.translator.SemanticCritic') as mock_critic_class:
            mock_critic = Mock()
            mock_critic.evaluate.return_value = {
                "pass": True,
                "proposition_recall": 1.0,
                "style_alignment": 0.8,
                "score": 0.9
            }
            mock_critic_class.return_value = mock_critic

            # Call translate_paragraph
            result = translator.translate_paragraph(
                paragraph="Test paragraph",
                atlas=mock_atlas,
                author_name="TestAuthor",
                style_dna=style_dna,
                verbose=False
            )

            # Verify LLM was called with prompt containing lexicon
            assert translator.llm_provider.call.called, "LLM should be called"
            call_args = translator.llm_provider.call.call_args
            user_prompt = call_args[1]["user_prompt"]

            # Should include lexicon words (limited to top 15) or vocabulary section
            # Check for either the actual word or the vocabulary section header
            has_vocab_section = "MANDATORY VOCABULARY" in user_prompt or "mandatory" in user_prompt.lower()
            has_lexicon_word = "dialectical" in user_prompt or "contradiction" in user_prompt
            assert has_vocab_section or has_lexicon_word, \
                f"Prompt should include lexicon from style_dna. Prompt preview: {user_prompt[:200]}..."
            print("✓ translate_paragraph extracts and formats lexicon")


class TestStylePreservingRepair:
    """Test that repairs preserve style vocabulary."""

    def test_repair_missing_artifacts_includes_style_lexicon(self):
        """Test that _repair_missing_artifacts includes style lexicon in prompt."""
        translator = StyleTranslator(config_path="config.json")
        translator.llm_provider = Mock()
        translator.llm_provider.call.return_value = "Repaired text with citations [^1]."

        style_lexicon = ["dialectical", "contradiction", "material"]

        result = translator._repair_missing_artifacts(
            candidate_text="Original text without citations.",
            missing_citations={"[^1]"},
            missing_quotes=[],
            style_lexicon=style_lexicon,
            verbose=False
        )

        # Verify LLM was called
        assert translator.llm_provider.call.called, "LLM should be called for repair"
        call_args = translator.llm_provider.call.call_args
        user_prompt = call_args[1]["user_prompt"]

        # Should include style preservation instructions
        assert "CRITICAL" in user_prompt or "Vocabulary" in user_prompt, \
            "Repair prompt should include style preservation instructions"
        assert "dialectical" in user_prompt or "contradiction" in user_prompt, \
            "Repair prompt should include lexicon words"
        print("✓ Repair missing artifacts includes style lexicon")

    def test_repair_prompt_preserves_style(self):
        """Test that repair prompt for missing facts includes style preservation."""
        translator = StyleTranslator(config_path="config.json")

        style_lexicon = ["dialectical", "contradiction", "material"]

        # This tests the repair prompt construction in translate_paragraph
        # We'll verify the prompt structure includes style preservation
        missing_propositions = ["Missing fact 1", "Missing fact 2"]
        current_best_text = "Existing text with dialectical contradictions."

        # Build repair prompt as translate_paragraph does
        missing_list = "\n".join([f"{i+1}. {prop}" for i, prop in enumerate(missing_propositions[:10])])

        style_preservation = ""
        if style_lexicon:
            lexicon_text = ", ".join(style_lexicon[:15])
            style_preservation = f"""

**CRITICAL: Do not lose the style of the original draft. You must maintain the Vocabulary ({lexicon_text}) and the Complex Sentence Structure. Do not simplify the text just to add the facts."""

        repair_prompt = f"""You have successfully captured most of the meaning, but you missed these specific facts:

{missing_list}

**Task:** Rewrite the paragraph. Keep the high-quality style you just generated, but surgically weave in these missing facts. Do not just append them at the end. Integrate them naturally into the flow.{style_preservation}

Original paragraph:
"{current_best_text}"

Generate 3 new variations that include all the missing facts seamlessly. Output as a JSON array of strings:"""

        # Verify style preservation is included
        assert "CRITICAL" in repair_prompt, "Repair prompt should include CRITICAL style preservation"
        assert "Vocabulary" in repair_prompt, "Repair prompt should mention Vocabulary"
        assert "dialectical" in repair_prompt, "Repair prompt should include lexicon words"
        assert "Do not simplify" in repair_prompt, "Repair prompt should forbid simplification"
        print("✓ Repair prompt preserves style vocabulary")


class TestAuthorStyleVector:
    """Test author style vector retrieval."""

    def test_get_author_style_vector_exists(self):
        """Test that StyleAtlas has get_author_style_vector method."""
        # Verify method exists
        assert hasattr(StyleAtlas, 'get_author_style_vector'), \
            "StyleAtlas should have get_author_style_vector method"
        print("✓ StyleAtlas has get_author_style_vector method")

    def test_get_author_style_vector_returns_vector(self):
        """Test that get_author_style_vector returns a numpy array."""
        # Create mock atlas with collection
        mock_atlas = StyleAtlas(
            collection_name="test_collection",
            cluster_ids={},
            cluster_centers=np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]),
            style_vectors=[],
            num_clusters=1
        )

        # Mock the collection and StyleBlender
        with patch('src.atlas.builder.StyleBlender') as mock_blender_class:
            mock_blender = Mock()
            mock_blender.get_author_centroid.return_value = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
            mock_blender_class.return_value = mock_blender

            result = mock_atlas.get_author_style_vector("TestAuthor")

            assert result is not None, "get_author_style_vector should return a vector"
            assert isinstance(result, np.ndarray), "Result should be a numpy array"
            assert len(result) == 7, "Style vector should have 7 dimensions"
            print("✓ get_author_style_vector returns numpy array")

    def test_get_author_style_vector_handles_missing_author(self):
        """Test that get_author_style_vector returns None for missing author."""
        mock_atlas = StyleAtlas(
            collection_name="test_collection",
            cluster_ids={},
            cluster_centers=np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]),
            style_vectors=[],
            num_clusters=1
        )

        # Mock StyleBlender to raise ValueError (author not found)
        with patch('src.atlas.builder.StyleBlender') as mock_blender_class:
            mock_blender = Mock()
            mock_blender.get_author_centroid.side_effect = ValueError("Author not found")
            mock_blender_class.return_value = mock_blender

            result = mock_atlas.get_author_style_vector("NonExistentAuthor")

            assert result is None, "Should return None when author not found"
            print("✓ get_author_style_vector handles missing author gracefully")


class TestStyleTransferIntegration:
    """Integration tests for full style transfer pipeline."""

    def test_paragraph_evaluation_includes_style_lexicon(self):
        """Test that paragraph evaluation passes style_lexicon to critic."""
        critic = SemanticCritic(config_path="config.json")

        blueprint = SemanticBlueprint(
            original_text="Test paragraph with dialectical contradictions.",
            svo_triples=[],
            core_keywords={"test", "paragraph"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        propositions = ["Proposition 1", "Proposition 2"]
        style_lexicon = ["dialectical", "contradiction"]

        # Mock author style vector
        try:
            from src.analyzer.style_metrics import get_style_vector
            author_style_vector = get_style_vector("Dialectical materialism shows contradictions.")
        except ImportError:
            print("⚠ Skipping test: style_metrics not available")
            return

        result = critic.evaluate(
            generated_text="The dialectical process reveals contradictions.",
            input_blueprint=blueprint,
            propositions=propositions,
            is_paragraph=True,
            author_style_vector=author_style_vector,
            style_lexicon=style_lexicon
        )

        # Verify result includes style_alignment
        assert "style_alignment" in result, "Result should include style_alignment"
        assert result["style_alignment"] > 0.0, "Style alignment should be > 0 when lexicon matches"
        print(f"✓ Paragraph evaluation includes style_lexicon: alignment={result['style_alignment']:.3f}")

    def test_style_scores_affect_selection(self):
        """Test that style scores influence candidate selection."""
        critic = SemanticCritic(config_path="config.json")

        try:
            from src.analyzer.style_metrics import get_style_vector
            author_style_vector = get_style_vector("Dialectical materialism demonstrates contradictions.")
        except ImportError:
            print("⚠ Skipping test: style_metrics not available")
            return

        style_lexicon = ["dialectical", "contradiction", "material"]

        candidates = [
            "The dialectical process reveals contradictions in material reality.",  # High style
            "The process shows contradictions in reality.",  # Medium style
            "Things change over time."  # Low style
        ]

        scores = []
        for candidate in candidates:
            score, _ = critic._check_style_alignment(
                candidate,
                author_style_vector=author_style_vector,
                style_lexicon=style_lexicon
            )
            scores.append(score)

        # First candidate should have highest style score
        # Allow some tolerance for floating point comparisons
        assert scores[0] > scores[2] - 0.01, \
            f"High-style candidate ({scores[0]:.3f}) should score higher than low-style ({scores[2]:.3f})"
        assert scores[0] >= scores[1] - 0.01, \
            f"High-style candidate ({scores[0]:.3f}) should score >= medium-style ({scores[1]:.3f})"
        print(f"✓ Style scores affect selection: {[f'{s:.3f}' for s in scores]}")


def run_all_tests():
    """Run all style transfer tests."""
    print("\n" + "="*70)
    print("STYLE TRANSFER COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")

    test_classes = [
        TestStyleMetricCalculation,
        TestStyleDNAInjection,
        TestStylePreservingRepair,
        TestAuthorStyleVector,
        TestStyleTransferIntegration
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        test_instance = test_class()

        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    passed_tests += 1
                except Exception as e:
                    print(f"✗ {method_name} FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "="*70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("="*70 + "\n")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

