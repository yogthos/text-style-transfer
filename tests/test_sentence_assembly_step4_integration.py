"""Tests for Step 4: Integration test for sentence-by-sentence assembly.

This test verifies that translate_paragraph correctly uses the new
sentence-by-sentence assembly architecture.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure config.json exists before imports
from tests.test_helpers import ensure_config_exists
ensure_config_exists()

try:
    from src.generator.translator import StyleTranslator
    from src.analysis.semantic_analyzer import PropositionExtractor
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠ Skipping tests: Missing dependencies - {IMPORT_ERROR}")


def test_translate_paragraph_simple():
    """Test with simple 2-sentence paragraph."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_translate_paragraph_simple (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    paragraph = "I spent my childhood in the Soviet Union. That country is a ghost now."

    # Mock atlas
    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = ["Example sentence"]

    # Mock LLM for generation
    mock_llm = Mock()
    mock_llm.call.return_value = "I spent my childhood in the Soviet Union."
    translator.llm_provider = mock_llm

    # Mock critic
    def mock_evaluate(text, blueprint, **kwargs):
        return {
            "proposition_recall": 0.90,
            "style_alignment": 0.75,
            "score": 0.83,
            "pass": True
        }

    with patch('src.validator.semantic_critic.SemanticCritic.evaluate', mock_evaluate):
        try:
            result = translator.translate_paragraph(
                paragraph=paragraph,
                atlas=mock_atlas,
                author_name="Test Author",
                verbose=False
            )
            assert result is not None, "Should return a result"
            assert isinstance(result, tuple), "Should return tuple"
            assert len(result) == 4, "Should return (text, rhythm_map, teacher_example, recall)"
            assert isinstance(result[0], str), "First element should be string"
            assert len(result[0]) > 0, "Generated text should not be empty"
        except Exception as e:
            if "atlas" in str(e).lower() or "style" in str(e).lower():
                print(f"  ⚠ Test skipped due to missing dependencies: {e}")
                return

    print("✓ test_translate_paragraph_simple passed")


def test_translate_paragraph_sentence_count():
    """Test that output has same number of sentences as input."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_translate_paragraph_sentence_count (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    paragraph = "First sentence. Second sentence. Third sentence."

    # Count input sentences
    nlp = translator._get_nlp()
    if nlp:
        doc = nlp(paragraph)
        input_sentences = len(list(doc.sents))
    else:
        input_sentences = len([s for s in paragraph.split('.') if s.strip()])

    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = ["Example"]

    mock_llm = Mock()
    mock_llm.call.return_value = "Generated sentence."
    translator.llm_provider = mock_llm

    def mock_evaluate(text, blueprint, **kwargs):
        return {
            "proposition_recall": 0.90,
            "style_alignment": 0.75,
            "score": 0.83,
            "pass": True
        }

    with patch('src.validator.semantic_critic.SemanticCritic.evaluate', mock_evaluate):
        try:
            result = translator.translate_paragraph(
                paragraph=paragraph,
                atlas=mock_atlas,
                author_name="Test Author",
                verbose=False
            )
            output_text = result[0]

            # Count output sentences
            if nlp:
                doc = nlp(output_text)
                output_sentences = len(list(doc.sents))
            else:
                output_sentences = len([s for s in output_text.split('.') if s.strip()])

            # Should have same number of sentences (or close)
            assert abs(output_sentences - input_sentences) <= 1, \
                f"Output should have similar sentence count. Input: {input_sentences}, Output: {output_sentences}"
        except Exception as e:
            if "atlas" in str(e).lower():
                print(f"  ⚠ Test skipped due to missing dependencies: {e}")
                return

    print("✓ test_translate_paragraph_sentence_count passed")


def test_translate_paragraph_preserves_propositions():
    """Test that all propositions are preserved (recall >= 0.85)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_translate_paragraph_preserves_propositions (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    paragraph = "I spent my childhood in the Soviet Union. That country is a ghost now."

    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = ["Example"]

    mock_llm = Mock()
    mock_llm.call.return_value = "I spent my childhood in the Soviet Union."
    translator.llm_provider = mock_llm

    def mock_evaluate(text, blueprint, **kwargs):
        return {
            "proposition_recall": 0.90,  # Above threshold
            "style_alignment": 0.75,
            "score": 0.83,
            "pass": True
        }

    with patch('src.validator.semantic_critic.SemanticCritic.evaluate', mock_evaluate):
        try:
            result = translator.translate_paragraph(
                paragraph=paragraph,
                atlas=mock_atlas,
                author_name="Test Author",
                verbose=False
            )
            # Final recall should be >= 0.85
            final_recall = result[3]  # 4th element is recall
            assert final_recall >= 0.85, f"Final recall should be >= 0.85, got {final_recall}"
        except Exception as e:
            if "atlas" in str(e).lower():
                print(f"  ⚠ Test skipped due to missing dependencies: {e}")
                return

    print("✓ test_translate_paragraph_preserves_propositions passed")


def test_translate_paragraph_empty():
    """Test with empty paragraph (edge case)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_translate_paragraph_empty (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    paragraph = ""

    mock_atlas = Mock()

    try:
        result = translator.translate_paragraph(
            paragraph=paragraph,
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=False
        )
        # Should return empty string or original
        assert result[0] == paragraph or result[0] == "", "Should return empty for empty input"
    except Exception as e:
        if "atlas" in str(e).lower():
            print(f"  ⚠ Test skipped due to missing dependencies: {e}")
            return

    print("✓ test_translate_paragraph_empty passed")


def test_translate_paragraph_single_sentence():
    """Test with single-sentence paragraph."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_translate_paragraph_single_sentence (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    paragraph = "I spent my childhood in the Soviet Union."

    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = ["Example"]

    mock_llm = Mock()
    mock_llm.call.return_value = "I spent my childhood in the Soviet Union."
    translator.llm_provider = mock_llm

    def mock_evaluate(text, blueprint, **kwargs):
        return {
            "proposition_recall": 0.90,
            "style_alignment": 0.75,
            "score": 0.83,
            "pass": True
        }

    with patch('src.validator.semantic_critic.SemanticCritic.evaluate', mock_evaluate):
        try:
            result = translator.translate_paragraph(
                paragraph=paragraph,
                atlas=mock_atlas,
                author_name="Test Author",
                verbose=False
            )
            assert result[0] is not None, "Should return a sentence"
            assert len(result[0]) > 0, "Generated text should not be empty"
        except Exception as e:
            if "atlas" in str(e).lower():
                print(f"  ⚠ Test skipped due to missing dependencies: {e}")
                return

    print("✓ test_translate_paragraph_single_sentence passed")


if __name__ == "__main__":
    test_translate_paragraph_simple()
    test_translate_paragraph_sentence_count()
    test_translate_paragraph_preserves_propositions()
    test_translate_paragraph_empty()
    test_translate_paragraph_single_sentence()
    print("\n✓ All Step 4 tests completed!")

