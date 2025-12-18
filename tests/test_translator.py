"""Tests for style translator."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType
from src.generator.translator import StyleTranslator


def test_basic_translation():
    """Test basic translation with mocked LLM."""
    # Create a simple blueprint
    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("the cat", "sit", "the mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    # Mock LLM provider
    mock_llm = Mock()
    mock_llm.call.return_value = "The feline rested upon the rug."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    examples = ["The feline rested upon the rug.", "A cat positioned itself on a mat."]

    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Formal and precise.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=examples
    )

    assert result == "The feline rested upon the rug."
    assert mock_llm.call.called

    print("✓ test_basic_translation passed")


def test_complex_translation():
    """Test complex translation."""
    blueprint = SemanticBlueprint(
        original_text="Human experience reinforces the rule of finitude.",
        svo_triples=[("human experience", "reinforce", "the rule of finitude")],
        named_entities=[],
        core_keywords={"human", "experience", "reinforce", "rule", "finitude"},
        citations=[],
        quotes=[]
    )

    mock_llm = Mock()
    mock_llm.call.return_value = "Human practice reinforces the rule of finitude."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    examples = ["Practice makes perfect.", "The rule stands firm."]

    result = translator.translate(
        blueprint=blueprint,
        author_name="Mao",
        style_dna="Revolutionary and direct.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=examples
    )

    assert "practice" in result.lower() or "experience" in result.lower()
    assert mock_llm.call.called

    print("✓ test_complex_translation passed")


def test_rhetorical_mode_matching():
    """Test that rhetorical mode is included in prompt."""
    blueprint = SemanticBlueprint(
        original_text="A revolution is a process.",
        svo_triples=[("a revolution", "be", "a process")],
        named_entities=[],
        core_keywords={"revolution", "process"},
        citations=[],
        quotes=[]
    )

    mock_llm = Mock()
    mock_llm.call.return_value = "A revolution represents a process."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    examples = ["A war is a struggle.", "A battle is a conflict."]

    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Academic style.",
        rhetorical_type=RhetoricalType.DEFINITION,
        examples=examples
    )

    # Verify that the prompt included the rhetorical mode
    call_args = mock_llm.call.call_args
    prompt = call_args[1]['prompt'] if 'prompt' in call_args[1] else call_args[0][0]
    assert "DEFINITION" in prompt or "definition" in prompt.lower()

    print("✓ test_rhetorical_mode_matching passed")


def test_error_handling():
    """Test error handling for edge cases."""
    blueprint = SemanticBlueprint(
        original_text="Test sentence.",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Test with empty examples
    mock_llm = Mock()
    mock_llm.call.return_value = "Translated text."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=[]  # Empty examples
    )

    assert result == "Translated text."

    # Test with LLM failure
    mock_llm.call.side_effect = Exception("LLM error")
    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example"]
    )

    # Should fall back to literal translation
    assert isinstance(result, str)

    print("✓ test_error_handling passed")


def test_translate_literal():
    """Test literal translation fallback."""
    blueprint = SemanticBlueprint(
        original_text="The server crashed.",
        svo_triples=[("the server", "crash", "")],
        named_entities=[],
        core_keywords={"server", "crash"},
        citations=[],
        quotes=[]
    )

    mock_llm = Mock()
    mock_llm.call.return_value = "The server stopped working."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    result = translator.translate_literal(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Clear and direct."
    )

    assert result == "The server stopped working."
    assert mock_llm.call.called

    # Test with LLM failure
    mock_llm.call.side_effect = Exception("LLM error")
    result = translator.translate_literal(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style."
    )

    # Should return original text
    assert result == blueprint.original_text

    print("✓ test_translate_literal passed")


if __name__ == "__main__":
    test_basic_translation()
    test_complex_translation()
    test_rhetorical_mode_matching()
    test_error_handling()
    test_translate_literal()
    print("\n✓ All translator tests completed!")

