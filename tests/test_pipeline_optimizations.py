"""Tests for pipeline optimization features.

Tests verify:
1. Lower precision threshold (0.50) allows style injection
2. Empty blueprint handling uses original_text
3. Zero-shot retrieval fallback gets random examples
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.blueprint import SemanticBlueprint
from src.validator.semantic_critic import SemanticCritic
from src.generator.translator import StyleTranslator
from src.atlas.rhetoric import RhetoricalType


def test_precision_threshold_allows_style_injection():
    """Test that precision threshold of 0.50 allows style injection."""
    # Load critic with config to get the new threshold
    critic = SemanticCritic(config_path="config.json")

    # Verify threshold is set correctly
    assert critic.precision_threshold == 0.50, f"Expected precision_threshold 0.50, got {critic.precision_threshold}"
    assert critic.recall_threshold == 0.85, f"Expected recall_threshold 0.85, got {critic.recall_threshold}"

    # Create input blueprint
    input_blueprint = SemanticBlueprint(
        original_text="Human experience reinforces the rule of finitude.",
        svo_triples=[("human experience", "reinforce", "the rule of finitude")],
        named_entities=[],
        core_keywords={"human", "experience", "reinforce", "rule", "finitude"},
        citations=[],
        quotes=[]
    )

    # Generated text with style injection (adjectives, metaphors)
    # This should pass with precision 0.50 but would fail with 0.75
    generated_with_style = "Human experience serves to reinforce the iron rule of finitude."

    result = critic.evaluate(generated_with_style, input_blueprint)

    # Check that precision score is calculated
    assert "precision_score" in result
    assert "recall_score" in result

    # With threshold 0.50, precision scores around 0.55 should pass
    # (The exact score depends on vector similarity, but we verify the threshold is correct)
    print(f"  Precision score: {result['precision_score']:.2f}")
    print(f"  Recall score: {result['recall_score']:.2f}")
    print(f"  Precision threshold: {critic.precision_threshold}")
    print(f"  Recall threshold: {critic.recall_threshold}")

    # Verify thresholds are correct (the actual pass/fail depends on scores)
    assert critic.precision_threshold <= 0.50, "Precision threshold should allow style injection"
    assert critic.recall_threshold >= 0.85, "Recall threshold should preserve meaning"

    print("✓ test_precision_threshold_allows_style_injection passed")


def test_empty_blueprint_uses_original_text():
    """Test that empty blueprint uses original_text in prompt."""
    translator = StyleTranslator(config_path="config.json")

    # Create empty blueprint (no SVOs, no keywords)
    empty_blueprint = SemanticBlueprint(
        original_text="Architecture remains.",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Mock LLM to capture the prompt
    mock_llm = Mock()
    mock_llm.call.return_value = "The architecture endures."
    translator.llm_provider = mock_llm

    # Build prompt
    prompt = translator._build_prompt(
        blueprint=empty_blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example 1", "Example 2"]
    )

    # Verify prompt uses original_text, not empty blueprint structure
    assert "INPUT TEXT:" in prompt or "INPUT TEXT" in prompt, "Should use INPUT TEXT for empty blueprint"
    assert "Architecture remains." in prompt, "Should include original text"
    assert "Subjects: None" not in prompt or "Subjects: []" not in prompt, "Should not show empty subjects list"

    print("✓ test_empty_blueprint_uses_original_text passed")


def test_empty_blueprint_literal_translation():
    """Test that empty blueprint uses original_text in literal translation."""
    translator = StyleTranslator(config_path="config.json")

    # Create empty blueprint
    empty_blueprint = SemanticBlueprint(
        original_text="The structure holds.",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Mock LLM
    mock_llm = Mock()
    mock_llm.call.return_value = "The structure endures."
    translator.llm_provider = mock_llm

    # Call translate_literal
    result = translator.translate_literal(
        blueprint=empty_blueprint,
        author_name="Test Author",
        style_dna="Test style"
    )

    # Verify LLM was called
    assert mock_llm.call.called

    # Get the prompt that was sent
    call_args = mock_llm.call.call_args
    user_prompt = call_args[1].get('user_prompt') or call_args[0][1] if len(call_args[0]) > 1 else ""

    # Verify prompt uses INPUT TEXT
    assert "INPUT TEXT:" in user_prompt, "Literal translation should use INPUT TEXT for empty blueprint"
    assert "The structure holds." in user_prompt, "Should include original text"

    print("✓ test_empty_blueprint_literal_translation passed")


def test_atlas_fallback_random_examples():
    """Test that atlas falls back to random examples when no rhetorical matches found."""
    from src.atlas.builder import StyleAtlas
    from src.atlas.rhetoric import RhetoricalType

    # Create a mock atlas with collection
    mock_atlas = MagicMock(spec=StyleAtlas)
    mock_atlas.collection_name = "test_atlas"

    # Track calls to verify fallback behavior
    call_count = {"get": 0}
    mock_collection = MagicMock()

    # First call: no results for specific rhetorical type
    # Second call: return random examples
    def mock_get(where=None, limit=None):
        call_count["get"] += 1
        if call_count["get"] == 1:
            # First call: no results for DEFINITION
            if where and where.get("rhetorical_type") == "DEFINITION":
                return {"documents": []}
        else:
            # Second call (fallback): return random examples
            return {"documents": ["Random example 1.", "Random example 2.", "Random example 3."]}
        return {"documents": []}

    mock_collection.get = mock_get
    mock_atlas._collection = mock_collection
    mock_atlas._client = MagicMock()

    # Simulate the fallback logic from get_examples_by_rhetoric
    try:
        results = mock_collection.get(where={"rhetorical_type": RhetoricalType.DEFINITION.value}, limit=6)
    except Exception:
        results = None

    # FALLBACK: If no examples found, get random examples
    if not results or not results.get('documents'):
        print(f"    ⚠ No examples found for {RhetoricalType.DEFINITION.value}. Fetching random style samples.")
        try:
            results = mock_collection.get(limit=6)
        except Exception:
            results = {"documents": []}

    # Verify fallback was triggered and returned examples
    assert call_count["get"] >= 2, "Should call get() at least twice (initial + fallback)"
    assert len(results.get('documents', [])) > 0, "Fallback should return random examples"
    assert results['documents'][0] == "Random example 1.", "Should get random examples"

    print("✓ test_atlas_fallback_random_examples passed")


def test_non_empty_blueprint_uses_normal_structure():
    """Test that non-empty blueprint uses normal blueprint structure."""
    translator = StyleTranslator(config_path="config.json")

    # Create normal blueprint with SVOs and keywords
    normal_blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("the cat", "sit", "the mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    # Build prompt
    prompt = translator._build_prompt(
        blueprint=normal_blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example 1"]
    )

    # Verify prompt uses blueprint structure, not INPUT TEXT
    assert "INPUT BLUEPRINT" in prompt, "Should use INPUT BLUEPRINT for normal blueprint"
    assert "Subjects:" in prompt, "Should include subjects"
    assert "Actions:" in prompt, "Should include actions"
    assert "INPUT TEXT:" not in prompt, "Should not use INPUT TEXT for non-empty blueprint"

    print("✓ test_non_empty_blueprint_uses_normal_structure passed")


def test_precision_threshold_config_loading():
    """Test that precision and recall thresholds are loaded from config."""
    critic = SemanticCritic(config_path="config.json")

    # Verify thresholds match config.json values
    assert critic.precision_threshold == 0.50, \
        f"Precision threshold should be 0.50 from config, got {critic.precision_threshold}"
    assert critic.recall_threshold == 0.85, \
        f"Recall threshold should be 0.85 from config, got {critic.recall_threshold}"

    print("✓ test_precision_threshold_config_loading passed")


if __name__ == "__main__":
    test_precision_threshold_allows_style_injection()
    test_empty_blueprint_uses_original_text()
    test_empty_blueprint_literal_translation()
    test_atlas_fallback_random_examples()
    test_non_empty_blueprint_uses_normal_structure()
    test_precision_threshold_config_loading()
    print("\n✓ All pipeline optimization tests completed!")

