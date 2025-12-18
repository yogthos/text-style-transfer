"""Verification test for contextual anchoring implementation.

This test verifies that:
1. Position tagging (OPENER, BODY, CLOSER, SINGLETON) works correctly
2. Context flow (previous_context) is properly passed between sentences
3. Prompts correctly inject context and position instructions
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Adjust path to find src if needed
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.blueprint import BlueprintExtractor
from src.pipeline import _split_into_paragraphs, _split_into_sentences_safe
from src.generator.translator import StyleTranslator


def test_tagging_logic():
    """Test 1: Verify position tagging is correct."""
    print("=== TEST 1: Tagging Logic ===")

    # Input: Two paragraphs.
    # Para 1: 3 sentences (Opener, Body, Closer).
    # Para 2: 1 sentence (Singleton).
    raw_text = (
        "The sky was blue. It had no clouds. The sun set.\n\n"
        "Night fell."
    )

    # Split into paragraphs and sentences, then extract blueprints with positions
    paragraphs = _split_into_paragraphs(raw_text)
    extractor = BlueprintExtractor()
    blueprints = []

    for para_idx, paragraph in enumerate(paragraphs):
        sentences = _split_into_sentences_safe(paragraph)
        previous_context = None

        for sent_idx, sentence in enumerate(sentences):
            # Determine position
            if len(sentences) == 1:
                position = "SINGLETON"
            elif sent_idx == 0:
                position = "OPENER"
            elif sent_idx == len(sentences) - 1:
                position = "CLOSER"
            else:
                position = "BODY"

            # Extract blueprint with metadata
            blueprint = extractor.extract(
                sentence,
                paragraph_id=para_idx,
                position=position,
                previous_context=previous_context
            )
            blueprints.append(blueprint)

            # Simulate generated text for next iteration
            previous_context = f"Generated: {sentence}"

    expected_tags = ["OPENER", "BODY", "CLOSER", "SINGLETON"]
    actual_tags = [b.position for b in blueprints]

    print(f"Tags Expected: {expected_tags}")
    print(f"Tags Actual:   {actual_tags}")

    assert actual_tags == expected_tags, f"FAILED: Position tags are incorrect. Expected {expected_tags}, got {actual_tags}"
    print("✓ PASS: Tagging logic is correct.\n")

    return blueprints


def test_context_flow(blueprints):
    """Test 2: Verify context flow and prompt injection."""
    print("=== TEST 2: The Daisy Chain (Context Flow) ===")

    # Create translator instance
    translator = StyleTranslator(config_path="config.json")

    # Mock the LLM provider to avoid actual API calls
    translator.llm_provider = MagicMock()
    translator.llm_provider.call.return_value = "Mock generated text"

    # Test that BODY sentence (index 1) receives context from OPENER (index 0)
    test_blueprint = blueprints[1]  # "It had no clouds" (BODY)

    # Set previous context (simulating what would come from sentence 0)
    previous_output = "MOCK_PREVIOUS_OUTPUT: The sky was blue (rewritten)"
    test_blueprint.previous_context = previous_output

    # Build prompt and check if context is injected
    try:
        # We need to provide the required arguments for _build_prompt
        from src.atlas.rhetoric import RhetoricalType
        prompt = translator._build_prompt(
            blueprint=test_blueprint,
            author_name="Test Author",
            style_dna="Test style",
            rhetorical_type=RhetoricalType.OBSERVATION,
            examples=["Example 1", "Example 2", "Example 3"]
        )

        print(f"Injected Context: {previous_output}")
        print(f"Position: {test_blueprint.position}")
        print(f"Prompt contains context: {previous_output in prompt}")
        print(f"Prompt contains position instruction: {'BODY' in prompt or 'CONTINUATION' in prompt}")

        # Check that context block is present for BODY position
        has_context_block = "PREVIOUS CONTEXT" in prompt or previous_output in prompt
        has_position_instruction = "CONTINUATION" in prompt or "BODY" in prompt or "INTERNAL BODY" in prompt

        if has_context_block and has_position_instruction:
            print("✓ PASS: Context and Instructions were successfully injected.")
            return True
        else:
            print(f"X FAIL: Context block present: {has_context_block}, Position instruction present: {has_position_instruction}")
            return False
    except Exception as e:
        print(f"X FAIL: Error building prompt: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_opener_no_context():
    """Test 3: Verify OPENER position does NOT include previous context."""
    print("\n=== TEST 3: OPENER Position (No Context) ===")

    extractor = BlueprintExtractor()
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()
    translator.llm_provider.call.return_value = "Mock generated text"

    # Create OPENER blueprint with no previous context
    opener_blueprint = extractor.extract(
        "The sky was blue.",
        paragraph_id=0,
        position="OPENER",
        previous_context=None
    )

    from src.atlas.rhetoric import RhetoricalType
    prompt = translator._build_prompt(
        blueprint=opener_blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example 1"]
    )

    # OPENER should NOT have PREVIOUS CONTEXT block
    has_context_block = "PREVIOUS CONTEXT" in prompt

    if not has_context_block:
        print("✓ PASS: OPENER position correctly excludes previous context.")
        return True
    else:
        print("X FAIL: OPENER position incorrectly includes previous context.")
        return False


def test_singleton_no_context():
    """Test 4: Verify SINGLETON position does NOT include previous context."""
    print("\n=== TEST 4: SINGLETON Position (No Context) ===")

    extractor = BlueprintExtractor()
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()
    translator.llm_provider.call.return_value = "Mock generated text"

    # Create SINGLETON blueprint with no previous context
    singleton_blueprint = extractor.extract(
        "Night fell.",
        paragraph_id=1,
        position="SINGLETON",
        previous_context=None
    )

    from src.atlas.rhetoric import RhetoricalType
    prompt = translator._build_prompt(
        blueprint=singleton_blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example 1"]
    )

    # SINGLETON should NOT have PREVIOUS CONTEXT block
    has_context_block = "PREVIOUS CONTEXT" in prompt

    if not has_context_block:
        print("✓ PASS: SINGLETON position correctly excludes previous context.")
        return True
    else:
        print("X FAIL: SINGLETON position incorrectly includes previous context.")
        return False


if __name__ == "__main__":
    try:
        # Run all tests
        blueprints = test_tagging_logic()
        result2 = test_context_flow(blueprints)
        result3 = test_opener_no_context()
        result4 = test_singleton_no_context()

        if result2 and result3 and result4:
            print("\n" + "="*60)
            print("✓ All Context Anchoring Tests Passed!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("X Some tests failed. Please review the output above.")
            print("="*60)
            sys.exit(1)
    except Exception as e:
        print(f"\nTEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

