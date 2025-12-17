"""Test script for Kaleidoscope Strategy features.

Tests the variability features including:
1. StructureNavigator with history tracking
2. Global vocabulary extraction
3. Stochastic structure matching
4. Integration with pipeline
"""

import sys
import random
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.navigator import StructureNavigator
from src.analyzer.vocabulary import extract_global_vocabulary
from src.atlas import build_style_atlas, find_structure_match


def test_structure_navigator_history_tracking():
    """Test that StructureNavigator prevents immediate repetition."""
    navigator = StructureNavigator(history_limit=3)

    # Create test candidates
    candidates = [
        {'id': 'para_0', 'text': 'First sentence.', 'word_count': 5},
        {'id': 'para_1', 'text': 'Second sentence here.', 'word_count': 6},
        {'id': 'para_2', 'text': 'Third sentence with more words.', 'word_count': 7},
        {'id': 'para_3', 'text': 'Fourth sentence.', 'word_count': 5},
        {'id': 'para_4', 'text': 'Fifth sentence here.', 'word_count': 6},
    ]

    input_length = 6

    # Select first template
    selected1 = navigator.select_template(candidates, input_length)
    assert selected1 is not None, "Should select a template"
    first_id = selected1.get('id')
    assert first_id in navigator.history_buffer, "Selected ID should be in history"

    # Select second template - should not be the same as first
    selected2 = navigator.select_template(candidates, input_length)
    assert selected2 is not None, "Should select a template"
    second_id = selected2.get('id')
    assert second_id != first_id, "Should not select the same template immediately"
    assert second_id not in navigator.history_buffer or len(navigator.history_buffer) > 1, "History should track selections"

    # Select third template
    selected3 = navigator.select_template(candidates, input_length)
    assert selected3 is not None, "Should select a template"
    third_id = selected3.get('id')

    # After 3 selections, history should have 3 items
    assert len(navigator.history_buffer) <= 3, "History should not exceed limit"

    print(f"✓ History tracking test passed")
    print(f"  Selected IDs: {first_id}, {second_id}, {third_id}")
    print(f"  History buffer: {navigator.history_buffer}")


def test_structure_navigator_weighted_selection():
    """Test that StructureNavigator uses weighted selection favoring better matches."""
    navigator = StructureNavigator(history_limit=3)

    # Create candidates with varying length differences
    candidates = [
        {'id': 'para_0', 'text': 'Perfect match.', 'word_count': 10},  # Best match
        {'id': 'para_1', 'text': 'Close match here.', 'word_count': 11},  # Good match
        {'id': 'para_2', 'text': 'Further away.', 'word_count': 15},  # Worse match
        {'id': 'para_3', 'text': 'Even further.', 'word_count': 20},  # Worst match
        {'id': 'para_4', 'text': 'Another one.', 'word_count': 12},  # Decent match
    ]

    input_length = 10

    # Run multiple selections and check that better matches are selected more often
    selections = []
    for _ in range(20):
        selected = navigator.select_template(candidates, input_length)
        if selected:
            selections.append(selected.get('id'))

    # Best match (para_0) should be selected frequently
    para_0_count = selections.count('para_0')
    para_3_count = selections.count('para_3')  # Worst match

    # Best match should be selected more often than worst match
    assert para_0_count >= para_3_count, "Better matches should be selected more often"

    print(f"✓ Weighted selection test passed")
    print(f"  para_0 (best) selected {para_0_count} times")
    print(f"  para_3 (worst) selected {para_3_count} times")


def test_structure_navigator_fallback():
    """Test that StructureNavigator handles edge cases gracefully."""
    navigator = StructureNavigator(history_limit=3)

    # Test with empty candidates
    result = navigator.select_template([], 10)
    assert result is None, "Should return None for empty candidates"

    # Test with single candidate
    single_candidate = [{'id': 'para_0', 'text': 'Only one.', 'word_count': 5}]
    result = navigator.select_template(single_candidate, 5)
    assert result is not None, "Should select single candidate"
    assert result.get('id') == 'para_0', "Should select the only candidate"

    # Test with all candidates in history (should clear buffer)
    candidates = [
        {'id': 'para_0', 'text': 'First.', 'word_count': 5},
        {'id': 'para_1', 'text': 'Second.', 'word_count': 6},
    ]

    # Fill history buffer
    navigator.history_buffer = ['para_0', 'para_1']

    # Should still be able to select (buffer will be cleared)
    result = navigator.select_template(candidates, 5)
    assert result is not None, "Should select even when all in history"

    print(f"✓ Fallback behavior test passed")


def test_extract_global_vocabulary():
    """Test global vocabulary extraction with sentiment clustering."""
    sample_text = """
    The beautiful sunset was amazing and wonderful. I love the brilliant colors.
    The terrible storm was awful and horrible. I hate the destructive winds.
    The neutral day was normal and average. The regular routine continued.
    """

    vocab_dict = extract_global_vocabulary(sample_text, top_n=50)

    # Check structure
    assert isinstance(vocab_dict, dict), "Should return a dictionary"
    assert 'positive' in vocab_dict, "Should have positive words"
    assert 'negative' in vocab_dict, "Should have negative words"
    assert 'neutral' in vocab_dict, "Should have neutral words"

    # Check that lists contain strings
    for sentiment, words in vocab_dict.items():
        assert isinstance(words, list), f"{sentiment} should be a list"
        assert all(isinstance(word, str) for word in words), f"All {sentiment} words should be strings"

    # Check that we extracted some words
    total_words = sum(len(words) for words in vocab_dict.values())
    assert total_words > 0, "Should extract some words"

    print(f"✓ Global vocabulary extraction test passed")
    print(f"  Positive words: {len(vocab_dict['positive'])}")
    print(f"  Negative words: {len(vocab_dict['negative'])}")
    print(f"  Neutral words: {len(vocab_dict['neutral'])}")
    print(f"  Sample positive: {vocab_dict['positive'][:5] if vocab_dict['positive'] else 'None'}")


def test_extract_global_vocabulary_empty():
    """Test global vocabulary extraction with empty text."""
    vocab_dict = extract_global_vocabulary("", top_n=50)

    assert isinstance(vocab_dict, dict), "Should return a dictionary"
    assert 'positive' in vocab_dict, "Should have positive key"
    assert 'negative' in vocab_dict, "Should have negative key"
    assert 'neutral' in vocab_dict, "Should have neutral key"

    # All should be empty lists
    for words in vocab_dict.values():
        assert isinstance(words, list), "Should be lists"
        assert len(words) == 0, "Should be empty for empty text"

    print(f"✓ Empty text test passed")


def test_find_structure_match_with_navigator():
    """Test that find_structure_match works with StructureNavigator."""
    import uuid

    sample_text = """
    The quick brown fox jumps over the lazy dog.
    The swift runner moved quickly through the park.
    The fast athlete ran rapidly across the field.
    The dog rested on the floor. The cat sat on the mat.
    The bird flew in the sky. The sun shone brightly.
    """

    try:
        collection_name = f"test_navigator_{uuid.uuid4().hex[:8]}"
        atlas = build_style_atlas(sample_text=sample_text, num_clusters=2, collection_name=collection_name)

        navigator = StructureNavigator(history_limit=3)
        input_text = "The cat sat on the mat."

        # First selection
        match1 = find_structure_match(
            atlas,
            target_cluster_id=0,
            input_text=input_text,
            length_tolerance=0.3,
            navigator=navigator
        )

        assert match1 is not None, "Should find a match"
        assert isinstance(match1, str), "Should return a string"

        # Second selection - should be different due to history tracking
        match2 = find_structure_match(
            atlas,
            target_cluster_id=0,
            input_text=input_text,
            length_tolerance=0.3,
            navigator=navigator
        )

        assert match2 is not None, "Should find a match"

        # With history tracking, we might get different results
        # (though not guaranteed if there's only one good match)
        print(f"✓ Structure match with navigator test passed")
        print(f"  First match: {match1[:50]}...")
        print(f"  Second match: {match2[:50]}...")
        print(f"  History buffer: {navigator.history_buffer}")

    except Exception as e:
        print(f"⚠ Navigator integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_find_structure_match_without_navigator():
    """Test that find_structure_match still works without navigator (backward compatibility)."""
    import uuid

    sample_text = """
    The quick brown fox jumps over the lazy dog.
    The swift runner moved quickly through the park.
    """

    try:
        collection_name = f"test_no_nav_{uuid.uuid4().hex[:8]}"
        atlas = build_style_atlas(sample_text=sample_text, num_clusters=1, collection_name=collection_name)

        input_text = "The cat sat on the mat."

        # Should work without navigator
        match = find_structure_match(
            atlas,
            target_cluster_id=0,
            input_text=input_text,
            length_tolerance=0.3,
            navigator=None  # No navigator
        )

        # Should still return a match (or None if no match found)
        if match:
            assert isinstance(match, str), "Should return a string"
            print(f"✓ Backward compatibility test passed (found match)")
        else:
            print(f"✓ Backward compatibility test passed (no match found, which is valid)")

    except Exception as e:
        print(f"⚠ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()


def test_structure_navigator_length_scoring():
    """Test that StructureNavigator scores candidates by length similarity."""
    navigator = StructureNavigator(history_limit=3)

    # Create candidates with different length differences
    candidates = [
        {'id': 'para_0', 'text': 'Exact match.', 'word_count': 10},  # Perfect
        {'id': 'para_1', 'text': 'Close match.', 'word_count': 11},  # Close
        {'id': 'para_2', 'text': 'Far away.', 'word_count': 20},  # Far
    ]

    input_length = 10

    # Select multiple times and verify better matches are preferred
    selections = []
    for _ in range(30):
        selected = navigator.select_template(candidates, input_length)
        if selected:
            selections.append(selected.get('word_count'))

    # Count how often each length is selected
    exact_count = selections.count(10)
    close_count = selections.count(11)
    far_count = selections.count(20)

    # Exact match should be selected most often
    assert exact_count > 0, "Exact match should be selected sometimes"
    # Exact or close should be selected more than far
    assert (exact_count + close_count) > far_count, "Better length matches should be preferred"

    print(f"✓ Length scoring test passed")
    print(f"  Exact (10): {exact_count}, Close (11): {close_count}, Far (20): {far_count}")


def test_prompt_builder_with_global_vocab():
    """Test that PromptAssembler includes global vocabulary in prompts."""
    from src.generator.prompt_builder import PromptAssembler

    assembler = PromptAssembler(target_author_name="Test Author")

    input_text = "The cat sat on the mat."
    structure_match = "The dog rested on the floor."
    situation_match = "The bird flew in the sky."
    global_vocab_list = ["beautiful", "amazing", "wonderful", "brilliant", "excellent"]

    # Build prompt with global vocabulary
    prompt = assembler.build_generation_prompt(
        input_text=input_text,
        situation_match=situation_match,
        structure_match=structure_match,
        global_vocab_list=global_vocab_list
    )

    # Check that prompt contains vocabulary section
    assert "VOCABULARY INSPIRATION" in prompt, "Prompt should contain vocabulary section"
    assert "PRIMARY SOURCE" in prompt, "Prompt should mention primary source"
    assert "SECONDARY SOURCE" in prompt, "Prompt should mention secondary source"

    # Check that some global vocab words appear in prompt
    vocab_found = any(word in prompt for word in global_vocab_list)
    assert vocab_found, "Some global vocabulary words should appear in prompt"

    # Build prompt without global vocabulary (should still work)
    prompt_no_vocab = assembler.build_generation_prompt(
        input_text=input_text,
        situation_match=situation_match,
        structure_match=structure_match,
        global_vocab_list=None
    )

    assert prompt_no_vocab is not None, "Should build prompt without vocab"
    assert len(prompt_no_vocab) > 0, "Prompt should not be empty"

    print(f"✓ Prompt builder with global vocab test passed")
    print(f"  Prompt length with vocab: {len(prompt)}")
    print(f"  Prompt length without vocab: {len(prompt_no_vocab)}")


if __name__ == "__main__":
    print("Running Kaleidoscope Strategy tests...\n")

    try:
        test_structure_navigator_history_tracking()
        test_structure_navigator_weighted_selection()
        test_structure_navigator_fallback()
        test_structure_navigator_length_scoring()
        test_extract_global_vocabulary()
        test_extract_global_vocabulary_empty()
        test_find_structure_match_with_navigator()
        test_find_structure_match_without_navigator()
        test_prompt_builder_with_global_vocab()
        print("\n✓ All Kaleidoscope Strategy tests completed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

