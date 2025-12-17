"""Test script for vocabulary mapping functionality."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.vocabulary import build_vocab_map, extract_global_vocabulary


def test_build_vocab_map():
    """Test that build_vocab_map creates a valid vocabulary mapping."""
    # Sample text with some repeated words and synonyms
    sample_text = """
    The swift runner moved quickly across the field.
    The fast athlete ran rapidly through the track.
    The quick jogger sprinted briskly along the path.
    The rapid movement was impressive. The speedy dash was remarkable.
    """

    vocab_map = build_vocab_map(sample_text, similarity_threshold=0.6)

    # Assertions
    assert isinstance(vocab_map, dict), "vocab_map should be a dictionary"

    # Check that all values are lists
    for key, value in vocab_map.items():
        assert isinstance(key, str), f"Key should be string, got {type(key)}"
        assert isinstance(value, list), f"Value should be list, got {type(value)}"
        assert all(isinstance(word, str) for word in value), "All synonyms should be strings"

    print(f"✓ Vocabulary map created with {len(vocab_map)} entries")
    print(f"  Sample entries: {dict(list(vocab_map.items())[:3])}")
    print("✓ build_vocab_map test passed")


def test_build_vocab_map_empty():
    """Test that build_vocab_map handles empty text gracefully."""
    vocab_map = build_vocab_map("")
    assert vocab_map == {}, "Empty text should return empty dict"
    print("✓ Empty text test passed")


def test_build_vocab_map_no_content_words():
    """Test that build_vocab_map handles text with no content words."""
    vocab_map = build_vocab_map("the and or but")
    # Should return empty or very small dict since these are mostly stop words
    assert isinstance(vocab_map, dict), "Should return a dict even with no content words"
    print("✓ No content words test passed")


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
    print(f"  Positive: {len(vocab_dict['positive'])}, "
          f"Negative: {len(vocab_dict['negative'])}, "
          f"Neutral: {len(vocab_dict['neutral'])}")


if __name__ == "__main__":
    print("Running vocabulary mapping tests...\n")

    try:
        test_build_vocab_map()
        test_build_vocab_map_empty()
        test_build_vocab_map_no_content_words()
        test_extract_global_vocabulary()
        print("\n✓ All vocabulary tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

