"""Test script to verify that StyleProfile and ContentUnit models work correctly."""

import sys
from pathlib import Path

import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import ContentUnit, StyleProfile


def test_style_profile():
    """Test that StyleProfile can be instantiated and holds values correctly."""
    # Create dummy data
    vocab_map = {"fast": ["swift", "brisk"], "big": ["massive", "enormous"]}

    # Create a 2x2 POS Markov chain (e.g., ADJ -> NOUN, ADJ -> ADJ)
    pos_markov_chain = np.array([
        [0.7, 0.3],  # ADJ -> ADJ: 0.7, ADJ -> NOUN: 0.3
        [0.4, 0.6]   # NOUN -> ADJ: 0.4, NOUN -> NOUN: 0.6
    ])

    # Create a 2x2 sentence flow Markov chain (e.g., Simple -> Complex, Simple -> Fragment)
    sentence_flow_markov = np.array([
        [0.5, 0.5],  # Simple -> Simple: 0.5, Simple -> Complex: 0.5
        [0.3, 0.7]   # Complex -> Simple: 0.3, Complex -> Complex: 0.7
    ])

    # Instantiate StyleProfile
    profile = StyleProfile(
        vocab_map=vocab_map,
        pos_markov_chain=pos_markov_chain,
        sentence_flow_markov=sentence_flow_markov
    )

    # Assertions
    assert isinstance(profile, StyleProfile)
    assert profile.vocab_map == vocab_map
    assert isinstance(profile.vocab_map, dict)
    assert isinstance(profile.vocab_map["fast"], list)
    assert profile.vocab_map["fast"] == ["swift", "brisk"]

    assert isinstance(profile.pos_markov_chain, np.ndarray)
    assert profile.pos_markov_chain.shape == (2, 2)
    assert np.allclose(profile.pos_markov_chain, pos_markov_chain)

    assert isinstance(profile.sentence_flow_markov, np.ndarray)
    assert profile.sentence_flow_markov.shape == (2, 2)
    assert np.allclose(profile.sentence_flow_markov, sentence_flow_markov)

    print("✓ StyleProfile test passed")


def test_content_unit():
    """Test that ContentUnit can be instantiated and holds values correctly."""
    # Create dummy data
    svo_triples = [("fox", "jump", "dog"), ("cat", "chase", "mouse")]
    entities = ["John", "London"]
    original_text = "The quick brown fox jumps over the dog."
    content_words = ["quick", "brown", "fox", "jumps", "dog"]

    # Instantiate ContentUnit
    content = ContentUnit(
        svo_triples=svo_triples,
        entities=entities,
        original_text=original_text,
        content_words=content_words
    )

    # Assertions
    assert isinstance(content, ContentUnit)
    assert content.svo_triples == svo_triples
    assert isinstance(content.svo_triples, list)
    assert isinstance(content.svo_triples[0], tuple)
    assert len(content.svo_triples[0]) == 3
    assert content.svo_triples[0] == ("fox", "jump", "dog")

    assert content.entities == entities
    assert isinstance(content.entities, list)
    assert content.entities[0] == "John"

    assert content.original_text == original_text
    assert isinstance(content.original_text, str)

    assert content.content_words == content_words
    assert isinstance(content.content_words, list)
    assert len(content.content_words) > 0

    print("✓ ContentUnit test passed")


if __name__ == "__main__":
    print("Running model verification tests...\n")
    test_style_profile()
    test_content_unit()
    print("\n✓ All tests passed!")

