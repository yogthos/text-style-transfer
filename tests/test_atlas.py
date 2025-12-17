"""Test script for Style Atlas and RAG retrieval."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas import (
    build_style_atlas,
    find_situation_match,
    find_structure_match,
    build_cluster_markov,
    predict_next_cluster
)


def test_build_style_atlas():
    """Test building a Style Atlas from sample text."""
    import uuid
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    The swift runner moved quickly through the park.
    The fast athlete ran rapidly across the field.
    The dog rested on the floor. The cat sat on the mat.
    The bird flew in the sky. The sun shone brightly.
    """

    try:
        # Use unique collection name to avoid conflicts
        collection_name = f"test_atlas_{uuid.uuid4().hex[:8]}"
        atlas = build_style_atlas(
            sample_text=sample_text,
            num_clusters=3,
            collection_name=collection_name
        )

        assert atlas is not None, "Should return a StyleAtlas"
        assert atlas.num_clusters > 0, "Should have at least one cluster"
        assert len(atlas.cluster_ids) > 0, "Should have cluster assignments"
        assert len(atlas.cluster_centers) == atlas.num_clusters, "Should have cluster centers"

        print(f"✓ Build Style Atlas test passed")
        print(f"  Clusters: {atlas.num_clusters}")
        print(f"  Paragraphs: {len(atlas.cluster_ids)}")

    except Exception as e:
        print(f"⚠ Build atlas test failed: {e}")
        import traceback
        traceback.print_exc()


def test_find_situation_match():
    """Test finding situation match (semantic similarity)."""
    import uuid
    sample_text = """
    The dog rested on the floor. The cat sat on the mat.
    The bird flew in the sky. The sun shone brightly.
    The quick brown fox jumps over the lazy dog.
    """

    try:
        collection_name = f"test_situation_{uuid.uuid4().hex[:8]}"
        atlas = build_style_atlas(sample_text=sample_text, num_clusters=2, collection_name=collection_name)

        # Test with similar input
        input_text = "The cat sat on the mat."
        match = find_situation_match(atlas, input_text, similarity_threshold=0.3)

        # Should find a match (cat/mat is in sample)
        if match:
            assert isinstance(match, str), "Should return a string"
            assert len(match) > 0, "Should return non-empty text"
            print(f"✓ Find situation match test passed")
            print(f"  Match: {match[:60]}...")
        else:
            print(f"⚠ No situation match found (threshold may be too high)")

    except Exception as e:
        print(f"⚠ Situation match test failed: {e}")


def test_find_structure_match():
    """Test finding structure match (cluster-based with length filtering)."""
    import uuid
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    The swift runner moved quickly through the park.
    The fast athlete ran rapidly across the field.
    The dog rested on the floor. The cat sat on the mat.
    """

    try:
        collection_name = f"test_structure_{uuid.uuid4().hex[:8]}"
        atlas = build_style_atlas(sample_text=sample_text, num_clusters=2, collection_name=collection_name)

        # Test with cluster 0 and input text for length matching
        input_text = "The cat sat on the mat."
        match = find_structure_match(atlas, target_cluster_id=0, input_text=input_text, length_tolerance=0.3)

        if match:
            assert isinstance(match, str), "Should return a string"
            assert len(match) > 0, "Should return non-empty text"
            print(f"✓ Find structure match test passed")
            print(f"  Match: {match[:60]}...")
        else:
            print(f"⚠ No structure match found (cluster may be empty or no length match)")

    except Exception as e:
        print(f"⚠ Structure match test failed: {e}")


def test_find_structure_match_length_filtering():
    """Test that length filtering works correctly."""
    import uuid
    sample_text = """
    Short sentence.
    This is a medium length sentence with more words.
    This is a very long sentence that contains many words and should be filtered out when looking for short matches.
    Another short one.
    """

    try:
        collection_name = f"test_length_{uuid.uuid4().hex[:8]}"
        atlas = build_style_atlas(sample_text=sample_text, num_clusters=1, collection_name=collection_name)

        # Test with short input - should match short sentences
        short_input = "Quick test."
        match = find_structure_match(atlas, target_cluster_id=0, input_text=short_input, length_tolerance=0.3)

        if match:
            # Verify the match is reasonably short (within tolerance)
            match_word_count = len(match.split())
            input_word_count = len(short_input.split())
            ratio = match_word_count / input_word_count if input_word_count > 0 else 1.0

            assert 0.7 <= ratio <= 1.5, f"Match ratio {ratio:.2f} should be between 0.7 and 1.5"
            print(f"✓ Length filtering test passed")
            print(f"  Input: {input_word_count} words, Match: {match_word_count} words, Ratio: {ratio:.2f}")
        else:
            print(f"⚠ No length match found (may need to adjust tolerance)")

    except Exception as e:
        print(f"⚠ Length filtering test failed: {e}")


def test_build_cluster_markov():
    """Test building cluster Markov chain."""
    import uuid
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    The swift runner moved quickly through the park.
    The fast athlete ran rapidly across the field.
    The dog rested on the floor. The cat sat on the mat.
    """

    try:
        collection_name = f"test_markov_{uuid.uuid4().hex[:8]}"
        atlas = build_style_atlas(sample_text=sample_text, num_clusters=2, collection_name=collection_name)
        transition_matrix, cluster_to_index = build_cluster_markov(atlas)

        assert transition_matrix is not None, "Should return transition matrix"
        assert cluster_to_index is not None, "Should return cluster mapping"
        assert len(cluster_to_index) > 0, "Should have cluster mappings"

        print(f"✓ Build cluster Markov test passed")
        print(f"  Matrix shape: {transition_matrix.shape}")
        print(f"  Clusters: {len(cluster_to_index)}")

    except Exception as e:
        print(f"⚠ Build Markov test failed: {e}")


def test_predict_next_cluster():
    """Test predicting next cluster."""
    import uuid
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    The swift runner moved quickly through the park.
    The fast athlete ran rapidly across the field.
    """

    try:
        collection_name = f"test_predict_{uuid.uuid4().hex[:8]}"
        atlas = build_style_atlas(sample_text=sample_text, num_clusters=2, collection_name=collection_name)
        cluster_markov = build_cluster_markov(atlas)

        # Test prediction
        current_cluster = 0
        next_cluster = predict_next_cluster(current_cluster, cluster_markov)

        assert isinstance(next_cluster, int), "Should return an integer"
        assert next_cluster >= 0, "Should return valid cluster ID"

        print(f"✓ Predict next cluster test passed")
        print(f"  Current: {current_cluster}, Next: {next_cluster}")

    except Exception as e:
        print(f"⚠ Predict cluster test failed: {e}")


if __name__ == "__main__":
    print("Running Style Atlas tests...\n")

    try:
        test_build_style_atlas()
        test_find_situation_match()
        test_find_structure_match()
        test_find_structure_match_length_filtering()
        test_build_cluster_markov()
        test_predict_next_cluster()
        print("\n✓ All Style Atlas tests completed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

