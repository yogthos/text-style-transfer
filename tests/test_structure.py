"""Test script for structural Markov models."""

import sys
from pathlib import Path

import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.structure import build_pos_markov, build_flow_markov


def test_build_pos_markov():
    """Test that build_pos_markov creates a valid transition matrix."""
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    A beautiful red rose blooms in the garden.
    The cat sat on the mat.
    """

    transition_matrix, tag_to_index = build_pos_markov(sample_text)

    # Assertions
    assert isinstance(transition_matrix, np.ndarray), "Should return numpy array"
    assert isinstance(tag_to_index, dict), "Should return tag to index mapping"

    if transition_matrix.size > 0:
        assert transition_matrix.ndim == 2, "Should be 2D matrix"
        assert len(tag_to_index) == transition_matrix.shape[0], "Matrix rows should match tag count"
        assert len(tag_to_index) == transition_matrix.shape[1], "Matrix should be square"

        # Check that rows sum to approximately 1.0 (allowing for floating point errors)
        row_sums = transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10, err_msg="Rows should sum to 1.0")

        # Check that all values are between 0 and 1
        assert np.all(transition_matrix >= 0), "All probabilities should be >= 0"
        assert np.all(transition_matrix <= 1), "All probabilities should be <= 1"

    print("✓ build_pos_markov test passed")


def test_build_flow_markov():
    """Test that build_flow_markov creates valid transition matrices."""
    sample_text = """
    This is a simple sentence. It has one clause.

    This is a compound sentence, and it has multiple clauses.
    The weather is nice today, but it might rain later.

    Although it was raining, we went outside.
    Because the sun was shining, we felt happy.

    Short fragment. Another one.
    """

    transition_matrix, type_to_index, context_matrices = build_flow_markov(sample_text)

    # Assertions
    assert isinstance(transition_matrix, np.ndarray), "Should return numpy array"
    assert isinstance(type_to_index, dict), "Should return type to index mapping"
    assert isinstance(context_matrices, dict), "Should return context matrices"

    if transition_matrix.size > 0:
        assert transition_matrix.ndim == 2, "Should be 2D matrix"
        assert len(type_to_index) == transition_matrix.shape[0], "Matrix rows should match type count"
        assert len(type_to_index) == transition_matrix.shape[1], "Matrix should be square"

        # Check that rows sum to approximately 1.0
        row_sums = transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10, err_msg="Rows should sum to 1.0")

        # Check that all values are between 0 and 1
        assert np.all(transition_matrix >= 0), "All probabilities should be >= 0"
        assert np.all(transition_matrix <= 1), "All probabilities should be <= 1"

        # Check context matrices
        assert 'paragraph_start' in context_matrices, "Should have paragraph_start matrix"
        assert 'paragraph_end' in context_matrices, "Should have paragraph_end matrix"

        para_start = context_matrices['paragraph_start']
        para_end = context_matrices['paragraph_end']

        if para_start.size > 0:
            assert np.all(para_start >= 0), "Paragraph start probabilities should be >= 0"
            assert np.all(para_start <= 1), "Paragraph start probabilities should be <= 1"
            # Paragraph start should sum to 1.0 (single row)
            if para_start.shape[0] == 1:
                np.testing.assert_allclose(para_start.sum(), 1.0, rtol=1e-10)

        if para_end.size > 0:
            assert np.all(para_end >= 0), "Paragraph end probabilities should be >= 0"
            assert np.all(para_end <= 1), "Paragraph end probabilities should be <= 1"

    print("✓ build_flow_markov test passed")
    print(f"  Sentence types found: {list(type_to_index.keys())}")


def test_build_flow_markov_empty():
    """Test that build_flow_markov handles empty text gracefully."""
    transition_matrix, type_to_index, context_matrices = build_flow_markov("")

    assert isinstance(transition_matrix, np.ndarray), "Should return numpy array even if empty"
    assert isinstance(type_to_index, dict), "Should return dict even if empty"
    assert isinstance(context_matrices, dict), "Should return context matrices dict"
    print("✓ Empty text test passed")


if __name__ == "__main__":
    print("Running structural Markov model tests...\n")

    try:
        test_build_pos_markov()
        test_build_flow_markov()
        test_build_flow_markov_empty()
        print("\n✓ All structure tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

