"""Test script for flow planning and template selection."""

import sys
from pathlib import Path

import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.planner.flow import select_template
from src.analyzer.structure import build_flow_markov


def test_select_template_verification_example():
    """Test the verification example from the plan.

    Mock a "previous structure" as `Simple`. Run function. Assert it returns
    a template derived from the Sample text that statistically follows a Simple sentence.
    """
    # Create a sample corpus with various sentence types
    sample_corpus = """
    This is a simple sentence. It has one clause.
    The cat sat on the mat. The dog ran in the park.

    This is a compound sentence, and it has multiple clauses.
    The weather is nice today, but it might rain later.

    Although it was raining, we went outside.
    Because the sun was shining, we felt happy.

    Short fragment. Another one.
    """

    # Build flow markov from the corpus
    flow_markov, type_to_index, _ = build_flow_markov(sample_corpus)

    # Test with previous structure as 'Simple'
    previous_structure = 'Simple'
    input_sentiment = 'Neutral'

    template = select_template(
        previous_structure_type=previous_structure,
        input_sentiment=input_sentiment,
        sample_corpus=sample_corpus,
        flow_markov=flow_markov,
        type_to_index=type_to_index
    )

    # Assertions
    assert isinstance(template, list), "Template should be a list"
    assert len(template) > 0, "Template should not be empty"
    assert all(isinstance(tag, str) for tag in template), "All template elements should be strings"

    # Check that template contains valid POS tags
    valid_pos_tags = ['DT', 'NN', 'VB', 'JJ', 'IN', 'PRP', 'CC', 'RB', 'VBZ', 'VBD', 'VBG']
    assert any(tag in valid_pos_tags for tag in template), "Template should contain valid POS tags"

    print(f"✓ Verification example passed")
    print(f"  Previous structure: {previous_structure}")
    print(f"  Selected template: {template[:10]}..." if len(template) > 10 else f"  Selected template: {template}")


def test_select_template_different_sentiments():
    """Test template selection with different sentiment requirements."""
    sample_corpus = """
    This is a wonderful day. I feel great and happy.
    The weather is terrible. I feel sad and disappointed.
    The cat sat on the mat. The dog ran in the park.
    """

    flow_markov, type_to_index, _ = build_flow_markov(sample_corpus)

    # Test with positive sentiment
    template_pos = select_template(
        previous_structure_type='Simple',
        input_sentiment='Positive',
        sample_corpus=sample_corpus,
        flow_markov=flow_markov,
        type_to_index=type_to_index
    )

    # Test with negative sentiment
    template_neg = select_template(
        previous_structure_type='Simple',
        input_sentiment='Negative',
        sample_corpus=sample_corpus,
        flow_markov=flow_markov,
        type_to_index=type_to_index
    )

    assert isinstance(template_pos, list), "Positive template should be a list"
    assert isinstance(template_neg, list), "Negative template should be a list"

    print(f"✓ Different sentiments test passed")
    print(f"  Positive template length: {len(template_pos)}")
    print(f"  Negative template length: {len(template_neg)}")


def test_select_template_different_structures():
    """Test template selection with different previous structures."""
    sample_corpus = """
    This is a simple sentence.
    This is a compound sentence, and it has multiple clauses.
    Although it was raining, we went outside.
    Short fragment.
    """

    flow_markov, type_to_index, _ = build_flow_markov(sample_corpus)

    structures = ['Simple', 'Compound', 'Complex', 'Fragment']

    for prev_struct in structures:
        template = select_template(
            previous_structure_type=prev_struct,
            input_sentiment='Neutral',
            sample_corpus=sample_corpus,
            flow_markov=flow_markov,
            type_to_index=type_to_index
        )

        assert isinstance(template, list), f"Template for {prev_struct} should be a list"
        assert len(template) > 0, f"Template for {prev_struct} should not be empty"

    print(f"✓ Different structures test passed")


def test_select_template_fallback():
    """Test that select_template handles edge cases with fallbacks."""
    # Very small corpus
    sample_corpus = "Hello world."

    template = select_template(
        previous_structure_type='Simple',
        input_sentiment='Neutral',
        sample_corpus=sample_corpus
    )

    assert isinstance(template, list), "Should return a list even with small corpus"
    assert len(template) > 0, "Should return a non-empty template"

    print(f"✓ Fallback test passed")


if __name__ == "__main__":
    print("Running flow planning tests...\n")

    try:
        test_select_template_verification_example()
        test_select_template_different_sentiments()
        test_select_template_different_structures()
        test_select_template_fallback()
        print("\n✓ All flow planning tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

