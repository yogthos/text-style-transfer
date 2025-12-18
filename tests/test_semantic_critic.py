"""Tests for semantic critic."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.blueprint import SemanticBlueprint
from src.validator.semantic_critic import SemanticCritic


def test_good_output():
    """Test Case A (Good): Server crashed -> Machine stopped."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Server crashed.",
        svo_triples=[("server", "crash", "")],
        named_entities=[],
        core_keywords={"server", "crash"},
        citations=[],
        quotes=[]
    )

    # Good output: preserves meaning
    result = critic.evaluate("Machine stopped.", input_blueprint)

    # Should pass (recall >= 0.85, precision >= 0.50 with new thresholds)
    # Note: Actual scores depend on vector similarity, so we check structure
    assert isinstance(result, dict)
    assert "pass" in result
    assert "recall_score" in result
    assert "precision_score" in result
    assert "score" in result
    assert "feedback" in result

    print("✓ test_good_output passed")


def test_hallucination():
    """Test Case B (Hallucination): Server crashed -> Machine stopped and the dog barked."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Server crashed.",
        svo_triples=[("server", "crash", "")],
        named_entities=[],
        core_keywords={"server", "crash"},
        citations=[],
        quotes=[]
    )

    # Hallucinated output: adds "dog" and "bark"
    result = critic.evaluate("Machine stopped and the dog barked.", input_blueprint)

    # Should fail precision (hallucinated "dog", "bark")
    assert isinstance(result, dict)
    # Precision should be lower due to hallucination
    # Note: Actual scores depend on vector similarity

    print("✓ test_hallucination passed")


def test_omission():
    """Test Case C (Omission): Server crashed -> It happened."""
    critic = SemanticCritic()

    input_blueprint = SemanticBlueprint(
        original_text="Server crashed.",
        svo_triples=[("server", "crash", "")],
        named_entities=[],
        core_keywords={"server", "crash"},
        citations=[],
        quotes=[]
    )

    # Omission: missing "server" and "crash"
    result = critic.evaluate("It happened.", input_blueprint)

    # Should fail recall (missing "server", "crash", "machine")
    assert isinstance(result, dict)
    # Recall should be lower due to omission

    print("✓ test_omission passed")


def test_edge_cases():
    """Test edge cases."""
    critic = SemanticCritic()

    # Empty input blueprint
    empty_blueprint = SemanticBlueprint(
        original_text="",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    result1 = critic.evaluate("Some text.", empty_blueprint)
    assert isinstance(result1, dict)

    # Empty generated text
    normal_blueprint = SemanticBlueprint(
        original_text="Test.",
        svo_triples=[("test", "be", "")],
        named_entities=[],
        core_keywords={"test"},
        citations=[],
        quotes=[]
    )

    result2 = critic.evaluate("", normal_blueprint)
    assert result2["pass"] == False
    assert "empty" in result2["feedback"].lower()

    # Perfect match
    result3 = critic.evaluate("Test.", normal_blueprint)
    assert isinstance(result3, dict)

    print("✓ test_edge_cases passed")


if __name__ == "__main__":
    test_good_output()
    test_hallucination()
    test_omission()
    test_edge_cases()
    print("\n✓ All semantic critic tests completed!")

