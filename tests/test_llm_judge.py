"""Tests for LLM-based judge component."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after setting up path
try:
    from src.critic.judge import LLMJudge
except ImportError:
    # If dependencies are missing, we'll skip tests
    LLMJudge = None


def test_judge_ranks_candidates():
    """Test that judge correctly ranks candidates."""
    if LLMJudge is None:
        print("⚠ Skipping test: LLMJudge not available (missing dependencies)")
        return

    with patch('src.critic.judge.LLMProvider') as mock_provider_class:
        mock_llm = Mock()
        mock_llm.call.return_value = "B"  # Judge selects candidate B
        mock_provider_class.return_value = mock_llm

        judge = LLMJudge(config_path="config.json")
        judge.llm_provider = mock_llm

    source_text = "Every object we touch eventually breaks."
    candidates = [
        "We touch breaks.",  # A - missing "object"
        "Every object we touch breaks.",  # B - complete
        "Objects break when touched."  # C - different phrasing
    ]

    result = judge.rank_candidates(
        source_text=source_text,
        candidates=candidates,
        style_dna="Natural and clear",
        rhetorical_type="OBSERVATION",
        verbose=False
    )

    # Judge selected B (index 1)
    assert result == 1, f"Judge should select candidate B (index 1), got {result}"
    assert mock_llm.call.called, "LLM should be called for ranking"

    print("✓ test_judge_ranks_candidates passed")


def test_judge_parses_various_responses():
    """Test that judge correctly parses various response formats."""
    if LLMJudge is None:
        print("⚠ Skipping test: LLMJudge not available (missing dependencies)")
        return

    with patch('src.critic.judge.LLMProvider') as mock_provider_class:
        mock_llm = Mock()
        mock_provider_class.return_value = mock_llm

        judge = LLMJudge(config_path="config.json")
        judge.llm_provider = mock_llm

    source_text = "Test sentence."
    candidates = ["Candidate A", "Candidate B", "Candidate C"]

    # Test various response formats
    test_cases = [
        ("A", 0),  # Simple letter
        ("Candidate A", 0),  # With word
        ("A is best", 0),  # With explanation
        ("B", 1),  # Different letter
        ("C", 2),  # Third candidate
        ("NONE", -1),  # Rejection
        ("REJECT ALL", -1),  # Rejection variant
    ]

    for response, expected_index in test_cases:
        mock_llm.call.return_value = response
        result = judge.rank_candidates(
            source_text=source_text,
            candidates=candidates,
            verbose=False
        )
        assert result == expected_index, \
            f"Response '{response}' should map to index {expected_index}, got {result}"

    print("✓ test_judge_parses_various_responses passed")


def test_judge_handles_single_candidate():
    """Test that judge handles single candidate without calling LLM."""
    if LLMJudge is None:
        print("⚠ Skipping test: LLMJudge not available (missing dependencies)")
        return

    with patch('src.critic.judge.LLMProvider') as mock_provider_class:
        mock_llm = Mock()
        mock_provider_class.return_value = mock_llm

        judge = LLMJudge(config_path="config.json")
        judge.llm_provider = mock_llm

    source_text = "Test sentence."
    candidates = ["Only candidate"]

    result = judge.rank_candidates(
        source_text=source_text,
        candidates=candidates,
        verbose=False
    )

    # Should return 0 (first candidate) without calling LLM
    assert result == 0, "Single candidate should return index 0"
    assert not mock_llm.call.called, "LLM should not be called for single candidate"

    print("✓ test_judge_handles_single_candidate passed")


def test_judge_handles_empty_candidates():
    """Test that judge handles empty candidate list."""
    if LLMJudge is None:
        print("⚠ Skipping test: LLMJudge not available (missing dependencies)")
        return

    with patch('src.critic.judge.LLMProvider') as mock_provider_class:
        mock_llm = Mock()
        mock_provider_class.return_value = mock_llm

        judge = LLMJudge(config_path="config.json")
        judge.llm_provider = mock_llm

    source_text = "Test sentence."
    candidates = []

    result = judge.rank_candidates(
        source_text=source_text,
        candidates=candidates,
        verbose=False
    )

    # Should return -1 (rejection)
    assert result == -1, "Empty candidates should return -1"
    assert not mock_llm.call.called, "LLM should not be called for empty candidates"

    print("✓ test_judge_handles_empty_candidates passed")


def test_judge_handles_llm_error():
    """Test that judge handles LLM errors gracefully."""
    if LLMJudge is None:
        print("⚠ Skipping test: LLMJudge not available (missing dependencies)")
        return

    with patch('src.critic.judge.LLMProvider') as mock_provider_class:
        mock_llm = Mock()
        mock_llm.call.side_effect = Exception("LLM error")
        mock_provider_class.return_value = mock_llm

        judge = LLMJudge(config_path="config.json")
        judge.llm_provider = mock_llm

    source_text = "Test sentence."
    candidates = ["Candidate A", "Candidate B"]

    result = judge.rank_candidates(
        source_text=source_text,
        candidates=candidates,
        verbose=False
    )

    # Should fallback to first candidate (index 0) on error
    assert result == 0, "Should fallback to first candidate on error"

    print("✓ test_judge_handles_llm_error passed")


def test_judge_handles_malformed_response():
    """Test that judge handles malformed LLM responses."""
    if LLMJudge is None:
        print("⚠ Skipping test: LLMJudge not available (missing dependencies)")
        return

    with patch('src.critic.judge.LLMProvider') as mock_provider_class:
        mock_llm = Mock()
        mock_llm.call.return_value = "This is not a valid response format"  # No letter
        mock_provider_class.return_value = mock_llm

        judge = LLMJudge(config_path="config.json")
        judge.llm_provider = mock_llm

    source_text = "Test sentence."
    candidates = ["Candidate A", "Candidate B"]

    result = judge.rank_candidates(
        source_text=source_text,
        candidates=candidates,
        verbose=False
    )

    # Should fallback to first candidate (index 0) on malformed response
    assert result == 0, "Should fallback to first candidate on malformed response"

    print("✓ test_judge_handles_malformed_response passed")


def test_judge_rejects_empty_candidate():
    """Test that judge rejects empty candidate even if selected."""
    if LLMJudge is None:
        print("⚠ Skipping test: LLMJudge not available (missing dependencies)")
        return

    with patch('src.critic.judge.LLMProvider') as mock_provider_class:
        mock_llm = Mock()
        mock_llm.call.return_value = "B"  # Judge selects B
        mock_provider_class.return_value = mock_llm

        judge = LLMJudge(config_path="config.json")
        judge.llm_provider = mock_llm

    source_text = "Test sentence."
    candidates = [
        "Candidate A",
        "",  # Empty candidate B
        "Candidate C"
    ]

    result = judge.rank_candidates(
        source_text=source_text,
        candidates=candidates,
        verbose=False
    )

    # Should reject empty candidate and return -1
    assert result == -1, "Should reject empty candidate even if selected"

    print("✓ test_judge_rejects_empty_candidate passed")


if __name__ == "__main__":
    test_judge_ranks_candidates()
    test_judge_parses_various_responses()
    test_judge_handles_single_candidate()
    test_judge_handles_empty_candidates()
    test_judge_handles_llm_error()
    test_judge_handles_malformed_response()
    test_judge_rejects_empty_candidate()
    print("\n✓ All LLM Judge tests completed!")

