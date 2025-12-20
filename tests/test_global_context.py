"""Unit tests for Global Context Awareness functionality.

Tests the GlobalContextAnalyzer and its integration with the pipeline,
translator, and semantic critic.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer.global_context import GlobalContextAnalyzer
from src.validator.semantic_critic import SemanticCritic
from src.ingestion.blueprint import SemanticBlueprint


class MockLLMProvider:
    """Mock LLM provider for testing GlobalContextAnalyzer."""

    def __init__(self, mock_response=None):
        self.call_count = 0
        self.call_history = []
        self.mock_response = mock_response or json.dumps({
            "thesis": "This document explores the relationship between technology and society.",
            "intent": "informing",
            "keywords": ["technology", "society", "relationship", "impact", "change"]
        })

    def call(self, system_prompt, user_prompt, model_type="editor", require_json=False,
             temperature=0.3, max_tokens=500, timeout=None, top_p=None):
        self.call_count += 1
        self.call_history.append({
            "system_prompt": system_prompt[:100] if len(system_prompt) > 100 else system_prompt,
            "user_prompt": user_prompt[:200] if len(user_prompt) > 200 else user_prompt,
            "model_type": model_type,
            "require_json": require_json
        })
        return self.mock_response


def test_global_context_analyzer_success():
    """Test GlobalContextAnalyzer with successful LLM response."""
    mock_response = json.dumps({
        "thesis": "This document explores the relationship between technology and society.",
        "intent": "informing",
        "keywords": ["technology", "society", "relationship", "impact", "change"]
    })

    with patch('src.analyzer.global_context.LLMProvider', return_value=MockLLMProvider(mock_response)):
        analyzer = GlobalContextAnalyzer(config_path="config.json")
        result = analyzer.analyze_document("This is a test document about technology and society.", verbose=False)

        assert isinstance(result, dict)
        assert "thesis" in result
        assert "intent" in result
        assert "keywords" in result
        assert result["thesis"] == "This document explores the relationship between technology and society."
        assert result["intent"] == "informing"
        assert len(result["keywords"]) == 5
        assert "technology" in result["keywords"]

    print("✓ test_global_context_analyzer_success passed")


def test_global_context_analyzer_empty_input():
    """Test GlobalContextAnalyzer with empty input."""
    with patch('src.analyzer.global_context.LLMProvider', return_value=MockLLMProvider()):
        analyzer = GlobalContextAnalyzer(config_path="config.json")
        result = analyzer.analyze_document("", verbose=False)

        assert isinstance(result, dict)
        assert result["thesis"] == ""
        assert result["intent"] == "informing"
        assert result["keywords"] == []

    print("✓ test_global_context_analyzer_empty_input passed")


def test_global_context_analyzer_long_input():
    """Test GlobalContextAnalyzer with very long input (truncation)."""
    # Create a very long document (>50k chars)
    long_text = "Introduction. " * 5000 + "Main content. " * 5000 + "Conclusion. " * 5000

    mock_response = json.dumps({
        "thesis": "This is a long document about various topics.",
        "intent": "informing",
        "keywords": ["topic1", "topic2", "topic3", "topic4", "topic5"]
    })

    with patch('src.analyzer.global_context.LLMProvider', return_value=MockLLMProvider(mock_response)):
        analyzer = GlobalContextAnalyzer(config_path="config.json")
        result = analyzer.analyze_document(long_text, verbose=False)

        # Should still work (truncation happens internally)
        assert isinstance(result, dict)
        assert "thesis" in result

    print("✓ test_global_context_analyzer_long_input passed")


def test_global_context_analyzer_llm_failure():
    """Test GlobalContextAnalyzer when LLM call fails."""
    mock_llm = MockLLMProvider()
    mock_llm.call = Mock(side_effect=Exception("LLM API error"))

    with patch('src.analyzer.global_context.LLMProvider', return_value=mock_llm):
        analyzer = GlobalContextAnalyzer(config_path="config.json")
        result = analyzer.analyze_document("Test document.", verbose=False)

        # Should return default context on failure
        assert isinstance(result, dict)
        assert result["thesis"] == ""
        assert result["intent"] == "informing"
        assert result["keywords"] == []

    print("✓ test_global_context_analyzer_llm_failure passed")


def test_global_context_analyzer_invalid_json():
    """Test GlobalContextAnalyzer with invalid JSON response."""
    mock_response = "This is not valid JSON"

    with patch('src.analyzer.global_context.LLMProvider', return_value=MockLLMProvider(mock_response)):
        analyzer = GlobalContextAnalyzer(config_path="config.json")
        result = analyzer.analyze_document("Test document.", verbose=False)

        # Should return default context on JSON parse failure
        assert isinstance(result, dict)
        assert result["thesis"] == ""
        assert result["intent"] == "informing"
        assert result["keywords"] == []

    print("✓ test_global_context_analyzer_invalid_json passed")


def test_global_context_analyzer_invalid_intent():
    """Test GlobalContextAnalyzer with invalid intent value."""
    mock_response = json.dumps({
        "thesis": "Test thesis.",
        "intent": "invalid_intent",
        "keywords": ["keyword1", "keyword2"]
    })

    with patch('src.analyzer.global_context.LLMProvider', return_value=MockLLMProvider(mock_response)):
        analyzer = GlobalContextAnalyzer(config_path="config.json")
        result = analyzer.analyze_document("Test document.", verbose=False)

        # Should normalize invalid intent to "informing"
        assert result["intent"] == "informing"

    print("✓ test_global_context_analyzer_invalid_intent passed")


def test_semantic_critic_with_global_context():
    """Test SemanticCritic.evaluate() accepts and uses global_context."""
    critic = SemanticCritic(config_path="config.json")

    input_blueprint = SemanticBlueprint(
        original_text="The economy crashed.",
        svo_triples=[("economy", "crash", "")],
        named_entities=[],
        core_keywords={"economy", "crash"},
        citations=[],
        quotes=[]
    )

    global_context = {
        "thesis": "This document is about economic systems and their failures.",
        "intent": "informing",
        "keywords": ["economy", "system", "failure", "market", "crisis"]
    }

    # Test that evaluate accepts global_context parameter
    result = critic.evaluate(
        "The economic system experienced a failure.",
        input_blueprint,
        global_context=global_context
    )

    assert isinstance(result, dict)
    assert "pass" in result
    assert "recall_score" in result
    assert "score" in result

    print("✓ test_semantic_critic_with_global_context passed")


def test_semantic_critic_verification_with_context():
    """Test that _verify_meaning_with_llm uses global_context in prompt."""
    critic = SemanticCritic(config_path="config.json")

    global_context = {
        "thesis": "This document is about economic systems.",
        "intent": "informing",
        "keywords": ["economy", "system"]
    }

    # Mock the LLM provider to capture the prompt
    mock_llm = Mock()
    mock_llm.call.return_value = json.dumps({
        "meaning_preserved": True,
        "confidence": 0.9,
        "explanation": "Meaning preserved"
    })

    with patch.object(critic, 'llm_provider', mock_llm):
        critic._verify_meaning_with_llm(
            "Original text.",
            "Generated text.",
            global_context=global_context
        )

        # Verify LLM was called
        assert mock_llm.call.called

        # Verify context was included in prompt
        call_args = mock_llm.call.call_args
        user_prompt = call_args[1]['user_prompt'] if 'user_prompt' in call_args[1] else call_args[0][1]
        assert "CONTEXT" in user_prompt or "economic systems" in user_prompt

    print("✓ test_semantic_critic_verification_with_context passed")


def test_semantic_critic_logic_verification_with_context():
    """Test that _verify_logic uses global_context in prompt."""
    critic = SemanticCritic(config_path="config.json")

    global_context = {
        "thesis": "This document is about economic systems.",
        "intent": "informing",
        "keywords": ["economy", "system"]
    }

    # Mock the LLM provider to capture the prompt
    mock_llm = Mock()
    mock_llm.call.return_value = json.dumps({
        "logic_fail": False,
        "reason": "Logic preserved"
    })

    with patch.object(critic, 'llm_provider', mock_llm):
        critic._verify_logic(
            "Original text.",
            "Generated text.",
            "DECLARATIVE",
            global_context=global_context
        )

        # Verify LLM was called
        assert mock_llm.call.called

        # Verify context was included in prompt
        call_args = mock_llm.call.call_args
        user_prompt = call_args[1]['user_prompt'] if 'user_prompt' in call_args[1] else call_args[0][1]
        assert "CONTEXT" in user_prompt or "economic systems" in user_prompt

    print("✓ test_semantic_critic_logic_verification_with_context passed")


def test_global_context_keywords_limit():
    """Test that GlobalContextAnalyzer limits keywords to 5."""
    mock_response = json.dumps({
        "thesis": "Test thesis.",
        "intent": "informing",
        "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5", "kw6", "kw7"]  # 7 keywords
    })

    with patch('src.analyzer.global_context.LLMProvider', return_value=MockLLMProvider(mock_response)):
        analyzer = GlobalContextAnalyzer(config_path="config.json")
        result = analyzer.analyze_document("Test document.", verbose=False)

        # Should limit to 5 keywords
        assert len(result["keywords"]) == 5
        assert result["keywords"] == ["kw1", "kw2", "kw3", "kw4", "kw5"]

    print("✓ test_global_context_keywords_limit passed")


def run_all_tests():
    """Run all global context tests."""
    print("\n" + "="*60)
    print("Running Global Context Awareness Tests")
    print("="*60 + "\n")

    tests = [
        test_global_context_analyzer_success,
        test_global_context_analyzer_empty_input,
        test_global_context_analyzer_long_input,
        test_global_context_analyzer_llm_failure,
        test_global_context_analyzer_invalid_json,
        test_global_context_analyzer_invalid_intent,
        test_global_context_keywords_limit,
        test_semantic_critic_with_global_context,
        test_semantic_critic_verification_with_context,
        test_semantic_critic_logic_verification_with_context,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

