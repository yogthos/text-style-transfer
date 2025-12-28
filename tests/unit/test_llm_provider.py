"""Unit tests for LLM provider abstraction."""

import pytest
import json
from unittest.mock import patch, MagicMock

from src.models import Message, MessageRole
from src.config import LLMProviderConfig
from src.llm.provider import (
    LLMProvider,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    get_provider,
)
from tests.fixtures.mock_llm import MockLLMProvider, create_mock_provider


class TestMockProvider:
    """Test the mock provider itself."""

    def test_basic_call(self):
        """Test basic call returns mock response."""
        provider = create_mock_provider(responses=["Hello, world!"])
        response = provider.call("System", "User prompt")
        assert response == "Hello, world!"

    def test_call_count(self):
        """Test that call count is tracked."""
        provider = create_mock_provider()
        assert provider.call_count == 0
        provider.call("System", "First")
        assert provider.call_count == 1
        provider.call("System", "Second")
        assert provider.call_count == 2

    def test_cycles_through_responses(self):
        """Test that responses cycle."""
        provider = create_mock_provider(responses=["First", "Second", "Third"])
        assert provider.call("S", "U") == "First"
        assert provider.call("S", "U") == "Second"
        assert provider.call("S", "U") == "Third"
        assert provider.call("S", "U") == "First"  # Cycles back

    def test_tracks_last_messages(self):
        """Test that last messages are tracked."""
        provider = create_mock_provider()
        provider.call("System prompt", "User prompt")
        assert len(provider.last_messages) == 2
        assert provider.last_messages[0].role == MessageRole.SYSTEM
        assert provider.last_messages[1].role == MessageRole.USER


class TestProviderRetryLogic:
    """Test retry logic in LLM provider."""

    def test_no_retry_on_success(self):
        """Test that successful calls don't retry."""
        provider = create_mock_provider(responses=["Success"])
        response = provider.call("S", "U")
        assert response == "Success"
        assert provider.call_count == 1

    def test_retry_on_rate_limit(self):
        """Test that rate limit errors trigger retry."""
        # Create a provider that fails on first call with rate limit
        provider = MockLLMProvider(
            error_on_call=1,
            error_type=LLMRateLimitError,
            retry_config={"max_retries": 3, "base_delay": 0.01, "max_delay": 0.1}
        )

        # Should succeed on second call
        with patch.object(provider, '_call_api') as mock_call:
            from src.models import LLMResponse
            mock_call.side_effect = [
                LLMRateLimitError("Rate limited"),
                LLMResponse(content="Success", input_tokens=10, output_tokens=5)
            ]
            response = provider.call("S", "U")
            assert response == "Success"
            assert mock_call.call_count == 2

    def test_max_retries_exhausted(self):
        """Test that max retries results in error."""
        provider = MockLLMProvider(
            retry_config={"max_retries": 2, "base_delay": 0.01, "max_delay": 0.1}
        )

        with patch.object(provider, '_call_api') as mock_call:
            mock_call.side_effect = LLMRateLimitError("Rate limited")
            with pytest.raises(LLMRateLimitError):
                provider.call("S", "U")
            assert mock_call.call_count == 2


class TestProviderTokenEstimation:
    """Test token estimation."""

    def test_mock_provider_estimation(self):
        """Test mock provider token estimation (word-based)."""
        provider = create_mock_provider()
        assert provider.estimate_tokens("Hello world") == 2
        assert provider.estimate_tokens("One two three four five") == 5


class TestProviderJSONResponse:
    """Test JSON response handling."""

    def test_valid_json_response(self):
        """Test parsing valid JSON response."""
        json_response = '{"key": "value", "number": 42}'
        provider = create_mock_provider(responses=[json_response])
        result = provider.call_json("S", "U")
        assert result == {"key": "value", "number": 42}

    def test_json_in_markdown_block(self):
        """Test extracting JSON from markdown code block."""
        markdown_response = '```json\n{"key": "value"}\n```'
        provider = create_mock_provider(responses=[markdown_response])
        result = provider.call_json("S", "U")
        assert result == {"key": "value"}

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises LLMResponseError."""
        provider = create_mock_provider(responses=["Not valid JSON"])
        with pytest.raises(LLMResponseError):
            provider.call_json("S", "U")


class TestProviderUsageStats:
    """Test usage statistics tracking."""

    def test_tracks_usage(self):
        """Test that usage stats are accumulated."""
        provider = create_mock_provider(responses=["Response one", "Response two"])

        provider.call("System", "First request")
        provider.call("System", "Second request")

        stats = provider.get_usage_stats()
        assert stats["total_calls"] == 2
        assert stats["total_input_tokens"] > 0
        assert stats["total_output_tokens"] > 0

    def test_reset_usage(self):
        """Test that usage stats can be reset."""
        provider = create_mock_provider()
        provider.call("S", "U")

        provider.reset_usage_stats()
        stats = provider.get_usage_stats()
        assert stats["total_calls"] == 0
        assert stats["total_tokens"] == 0


class TestProviderFactory:
    """Test provider factory function."""

    def test_get_mock_provider(self):
        """Test getting mock provider by name."""
        config = LLMProviderConfig(model="test-model")
        provider = get_provider("mock", config)
        assert isinstance(provider, MockLLMProvider)

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider name raises ValueError."""
        config = LLMProviderConfig()
        with pytest.raises(ValueError) as exc_info:
            get_provider("unknown_provider", config)
        assert "Unknown LLM provider" in str(exc_info.value)


class TestCallWithHistory:
    """Test conversation history support."""

    def test_call_with_history(self):
        """Test calling with conversation history."""
        provider = create_mock_provider(responses=["Response"])

        history = [
            Message(role=MessageRole.USER, content="First user message"),
            Message(role=MessageRole.ASSISTANT, content="First response"),
            Message(role=MessageRole.USER, content="Second user message"),
        ]

        response = provider.call_with_history(
            system_prompt="System prompt",
            messages=history
        )

        assert response == "Response"
        # Check that all messages were passed (system + history)
        assert len(provider.last_messages) == 4
        assert provider.last_messages[0].role == MessageRole.SYSTEM
