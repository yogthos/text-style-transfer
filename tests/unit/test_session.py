"""Unit tests for LLM session management."""

import pytest
from src.models import Message, MessageRole
from src.llm.session import LLMSession, SessionConfig
from tests.fixtures.mock_llm import create_mock_provider


class TestLLMSession:
    """Test LLMSession conversation management."""

    def test_basic_send(self):
        """Test basic send and response."""
        provider = create_mock_provider(responses=["Response 1"])
        session = LLMSession(provider, "System prompt")

        response = session.send("User message")
        assert response == "Response 1"

    def test_maintains_history(self):
        """Test that conversation history is maintained."""
        provider = create_mock_provider(responses=["R1", "R2", "R3"])
        session = LLMSession(provider, "System prompt")

        session.send("Message 1")
        session.send("Message 2")
        session.send("Message 3")

        history = session.get_history()
        assert len(history) == 6  # 3 user + 3 assistant

        # Check alternating user/assistant
        assert history[0].role == MessageRole.USER
        assert history[1].role == MessageRole.ASSISTANT
        assert history[2].role == MessageRole.USER
        assert history[3].role == MessageRole.ASSISTANT

    def test_history_passed_to_provider(self):
        """Test that full history is passed to provider."""
        provider = create_mock_provider(responses=["R1", "R2"])
        session = LLMSession(provider, "System prompt")

        session.send("First message")
        session.send("Second message")

        # Check last messages include history
        last_msgs = provider.last_messages
        # Should have: system + user1 + assistant1 + user2
        assert len(last_msgs) == 4

    def test_token_counting(self):
        """Test that token count is tracked."""
        provider = create_mock_provider(responses=["Short response"])
        session = LLMSession(provider, "System prompt")

        initial_tokens = session.get_token_count()
        session.send("User message here")
        after_tokens = session.get_token_count()

        assert after_tokens > initial_tokens

    def test_clear_history(self):
        """Test clearing conversation history."""
        provider = create_mock_provider()
        session = LLMSession(provider, "System prompt")

        session.send("Message 1")
        session.send("Message 2")
        assert len(session) == 4

        session.clear()
        assert len(session) == 0
        assert session.get_history() == []

    def test_fork_session(self):
        """Test forking a session creates independent copy."""
        provider = create_mock_provider(responses=["R1", "R2", "R3"])
        session = LLMSession(provider, "System prompt")

        session.send("Original message")

        forked = session.fork()
        forked.send("Forked message")

        # Original should have 2 messages, forked should have 4
        assert len(session) == 2
        assert len(forked) == 4

    def test_add_assistant_response(self):
        """Test manually adding assistant response."""
        provider = create_mock_provider()
        session = LLMSession(provider, "System prompt")

        session.add_assistant_response("Injected response")

        history = session.get_history()
        assert len(history) == 1
        assert history[0].role == MessageRole.ASSISTANT
        assert history[0].content == "Injected response"

    def test_get_last_response(self):
        """Test getting last assistant response."""
        provider = create_mock_provider(responses=["First", "Second"])
        session = LLMSession(provider, "System prompt")

        assert session.get_last_response() is None

        session.send("Message 1")
        assert session.get_last_response() == "First"

        session.send("Message 2")
        assert session.get_last_response() == "Second"

    def test_len(self):
        """Test __len__ returns message count."""
        provider = create_mock_provider(responses=["R1", "R2"])
        session = LLMSession(provider, "System prompt")

        assert len(session) == 0
        session.send("M1")
        assert len(session) == 2  # user + assistant
        session.send("M2")
        assert len(session) == 4


class TestSessionCompression:
    """Test session history compression."""

    def test_compression_triggers_at_threshold(self):
        """Test that compression triggers when approaching token limit."""
        provider = create_mock_provider(responses=["Response"] * 20)

        # Low token limit to trigger compression
        config = SessionConfig(
            max_conversation_tokens=50,
            keep_last_n_messages=2,
            compress_threshold=0.5
        )
        session = LLMSession(provider, "Sys", config)

        # Send enough messages to trigger compression
        for i in range(10):
            session.send(f"Message {i} with some content")

        # History should be compressed
        # Exact count depends on compression logic
        assert len(session) < 20  # Less than 10 pairs

    def test_keeps_recent_messages(self):
        """Test that recent messages are preserved during compression."""
        provider = create_mock_provider(responses=["R"] * 10)

        config = SessionConfig(
            max_conversation_tokens=100,
            keep_last_n_messages=2,
            compress_threshold=0.3
        )
        session = LLMSession(provider, "Sys", config)

        # Send messages
        for i in range(5):
            session.send(f"Message {i}")

        # Should have at least the last 2 pairs (4 messages)
        # Plus potentially a summary message
        history = session.get_history()
        assert len(history) >= 4


class TestSessionConfig:
    """Test session configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SessionConfig()
        assert config.max_conversation_tokens == 8000
        assert config.keep_last_n_messages == 10
        assert config.compress_threshold == 0.8

    def test_custom_config(self):
        """Test custom configuration."""
        config = SessionConfig(
            max_conversation_tokens=4000,
            keep_last_n_messages=5,
            compress_threshold=0.7
        )
        provider = create_mock_provider()
        session = LLMSession(provider, "Sys", config)

        assert session.config.max_conversation_tokens == 4000
        assert session.config.keep_last_n_messages == 5
