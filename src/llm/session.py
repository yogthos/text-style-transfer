"""LLM Session for conversation-based context management."""

from dataclasses import dataclass, field
from typing import List, Optional, Callable

from ..models import Message, MessageRole
from ..utils.logging import get_logger
from .provider import LLMProvider

logger = get_logger(__name__)


@dataclass
class SessionConfig:
    """Configuration for LLM sessions."""
    max_conversation_tokens: int = 8000
    keep_last_n_messages: int = 10  # Keep last N user/assistant pairs
    compress_threshold: float = 0.8  # Compress when at 80% of max tokens


class LLMSession:
    """Manages a multi-turn conversation with context reuse.

    The session maintains conversation history and automatically manages
    the context window by compressing older messages when approaching limits.

    Usage:
        session = LLMSession(provider, system_prompt)
        response1 = session.send("Generate sentence 1: ...")
        response2 = session.send("Generate sentence 2: ...")  # Has full context
        response3 = session.send("Repair sentence 1: ...")    # Still has context
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str,
        config: Optional[SessionConfig] = None
    ):
        """Initialize a new session.

        Args:
            provider: LLM provider to use for calls.
            system_prompt: System prompt (sent with every call).
            config: Session configuration.
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.config = config or SessionConfig()
        self.messages: List[Message] = []
        self._token_count = 0
        self._system_tokens = provider.estimate_tokens(system_prompt)

    def send(self, user_prompt: str, require_json: bool = False) -> str:
        """Send a message and get a response.

        The conversation history is automatically maintained.

        Args:
            user_prompt: The user message to send.
            require_json: If True, request JSON response format.

        Returns:
            The assistant's response text.
        """
        # Add user message
        user_message = Message(role=MessageRole.USER, content=user_prompt)
        self.messages.append(user_message)
        self._token_count += self.provider.estimate_tokens(user_prompt)

        # Check if we need to compress history
        self._manage_context_window()

        # Make the call with full history
        response = self.provider.call_with_history(
            system_prompt=self.system_prompt,
            messages=self.messages,
            require_json=require_json
        )

        # Add assistant response to history
        assistant_message = Message(role=MessageRole.ASSISTANT, content=response)
        self.messages.append(assistant_message)
        self._token_count += self.provider.estimate_tokens(response)

        return response

    def _manage_context_window(self) -> None:
        """Compress history if approaching context limit."""
        threshold = self.config.max_conversation_tokens * self.config.compress_threshold
        total_tokens = self._system_tokens + self._token_count

        if total_tokens > threshold:
            self._compress_history()

    def _compress_history(self) -> None:
        """Compress older messages to stay within context limits.

        Strategy:
        1. Keep the last N message pairs (user/assistant)
        2. Summarize older messages into a single context message
        """
        keep_count = self.config.keep_last_n_messages * 2  # pairs

        if len(self.messages) <= keep_count:
            return  # Nothing to compress

        old_messages = self.messages[:-keep_count]
        recent_messages = self.messages[-keep_count:]

        # Create summary of old messages
        summary_parts = []
        for msg in old_messages:
            if msg.role == MessageRole.ASSISTANT:
                # Keep assistant responses (these are the generated sentences)
                content = msg.content
                if len(content) > 100:
                    content = content[:100] + "..."
                summary_parts.append(f"[Generated: {content}]")

        if summary_parts:
            summary = "Previous context: " + " ".join(summary_parts)
            summary_message = Message(role=MessageRole.USER, content=summary)

            # Replace history with summary + recent messages
            self.messages = [summary_message] + recent_messages

            # Recalculate token count
            self._token_count = sum(
                self.provider.estimate_tokens(m.content)
                for m in self.messages
            )

            logger.info(
                f"Compressed session history: {len(old_messages)} messages -> 1 summary",
                extra_data={
                    "old_message_count": len(old_messages),
                    "new_token_count": self._token_count
                }
            )

    def get_history(self) -> List[Message]:
        """Get the current conversation history."""
        return self.messages.copy()

    def get_token_count(self) -> int:
        """Get the current estimated token count."""
        return self._system_tokens + self._token_count

    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        self._token_count = 0

    def fork(self) -> "LLMSession":
        """Create a copy of this session with the same history.

        Useful for exploring alternative generation paths.
        """
        new_session = LLMSession(
            provider=self.provider,
            system_prompt=self.system_prompt,
            config=self.config
        )
        new_session.messages = self.messages.copy()
        new_session._token_count = self._token_count
        return new_session

    def add_assistant_response(self, response: str) -> None:
        """Manually add an assistant response to history.

        Useful for injecting pre-generated or modified responses.
        """
        message = Message(role=MessageRole.ASSISTANT, content=response)
        self.messages.append(message)
        self._token_count += self.provider.estimate_tokens(response)

    def get_last_response(self) -> Optional[str]:
        """Get the last assistant response, if any."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg.content
        return None

    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self.messages)
