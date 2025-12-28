"""Mock LLM provider for testing."""

from typing import List, Optional, Dict
from src.models import Message, LLMResponse
from src.config import LLMProviderConfig
from src.llm.provider import LLMProvider, register_provider, LLMError


@register_provider("mock")
class MockLLMProvider(LLMProvider):
    """Mock LLM provider for unit testing.

    Can be configured to return specific responses or simulate errors.
    """

    def __init__(
        self,
        config: LLMProviderConfig = None,
        retry_config: Optional[Dict] = None,
        responses: Optional[List[str]] = None,
        error_on_call: Optional[int] = None,
        error_type: type = LLMError
    ):
        if config is None:
            config = LLMProviderConfig(model="mock-model")
        super().__init__(config, retry_config)

        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.error_on_call = error_on_call
        self.error_type = error_type
        self.last_messages: List[Message] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def estimate_tokens(self, text: str) -> int:
        """Simple token estimation for testing."""
        return len(text.split())

    def _call_api(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False
    ) -> LLMResponse:
        """Return mock response or raise configured error."""
        self.last_messages = messages
        self.call_count += 1

        if self.error_on_call is not None and self.call_count == self.error_on_call:
            raise self.error_type("Simulated error")

        # Cycle through responses
        response_idx = (self.call_count - 1) % len(self.responses)
        content = self.responses[response_idx]

        # Estimate tokens
        input_text = " ".join(m.content for m in messages)
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(content)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="mock-model"
        )


def create_mock_provider(
    responses: Optional[List[str]] = None,
    error_on_call: Optional[int] = None,
    error_type: type = LLMError
) -> MockLLMProvider:
    """Factory function to create a mock provider."""
    return MockLLMProvider(
        responses=responses,
        error_on_call=error_on_call,
        error_type=error_type
    )
