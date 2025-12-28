"""Abstract base class for LLM providers."""

import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type

from ..models import Message, MessageRole, LLMResponse
from ..config import LLMProviderConfig, LLMConfig
from ..utils.logging import get_logger, log_llm_call

logger = get_logger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is hit."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when request times out."""
    pass


class LLMResponseError(LLMError):
    """Raised when response is malformed or invalid."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations must provide:
    - _call_api: Make the actual API call
    - estimate_tokens: Estimate token count for text
    """

    def __init__(self, config: LLMProviderConfig, retry_config: Optional[Dict] = None):
        """Initialize the provider.

        Args:
            config: Provider-specific configuration.
            retry_config: Retry settings (max_retries, base_delay, max_delay).
        """
        self.config = config
        self.retry_config = retry_config or {
            "max_retries": 5,
            "base_delay": 2.0,
            "max_delay": 60.0
        }
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    def _call_api(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False
    ) -> LLMResponse:
        """Make the actual API call.

        Args:
            messages: List of messages in the conversation.
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Maximum tokens in response (uses config default if None).
            require_json: If True, request JSON response format.

        Returns:
            LLMResponse with the generated content.

        Raises:
            LLMRateLimitError: If rate limit is hit.
            LLMTimeoutError: If request times out.
            LLMResponseError: If response is invalid.
            LLMError: For other errors.
        """
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        pass

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False
    ) -> str:
        """Make a single LLM call with system and user prompts.

        Args:
            system_prompt: System prompt setting context.
            user_prompt: User prompt with the request.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            require_json: If True, request JSON response format.

        Returns:
            Generated text content.
        """
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt)
        ]
        response = self._call_with_retry(messages, temperature, max_tokens, require_json)
        return response.content

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make an LLM call expecting JSON response.

        Args:
            system_prompt: System prompt setting context.
            user_prompt: User prompt with the request.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            LLMResponseError: If response is not valid JSON.
        """
        content = self.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            require_json=True
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                json_match = content.split("```json")[1].split("```")[0].strip()
                try:
                    return json.loads(json_match)
                except json.JSONDecodeError:
                    pass
            elif "```" in content:
                json_match = content.split("```")[1].split("```")[0].strip()
                try:
                    return json.loads(json_match)
                except json.JSONDecodeError:
                    pass

            raise LLMResponseError(f"Failed to parse JSON response: {e}\nContent: {content[:500]}")

    def call_with_history(
        self,
        system_prompt: str,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False
    ) -> str:
        """Make an LLM call with conversation history.

        Args:
            system_prompt: System prompt setting context.
            messages: List of previous messages (user/assistant pairs).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            require_json: If True, request JSON response format.

        Returns:
            Generated text content.
        """
        full_messages = [Message(role=MessageRole.SYSTEM, content=system_prompt)]
        full_messages.extend(messages)

        response = self._call_with_retry(full_messages, temperature, max_tokens, require_json)
        return response.content

    def _call_with_retry(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False
    ) -> LLMResponse:
        """Make API call with retry logic.

        Uses exponential backoff for rate limits and transient errors.
        """
        max_retries = self.retry_config["max_retries"]
        base_delay = self.retry_config["base_delay"]
        max_delay = self.retry_config["max_delay"]

        last_error = None

        for attempt in range(max_retries):
            start_time = time.time()
            try:
                response = self._call_api(messages, temperature, max_tokens, require_json)

                # Track usage
                self._total_input_tokens += response.input_tokens
                self._total_output_tokens += response.output_tokens
                self._total_calls += 1

                duration_ms = int((time.time() - start_time) * 1000)
                log_llm_call(
                    logger=logger,
                    provider=self.provider_name,
                    model=response.model or self.config.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    duration_ms=duration_ms,
                    success=True
                )

                return response

            except LLMRateLimitError as e:
                last_error = e
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})",
                    extra_data={"provider": self.provider_name, "delay": delay}
                )
                time.sleep(delay)

            except LLMTimeoutError as e:
                last_error = e
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Request timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})",
                    extra_data={"provider": self.provider_name, "delay": delay}
                )
                time.sleep(delay)

            except LLMError as e:
                # Don't retry on other LLM errors (malformed response, etc.)
                duration_ms = int((time.time() - start_time) * 1000)
                log_llm_call(
                    logger=logger,
                    provider=self.provider_name,
                    model=self.config.model,
                    input_tokens=0,
                    output_tokens=0,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e)
                )
                raise

        # All retries exhausted
        duration_ms = int((time.time() - start_time) * 1000)
        log_llm_call(
            logger=logger,
            provider=self.provider_name,
            model=self.config.model,
            input_tokens=0,
            output_tokens=0,
            duration_ms=duration_ms,
            success=False,
            error=f"Max retries ({max_retries}) exhausted"
        )
        raise last_error or LLMError(f"Max retries ({max_retries}) exhausted")

    def get_usage_stats(self) -> Dict[str, int]:
        """Get cumulative usage statistics."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_calls": self._total_calls
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0


# Provider registry for factory function
_provider_registry: Dict[str, Type[LLMProvider]] = {}


def register_provider(name: str):
    """Decorator to register an LLM provider class."""
    def decorator(cls: Type[LLMProvider]):
        _provider_registry[name] = cls
        return cls
    return decorator


def get_provider(name: str, config: LLMProviderConfig, retry_config: Optional[Dict] = None) -> LLMProvider:
    """Get an LLM provider instance by name.

    Args:
        name: Provider name (e.g., "deepseek", "ollama").
        config: Provider configuration.
        retry_config: Retry settings.

    Returns:
        Initialized LLM provider.

    Raises:
        ValueError: If provider name is unknown.
    """
    if name not in _provider_registry:
        available = ", ".join(_provider_registry.keys())
        raise ValueError(f"Unknown LLM provider: {name}. Available: {available}")

    return _provider_registry[name](config, retry_config)


def create_provider_from_config(llm_config: LLMConfig) -> LLMProvider:
    """Create an LLM provider from configuration.

    Args:
        llm_config: LLM configuration section.

    Returns:
        Initialized LLM provider for the configured default provider.
    """
    provider_name = llm_config.provider
    provider_config = llm_config.get_provider_config(provider_name)
    retry_config = {
        "max_retries": llm_config.max_retries,
        "base_delay": llm_config.base_delay,
        "max_delay": llm_config.max_delay
    }
    return get_provider(provider_name, provider_config, retry_config)
