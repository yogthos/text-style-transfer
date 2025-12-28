"""DeepSeek LLM provider implementation."""

import requests
from typing import List, Optional, Dict

from ..models import Message, LLMResponse
from ..config import LLMProviderConfig
from ..utils.logging import get_logger
from .provider import (
    LLMProvider,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    register_provider
)

logger = get_logger(__name__)


@register_provider("deepseek")
class DeepSeekProvider(LLMProvider):
    """LLM provider for DeepSeek API.

    DeepSeek uses an OpenAI-compatible API format.
    """

    def __init__(self, config: LLMProviderConfig, retry_config: Optional[Dict] = None):
        super().__init__(config, retry_config)

        if not config.api_key:
            raise ValueError("DeepSeek API key is required")

        self.base_url = config.base_url.rstrip("/") or "https://api.deepseek.com"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

        # Try to use tiktoken for accurate token counting
        self._tokenizer = None
        try:
            import tiktoken
            # DeepSeek uses a similar tokenizer to GPT models
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not available, using character-based token estimation")

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken or fallback to character count."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: rough estimate of 4 characters per token
        return len(text) // 4

    def _call_api(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False,
        logit_bias: Optional[Dict[str, float]] = None
    ) -> LLMResponse:
        """Make API call to DeepSeek.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            require_json: Request JSON output format.
            logit_bias: Optional dict mapping token IDs (as strings) to bias values.
                       Values from -100 (ban) to 100 (force).
        """
        url = f"{self.base_url}/v1/chat/completions"

        # Build request payload
        payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }

        if require_json:
            payload["response_format"] = {"type": "json_object"}

        if logit_bias:
            payload["logit_bias"] = logit_bias

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout
            )

            # Handle rate limiting
            if response.status_code == 429:
                raise LLMRateLimitError("DeepSeek rate limit exceeded")

            # Handle other HTTP errors
            if response.status_code != 200:
                error_msg = f"DeepSeek API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg} - {error_data['error'].get('message', '')}"
                except Exception:
                    error_msg = f"{error_msg} - {response.text[:200]}"
                raise LLMError(error_msg)

            # Parse response
            data = response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise LLMResponseError("No choices in DeepSeek response")

            content = data["choices"][0].get("message", {}).get("content", "")
            if not content:
                raise LLMResponseError("Empty content in DeepSeek response")

            # Extract usage info
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                model=data.get("model", self.config.model)
            )

        except requests.exceptions.Timeout:
            raise LLMTimeoutError(f"DeepSeek request timed out after {self.config.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"DeepSeek connection error: {e}")
        except requests.exceptions.RequestException as e:
            raise LLMError(f"DeepSeek request error: {e}")

    def call_with_logit_bias(
        self,
        system_prompt: str,
        user_prompt: str,
        logit_bias: Dict[str, float],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Make an LLM call with vocabulary control via logit_bias.

        Args:
            system_prompt: System prompt setting context.
            user_prompt: User prompt with the request.
            logit_bias: Dict mapping token IDs to bias values (-100 to 100).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Generated text content.
        """
        from ..models import MessageRole

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt)
        ]

        response = self._call_api(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logit_bias=logit_bias
        )
        return response.content
