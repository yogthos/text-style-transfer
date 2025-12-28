"""Ollama LLM provider implementation."""

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


@register_provider("ollama")
class OllamaProvider(LLMProvider):
    """LLM provider for local Ollama instance.

    Ollama runs locally and provides an HTTP API for LLM inference.
    """

    def __init__(self, config: LLMProviderConfig, retry_config: Optional[Dict] = None):
        super().__init__(config, retry_config)

        self.base_url = config.base_url.rstrip("/") or "http://localhost:11434"

        # Verify Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.warning(f"Ollama may not be running at {self.base_url}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Cannot connect to Ollama at {self.base_url}")

    @property
    def provider_name(self) -> str:
        return "ollama"

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using character-based heuristic.

        Ollama doesn't provide a tokenizer, so we use a rough estimate.
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4

    def _call_api(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False
    ) -> LLMResponse:
        """Make API call to Ollama."""
        url = f"{self.base_url}/api/chat"

        # Build request payload
        payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
            }
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        if require_json:
            payload["format"] = "json"

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )

            # Handle errors
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg} - {error_data['error']}"
                except Exception:
                    error_msg = f"{error_msg} - {response.text[:200]}"
                raise LLMError(error_msg)

            # Parse response
            data = response.json()

            content = data.get("message", {}).get("content", "")
            if not content:
                raise LLMResponseError("Empty content in Ollama response")

            # Ollama provides token counts in response
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=data.get("model", self.config.model)
            )

        except requests.exceptions.Timeout:
            raise LLMTimeoutError(f"Ollama request timed out after {self.config.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise LLMError(f"Ollama connection error: {e}. Is Ollama running?")
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Ollama request error: {e}")
