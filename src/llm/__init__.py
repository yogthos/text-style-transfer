"""LLM provider abstraction layer."""

from .provider import (
    LLMProvider,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    get_provider,
    create_provider_from_config,
    create_writer_provider,
    create_critic_provider,
    register_provider,
)
from .mlx_provider import MLXGenerator

# Import providers to register them
from . import deepseek
from . import ollama

__all__ = [
    # Provider base
    "LLMProvider",
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMResponseError",
    "get_provider",
    "create_provider_from_config",
    "create_writer_provider",
    "create_critic_provider",
    "register_provider",
    # MLX
    "MLXGenerator",
]
