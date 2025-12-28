"""LLM provider abstraction layer."""

from .provider import (
    LLMProvider,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    get_provider,
    create_provider_from_config,
    register_provider,
)
from .session import LLMSession, SessionConfig
from .mlx_provider import MLXGenerator, TextNeutralizer, create_mlx_generator

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
    "register_provider",
    # Session
    "LLMSession",
    "SessionConfig",
    # MLX
    "MLXGenerator",
    "TextNeutralizer",
    "create_mlx_generator",
]
