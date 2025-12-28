"""DeepSeek-specific vocabulary controller.

Uses logit_bias parameter supported by DeepSeek's OpenAI-compatible API.
"""

from typing import List, Optional
import re

from .controller import VocabularyController
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DeepSeekVocabularyController(VocabularyController):
    """Vocabulary controller for DeepSeek API.

    Uses tiktoken (cl100k_base) for tokenization, which is compatible
    with DeepSeek's tokenizer.
    """

    def __init__(self, max_biases: int = 300):
        """Initialize DeepSeek vocabulary controller.

        Args:
            max_biases: Maximum token biases to include.
                       DeepSeek doesn't document a hard limit, but
                       keeping it reasonable avoids performance issues.
        """
        super().__init__(max_biases)
        self._tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load tiktoken tokenizer."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info("Loaded cl100k_base tokenizer for DeepSeek vocabulary control")
        except ImportError:
            logger.warning(
                "tiktoken not available. Vocabulary control disabled. "
                "Install with: pip install tiktoken"
            )

    def get_provider_name(self) -> str:
        return "deepseek"

    def get_token_ids(self, word: str) -> List[int]:
        """Get token IDs for a word.

        Returns token IDs for both:
        - Word with leading space (common in context)
        - Word without leading space (at sentence start)
        - Capitalized variants

        Args:
            word: Word to tokenize.

        Returns:
            List of unique token IDs.
        """
        if self._tokenizer is None:
            return []

        token_ids = set()

        # Try various word forms
        variants = [
            word.lower(),           # lowercase
            f" {word.lower()}",     # with leading space (most common)
            word.capitalize(),       # Capitalized
            f" {word.capitalize()}", # Capitalized with space
            word.upper(),            # UPPERCASE (rare but possible)
        ]

        for variant in variants:
            try:
                ids = self._tokenizer.encode(variant)
                # Only use if it's a single token (direct word match)
                if len(ids) == 1:
                    token_ids.add(ids[0])
                # For multi-token words, add the first token
                # (partial match is better than nothing)
                elif len(ids) > 1:
                    token_ids.add(ids[0])
            except Exception as e:
                logger.debug(f"Failed to encode '{variant}': {e}")

        return list(token_ids)

    def is_available(self) -> bool:
        """Check if vocabulary control is available."""
        return self._tokenizer is not None


class OllamaVocabularyController(VocabularyController):
    """Vocabulary controller for Ollama.

    Placeholder for future implementation.
    Ollama may support logit_bias or other mechanisms.
    """

    def __init__(self, max_biases: int = 300):
        super().__init__(max_biases)
        logger.warning(
            "OllamaVocabularyController is not yet implemented. "
            "Vocabulary control will be disabled."
        )

    def get_provider_name(self) -> str:
        return "ollama"

    def get_token_ids(self, word: str) -> List[int]:
        """Ollama tokenization - not yet implemented."""
        # Ollama uses different tokenizers per model
        # This would need model-specific tokenizer loading
        return []

    def is_available(self) -> bool:
        return False


def get_vocabulary_controller(provider_name: str) -> Optional[VocabularyController]:
    """Factory function to get vocabulary controller for a provider.

    Args:
        provider_name: Name of the LLM provider.

    Returns:
        VocabularyController if available, None otherwise.
    """
    if provider_name == "deepseek":
        controller = DeepSeekVocabularyController()
        if controller.is_available():
            return controller
        return None

    if provider_name == "ollama":
        controller = OllamaVocabularyController()
        if controller.is_available():
            return controller
        return None

    logger.warning(f"No vocabulary controller for provider: {provider_name}")
    return None
