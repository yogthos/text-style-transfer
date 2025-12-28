"""Abstract vocabulary controller for LLM generation.

Provides a provider-agnostic interface for vocabulary control.
Different LLM providers implement this differently:
- DeepSeek/OpenAI: logit_bias parameter
- Ollama: may use different mechanism
- Local models: could use constrained decoding
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from .analyzer import VocabularyAnalysis
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BiasType(Enum):
    """Type of vocabulary bias to apply."""
    BOOST = "boost"      # Increase probability
    PENALTY = "penalty"  # Decrease probability
    BAN = "ban"          # Completely prevent


@dataclass
class TokenBias:
    """Bias to apply to a specific token."""
    token_id: int
    word: str
    bias_type: BiasType
    strength: float  # 0.0 to 1.0 (normalized)
    raw_bias: float  # Provider-specific value


@dataclass
class VocabularyBiases:
    """Collection of token biases for generation."""
    biases: List[TokenBias] = field(default_factory=list)

    # Summary stats
    boost_count: int = 0
    penalty_count: int = 0
    ban_count: int = 0

    def to_logit_bias_dict(self) -> Dict[str, float]:
        """Convert to OpenAI/DeepSeek logit_bias format.

        Returns dict mapping token ID strings to bias values.
        """
        return {
            str(b.token_id): b.raw_bias
            for b in self.biases
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"VocabularyBiases: {self.boost_count} boosted, "
            f"{self.penalty_count} penalized, {self.ban_count} banned"
        )


class VocabularyController(ABC):
    """Abstract base class for vocabulary control.

    Implementations must provide:
    - get_token_id: Convert word to token ID(s)
    - create_biases: Create biases from analysis
    """

    def __init__(self, max_biases: int = 300):
        """Initialize controller.

        Args:
            max_biases: Maximum number of token biases to apply.
                       API limits may restrict this.
        """
        self.max_biases = max_biases

    @abstractmethod
    def get_token_ids(self, word: str) -> List[int]:
        """Get token ID(s) for a word.

        A word may map to multiple tokens depending on context
        (e.g., with/without leading space).

        Args:
            word: Word to tokenize.

        Returns:
            List of token IDs that could represent this word.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider this controller works with."""
        pass

    def create_biases(
        self,
        analysis: VocabularyAnalysis,
        boost_scale: float = 30.0,
        penalty_scale: float = -50.0,
        ban_value: float = -100.0,
    ) -> VocabularyBiases:
        """Create token biases from vocabulary analysis.

        Args:
            analysis: Vocabulary analysis with boost/penalty words.
            boost_scale: Max logit bias for boosted words (positive).
            penalty_scale: Max logit bias for penalized words (negative).
            ban_value: Logit bias for banned words.

        Returns:
            VocabularyBiases ready to use in generation.
        """
        biases = VocabularyBiases()
        all_token_biases: List[TokenBias] = []

        # Process penalty words first (more important)
        for word, strength in analysis.penalty_words.items():
            token_ids = self.get_token_ids(word)
            for token_id in token_ids:
                raw_bias = penalty_scale * strength
                all_token_biases.append(TokenBias(
                    token_id=token_id,
                    word=word,
                    bias_type=BiasType.PENALTY,
                    strength=strength,
                    raw_bias=raw_bias,
                ))
                biases.penalty_count += 1

        # Process boost words
        for word, strength in analysis.boost_words.items():
            token_ids = self.get_token_ids(word)
            for token_id in token_ids:
                raw_bias = boost_scale * strength
                all_token_biases.append(TokenBias(
                    token_id=token_id,
                    word=word,
                    bias_type=BiasType.BOOST,
                    strength=strength,
                    raw_bias=raw_bias,
                ))
                biases.boost_count += 1

        # Limit to max biases (prioritize penalties)
        if len(all_token_biases) > self.max_biases:
            # Sort: penalties first (more negative), then by strength
            all_token_biases.sort(key=lambda b: (b.raw_bias, -b.strength))
            all_token_biases = all_token_biases[:self.max_biases]
            logger.warning(
                f"Truncated biases from {len(all_token_biases)} to {self.max_biases}"
            )

        biases.biases = all_token_biases
        logger.debug(biases.get_summary())

        return biases

    def create_biases_for_text(
        self,
        analysis: VocabularyAnalysis,
        input_text: str,
        boost_scale: float = 30.0,
        penalty_scale: float = -50.0,
    ) -> VocabularyBiases:
        """Create biases tailored for specific input text.

        This filters out any words that appear in the input text
        to avoid penalizing content words.

        Args:
            analysis: Base vocabulary analysis.
            input_text: The text being transformed.
            boost_scale: Max logit bias for boosted words.
            penalty_scale: Max logit bias for penalized words.

        Returns:
            VocabularyBiases filtered for this input.
        """
        # Get words from input text
        input_words = set(input_text.lower().split())

        # Filter analysis to exclude input words
        filtered_penalties = {
            w: s for w, s in analysis.penalty_words.items()
            if w.lower() not in input_words
        }
        filtered_boosts = {
            w: s for w, s in analysis.boost_words.items()
            if w.lower() not in input_words
        }

        # Create modified analysis
        filtered_analysis = VocabularyAnalysis(
            author_name=analysis.author_name,
            boost_words=filtered_boosts,
            penalty_words=filtered_penalties,
        )

        return self.create_biases(
            filtered_analysis,
            boost_scale=boost_scale,
            penalty_scale=penalty_scale,
        )
