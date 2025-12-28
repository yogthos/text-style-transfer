"""Vocabulary-controlled text generation wrapper.

Wraps LLM generation with vocabulary control via logit_bias.
This keeps the generator code clean while adding vocabulary control.
"""

from typing import Callable, Optional, Dict, Any

from .controller import VocabularyController, VocabularyBiases
from .analyzer import VocabularyAnalyzer, VocabularyAnalysis
from .semantic_filter import SemanticFilter
from ..style.profile import AuthorStyleProfile
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ControlledGenerator:
    """Wraps LLM generation with vocabulary control.

    Usage:
        controller = DeepSeekVocabularyController()
        controlled = ControlledGenerator(
            llm_provider=deepseek_provider,
            profile=author_profile,
            controller=controller
        )

        # Use as drop-in replacement for llm_generate
        text = controlled.generate(prompt)
    """

    def __init__(
        self,
        llm_provider: Any,  # LLM provider with call_with_logit_bias method
        profile: AuthorStyleProfile,
        controller: Optional[VocabularyController] = None,
        boost_scale: float = 25.0,
        penalty_scale: float = -40.0,
        enabled: bool = True,
    ):
        """Initialize controlled generator.

        Args:
            llm_provider: LLM provider (must have call_with_logit_bias method).
            profile: Author style profile for vocabulary analysis.
            controller: VocabularyController for token ID mapping.
            boost_scale: Logit bias scale for boosted words.
            penalty_scale: Logit bias scale for penalized words.
            enabled: Whether vocabulary control is enabled.
        """
        self.llm_provider = llm_provider
        self.profile = profile
        self.controller = controller
        self.boost_scale = boost_scale
        self.penalty_scale = penalty_scale
        self.enabled = enabled and controller is not None

        # Analyze vocabulary once
        self._analyzer = VocabularyAnalyzer()
        self._base_analysis: Optional[VocabularyAnalysis] = None
        self._cached_biases: Optional[VocabularyBiases] = None
        self._current_input_text: Optional[str] = None

        if self.enabled:
            self._initialize_analysis()

    def _initialize_analysis(self):
        """Initialize vocabulary analysis from profile."""
        self._base_analysis = self._analyzer.analyze_profile(
            self.profile,
            input_text=None,  # Will be updated per-text
            boost_strength=0.5,
            penalty_strength=0.7,
        )
        logger.info(
            f"Vocabulary control initialized: "
            f"{len(self._base_analysis.boost_words)} boost, "
            f"{len(self._base_analysis.penalty_words)} penalty words"
        )

    def set_input_text(self, input_text: str):
        """Set the input text to filter out its content words.

        This ensures we don't penalize words that are part of the
        content being transferred.

        Args:
            input_text: The source text being style-transferred.
        """
        if not self.enabled or self.controller is None:
            return

        self._current_input_text = input_text

        # Re-analyze with input text filtering
        filtered_analysis = self._analyzer.analyze_profile(
            self.profile,
            input_text=input_text,
            boost_strength=0.5,
            penalty_strength=0.7,
        )

        # Create biases
        self._cached_biases = self.controller.create_biases(
            filtered_analysis,
            boost_scale=self.boost_scale,
            penalty_scale=self.penalty_scale,
        )

        logger.debug(
            f"Updated vocabulary biases for input: "
            f"{self._cached_biases.get_summary()}"
        )

    def generate(self, prompt: str) -> str:
        """Generate text with vocabulary control.

        Args:
            prompt: The generation prompt.

        Returns:
            Generated text.
        """
        if not self.enabled or self._cached_biases is None:
            # Fallback to regular generation
            return self.llm_provider.call(
                system_prompt="You are a skilled writer. Follow instructions exactly.",
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=200
            )

        # Generate with vocabulary control
        logit_bias = self._cached_biases.to_logit_bias_dict()

        return self.llm_provider.call_with_logit_bias(
            system_prompt="You are a skilled writer. Follow instructions exactly.",
            user_prompt=prompt,
            logit_bias=logit_bias,
            temperature=0.7,
            max_tokens=200
        )

    def get_generate_function(self) -> Callable[[str], str]:
        """Get a function that can be used as llm_generate parameter.

        Returns:
            Callable that takes prompt and returns generated text.
        """
        return self.generate

    def get_stats(self) -> Dict[str, Any]:
        """Get vocabulary control statistics."""
        if not self.enabled or self._base_analysis is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "boost_words": len(self._base_analysis.boost_words),
            "penalty_words": len(self._base_analysis.penalty_words),
            "cached_biases": (
                self._cached_biases.get_summary()
                if self._cached_biases else "None"
            ),
        }


def create_controlled_generator(
    llm_provider: Any,
    profile: AuthorStyleProfile,
    provider_name: str = "deepseek",
) -> ControlledGenerator:
    """Factory function to create a controlled generator.

    Args:
        llm_provider: LLM provider instance.
        profile: Author style profile.
        provider_name: Name of the provider for controller selection.

    Returns:
        ControlledGenerator ready to use.
    """
    from .deepseek_controller import get_vocabulary_controller

    controller = get_vocabulary_controller(provider_name)

    return ControlledGenerator(
        llm_provider=llm_provider,
        profile=profile,
        controller=controller,
        enabled=controller is not None,
    )
