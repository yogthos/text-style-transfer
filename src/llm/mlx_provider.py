"""MLX-based LLM provider for text generation.

Uses the same Qwen model as the LoRA pipeline for consistency.
This allows the entire pipeline to be self-contained without external services.
"""

import json
from typing import Optional, Callable
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


def _load_mlx_config() -> dict:
    """Load MLX config from config.json."""
    config_path = Path(__file__).parent.parent.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            return config.get("llm", {}).get("providers", {}).get("mlx", {})
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    return {}

# Check MLX availability
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. Install with: pip install mlx mlx-lm")


class MLXGenerator:
    """MLX-based text generator using Qwen model.

    Can be used for:
    - Neutralizing author text (converting to plain English)
    - Any other text generation tasks in the pipeline

    Configuration is loaded from config.json under llm.providers.mlx:
        {
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9
        }

    Example:
        generator = MLXGenerator()
        neutral = generator.generate(
            prompt="Convert to plain English: ...",
            max_tokens=200,
        )
    """

    DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the MLX generator.

        Args:
            model_name: Model to use (from config or default).
            temperature: Generation temperature (from config or 0.3).
            top_p: Top-p sampling parameter (from config or 0.9).
            max_tokens: Default max tokens (from config or 512).
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx mlx-lm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )

        # Load config defaults
        config = _load_mlx_config()

        self.model_name = model_name or config.get("model", self.DEFAULT_MODEL)
        self.temperature = temperature if temperature is not None else config.get("temperature", 0.3)
        self.top_p = top_p if top_p is not None else config.get("top_p", 0.9)
        self.default_max_tokens = max_tokens if max_tokens is not None else config.get("max_tokens", 512)

        # Detect if this is a base model (no chat template)
        self.is_base_model = "instruct" not in self.model_name.lower() and "chat" not in self.model_name.lower()

        logger.info(f"MLX config: model={self.model_name}, base_model={self.is_base_model}, temp={self.temperature}")

        # Lazy load model
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is not None:
            return

        logger.info(f"Loading MLX model: {self.model_name}")
        self._model, self._tokenizer = load(self.model_name)
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt.
            max_tokens: Maximum tokens to generate (from config if not specified).
            system_prompt: Optional system prompt.
            temperature: Override temperature for this call.

        Returns:
            Generated text.
        """
        self._ensure_loaded()
        max_tokens = max_tokens or self.default_max_tokens

        # For base models, use raw text completion
        if self.is_base_model:
            # Build a simple prompt format for base models
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                formatted_prompt = prompt
        else:
            # Build messages for instruct models
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Create sampler
        temp = temperature if temperature is not None else self.temperature
        sampler = make_sampler(temp=temp, top_p=self.top_p)

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        return response.strip()

    def generate_raw(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a raw prompt (no chat template).

        Args:
            prompt: The raw prompt text.
            max_tokens: Maximum tokens to generate (from config if not specified).
            temperature: Override temperature for this call.

        Returns:
            Generated text.
        """
        self._ensure_loaded()
        max_tokens = max_tokens or self.default_max_tokens

        # Create sampler
        temp = temperature if temperature is not None else self.temperature
        sampler = make_sampler(temp=temp, top_p=self.top_p)

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        return response.strip()

    def unload(self):
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        logger.info("Model unloaded")


def create_mlx_generator(
    model: str = None,
    temperature: float = 0.3,
) -> Callable[[str], str]:
    """Create a simple generator function using MLX.

    This returns a function compatible with the training scripts.

    Args:
        model: Model name (from config.json, defaults to Qwen3-8B-4bit base model).
        temperature: Generation temperature.

    Returns:
        Function that takes a prompt and returns generated text.
    """
    generator = MLXGenerator(model_name=model, temperature=temperature)

    def generate(prompt: str) -> str:
        """Generate text using MLX."""
        try:
            return generator.generate(prompt, max_tokens=512)
        except Exception as e:
            logger.error(f"MLX generation error: {e}")
            return ""

    return generate


class TextNeutralizer:
    """Neutralize author-specific text to plain English using MLX.

    This converts distinctive prose into neutral/generic text while
    preserving the meaning. Used for generating training data.

    Example:
        neutralizer = TextNeutralizer()
        neutral = neutralizer.neutralize(
            "The cosmos is vast beyond imagining. We are tiny specks..."
        )
        # -> "The universe is very large. We are small parts..."
    """

    NEUTRALIZE_PROMPT = """Convert this distinctive prose to plain neutral English. Keep approximately the SAME word count.

EXAMPLE:
Distinctive: "The cosmos is vast beyond imagining. We are, each of us, a tiny speck in an ocean of stars. And yet - in this vastness - our minds can grasp infinity itself."
Neutral: "The universe is very large. Each person is a small part of a huge number of stars. Despite this large scale, human minds are able to understand the concept of infinity."

NOW CONVERT THIS TEXT:
Distinctive: "{text}"
Neutral:"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """Initialize the neutralizer.

        Args:
            model_name: Model to use.
            temperature: Generation temperature.
        """
        self._generator = MLXGenerator(
            model_name=model_name,
            temperature=temperature,
        )

    def neutralize(self, text: str, max_retries: int = 2) -> str:
        """Convert distinctive text to neutral/plain English.

        Args:
            text: The distinctive/styled text to neutralize.
            max_retries: Number of retries if output is too short.

        Returns:
            Neutral/plain English version of the text.
        """
        import re

        original_words = len(text.split())
        prompt = self.NEUTRALIZE_PROMPT.format(text=text[:2000])

        # Estimate tokens needed
        max_tokens = max(100, int(original_words * 2))

        for attempt in range(max_retries + 1):
            response = self._generator.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3 + (attempt * 0.1),  # Slightly increase temp on retry
            )

            # Clean up response
            neutral = response.strip()

            # Remove any prefix like 'Neutral: "...'
            neutral = re.sub(r'^Neutral\s*:\s*', '', neutral, flags=re.IGNORECASE)
            if neutral.startswith('"') and neutral.endswith('"'):
                neutral = neutral[1:-1]

            # Check if reasonable length
            neutral_words = len(neutral.split())
            if neutral_words >= original_words * 0.5:
                return neutral.strip()

            logger.warning(
                f"Neutral too short ({neutral_words} vs {original_words} words), "
                f"retry {attempt + 1}/{max_retries}"
            )

        # Fallback to original if all retries failed
        logger.warning("All neutralization attempts failed, using original text")
        return text

    def neutralize_batch(
        self,
        texts: list,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list:
        """Neutralize multiple texts.

        Args:
            texts: List of texts to neutralize.
            on_progress: Optional progress callback (current, total).

        Returns:
            List of neutralized texts.
        """
        results = []
        for i, text in enumerate(texts):
            if on_progress:
                on_progress(i + 1, len(texts))
            results.append(self.neutralize(text))
        return results

    def unload(self):
        """Unload model to free memory."""
        self._generator.unload()
