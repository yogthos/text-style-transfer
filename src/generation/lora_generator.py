"""LoRA-based style transfer generator using MLX.

This module provides fast style transfer using a LoRA-adapted model.
Style is baked into the adapter weights, eliminating the need for:
- Multi-candidate evolutionary search
- Example-based prompting
- Statistical style verification

Performance target: ~5-10 seconds per paragraph generation.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger
from ..utils.prompts import format_prompt

logger = get_logger(__name__)


def generate_style_tag(text: str) -> str:
    """Generate a structural style tag based on text analysis.

    Tags describe the structural features the model should produce:
    - Length pattern: Short & Punchy, Varied Lengths, Long & Flowing
    - Complexity: Simple Syntax, Complex Syntax, Baroque Syntax

    Example: [STYLE: Varied Lengths | Complex Syntax]
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return "[STYLE: Medium Length | Simple Syntax]"

    # Calculate sentence lengths
    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)

    # Calculate variance
    if len(lengths) > 1:
        variance = sum((x - avg_length) ** 2 for x in lengths) / len(lengths)
        std_dev = variance ** 0.5
    else:
        std_dev = 0

    # Detect complex syntax markers
    complex_markers = [';', '—', '--', ':', '(', ')']
    has_complex = any(m in text for m in complex_markers)

    # Detect literary connectives
    connective_words = ['however', 'although', 'yet', 'moreover', 'furthermore',
                        'nevertheless', 'whilst', 'whereas']
    has_literary_connectives = any(w in text.lower() for w in connective_words)

    # Determine length pattern
    if avg_length < 12:
        length_tag = "Short & Punchy"
    elif avg_length > 25:
        length_tag = "Long & Flowing"
    elif std_dev > 8:
        length_tag = "Varied Lengths"
    else:
        length_tag = "Medium Length"

    # Determine complexity
    if has_complex and has_literary_connectives:
        complexity_tag = "Baroque Syntax"
    elif has_complex:
        complexity_tag = "Complex Syntax"
    else:
        complexity_tag = "Simple Syntax"

    return f"[STYLE: {length_tag} | {complexity_tag}]"


# Check MLX availability at module level
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. LoRA generation will not work.")


@dataclass
class GenerationConfig:
    """Configuration for LoRA generation."""

    max_tokens: int = 512
    temperature: float = 0.4  # Higher temp helps complete sentences before repetition loops
    top_p: float = 0.9
    repetition_penalty: float = 1.4  # Strong penalty to prevent repetition loops
    min_tokens: int = 50  # Prevent too-short outputs
    lora_scale: float = 2.0  # LoRA influence: match training scale (alpha/rank = 128/64 = 2.0)
    skip_cleaning: bool = False  # If True, return raw output without _clean_response


@dataclass
class AdapterMetadata:
    """Metadata about a LoRA adapter."""

    author: str
    base_model: str
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    training_examples: int = 0

    @classmethod
    def from_file(cls, path: Path) -> "AdapterMetadata":
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            author=data.get("author", "Unknown"),
            base_model=data.get("base_model", "mlx-community/Qwen3-8B-Base-bf16"),
            lora_rank=data.get("lora_rank", 16),
            lora_alpha=data.get("lora_alpha", 32),
            epochs=data.get("epochs", 3),
            training_examples=data.get("training_examples", 0),
        )


class LoRAStyleGenerator:
    """Fast style transfer using LoRA-adapted model.

    Key advantages over prompted approach:
    - Style baked into weights (no examples needed in prompt)
    - Single forward pass (no evolutionary search)
    - Consistent voice (no mode collapse across calls)

    Example usage:
        generator = LoRAStyleGenerator(
            adapter_path="lora_adapters/sagan",
            config=GenerationConfig(temperature=0.7),
        )

        # Generate styled text
        output = generator.generate(
            content="The universe is vast. Stars are distant suns.",
            author="Carl Sagan",
        )
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_model: str = "mlx-community/Qwen3-8B-Base-bf16",
        config: Optional[GenerationConfig] = None,
    ):
        """Initialize the LoRA generator.

        Args:
            adapter_path: Path to LoRA adapter directory.
            base_model: Base model (overridden by adapter metadata if available).
            config: Generation configuration.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx mlx-lm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )

        self.config = config or GenerationConfig()
        self.adapter_path = adapter_path
        self.base_model_name = base_model
        self.metadata: Optional[AdapterMetadata] = None

        # Lazy load model
        self._model = None
        self._tokenizer = None

        # Load metadata if adapter exists
        if adapter_path:
            metadata_path = Path(adapter_path) / "metadata.json"
            if metadata_path.exists():
                self.metadata = AdapterMetadata.from_file(metadata_path)
                self.base_model_name = self.metadata.base_model
                logger.info(f"Loaded adapter metadata: {self.metadata.author}")

    def _is_model_cached(self, model_name: str) -> bool:
        """Check if model is already downloaded in HuggingFace cache."""
        try:
            from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
            # Check for config.json as indicator the model is cached
            result = try_to_load_from_cache(model_name, "config.json")
            return result is not None and result is not _CACHED_NO_EXIST
        except Exception:
            return False

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        import os

        if self._model is not None:
            return

        is_cached = self._is_model_cached(self.base_model_name)
        if is_cached:
            logger.info(f"Loading model: {self.base_model_name}")
            # Suppress progress bars for cached models
            old_hf_disable = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        else:
            logger.info(f"Downloading model: {self.base_model_name}")
            old_hf_disable = None

        try:
            if self.adapter_path:
                lora_scale = self.config.lora_scale
                logger.info(f"With LoRA adapter: {self.adapter_path} (scale={lora_scale})")
                self._model, self._tokenizer = load(
                    self.base_model_name,
                    adapter_path=self.adapter_path,
                )
                # Apply LoRA scale if not default (1.0)
                # This scales the adapter weights to control style influence
                if lora_scale != 1.0 and hasattr(self._model, 'model'):
                    self._apply_lora_scale(lora_scale)
            else:
                self._model, self._tokenizer = load(self.base_model_name)
        finally:
            # Restore progress bar setting
            if is_cached:
                if old_hf_disable is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = old_hf_disable

        logger.info("Model loaded successfully")

    def _apply_lora_scale(self, scale: float) -> None:
        """Apply scaling factor to LoRA adapter weights.

        This controls how much the LoRA adapter influences the base model:
        - scale=0.0: Base model only (no LoRA influence)
        - scale=0.5: Half LoRA influence (more base model)
        - scale=1.0: Full LoRA influence (default)
        - scale>1.0: Amplified LoRA influence (stronger style)

        Args:
            scale: Scaling factor for LoRA weights.
        """
        import mlx.core as mx

        def scale_lora_layers(module, path=""):
            """Recursively find and scale LoRA layers."""
            # Check if this module has LoRA weights
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b'):
                # Scale the LoRA output by adjusting lora_b (more efficient than scaling both)
                if hasattr(module, 'scale'):
                    # If module has a scale attribute, use it
                    module.scale = scale
                    logger.debug(f"Scaled {path}.scale = {scale}")
                else:
                    # Otherwise, scale lora_b directly
                    module.lora_b = module.lora_b * scale
                    logger.debug(f"Scaled {path}.lora_b by {scale}")

            # Recurse into children
            if hasattr(module, 'children'):
                for name, child in module.children().items():
                    scale_lora_layers(child, f"{path}.{name}" if path else name)
            elif hasattr(module, '__dict__'):
                for name, child in module.__dict__.items():
                    if hasattr(child, 'lora_a') or hasattr(child, 'children'):
                        scale_lora_layers(child, f"{path}.{name}" if path else name)

        try:
            scale_lora_layers(self._model)
            logger.info(f"Applied LoRA scale: {scale}")
        except Exception as e:
            logger.warning(f"Could not apply LoRA scale: {e}")

    def generate(
        self,
        content: str,
        author: str,
        max_tokens: Optional[int] = None,
        target_words: Optional[int] = None,
    ) -> str:
        """Generate styled text from content description.

        Args:
            content: What to express (neutral text to restyle).
            author: Author name (used in prompt).
            max_tokens: Override for max tokens (defaults to config).
            target_words: Target word count for output.

        Returns:
            Generated text in the author's style.
        """
        self._ensure_loaded()

        # Build user message - just the content
        user = content

        # Estimate target word count from input if not provided
        input_words = len(user.split())
        if target_words is None:
            target_words = input_words

        # Calculate tokens based on input length
        # Training data has ~1:1 ratio, but we need enough for complete sentences
        # Convert words to tokens (roughly 1.3 tokens per word)
        # Use 2x input to allow for style variation
        auto_max_tokens = max(100, int(input_words * 2.0 * 1.3))

        # Generate style tag from input to guide output structure
        style_tag = generate_style_tag(user)

        # Build prompt matching training data format EXACTLY
        prompt = format_prompt(
            "style_transfer",
            author=author,
            content=user,
            word_count=target_words,
            style_tag=style_tag,
        )

        # Create sampler with temperature and top_p
        sampler = make_sampler(
            temp=self.config.temperature,
            top_p=self.config.top_p,
        )

        # Create repetition penalty processor
        rep_penalty = make_repetition_penalty(
            penalty=self.config.repetition_penalty,
            context_size=50,
        )

        # Use provided max_tokens, or auto-calculated limit, or config default
        # Prefer tighter auto-calculated limit to prevent repetition
        if max_tokens:
            tokens_limit = max_tokens
        else:
            tokens_limit = min(auto_max_tokens, self.config.max_tokens)

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=tokens_limit,
            sampler=sampler,
            logits_processors=[rep_penalty],
        )

        response = response.strip()

        # Skip cleaning if configured (useful for debugging)
        if self.config.skip_cleaning:
            logger.debug("Skipping _clean_response (skip_cleaning=True)")
            return response

        # Clean up the response
        raw_response = response
        response = self._clean_response(response, input_words)

        # Log if cleaning removed significant content
        raw_words = len(raw_response.split())
        clean_words = len(response.split())
        if clean_words < raw_words * 0.7:
            logger.warning(f"_clean_response removed {raw_words - clean_words} words ({raw_words} → {clean_words})")
            logger.debug(f"Raw content before cleaning: {raw_response[:500]}...")

        return response

    def _clean_response(self, response: str, input_words: int = 0) -> str:
        """Clean model output of obvious garbage only.

        Removes:
        - ### markers and everything after (model repetition boundary)
        - Non-ASCII garbage (Thai, Cyrillic, Chinese characters)
        - <think> tags
        - Training format markers

        Does NOT aggressively detect repetition - preserves content.

        Args:
            response: Raw model output.
            input_words: Word count of input (unused, kept for compatibility).

        Returns:
            Cleaned response text.
        """
        # 1. Stop at ### markers (model uses these before repeating)
        if '###' in response:
            response = response.split('###')[0].strip()

        # 2. Stop at training format markers
        training_markers = [
            "[NEUTRAL INPUT]:",
            "[NEUTRAL INPUT]",
            "_OUTPUT]:",  # Any [AUTHOR_OUTPUT]: marker
            "\n\nRewrite the following",
            "\n\n---",
            "_NOTE:",
        ]
        for marker in training_markers:
            if marker in response:
                response = response.split(marker)[0].strip()

        # 3. Remove <think>...</think> blocks (Qwen3 thinking mode)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        if '<think>' in response:
            response = response.split('<think>')[0]

        # 4. Remove non-ASCII garbage characters (Thai, Cyrillic, Arabic, Chinese, etc.)
        # These appear when the model degenerates
        garbage_ranges = r'[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0E00-\u0E7F\uAC00-\uD7AF\u3040-\u30FF\u4E00-\u9FFF\u0080-\u009F\u3000-\u303F\uFF00-\uFFEF]'

        # Stop at first garbage character sequence
        match = re.search(garbage_ranges + r'+', response)
        if match:
            response = response[:match.start()].strip()

        # 5. Stop at Chinese punctuation that often precedes garbage
        for stop_char in ['：', '——']:
            if stop_char in response:
                response = response.split(stop_char)[0].strip()

        # 6. Clean up artifacts
        response = re.sub(r'\s{2,}', ' ', response)  # Multiple spaces
        response = re.sub(r'\n{3,}', '\n\n', response)  # Multiple newlines

        response = response.strip()

        # 7. Ensure text ends with a complete sentence
        response = self._ensure_complete_sentences(response)

        return response

    def _ensure_complete_sentences(self, text: str) -> str:
        """Ensure text ends with a complete sentence.

        If text ends mid-sentence, truncate to the last complete sentence.
        """
        text = text.strip()
        if not text:
            return text

        # If already ends with sentence punctuation, we're good
        if text[-1] in '.!?':
            return text

        # Find the last complete sentence by looking for period, !, or ?
        # followed by either end of string or a capital letter
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter to only complete sentences (end with punctuation)
        complete = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent[-1] in '.!?':
                complete.append(sent)

        if complete:
            return ' '.join(complete)

        # No complete sentences - try to add period if it looks like a complete thought
        if len(text.split()) > 10 and text[-1] not in ',:;':
            return text + '.'

        return text

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        logger.info("Model unloaded")
