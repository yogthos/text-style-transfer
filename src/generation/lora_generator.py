"""LoRA-based style transfer generator using MLX.

This module provides fast style transfer using a LoRA-adapted model.
Style is baked into the adapter weights, eliminating the need for:
- Multi-candidate evolutionary search
- Example-based prompting
- Statistical style verification

Performance target: ~5-10 seconds per paragraph generation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..utils.logging import get_logger
from ..utils.prompts import format_prompt, get_prompt_with_fallback

logger = get_logger(__name__)

# Check MLX availability at module level
try:
    import mlx.core as mx
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
    temperature: float = 1.0  # Per paper: 1.0 allows creative vocabulary selection
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    min_tokens: int = 50  # Prevent too-short outputs


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
            base_model=data.get("base_model", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
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

        # Generate styled paragraph
        output = generator.generate_paragraph(
            propositions=["The universe is vast", "Stars are distant suns"],
            author="Carl Sagan",
        )
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
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
                logger.info(f"With LoRA adapter: {self.adapter_path}")
                self._model, self._tokenizer = load(
                    self.base_model_name,
                    adapter_path=self.adapter_path,
                )
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

    def generate(
        self,
        content: str,
        author: str,
        context: Optional[str] = None,
        system_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate styled text from content description.

        Args:
            content: What to express (propositions, summary, or description).
            author: Author name (used in system prompt).
            context: Optional previous paragraph for continuity.
            system_override: Optional custom system prompt.
            max_tokens: Override for max tokens (defaults to config).

        Returns:
            Generated text in the author's style.
        """
        self._ensure_loaded()

        # Build messages
        if system_override:
            system = system_override
        else:
            # Content-preserving prompt with explicit constraints
            system = format_prompt("style_transfer_system", author=author)

        # Build user message - just the content
        user = content

        # Estimate target word count from input (style transfer maintains similar length)
        input_words = len(user.split())
        # Allow some flexibility in output length
        target_words = input_words

        # Check if base model (no chat template)
        is_base_model = (
            "instruct" not in self.base_model_name.lower() and
            "chat" not in self.base_model_name.lower()
        )

        if is_base_model:
            # For base models, match the training format (instruction back-translation)
            instruction = format_prompt(
                "style_transfer_base_model",
                target_words=target_words,
                author=author
            )
            prompt = f"{instruction}\n\n{user}\n\n"
        else:
            # For instruct models, use chat template
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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

        # Use provided max_tokens or config default
        tokens_limit = max_tokens or self.config.max_tokens

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

        # For base models, extract just the first response (stop at continuation patterns)
        if is_base_model:
            # Stop at patterns that indicate model is starting a new example
            stop_patterns = ["\n\nWrite in the style of", "\nWrite in the style of", "\n\n---", "\n---"]
            for pattern in stop_patterns:
                if pattern in response:
                    response = response.split(pattern)[0].strip()
                    break

        return response

    def generate_paragraph(
        self,
        propositions: List[str],
        author: str,
        previous_paragraph: Optional[str] = None,
    ) -> str:
        """Generate a full paragraph from propositions.

        This is the main entry point for style transfer. Takes a list
        of semantic propositions (what to say) and renders them in
        the author's style.

        Args:
            propositions: List of content propositions to express.
            author: Author name.
            previous_paragraph: Previous output paragraph for continuity.

        Returns:
            Styled paragraph expressing all propositions.
        """
        # Join propositions as the content to translate
        content = " ".join(propositions)

        # Strict max tokens based on input length (1.3x to allow for style variation)
        input_words = len(content.split())
        estimated_tokens = int(input_words * 1.5)  # ~1.1 tokens per word + small margin
        max_tokens = max(50, min(estimated_tokens, self.config.max_tokens))

        return self.generate(
            content=content,
            author=author,
            context=previous_paragraph,
            max_tokens=max_tokens,
        )

    def generate_with_repair(
        self,
        propositions: List[str],
        author: str,
        verify_fn=None,
        threshold: float = 0.7,
        previous_paragraph: Optional[str] = None,
    ) -> str:
        """Generate with optional verification and repair.

        If verification fails, attempts a single repair pass with
        explicit instruction to include missing content.

        Args:
            propositions: Content propositions.
            author: Author name.
            verify_fn: Optional function (original, output) -> score.
            threshold: Minimum verification score.
            previous_paragraph: Previous paragraph for context.

        Returns:
            Generated (and possibly repaired) text.
        """
        # First generation attempt
        output = self.generate_paragraph(
            propositions=propositions,
            author=author,
            previous_paragraph=previous_paragraph,
        )

        # Skip verification if no function provided
        if verify_fn is None:
            return output

        # Verify output
        original = " ".join(propositions)
        score = verify_fn(original, output)

        if score >= threshold:
            logger.debug(f"Verification passed: {score:.2f}")
            return output

        # Repair attempt
        logger.warning(f"Verification failed ({score:.2f}), attempting repair...")

        # Add explicit requirement to system prompt
        repair_system = format_prompt("repair_with_content", author=author)

        return self.generate(
            content="\n".join(f"- {p}" for p in propositions),
            author=author,
            context=previous_paragraph,
            system_override=repair_system,
        )

    def switch_adapter(self, adapter_path: str) -> None:
        """Hot-swap to a different author's adapter.

        Args:
            adapter_path: Path to new adapter directory.
        """
        logger.info(f"Switching adapter to: {adapter_path}")

        self.adapter_path = adapter_path
        self._model = None  # Force reload

        # Update metadata
        metadata_path = Path(adapter_path) / "metadata.json"
        if metadata_path.exists():
            self.metadata = AdapterMetadata.from_file(metadata_path)
            self.base_model_name = self.metadata.base_model

    def get_author(self) -> str:
        """Get the current author name from metadata."""
        if self.metadata:
            return self.metadata.author
        return "Unknown"

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        logger.info("Model unloaded")
