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

logger = get_logger(__name__)

# Check MLX availability at module level
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. LoRA generation will not work.")


@dataclass
class GenerationConfig:
    """Configuration for LoRA generation."""

    max_tokens: int = 512
    temperature: float = 0.5  # Lower for more faithful translation
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

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is not None:
            return

        logger.info(f"Loading model: {self.base_model_name}")

        if self.adapter_path:
            logger.info(f"With LoRA adapter: {self.adapter_path}")
            self._model, self._tokenizer = load(
                self.base_model_name,
                adapter_path=self.adapter_path,
            )
        else:
            self._model, self._tokenizer = load(self.base_model_name)

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
            system = f"""You are {author}. Rewrite the following text in your distinctive voice.

CRITICAL RULES:
1. COMPLETE all sentences - never truncate or leave sentences unfinished
2. PRESERVE all facts, names, and key information from the source
3. DO NOT add new claims, examples, or ideas not in the original
4. DO NOT substitute technical terms with synonyms
5. VARY your word choice - avoid repeating the same adjectives and qualifiers"""

        # Build user message - just the content
        user = content

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Apply chat template
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

        # Use provided max_tokens or config default
        tokens_limit = max_tokens or self.config.max_tokens

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=tokens_limit,
            sampler=sampler,
        )

        return response.strip()

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
        repair_system = (
            f"You write in the style of {author}. "
            "IMPORTANT: You MUST include all the specific content mentioned. "
            "Preserve names, numbers, and key claims exactly."
        )

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
