"""LoRA-based style transfer generator using MLX.

This module provides fast style transfer using a LoRA-adapted model.
Style is baked into the adapter weights, eliminating the need for:
- Multi-candidate evolutionary search
- Example-based prompting
- Statistical style verification

Performance target: ~5-10 seconds per paragraph generation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..utils.logging import get_logger
from ..utils.prompts import format_prompt

logger = get_logger(__name__)

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

        # Build prompt matching training data format EXACTLY
        author_tag = author.upper().replace(' ', '_').replace('.', '')
        prompt = format_prompt(
            "style_transfer",
            author=author,
            content=user,
            author_tag=author_tag,
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

        # Clean up the response
        response = self._clean_response(response, input_words)

        return response

    def _clean_response(self, response: str, input_words: int = 0) -> str:
        """Clean model output of thinking tags, meta-commentary, and garbage.

        Args:
            response: Raw model output.
            input_words: Word count of input (for length-based cleaning).

        Returns:
            Cleaned response text.
        """
        import re

        # 0. VERY FIRST: Truncate at training format markers
        # The model may continue generating training examples
        training_markers = [
            "[NEUTRAL INPUT]:",
            "[NEUTRAL INPUT]",
            "\n\n[HP_LOVECRAFT OUTPUT]:",  # If it loops back
            "\n\nRewrite the following",  # New training example
            "\n\n---",  # Section breaks
            "HP_LOVECRAFT_NOTE:",  # Hallucinated meta-comment
            "_NOTE:",  # Any author note
            "\nNote:",  # Note section
            "\n\nNOTE:",  # Note section
        ]
        for marker in training_markers:
            if marker in response:
                response = response.split(marker)[0].strip()

        # 0a. FIRST: Detect repetition and garbage
        # Split by double newline (paragraph boundaries) first
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            # Check for paragraph-level repetition (first 50 chars match)
            first_para_key = paragraphs[0].strip().lower()[:50]
            clean_paras = [paragraphs[0]]
            for para in paragraphs[1:]:
                para_key = para.strip().lower()[:50]
                # Stop if this paragraph starts like one we've seen
                if para_key == first_para_key or para_key[:30] in first_para_key:
                    break
                # Stop if paragraph starts with garbage
                if para and ord(para[0]) > 127:
                    break
                clean_paras.append(para)
            response = '\n\n'.join(clean_paras)

        # Also check sentence-level repetition
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > 3:
            seen = {}
            clean_sentences = []
            for sent in sentences:
                sent_key = sent.strip().lower()[:50]  # Use first 50 chars
                if sent_key in seen and len(sent_key) > 20:
                    break
                seen[sent_key] = True
                clean_sentences.append(sent)
            response = ' '.join(clean_sentences)

        # 0b. Stop at inline garbage markers (random years, garbage sequences)
        # These often appear right before the model degenerates
        garbage_markers = [
            r'\(\d{4}\)\s*[\u0400-\u04FF\u0E00-\u0E7F]',  # (2015) followed by garbage
            r'\n\s*!\s*\n',  # Isolated exclamation marks on newlines
            r'\n\s*I\s*\n',  # Isolated "I" on newlines
        ]
        for marker in garbage_markers:
            match = re.search(marker, response)
            if match:
                response = response[:match.start()].strip()

        # 1. Remove <think>...</think> blocks (Qwen3 thinking mode)
        # Handle both single-line and multi-line think blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # 2. Remove unclosed <think> tags and everything after
        if '<think>' in response:
            response = response.split('<think>')[0]

        # 2b. Remove hallucinated parenthetical additions
        # These often contain wrong information like "(later known as X)" or "(see Figure 1)"
        parenthetical_patterns = [
            # Biographical additions
            r'\(later known as [^)]+\)',
            r'\(better known as [^)]+\)',
            r'\(also known as [^)]+\)',
            r'\(the pen name [^)]+\)',
            r'\(a\.k\.a\. [^)]+\)',
            r'\(born [^)]+\)',
            r'\(died [^)]+\)',
            r'\(\d{4}[–-]\d{4}\)',  # Date ranges like (1818-1893)
            # Hallucinated references
            r'\(see [Ff]igure \d+\)',
            r'\(see [Cc]hapter \d+\)',
            r'\(see [Tt]able \d+\)',
            r'\(see [Aa]ppendix [A-Z]?\)',
            r'\(see above\)',
            r'\(see below\)',
            r'\(i\.[Ee]\.,? [^)]+\)',  # Remove "(i.e., ...)" explanations
            # Without parentheses (comma-separated)
            r',\s*later known as [^.,]+',
            r',\s*better known as [^.,]+',
            r',\s*also known as [^.,]+',
            r',\s*a\.k\.a\. [^.,]+',
        ]
        for pattern in parenthetical_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        # 2c. Remove trailing non-ASCII garbage (Thai, Arabic, Korean, Chinese, Cyrillic, CJK punctuation, Hebrew)
        # These appear when the model generates garbage tokens
        # Include CJK punctuation: ：。，！？《》""''【】 etc.
        # Include Hebrew: \u0590-\u05FF
        garbage_ranges = r'[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0E00-\u0E7F\uAC00-\uD7AF\u3040-\u30FF\u4E00-\u9FFF\u0080-\u009F\u3000-\u303F\uFF00-\uFFEF]'
        response = re.sub(garbage_ranges + r'+[.!?]?\s*$', '', response)

        # 2d. Remove mid-text non-ASCII garbage and mixed garbage
        # Also stop at Chinese colon (：) or em-dash (——) which often precedes garbage
        for stop_char in ['：', '——']:
            if stop_char in response:
                response = response.split(stop_char)[0].strip()
        response = re.sub(garbage_ranges + r'+[.!?]*', '', response)
        response = re.sub(r'\b[A-Z]?[a-z]*[Pp]id\b', '', response)  # Remove "mPid", "MPid" style garbage
        response = re.sub(r'SOEVER[,.]?\s*', '', response)  # Remove "SOEVER" garbage
        response = re.sub(r'togroup\s*', '', response)  # Remove "togroup" repetition
        response = re.sub(r'!+\s*', ' ', response)  # Remove repeated exclamation marks
        response = re.sub(r'\bприятн\b', '', response)  # Remove Cyrillic garbage

        # 2e. Fix double commas/spaces and orphaned punctuation from removed content
        response = re.sub(r',\s*,', ',', response)
        response = re.sub(r'\s+,', ',', response)  # Fix " ,"
        response = re.sub(r'in\s*,', 'in', response)  # Fix "in ,"
        response = re.sub(r'\(\s*,', '(', response)  # Fix "( ,"
        response = re.sub(r'\s{2,}', ' ', response)
        # Remove trailing empty quotes, stray punctuation
        response = re.sub(r'[""\'\']\.?\s*$', '', response)  # Trailing quotes
        response = re.sub(r'\.{2,}$', '.', response)  # Multiple periods at end

        # 3. Remove reasoning prefixes and leaked instructions
        thinking_prefixes = [
            r'^Okay,?\s+(so\s+)?let me.*?\.\s*',
            r'^Okay,?\s+(so\s+)?I need to.*?\.\s*',
            r'^First,?\s+I (need to|should|\'ll|will).*?\.\s*',
            r'^Let me (try to|start by|begin).*?\.\s*',
            r'^I\'ll (start|begin|try).*?\.\s*',
            r'^Now,?\s+(I need to|let me).*?\.\s*',
            r'^Hmm,?\s+.*?\.\s*',
            r'^The user (is asking|wants|provided).*?\.\s*',
            # Leaked system prompt patterns
            r'^Do not add.*?\.\s*',
            r'^Preserve every.*?\.\s*',
            r'^Keep all.*?\.\s*',
            r'^FORBIDDEN:.*?\.\s*',
            r'^Transform into.*?\.\s*',
            r'^Rewrite in.*?\.\s*',
        ]

        for prefix in thinking_prefixes:
            response = re.sub(prefix, '', response, flags=re.IGNORECASE | re.MULTILINE)

        # 4. Remove mid-text thinking patterns
        mid_thinking = [
            r'\n+Okay,?\s+(so\s+)?.*?\.\s*\n*',
            r'\n+First,?\s+I.*?\.\s*\n*',
            r'\n+Let me.*?\.\s*\n*',
            r'\n+I need to.*?\.\s*\n*',
            r'\n+Now,?\s+I.*?\.\s*\n*',
            r'\n+The (user|original|passage).*?\.\s*\n*',
        ]

        for pattern in mid_thinking:
            response = re.sub(pattern, '\n', response, flags=re.IGNORECASE)

        response = response.strip()

        # 5. Stop at patterns indicating meta-commentary or new turns
        stop_patterns = [
            "\n\nKeep",        # Echoing instructions
            "\nKeeping every",  # Meta-commentary
            "\n\nThis ",       # Starting explanation
            "\n\nNote:",       # Meta-note
            "\n\nuser",        # New turn marker
            "\n\nassistant",   # New turn marker
            "\nuser\n",        # Turn marker
            "\n\n---",         # Section break
            "\n---",           # Section break
            "\n\nWrite",       # New example
            "\n\nI need to",   # Thinking leaked
            "\n\nFirst,",      # Thinking leaked
            "\n\nOkay,",       # Thinking leaked
        ]

        for pattern in stop_patterns:
            if pattern in response:
                response = response.split(pattern)[0].strip()

        # 6. Stop at double newline followed by non-ASCII (often garbage)
        lines = response.split("\n\n")
        clean_lines = []
        for line in lines:
            # Stop if line starts with non-ASCII characters
            if line and ord(line[0]) > 127:
                break
            clean_lines.append(line)
        response = "\n\n".join(clean_lines).strip()

        # 7. For short inputs, only take first paragraph to avoid runaway generation
        # But don't limit sentences - style transfer may add atmospheric elaboration
        if input_words < 50 and "\n\n" in response:
            response = response.split("\n\n")[0].strip()

        # 8. Detect and remove duplicate phrases/sentences
        response = self._remove_duplicates(response)

        # 9. Ensure text ends with a complete sentence
        response = self._ensure_complete_sentences(response)

        return response

    def _remove_duplicates(self, text: str) -> str:
        """Remove duplicate sentences and phrases from text.

        Detects when the model repeats itself and removes the duplicate.
        """
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return text

        # Check for exact duplicate sentences
        seen = set()
        unique_sentences = []
        for sent in sentences:
            normalized = sent.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sent)

        # Check for mid-sentence repetition (phrase appears twice)
        result = ' '.join(unique_sentences)

        # Find repeated phrases of 10+ words
        words = result.split()
        if len(words) > 20:
            # Look for repeated sequences
            for phrase_len in range(15, 8, -1):  # Check phrases 15 down to 9 words
                for i in range(len(words) - phrase_len * 2):
                    phrase = ' '.join(words[i:i + phrase_len])
                    rest = ' '.join(words[i + phrase_len:])
                    if phrase.lower() in rest.lower():
                        # Found duplicate - remove second occurrence
                        pattern = re.escape(phrase)
                        # Remove second occurrence only
                        parts = re.split(pattern, result, flags=re.IGNORECASE)
                        if len(parts) > 2:
                            result = parts[0] + phrase + ''.join(parts[2:])
                        break

        return result.strip()

    def _ensure_complete_sentences(self, text: str) -> str:
        """Ensure text ends with a complete sentence.

        If text ends mid-sentence, truncate to the last complete sentence.
        """
        import re

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
