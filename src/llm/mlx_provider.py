"""MLX-based LLM provider for text generation.

Uses the same Qwen model as the LoRA pipeline for consistency.
This allows the entire pipeline to be self-contained without external services.
"""

import json
from typing import Optional, Callable
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.prompts import load_prompt

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

    def unload(self):
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        logger.info("Model unloaded")


class RTTNeutralizer:
    """Round-Trip Translation neutralizer using local MLX model.

    Uses Qwen2.5-3B-Instruct for English → Mandarin → English translation
    to strip style while preserving facts. The HSK constraint forces simple
    vocabulary, and the grammar distance flattens syntax.

    Configuration loaded from config.json under llm.providers.mlx_rtt.

    Example:
        neutralizer = RTTNeutralizer()
        neutral = neutralizer.neutralize(
            "The eldritch horror lurked in cyclopean shadows."
        )
        # -> "The strange monster hid in the big shadows."
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the RTT neutralizer.

        Args:
            model_name: Override model (defaults to config mlx_rtt.model).
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx mlx-lm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )

        # Load RTT-specific config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        rtt_config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                rtt_config = config.get("llm", {}).get("providers", {}).get("mlx_rtt", {})
            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}")

        self.model_name = model_name or rtt_config.get("model", "mlx-community/Qwen2.5-3B-Instruct-4bit")
        self.max_tokens = rtt_config.get("max_tokens", 512)
        self.temperature = rtt_config.get("temperature", 0.1)
        self.top_p = rtt_config.get("top_p", 0.9)

        logger.info(f"RTT neutralizer using: {self.model_name}")

        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Ensure model is loaded and warmed up."""
        if self._model is not None:
            return

        logger.info(f"Loading RTT model: {self.model_name}")
        self._model, self._tokenizer = load(self.model_name)

        # Create cached sampler (reused across calls)
        self._sampler = make_sampler(temp=self.temperature, top_p=self.top_p)

        # Warm up the model with a short generation (compiles MLX graphs)
        logger.info("Warming up model...")
        _ = generate(
            self._model,
            self._tokenizer,
            prompt="Hello",
            max_tokens=5,
            sampler=self._sampler,
        )
        logger.info("RTT model ready")

    def _generate(self, system: str, user: str, max_tokens: int) -> str:
        """Generate using Qwen2.5 chat format."""
        self._ensure_loaded()

        # Build Qwen2.5 chat format
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self._sampler,  # Use cached sampler
        )

        return response.strip()

    def _extract_entities(self, text: str) -> tuple:
        """Extract proper nouns and replace with placeholders.

        Preserves names like "Jervas Dudley", "New England", "Squire Brewster"
        through the RTT process by replacing them with __ENT0__, __ENT1__, etc.

        Returns:
            (masked_text, entity_map) where entity_map is {placeholder: original}
        """
        import re

        entity_map = {}
        counter = [0]  # Use list to allow mutation in nested function

        # Common words that shouldn't be treated as proper nouns
        skip_words = {
            'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'And', 'But', 'Or',
            'It', 'He', 'She', 'They', 'We', 'I', 'My', 'His', 'Her', 'Their',
            'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'Who',
            'How', 'Why', 'If', 'Then', 'Now', 'Here', 'There', 'So', 'Yet',
            'From', 'With', 'Into', 'Upon', 'Through', 'About', 'After', 'Before',
            'During', 'Within', 'Without', 'Between', 'Among', 'Against', 'Beyond',
            'Such', 'Each', 'Every', 'Some', 'Many', 'Most', 'Other', 'Another',
            'Both', 'All', 'Any', 'No', 'Not', 'Only', 'Just', 'Even', 'Still',
            'Already', 'Always', 'Never', 'Perhaps', 'Indeed', 'Thus', 'Hence',
            'Therefore', 'Moreover', 'Furthermore', 'Meanwhile', 'Otherwise',
            'Nonetheless', 'Nevertheless', 'However', 'Although', 'Though', 'While',
            'Because', 'Since', 'Unless', 'Until', 'Whether', 'Whenever', 'Wherever',
            'One', 'Two', 'Three', 'Four', 'Five', 'First', 'Second', 'Third',
        }

        def make_replacer():
            """Create a replacer function that tracks what it's seen."""
            def replace_entity(match):
                word = match.group(1).strip()
                if not word or word in skip_words:
                    return match.group(0)  # Keep original including any prefix
                # Check if already masked
                if word.startswith('__ENT'):
                    return match.group(0)
                placeholder = f"__ENT{counter[0]}__"
                entity_map[placeholder] = word
                counter[0] += 1
                # Preserve any whitespace/punctuation before the word
                prefix = match.group(0)[:-len(word)] if match.group(0).endswith(word) else ''
                return prefix + placeholder
            return replace_entity

        masked = text

        # Pattern 1: Multi-word proper nouns (2-3 consecutive capitalized words)
        # Matches: "Jervas Dudley", "New England", "Squire Brewster Hyde"
        masked = re.sub(
            r'(?:^|(?<=[.!?,;:\s"\'(]))([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?=[\s.,;:!?\'")\-]|$)',
            make_replacer(),
            masked
        )

        # Pattern 2: Single capitalized words (but not at sentence start unless clearly a name)
        # More conservative - only after certain punctuation that suggests mid-sentence
        masked = re.sub(
            r'(?<=[,;:\s"\'(])([A-Z][a-z]{2,})(?=[\s.,;:!?\'")\-]|$)',
            make_replacer(),
            masked
        )

        return masked, entity_map

    def _restore_entities(self, text: str, entity_map: dict) -> str:
        """Restore original entities from placeholders."""
        result = text
        for placeholder, original in entity_map.items():
            result = result.replace(placeholder, original)
        return result

    def neutralize(
        self,
        text: str,
        max_retries: int = 2,
        monotone: bool = False,
    ) -> Optional[str]:
        """Neutralize text via round-trip translation through Mandarin.

        Preserves proper nouns by masking them before RTT and restoring after.

        Args:
            text: Text to neutralize.
            max_retries: Number of retry attempts.
            monotone: If True, apply aggressive flattening to create
                uniform sentence lengths (for training burstiness).

        Returns:
            Neutralized text, or None if failed.
        """
        import re

        # Extract and mask entities before RTT
        masked_text, entity_map = self._extract_entities(text)
        if entity_map:
            logger.debug(f"Masked {len(entity_map)} entities: {list(entity_map.values())[:5]}...")

        word_count = len(text.split())

        # For very long texts (300+ words), split into chunks
        if word_count > 300:
            result = self._neutralize_chunked(masked_text, max_retries, monotone)
            if result:
                return self._restore_entities(result, entity_map)
            return None

        for attempt in range(max_retries):
            try:
                # Step 1: English → Mandarin ("Acid Bath" - strips Victorian syntax)
                # HSK5 vocabulary constraint forces simple word choices
                # Entity placeholders preserved through RTT
                max_mandarin_tokens = min(int(word_count * 2), 512)
                rtt_mandarin_prompt = load_prompt("rtt_to_mandarin")
                mandarin = self._generate(
                    system=rtt_mandarin_prompt,
                    user=f"Translate to simple Mandarin:\n\n{masked_text}",
                    max_tokens=max_mandarin_tokens,
                )

                if not mandarin or len(mandarin) < 10:
                    logger.debug(f"RTT Step 1 failed: empty Mandarin (attempt {attempt + 1})")
                    continue

                # Step 2: Mandarin → Plain English ("Flattener" - clinical SVO output)
                # Controlled English: short sentences, common words, monotonous tone
                max_english_tokens = min(int(word_count * 1.5) + 20, 400)
                rtt_english_prompt = load_prompt("rtt_to_english")
                english = self._generate(
                    system=rtt_english_prompt,
                    user=f"Translate to simple English:\n\n{mandarin}",
                    max_tokens=max_english_tokens,
                )

                if not english or len(english) < 10:
                    logger.debug(f"RTT Step 2 failed: empty English (attempt {attempt + 1})")
                    continue

                # Clean response
                english = english.strip()
                english = re.sub(r'^```\w*\n?', '', english)
                english = re.sub(r'\n?```$', '', english)

                # Check for Chinese characters (translation failed if present)
                if re.search(r'[\u4e00-\u9fff]', english):
                    logger.debug(f"RTT Step 2 failed: output contains Chinese (attempt {attempt + 1})")
                    continue

                # Validate length (lenient for long texts)
                neutral_words = len(english.split())
                if neutral_words < 3:
                    # Too short - generation failed
                    logger.debug(f"RTT too short: {neutral_words} words (attempt {attempt + 1})")
                    continue

                # Length validation: more lenient for long texts
                # Short texts (<100w): allow 2x variance
                # Long texts (>100w): allow any reasonable output (truncation ok)
                if word_count < 100:
                    max_diff = word_count * 1.0
                else:
                    max_diff = word_count * 2.0  # Very lenient for long texts

                if abs(neutral_words - word_count) > max_diff:
                    logger.debug(f"RTT length mismatch: {neutral_words} vs {word_count} (attempt {attempt + 1})")
                    continue

                # Step 3 (optional): Monotone flattening for training burstiness
                if monotone:
                    english = self._monotone_flatten(english)

                # Step 4: Restore original proper nouns
                if entity_map:
                    english = self._restore_entities(english, entity_map)

                logger.debug(f"RTT success: {word_count} → {len(english.split())} words")
                return english

            except Exception as e:
                logger.warning(f"RTT attempt {attempt + 1} failed: {e}")

        return None

    def _neutralize_chunked(
        self,
        text: str,
        max_retries: int,
        monotone: bool,
    ) -> Optional[str]:
        """Neutralize long text by splitting into ~150 word chunks.

        Args:
            text: Long text to neutralize (300+ words)
            max_retries: Retries per chunk
            monotone: Apply monotone flattening

        Returns:
            Combined neutralized text, or None if failed
        """
        import re

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Group sentences into ~150 word chunks
        chunks = []
        current_chunk = []
        current_words = 0

        for sent in sentences:
            sent_words = len(sent.split())
            if current_words + sent_words > 150 and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_words = sent_words
            else:
                current_chunk.append(sent)
                current_words += sent_words

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Neutralize each chunk
        results = []
        for chunk in chunks:
            # Call neutralize for short chunk (won't recurse since <300 words)
            result = self.neutralize(chunk, max_retries=max_retries, monotone=False)
            if result:
                results.append(result)
            else:
                logger.debug(f"Chunk failed: {len(chunk.split())} words")
                # Continue with other chunks

        if not results:
            return None

        combined = ' '.join(results)

        # Apply monotone flattening once at the end
        if monotone:
            combined = self._monotone_flatten(combined)

        return combined

    def _monotone_flatten(self, text: str) -> str:
        """Flatten text into uniform short sentences using rules (fast, no LLM).

        This creates "The Monotone" - boring, repetitive input that maximizes
        the delta from the stylized output. The model learns its job is to
        break this monotony.

        Rules applied:
        - Split on sentence boundaries
        - Break at conjunctions (and, but, or, yet, so)
        - Break at semicolons and em-dashes
        - Remove parentheticals
        - Target 8-15 words per sentence
        """
        import re

        # Remove parentheticals
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'—[^—]*—', '', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        result = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Break on semicolons
            parts = re.split(r'\s*;\s*', sent)

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Break on conjunctions if sentence is long
                words = part.split()
                if len(words) > 15:
                    # Split at conjunctions
                    conj_pattern = r'\s*,?\s*\b(and|but|or|yet|so|however|although|while|whereas)\b\s*'
                    sub_parts = re.split(conj_pattern, part, flags=re.IGNORECASE)
                    for sub in sub_parts:
                        sub = sub.strip(' ,')
                        if sub and len(sub.split()) >= 3 and sub.lower() not in ['and', 'but', 'or', 'yet', 'so', 'however', 'although', 'while', 'whereas']:
                            # Ensure ends with period
                            if not sub.endswith(('.', '!', '?')):
                                sub += '.'
                            # Capitalize first letter
                            sub = sub[0].upper() + sub[1:] if len(sub) > 1 else sub.upper()
                            result.append(sub)
                else:
                    # Short enough, keep as is
                    if not part.endswith(('.', '!', '?')):
                        part += '.'
                    part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
                    result.append(part)

        return ' '.join(result) if result else text


class DeepSeekRTTNeutralizer:
    """Round-Trip Translation neutralizer using DeepSeek API.

    Faster than local MLX for bulk processing. Supports batching multiple
    chunks in a single API call for efficiency.

    Configuration loaded from config.json under llm.providers.deepseek_rtt.
    """

    def __init__(self, batch_size: int = 5):
        """Initialize the DeepSeek RTT neutralizer.

        Args:
            batch_size: Number of texts to process in a single API call.
        """
        import os
        import requests

        # Load config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        rtt_config = {}
        deepseek_config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                rtt_config = config.get("llm", {}).get("providers", {}).get("deepseek_rtt", {})
                deepseek_config = config.get("llm", {}).get("providers", {}).get("deepseek", {})
            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}")

        # Get API key: try config first (with env var expansion), then direct env
        self.api_key = None
        config_api_key = deepseek_config.get("api_key", "")
        if config_api_key.startswith("${") and config_api_key.endswith("}"):
            # Expand ${VAR} syntax
            env_var = config_api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")
        elif config_api_key:
            self.api_key = config_api_key

        # Fallback to direct env var
        if not self.api_key:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. Set it in config.json or as environment variable."
            )

        self.model = rtt_config.get("model", "deepseek-chat")
        self.max_tokens = rtt_config.get("max_tokens", 8192)
        self.temperature = rtt_config.get("temperature", 0.1)
        self.batch_size = batch_size or rtt_config.get("batch_size", 5)
        self.timeout = rtt_config.get("timeout", 180)
        self.concurrent_batches = rtt_config.get("concurrent_batches", 4)

        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"DeepSeek RTT: model={self.model}, batch_size={self.batch_size}, concurrent={self.concurrent_batches}")

    def _call_api(self, system: str, user: str, max_tokens: int = None) -> str:
        """Make a single API call to DeepSeek."""
        import requests

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek API error: {response.status_code} - {response.text[:200]}")

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _extract_entities(self, text: str) -> tuple:
        """Extract proper nouns and replace with placeholders.

        Same logic as RTTNeutralizer for consistency.
        """
        import re

        entity_map = {}
        counter = [0]

        skip_words = {
            'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'And', 'But', 'Or',
            'It', 'He', 'She', 'They', 'We', 'I', 'My', 'His', 'Her', 'Their',
            'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'Who',
            'How', 'Why', 'If', 'Then', 'Now', 'Here', 'There', 'So', 'Yet',
            'From', 'With', 'Into', 'Upon', 'Through', 'About', 'After', 'Before',
            'During', 'Within', 'Without', 'Between', 'Among', 'Against', 'Beyond',
            'Such', 'Each', 'Every', 'Some', 'Many', 'Most', 'Other', 'Another',
            'Both', 'All', 'Any', 'No', 'Not', 'Only', 'Just', 'Even', 'Still',
            'Already', 'Always', 'Never', 'Perhaps', 'Indeed', 'Thus', 'Hence',
            'Therefore', 'Moreover', 'Furthermore', 'Meanwhile', 'Otherwise',
            'Nonetheless', 'Nevertheless', 'However', 'Although', 'Though', 'While',
            'Because', 'Since', 'Unless', 'Until', 'Whether', 'Whenever', 'Wherever',
            'One', 'Two', 'Three', 'Four', 'Five', 'First', 'Second', 'Third',
        }

        def make_replacer():
            def replace_entity(match):
                word = match.group(1).strip()
                if not word or word in skip_words:
                    return match.group(0)
                if word.startswith('__ENT'):
                    return match.group(0)
                placeholder = f"__ENT{counter[0]}__"
                entity_map[placeholder] = word
                counter[0] += 1
                prefix = match.group(0)[:-len(word)] if match.group(0).endswith(word) else ''
                return prefix + placeholder
            return replace_entity

        masked = text

        # Multi-word proper nouns
        masked = re.sub(
            r'(?:^|(?<=[.!?,;:\s"\'(]))([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?=[\s.,;:!?\'")\-]|$)',
            make_replacer(),
            masked
        )

        # Single capitalized words
        masked = re.sub(
            r'(?<=[,;:\s"\'(])([A-Z][a-z]{2,})(?=[\s.,;:!?\'")\-]|$)',
            make_replacer(),
            masked
        )

        return masked, entity_map

    def _restore_entities(self, text: str, entity_map: dict) -> str:
        """Restore original entities from placeholders."""
        result = text
        for placeholder, original in entity_map.items():
            result = result.replace(placeholder, original)
        return result

    def _monotone_flatten(self, text: str) -> str:
        """Flatten text into uniform short sentences (rule-based, fast)."""
        import re

        # Remove parentheticals
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'—[^—]*—', '', text)

        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            parts = re.split(r'\s*;\s*', sent)

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                words = part.split()
                if len(words) > 15:
                    conj_pattern = r'\s*,?\s*\b(and|but|or|yet|so|however|although|while|whereas)\b\s*'
                    sub_parts = re.split(conj_pattern, part, flags=re.IGNORECASE)
                    for sub in sub_parts:
                        sub = sub.strip(' ,')
                        if sub and len(sub.split()) >= 3 and sub.lower() not in ['and', 'but', 'or', 'yet', 'so', 'however', 'although', 'while', 'whereas']:
                            if not sub.endswith(('.', '!', '?')):
                                sub += '.'
                            sub = sub[0].upper() + sub[1:] if len(sub) > 1 else sub.upper()
                            result.append(sub)
                else:
                    if not part.endswith(('.', '!', '?')):
                        part += '.'
                    part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
                    result.append(part)

        return ' '.join(result) if result else text

    def neutralize(
        self,
        text: str,
        max_retries: int = 2,
        monotone: bool = False,
    ) -> Optional[str]:
        """Neutralize a single text via DeepSeek RTT.

        Args:
            text: Text to neutralize.
            max_retries: Number of retry attempts.
            monotone: If True, apply monotone flattening.

        Returns:
            Neutralized text, or None if failed.
        """
        import re

        # Extract and mask entities
        masked_text, entity_map = self._extract_entities(text)
        if entity_map:
            logger.debug(f"Masked {len(entity_map)} entities")

        word_count = len(text.split())

        for attempt in range(max_retries):
            try:
                # Single RTT call - DeepSeek handles English→Mandarin→English with optimized prompts
                rtt_deepseek_prompt = load_prompt("rtt_deepseek")
                result = self._call_api(
                    system=rtt_deepseek_prompt,
                    user=f"Input ({word_count} words - output must also be ~{word_count} words):\n\n{masked_text}",
                    max_tokens=min(int(word_count * 2) + 100, 2000),
                )

                if not result or len(result) < 10:
                    continue

                # Clean response
                result = result.strip()
                result = re.sub(r'^```\w*\n?', '', result)
                result = re.sub(r'\n?```$', '', result)

                # Check for Chinese characters
                if re.search(r'[\u4e00-\u9fff]', result):
                    logger.debug(f"RTT failed: contains Chinese (attempt {attempt + 1})")
                    continue

                # Apply monotone flattening if requested
                if monotone:
                    result = self._monotone_flatten(result)

                # Restore entities
                if entity_map:
                    result = self._restore_entities(result, entity_map)

                return result

            except Exception as e:
                logger.warning(f"DeepSeek RTT attempt {attempt + 1} failed: {e}")

        return None

    def _process_single_batch(
        self,
        batch_info: tuple,
        monotone: bool = False,
    ) -> list:
        """Process a single batch of texts (for parallel execution).

        Args:
            batch_info: Tuple of (batch_start, batch_texts)
            monotone: If True, apply monotone flattening.

        Returns:
            List of (global_index, result) tuples.
        """
        import re

        batch_start, batch = batch_info
        results = []

        # Extract entities for each text in batch
        masked_batch = []
        entity_maps = []
        for text in batch:
            masked, entity_map = self._extract_entities(text)
            masked_batch.append(masked)
            entity_maps.append(entity_map)

        # Build batched prompt
        batch_prompt = "Perform 'Chemical Dissolution' on each numbered text below.\n"
        batch_prompt += "Output the final neutral English on a new line prefixed with the same number.\n"
        batch_prompt += "Keep all __ENT*__ placeholders unchanged.\n\n"

        for i, masked in enumerate(masked_batch):
            batch_prompt += f"[{i+1}] {masked}\n\n"

        try:
            # Estimate tokens needed
            total_words = sum(len(t.split()) for t in batch)
            max_tokens = min(int(total_words * 2) + 100, 4000)

            rtt_batch_prompt = load_prompt("rtt_deepseek_batch")
            response = self._call_api(
                system=rtt_batch_prompt,
                user=batch_prompt,
                max_tokens=max_tokens,
            )

            # Parse numbered responses
            lines = response.strip().split('\n')
            current_num = None
            current_text = []

            for line in lines:
                # Check if line starts with a number
                match = re.match(r'^\[?(\d+)\]?\s*(.*)$', line.strip())
                if match:
                    # Save previous if exists
                    if current_num is not None and current_text:
                        idx = current_num - 1
                        if 0 <= idx < len(batch):
                            text = ' '.join(current_text).strip()
                            if not re.search(r'[\u4e00-\u9fff]', text):
                                if monotone:
                                    text = self._monotone_flatten(text)
                                text = self._restore_entities(text, entity_maps[idx])
                                results.append((batch_start + idx, text))

                    current_num = int(match.group(1))
                    current_text = [match.group(2)] if match.group(2) else []
                elif current_num is not None:
                    current_text.append(line)

            # Don't forget last item
            if current_num is not None and current_text:
                idx = current_num - 1
                if 0 <= idx < len(batch):
                    text = ' '.join(current_text).strip()
                    if not re.search(r'[\u4e00-\u9fff]', text):
                        if monotone:
                            text = self._monotone_flatten(text)
                        text = self._restore_entities(text, entity_maps[idx])
                        results.append((batch_start + idx, text))

        except Exception as e:
            batch_end = batch_start + len(batch)
            logger.warning(f"Batch {batch_start}-{batch_end} failed: {e}")
            # Fall back to individual processing for this batch
            for i, text in enumerate(batch):
                result = self.neutralize(text, monotone=monotone)
                if result:
                    results.append((batch_start + i, result))

        return results

    def neutralize_batch(
        self,
        texts: list,
        monotone: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list:
        """Neutralize multiple texts in parallel batched API calls.

        Uses concurrent_batches threads to process multiple batches in parallel,
        significantly improving throughput for large datasets.

        Args:
            texts: List of texts to neutralize.
            monotone: If True, apply monotone flattening.
            on_progress: Optional callback (processed, total).

        Returns:
            List of neutralized texts (None for failures).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        results = [None] * len(texts)
        processed_count = [0]  # Use list for mutable counter in closure
        progress_lock = threading.Lock()

        # Prepare all batches
        batches = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch = texts[batch_start:batch_end]
            batches.append((batch_start, batch))

        logger.info(f"Processing {len(texts)} texts in {len(batches)} batches ({self.concurrent_batches} concurrent)")

        def process_with_progress(batch_info):
            batch_results = self._process_single_batch(batch_info, monotone=monotone)
            # Update progress
            with progress_lock:
                processed_count[0] += len(batch_info[1])
                if on_progress:
                    on_progress(processed_count[0], len(texts))
            return batch_results

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.concurrent_batches) as executor:
            futures = [executor.submit(process_with_progress, batch) for batch in batches]

            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    # Merge results into main list
                    for idx, text in batch_results:
                        results[idx] = text
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")

        return results


def create_rtt_neutralizer(provider: str = None, batch_size: int = None):
    """Factory function to create the appropriate RTT neutralizer.

    Args:
        provider: 'mlx' or 'deepseek'. If None, reads from config.json.
        batch_size: Batch size for DeepSeek (ignored for MLX).

    Returns:
        RTTNeutralizer or DeepSeekRTTNeutralizer instance.
    """
    if provider is None:
        # Load from config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                provider = config.get("llm", {}).get("provider", {}).get("rtt", "deepseek")
            except Exception:
                provider = "deepseek"
        else:
            provider = "deepseek"

    if provider == "mlx":
        return RTTNeutralizer()
    elif provider == "deepseek":
        return DeepSeekRTTNeutralizer(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown RTT provider: {provider}")
