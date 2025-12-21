"""Structure Template Extractor.

This module provides functionality to "bleach" style samples by replacing
content words with placeholders while preserving functional words and syntax.
This prevents domain-specific vocabulary from leaking into narrative generation.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional
from src.generator.llm_provider import LLMProvider


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'structure_extractor_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


class StructureExtractor:
    """Extracts structural templates from text by replacing content with placeholders."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the structure extractor.

        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.llm_provider = LLMProvider(config_path=config_path)
        self.cache_file = Path(config_path).parent / "structure_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load template cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        """Save template cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except IOError:
            pass  # Silently fail if cache can't be written

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def extract_template(self, text: str) -> str:
        """Extract structural template from text by replacing content with placeholders.

        Replaces specific nouns, verbs, adjectives, and adverbs with placeholders
        ([NP], [VP], [ADJ], [ADV]) while preserving all functional words
        (auxiliaries, connectors, prepositions, articles).

        Args:
            text: Input text to extract template from.

        Returns:
            Structural template with placeholders.
        """
        if not text or not text.strip():
            return text

        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Build prompt with explicit keep list and pronoun handling
        system_prompt = _load_prompt_template("structure_extractor_system.md")
        user_prompt = f"Convert this text to a structural template:\n\n{text}"

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.3,  # Low temperature for consistent extraction
                max_tokens=500,
                timeout=30
            )

            # Clean response (remove quotes if wrapped)
            template = response.strip()
            if template.startswith('"') and template.endswith('"'):
                template = template[1:-1]
            if template.startswith("'") and template.endswith("'"):
                template = template[1:-1]

            # Validate template has placeholders
            if '[NP]' in template or '[VP]' in template or '[ADJ]' in template or '[ADV]' in template:
                # Cache the result
                self.cache[cache_key] = template
                self._save_cache()
                return template
            else:
                # If no placeholders found, return original (fallback)
                return text

        except Exception as e:
            # On any error, return original text as fallback
            return text

