"""Structure Template Extractor.

This module provides functionality to "bleach" style samples by replacing
content words with placeholders while preserving functional words and syntax.
This prevents domain-specific vocabulary from leaking into narrative generation.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional
from src.generator.llm_interface import LLMProvider


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
        system_prompt = """You are a Structural Linguist. Your goal is to strip all specific meaning from the text while preserving its exact grammatical structure, rhythm, and length.

**Step 1: Expand Contractions**
First, expand all contractions (e.g., 'You'll' -> 'You will', 'won't' -> 'will not', 'I'm' -> 'I am', 'We're' -> 'We are', 'They've' -> 'They have').

**Step 2: Replace Content Words**
1. Replace specific Nouns/Entities with `[NP]`.
2. Replace specific Verbs with `[VP]`.
3. Replace specific Adjectives with `[ADJ]`.
4. Replace specific Adverbs with `[ADV]`.

**Step 3: Replace Personal Pronouns**
Replace PERSONAL pronouns (I, you, we, he, she, they) with `[NP]`.

**Exception - Dummy Subject "It":**
PRESERVE the pronoun 'It' ONLY if used as a dummy subject (e.g., 'It is clear that...', 'It seems that...', 'It was raining', 'It is important to note'). If 'It' refers to a specific entity (e.g., 'It crashed' referring to a car), replace it with `[NP]`.

**CRITICAL - You MUST preserve:**
- All auxiliary verbs: is, are, was, were, has, have, had, do, does, did, will, would, could, should, may, might, must, can, shall
- All connectors: however, but, and, or, because, since, although, though, thus, therefore, hence, consequently, furthermore, moreover, nevertheless, nonetheless, meanwhile, alternatively, specifically, particularly, notably, indeed, in fact
- All prepositions: in, on, at, to, for, of, with, by, from, into, onto, upon, within, without, throughout, during, before, after, above, below, between, among, through, across, around, over, under, near, far, beside, behind, beyond
- All articles: the, a, an
- All punctuation exactly as it appears

**Examples:**
Input: "The violent shift to capitalism did not bring freedom."
Output: "The [ADJ] [NP] to [NP] did not bring [NP]."

Input: "You'll understand that the system operates correctly."
Output: "[NP] will [VP] that the [NP] [VP] [ADV]."

Input: "It is clear that the approach works."
Output: "It is [ADJ] that the [NP] [VP]."

**Your task:** Convert the following text into a structural template."""

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

