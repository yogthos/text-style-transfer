"""Dynamic style extractor for RAG-driven style injection.

This module extracts style DNA (lexicon, tone, structure) from ChromaDB examples
to enable author-agnostic style transfer without hardcoding.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from src.generator.llm_provider import LLMProvider


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'style_extractor_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


class StyleExtractor:
    """Extracts style DNA from example texts using LLM analysis."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the style extractor.

        Args:
            config_path: Path to configuration file.
        """
        self.llm_provider = LLMProvider(config_path=config_path)
        self.config_path = config_path
        # Manual cache for style DNA extraction (key: hash of sorted example texts)
        self._dna_cache = {}

    def extract_style_dna(self, examples: List[str]) -> Dict[str, any]:
        """Extract style DNA from example texts.

        Analyzes the provided examples to extract:
        - Lexicon: 5-10 distinct words/phrases the author uses frequently
        - Tone: One adjective describing the voice
        - Structure: One rule about sentence structure

        Args:
            examples: List of example text strings from ChromaDB.

        Returns:
            Dictionary with:
            - lexicon: List[str] - Distinct words/phrases (5-10 items)
            - tone: str - One adjective describing voice
            - structure: str - One rule about sentence structure
        """
        if not examples:
            # Fallback if no examples
            return {
                "lexicon": [],
                "tone": "Neutral",
                "structure": "Standard sentence structure"
            }

        # Check cache: use hash of sorted example texts as key
        cache_key = hash(tuple(sorted(examples)))
        if cache_key in self._dna_cache:
            return self._dna_cache[cache_key]

        # Limit to first 3 examples for analysis (faster, more focused)
        analysis_examples = examples[:3]
        examples_text = "\n\n".join([f"Example {i+1}: \"{ex}\"" for i, ex in enumerate(analysis_examples)])

        system_prompt = _load_prompt_template("style_extractor_system.md")
        user_template = _load_prompt_template("style_extractor_user.md")
        user_prompt = user_template.format(
            num_examples=len(analysis_examples),
            examples_text=examples_text
        )

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Low temperature for consistent extraction
                max_tokens=300
            )

            # Clean and parse JSON response
            response = response.strip()

            # Extract JSON if wrapped in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)

            # Try to find JSON object in response
            json_match = re.search(r'\{[^{}]*"lexicon"[^{}]*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            # Parse JSON
            style_dna = json.loads(response)

            # Validate and normalize
            if not isinstance(style_dna.get("lexicon"), list):
                style_dna["lexicon"] = []
            if not isinstance(style_dna.get("tone"), str):
                style_dna["tone"] = "Neutral"
            if not isinstance(style_dna.get("structure"), str):
                style_dna["structure"] = "Standard sentence structure"

            # Ensure lexicon is list of strings, limit to 10
            lexicon = style_dna.get("lexicon", [])
            if isinstance(lexicon, list):
                lexicon = [str(item).strip().lower() for item in lexicon if item][:10]
            else:
                lexicon = []

            result = {
                "lexicon": lexicon,
                "tone": str(style_dna.get("tone", "Neutral")).strip(),
                "structure": str(style_dna.get("structure", "Standard sentence structure")).strip()
            }

            # Store in cache before returning
            self._dna_cache[cache_key] = result
            return result

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fallback: extract basic lexicon using simple frequency analysis
            fallback_result = self._fallback_extract(examples)
            # Cache fallback result too
            self._dna_cache[cache_key] = fallback_result
            return fallback_result

    def _fallback_extract(self, examples: List[str]) -> Dict[str, any]:
        """Fallback extraction using simple frequency analysis.

        Args:
            examples: List of example texts.

        Returns:
            Basic style DNA dictionary.
        """
        # Use existing library function for stop word filtering
        from collections import Counter
        from src.validator.semantic_critic import _get_significant_tokens

        # Extract significant words using existing spaCy-based filtering
        all_words = []
        for example in examples[:3]:
            significant_tokens = _get_significant_tokens(example)
            all_words.extend(significant_tokens)

        # Get top 10 most frequent
        word_freq = Counter(all_words)
        lexicon = [word for word, _ in word_freq.most_common(10)]

        return {
            "lexicon": lexicon,
            "tone": "Authoritative",  # Default fallback
            "structure": "Standard sentence structure"
        }

