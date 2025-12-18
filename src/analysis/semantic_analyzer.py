"""Semantic analyzer for extracting atomic propositions from text.

This module provides functionality to decompose text into atomic propositions,
stripping away style and connectors to extract core factual statements.
"""

import json
import re
from typing import List, Optional
from src.generator.llm_provider import LLMProvider


class PropositionExtractor:
    """Extracts atomic propositions from text using LLM-based decomposition."""

    PROPOSITION_EXTRACTION_PROMPT = """You are a semantic analyzer. Your task is to extract every distinct fact or claim from the given text as standalone simple sentences.

### INSTRUCTIONS:
1. Extract every distinct factual statement or claim from the text.
2. Remove all style, connectors, rhetorical flourishes, and transitional phrases.
3. Convert each fact into a simple, standalone sentence.
4. Preserve the core meaning but simplify the language.
5. Do NOT combine multiple facts into one sentence.
6. Do NOT add information that wasn't in the original text.
7. **CRITICAL: PRESERVE all citation references** (e.g., `[^1]`, `[^2]`) and attach them to the specific fact they support. Do NOT strip citations from the propositions. If a fact has a citation in the original text, include it in the extracted proposition.

### INPUT TEXT:
{text}

### OUTPUT FORMAT:
Output a JSON array of strings, where each string is an atomic proposition (with citations preserved if present).

Example:
Input: "Stars burn [^1]. They eventually die [^2]. The universe expands."
Output: ["Stars burn [^1]", "Stars eventually die [^2]", "The universe expands"]

### OUTPUT:"""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the proposition extractor.

        Args:
            config_path: Path to configuration file for LLM provider.
        """
        self.llm_provider = LLMProvider(config_path=config_path)

    def extract_atomic_propositions(self, text: str) -> List[str]:
        """Extract atomic propositions from text.

        Args:
            text: Input text to analyze (can be paragraph or multiple sentences).

        Returns:
            List of atomic proposition strings (standalone factual statements).
        """
        if not text or not text.strip():
            return []

        # Format the prompt
        prompt = self.PROPOSITION_EXTRACTION_PROMPT.format(text=text.strip())

        try:
            # Call LLM with JSON mode
            response = self.llm_provider.call(
                system_prompt="You are a semantic analyzer that extracts atomic propositions from text. Output only valid JSON arrays.",
                user_prompt=prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3  # Low temperature for consistent extraction
            )

            # Parse JSON response
            propositions = self._parse_json_response(response)

            # Clean and validate propositions
            cleaned = [p.strip() for p in propositions if p and p.strip()]

            return cleaned if cleaned else [text.strip()]  # Fallback to original if extraction fails

        except Exception as e:
            # Fallback: if LLM extraction fails, try simple sentence splitting
            return self._fallback_extraction(text)

    def _parse_json_response(self, response: str) -> List[str]:
        """Parse JSON response from LLM.

        Args:
            response: LLM response string (may contain JSON or markdown).

        Returns:
            List of proposition strings.
        """
        # Try to extract JSON array from response
        # Handle cases where LLM wraps JSON in markdown code blocks
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                propositions = json.loads(json_match.group(0))
                if isinstance(propositions, list):
                    return [str(p) for p in propositions]
            except json.JSONDecodeError:
                pass

        # Try parsing entire response as JSON
        try:
            propositions = json.loads(response.strip())
            if isinstance(propositions, list):
                return [str(p) for p in propositions]
        except json.JSONDecodeError:
            pass

        # If all parsing fails, return empty list
        return []

    def _fallback_extraction(self, text: str) -> List[str]:
        """Fallback extraction using simple sentence splitting.

        Args:
            text: Input text.

        Returns:
            List of sentences (as atomic propositions).
        """
        # Simple sentence splitting as fallback
        sentences = re.split(r'[.!?]+\s+', text)
        cleaned = [s.strip() for s in sentences if s.strip()]
        return cleaned if cleaned else [text.strip()]

