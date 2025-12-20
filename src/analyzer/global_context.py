"""Global Context Analyzer for document-wide context extraction.

This module extracts the thesis, intent, and key terms from the full document
to provide context for paragraph fusion generation and evaluation.
"""

import json
from typing import Dict, Optional
from src.generator.llm_provider import LLMProvider


class GlobalContextAnalyzer:
    """Analyzes full documents to extract global context (thesis, intent, keywords)."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the global context analyzer.

        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.llm_provider = LLMProvider(config_path=config_path)

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.global_context_config = config.get("global_context", {})
        self.max_summary_tokens = self.global_context_config.get("max_summary_tokens", 500)

    def analyze_document(self, full_text: str, verbose: bool = False) -> Dict[str, any]:
        """Analyze full document to extract global context.

        Extracts:
        - Thesis: Main argument (max 2 sentences)
        - Intent: One of "persuading", "informing", "narrating"
        - Keywords: List of 5 central concepts that must remain consistent

        Args:
            full_text: Full input document text.
            verbose: Whether to print debug information.

        Returns:
            Dict with keys: 'thesis', 'intent', 'keywords'
        """
        if not full_text or not full_text.strip():
            return {'thesis': '', 'intent': 'informing', 'keywords': []}

        # Truncate if text is too long (>50k chars) - keep first 10k and last 5k
        text_to_analyze = full_text
        if len(full_text) > 50000:
            if verbose:
                print(f"  Document too long ({len(full_text)} chars), truncating for analysis...")
            # Keep first 10k and last 5k chars
            text_to_analyze = full_text[:10000] + "\n\n[... document truncated ...]\n\n" + full_text[-5000:]

        system_prompt = "You are a document analyst. Analyze the following text and extract its core structure. Return your analysis as valid JSON only."

        user_prompt = f"""Analyze the following text and extract:

1. **The Thesis:** What is the main argument or central theme? (Maximum 2 sentences)
2. **The Intent:** Is the author primarily persuading, informing, or narrating?
3. **Key Terminology:** List 5 central concepts or terms that must remain consistent throughout the document.

Text to analyze:
{text_to_analyze}

Return your analysis as JSON with this exact structure:
{{
    "thesis": "Main argument in 1-2 sentences",
    "intent": "persuading" or "informing" or "narrating",
    "keywords": ["concept1", "concept2", "concept3", "concept4", "concept5"]
}}"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,  # Low temperature for consistent analysis
                max_tokens=self.max_summary_tokens
            )

            # Parse JSON response
            try:
                # Try to extract JSON from response (may have extra text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    context = json.loads(json_str)
                else:
                    # Fallback: try parsing entire response
                    context = json.loads(response)
            except json.JSONDecodeError:
                if verbose:
                    print(f"  ⚠ Failed to parse JSON response, using defaults")
                return {'thesis': '', 'intent': 'informing', 'keywords': []}

            # Validate and normalize response
            thesis = context.get('thesis', '').strip()
            intent = context.get('intent', 'informing').strip().lower()
            keywords = context.get('keywords', [])

            # Ensure intent is valid
            if intent not in ['persuading', 'informing', 'narrating']:
                intent = 'informing'

            # Ensure keywords is a list
            if not isinstance(keywords, list):
                keywords = []

            # Limit keywords to 5
            keywords = keywords[:5]

            return {
                'thesis': thesis,
                'intent': intent,
                'keywords': keywords
            }

        except Exception as e:
            if verbose:
                print(f"  ⚠ Global context extraction failed: {e}, using defaults")
            # Return default context on failure
            return {'thesis': '', 'intent': 'informing', 'keywords': []}

