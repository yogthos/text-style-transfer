"""Global Context Analyzer for document-wide context extraction.

This module extracts the thesis, intent, and key terms from the full document
to provide context for paragraph fusion generation and evaluation.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from src.generator.llm_provider import LLMProvider


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'global_context_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


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
        - Counter-Arguments: List of positions/views the author opposes
        - Stance Markers: List of phrases showing how author positions cited figures

        Args:
            full_text: Full input document text.
            verbose: Whether to print debug information.

        Returns:
            Dict with keys: 'thesis', 'intent', 'keywords', 'counter_arguments', 'stance_markers'
        """
        if not full_text or not full_text.strip():
            return {'thesis': '', 'intent': 'informing', 'keywords': [], 'counter_arguments': [], 'stance_markers': []}

        # Truncate if text is too long (>50k chars) - keep first 10k and last 5k
        text_to_analyze = full_text
        if len(full_text) > 50000:
            if verbose:
                print(f"  Document too long ({len(full_text)} chars), truncating for analysis...")
            # Keep first 10k and last 5k chars
            text_to_analyze = full_text[:10000] + "\n\n[... document truncated ...]\n\n" + full_text[-5000:]

        system_prompt = _load_prompt_template("global_context_system.md")
        user_template = _load_prompt_template("global_context_user.md")
        user_prompt = user_template.format(text_to_analyze=text_to_analyze)

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
                return {'thesis': '', 'intent': 'informing', 'keywords': [], 'counter_arguments': [], 'stance_markers': []}

            # Validate and normalize response
            thesis = context.get('thesis', '').strip()
            intent = context.get('intent', 'informing').strip().lower()
            keywords = context.get('keywords', [])
            counter_arguments = context.get('counter_arguments', [])
            stance_markers = context.get('stance_markers', [])

            # Ensure intent is valid
            if intent not in ['persuading', 'informing', 'narrating']:
                intent = 'informing'

            # Ensure keywords is a list
            if not isinstance(keywords, list):
                keywords = []

            # Limit keywords to 5
            keywords = keywords[:5]

            # Ensure counter_arguments is a list (backward compatibility)
            if not isinstance(counter_arguments, list):
                counter_arguments = []

            # Ensure stance_markers is a list (backward compatibility)
            if not isinstance(stance_markers, list):
                stance_markers = []

            return {
                'thesis': thesis,
                'intent': intent,
                'keywords': keywords,
                'counter_arguments': counter_arguments,
                'stance_markers': stance_markers
            }

        except Exception as e:
            if verbose:
                print(f"  ⚠ Global context extraction failed: {e}, using defaults")
            # Return default context on failure
            return {'thesis': '', 'intent': 'informing', 'keywords': [], 'counter_arguments': [], 'stance_markers': []}

