"""Semantic Translator for extracting neutral logical summaries.

This module converts input text into a "Neutral Logical Summary" to preserve
meaning without style interference.
"""

from pathlib import Path
from typing import Optional
from src.generator.llm_provider import LLMProvider


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'semantic_translator_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


class SemanticTranslator:
    """Converts text into neutral logical summaries."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the semantic translator.

        Args:
            config_path: Path to configuration file.
        """
        self.llm_provider = LLMProvider(config_path=config_path)
        self.config_path = config_path

        # Load config
        import json
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.semantic_config = self.config.get("semantic_translation", {})
        self.temperature = self.semantic_config.get("temperature", 0.3)
        self.max_tokens = self.semantic_config.get("max_tokens", 1000)

    def extract_neutral_summary(self, text: str, target_perspective: str = "auto") -> str:
        """Extract neutral logical summary from text.

        Args:
            text: Input text to neutralize
            target_perspective: Target perspective for the summary. Options:
                - "first_person_singular": Use I/Me/My
                - "first_person_plural": Use We/Us/Our
                - "third_person": Use The subject/The narrator
                - "auto": Detect from input text

        Returns:
            Neutral logical summary string
        """
        if not text or not text.strip():
            return text

        # Handle "auto" perspective by detecting from input
        if target_perspective == "auto":
            # Simple detection: look for first person pronouns
            text_lower = text.lower()
            if any(pronoun in text_lower for pronoun in [' i ', ' me ', ' my ', ' mine ', ' myself ']):
                target_perspective = "first_person_singular"
            elif any(pronoun in text_lower for pronoun in [' we ', ' us ', ' our ', ' ours ', ' ourselves ']):
                target_perspective = "first_person_plural"
            else:
                target_perspective = "third_person"

        # Map perspective to pronoun guidance
        perspective_guidance = {
            "first_person_singular": "Use 'I', 'Me', 'My', 'Myself', 'Mine'.",
            "first_person_plural": "Use 'We', 'Us', 'Our', 'Ourselves', 'Ours'.",
            "third_person": "Use 'The subject', 'The narrator', or specific names."
        }
        pronoun_guidance = perspective_guidance.get(target_perspective, "Use 'The subject', 'The narrator', or specific names.")

        # Load prompt templates
        try:
            system_prompt = _load_prompt_template("semantic_translator_system.md")
        except FileNotFoundError:
            # Fallback prompt
            system_prompt = "You are a semantic neutralizer. Extract logical meaning without style."

        try:
            user_template = _load_prompt_template("semantic_translator_user.md")
            user_prompt = user_template.format(
                text=text,
                target_perspective=target_perspective,
                pronoun_guidance=pronoun_guidance
            )
        except (FileNotFoundError, KeyError):
            # Fallback prompt
            user_prompt = f"""Distill this text into a neutral, logical summary. Preserve causal links. Remove all rhetoric.

**Perspective Constraint:**
You must write this summary from the **{target_perspective}** point of view.
{pronoun_guidance}
Do NOT use 'The text says' or 'The author describes'. Be direct. Preserve the subject's agency and perspective.

Text:
{text}"""

        # Call LLM
        response = self.llm_provider.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_type="editor",
            require_json=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.strip()

