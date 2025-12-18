"""LLM-based judge for ranking translation candidates.

This module provides an LLM Judge that ranks candidates based on quality
rather than numerical scores. The judge naturally prefers fluent, natural
English over stilted translations.
"""

import json
import re
from pathlib import Path
from typing import List, Optional
from src.generator.llm_provider import LLMProvider


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'judge_ranking.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


class LLMJudge:
    """LLM-based judge for ranking translation candidates."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the LLM Judge.

        Args:
            config_path: Path to configuration file.
        """
        self.llm_provider = LLMProvider(config_path=config_path)
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        judge_config = self.config.get("judge", {})
        self.temperature = judge_config.get("temperature", 0.1)  # Low temp for consistent ranking
        self.max_tokens = judge_config.get("max_tokens", 50)  # Short response (just a letter)

    def rank_candidates(
        self,
        source_text: str,
        candidates: List[str],
        style_dna: str = "",
        rhetorical_type: str = "OBSERVATION",
        verbose: bool = False
    ) -> int:
        """Rank candidates and return index of best one.

        Args:
            source_text: Original source text to translate.
            candidates: List of candidate translations (2-3 candidates).
            style_dna: Target style description.
            rhetorical_type: Rhetorical type (OBSERVATION, ARGUMENT, etc.).
            verbose: Enable verbose logging.

        Returns:
            Index of best candidate (0, 1, or 2), or -1 if all are unacceptable.
            If only 1 candidate provided, returns 0.
            If 0 candidates provided, returns -1.
        """
        if not candidates:
            if verbose:
                print("    Judge: No candidates provided")
            return -1

        if len(candidates) == 1:
            if verbose:
                print("    Judge: Only 1 candidate, returning it")
            return 0

        # Limit to 3 candidates (A, B, C)
        candidates = candidates[:3]
        num_candidates = len(candidates)

        # Load judge prompt template
        try:
            prompt_template = _load_prompt_template("judge_ranking.md")
        except FileNotFoundError:
            # Fallback if template doesn't exist yet
            if verbose:
                print("    Judge: Template not found, using fallback")
            return 0  # Fallback to first candidate

        # Format prompt with candidates
        candidate_labels = ["A", "B", "C"][:num_candidates]
        candidate_sections = []
        for label, candidate in zip(candidate_labels, candidates):
            candidate_sections.append(f"{label}: \"{candidate}\"")

        prompt = prompt_template.format(
            source_text=source_text,
            candidate_a=candidates[0] if num_candidates >= 1 else "",
            candidate_b=candidates[1] if num_candidates >= 2 else "",
            candidate_c=candidates[2] if num_candidates >= 3 else "",
            style_dna=style_dna or "Natural and clear",
            rhetorical_type=rhetorical_type
        )

        # System prompt for judge
        system_prompt = """You are an Expert Editor specializing in translation quality assessment.
Your task is to rank translation candidates based on accuracy, fluency, and style.
Be decisive and consistent in your judgments."""

        try:
            # Call LLM for ranking
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Parse response to extract letter (A, B, C, or NONE)
            response = response.strip().upper()

            # Extract letter from response
            # Look for patterns like "A", "Candidate A", "A is best", etc.
            letter_match = re.search(r'\b([ABC])\b', response)
            if letter_match:
                selected_letter = letter_match.group(1)
                # Map letter to index
                letter_to_index = {"A": 0, "B": 1, "C": 2}
                selected_index = letter_to_index.get(selected_letter, -1)

                # Only accept if the selected candidate exists and is not empty
                if selected_index >= 0 and selected_index < num_candidates:
                    # Check if candidate is not empty
                    if candidates[selected_index] and candidates[selected_index].strip():
                        if verbose:
                            print(f"    Judge: Selected Candidate {selected_letter} (index {selected_index})")
                        return selected_index
                    else:
                        if verbose:
                            print(f"    Judge: Selected Candidate {selected_letter} but it's empty, rejecting")
                        return -1

            # Check for "NONE" or rejection
            if "NONE" in response or "REJECT" in response or "ALL" in response.upper():
                if verbose:
                    print(f"    Judge: Rejected all candidates (response: {response})")
                return -1

            # Fallback: if we can't parse, return first candidate
            if verbose:
                print(f"    Judge: Could not parse response '{response}', defaulting to first candidate")
            return 0

        except Exception as e:
            if verbose:
                print(f"    Judge: Error during ranking: {e}, defaulting to first candidate")
            # Fallback to first candidate on error
            return 0

