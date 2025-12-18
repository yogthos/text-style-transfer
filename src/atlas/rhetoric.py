"""Rhetorical indexing module for style atlas.

This module classifies text into rhetorical modes (DEFINITION, ARGUMENT,
OBSERVATION, IMPERATIVE) to enable better style example retrieval.
"""

from enum import Enum
from typing import Optional


class RhetoricalType(Enum):
    """Rhetorical modes for text classification."""
    DEFINITION = "DEFINITION"  # Explains what something is
    ARGUMENT = "ARGUMENT"  # Makes a claim/logic
    OBSERVATION = "OBSERVATION"  # Describes a state
    IMPERATIVE = "IMPERATIVE"  # Commands/instructions
    UNKNOWN = "UNKNOWN"  # Fallback for unclassifiable text


class RhetoricalClassifier:
    """Classifies text into rhetorical modes using heuristics and LLM."""

    def classify_heuristic(self, text: str) -> RhetoricalType:
        """Classify using simple heuristics (fast, no LLM call).

        Args:
            text: Text to classify.

        Returns:
            RhetoricalType classification.
        """
        if not text or not text.strip():
            return RhetoricalType.UNKNOWN

        text_lower = text.lower()

        # DEFINITION: "is a", "is an", "means", "refers to"
        definition_phrases = ["is a", "is an", "means", "refers to", "defined as", "is defined"]
        if any(phrase in text_lower for phrase in definition_phrases):
            return RhetoricalType.DEFINITION

        # IMPERATIVE: Commands, instructions (often start with verbs)
        imperative_starters = ("study", "learn", "consider", "remember", "note", "observe",
                              "understand", "recognize", "examine", "analyze", "think")
        first_word = text.strip().split()[0].lower() if text.strip().split() else ""
        if first_word in imperative_starters:
            return RhetoricalType.IMPERATIVE

        # ARGUMENT: "Therefore", "Thus", "Hence", "Because", "Since"
        argument_words = ["therefore", "thus", "hence", "because", "since", "consequently",
                         "accordingly", "as a result", "for this reason", "so"]
        if any(word in text_lower for word in argument_words):
            return RhetoricalType.ARGUMENT

        # Default to OBSERVATION (descriptive statements)
        return RhetoricalType.OBSERVATION

    def classify_llm(self, text: str, llm_provider) -> RhetoricalType:
        """Classify using LLM (more accurate, slower).

        Args:
            text: Text to classify.
            llm_provider: LLM provider instance for classification.

        Returns:
            RhetoricalType classification.
        """
        if not text or not text.strip():
            return RhetoricalType.UNKNOWN

        prompt = f"""Classify the following text into one of these rhetorical modes:
- DEFINITION: Explains what something is (e.g., "A revolution is...")
- ARGUMENT: Makes a claim/logic (e.g., "Therefore, we must...")
- OBSERVATION: Describes a state (e.g., "The enemy advances...")
- IMPERATIVE: Commands/Instructions (e.g., "Study the problem...")

Text: "{text}"

Respond with ONLY one word: DEFINITION, ARGUMENT, OBSERVATION, or IMPERATIVE."""

        try:
            response = llm_provider.call(
                prompt=prompt,
                system_prompt="You are a text classifier. Respond with only the classification word.",
                max_tokens=10
            )

            response_upper = response.strip().upper()
            for rtype in RhetoricalType:
                if rtype.value in response_upper:
                    return rtype
        except Exception:
            # If LLM call fails, fall back to heuristic
            return self.classify_heuristic(text)

        return RhetoricalType.UNKNOWN

