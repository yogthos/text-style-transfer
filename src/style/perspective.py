"""Perspective transformation for style transfer.

Converts text between different narrative perspectives:
- first_person_singular: I/me/my
- first_person_plural: we/us/our
- third_person: he/she/they/it
- preserve: keep original perspective
"""

import re
from typing import Optional, Dict, List, Tuple
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class Perspective(Enum):
    """Narrative perspective options."""
    PRESERVE = "preserve"
    FIRST_PERSON_SINGULAR = "first_person_singular"
    FIRST_PERSON_PLURAL = "first_person_plural"
    THIRD_PERSON = "third_person"


# Pronoun mappings for each perspective transformation
# Format: (pattern, replacement) - case-insensitive
PRONOUN_MAPS = {
    # To first person singular
    Perspective.FIRST_PERSON_SINGULAR: {
        # From first person plural
        r'\bwe\b': 'I',
        r'\bus\b': 'me',
        r'\bour\b': 'my',
        r'\bours\b': 'mine',
        r'\bourselves\b': 'myself',
        # From third person
        r'\bhe\b': 'I',
        r'\bshe\b': 'I',
        r'\bthey\b': 'I',
        r'\bhim\b': 'me',
        r'\bher\b': 'me',
        r'\bthem\b': 'me',
        r'\bhis\b': 'my',
        r'\bhers\b': 'mine',
        r'\btheir\b': 'my',
        r'\btheirs\b': 'mine',
        r'\bhimself\b': 'myself',
        r'\bherself\b': 'myself',
        r'\bthemselves\b': 'myself',
    },

    # To first person plural
    Perspective.FIRST_PERSON_PLURAL: {
        # From first person singular
        r'\bI\b': 'we',
        r'\bme\b': 'us',
        r'\bmy\b': 'our',
        r'\bmine\b': 'ours',
        r'\bmyself\b': 'ourselves',
        # From third person
        r'\bhe\b': 'we',
        r'\bshe\b': 'we',
        r'\bthey\b': 'we',
        r'\bhim\b': 'us',
        r'\bher\b': 'us',
        r'\bthem\b': 'us',
        r'\bhis\b': 'our',
        r'\bhers\b': 'ours',
        r'\btheir\b': 'our',
        r'\btheirs\b': 'ours',
        r'\bhimself\b': 'ourselves',
        r'\bherself\b': 'ourselves',
        r'\bthemselves\b': 'ourselves',
    },

    # To third person (default to "they" for gender neutrality)
    Perspective.THIRD_PERSON: {
        # From first person singular
        r'\bI\b': 'they',
        r'\bme\b': 'them',
        r'\bmy\b': 'their',
        r'\bmine\b': 'theirs',
        r'\bmyself\b': 'themselves',
        # From first person plural
        r'\bwe\b': 'they',
        r'\bus\b': 'them',
        r'\bour\b': 'their',
        r'\bours\b': 'theirs',
        r'\bourselves\b': 'themselves',
    },
}

# Verb conjugations that need adjustment
# Only the most common verbs that change form
VERB_ADJUSTMENTS = {
    # First person -> Third person
    ('am', 'is'),
    ('are', 'is'),
    ("'m", "'s"),
    ("'re", "'s"),
    ('have', 'has'),
    ("'ve", "'s"),
    ('do', 'does'),
    ('was', 'was'),  # No change for past tense
    ('were', 'was'),
}


def detect_perspective(text: str) -> Perspective:
    """Detect the dominant perspective in text.

    Args:
        text: Input text to analyze.

    Returns:
        Detected Perspective enum value.
    """
    text_lower = text.lower()

    # Count perspective indicators
    first_singular = len(re.findall(r'\b(i|me|my|mine|myself)\b', text_lower))
    first_plural = len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text_lower))
    third_person = len(re.findall(
        r'\b(he|she|they|him|her|them|his|hers|their|theirs)\b', text_lower
    ))

    # Determine dominant perspective
    counts = {
        Perspective.FIRST_PERSON_SINGULAR: first_singular,
        Perspective.FIRST_PERSON_PLURAL: first_plural,
        Perspective.THIRD_PERSON: third_person,
    }

    max_perspective = max(counts, key=counts.get)

    if counts[max_perspective] == 0:
        return Perspective.PRESERVE

    logger.debug(f"Detected perspective: {max_perspective.value} "
                f"(1st-sing={first_singular}, 1st-pl={first_plural}, 3rd={third_person})")

    return max_perspective


def transform_perspective(
    text: str,
    target: Perspective,
    source: Optional[Perspective] = None,
) -> str:
    """Transform text to a target perspective.

    Args:
        text: Input text to transform.
        target: Target perspective to convert to.
        source: Optional source perspective (auto-detected if not provided).

    Returns:
        Transformed text in target perspective.
    """
    if target == Perspective.PRESERVE:
        return text

    if source is None:
        source = detect_perspective(text)

    if source == target:
        logger.debug("Source and target perspective match, no transformation needed")
        return text

    if source == Perspective.PRESERVE:
        logger.debug("No clear source perspective detected, skipping transformation")
        return text

    logger.debug(f"Transforming perspective: {source.value} -> {target.value}")

    # Get pronoun mappings for target perspective
    pronoun_map = PRONOUN_MAPS.get(target, {})

    result = text

    # Apply pronoun transformations
    for pattern, replacement in pronoun_map.items():
        # Handle case preservation
        def replace_with_case(match):
            original = match.group(0)
            if original.isupper():
                return replacement.upper()
            elif original[0].isupper():
                return replacement.capitalize()
            else:
                return replacement.lower()

        result = re.sub(pattern, replace_with_case, result, flags=re.IGNORECASE)

    # Handle verb adjustments for first -> third person
    if source in (Perspective.FIRST_PERSON_SINGULAR, Perspective.FIRST_PERSON_PLURAL):
        if target == Perspective.THIRD_PERSON:
            for first_form, third_form in VERB_ADJUSTMENTS:
                # Only replace when following converted pronoun
                # This is a simplified heuristic
                pattern = rf'\b(they|he|she|it)\s+{first_form}\b'
                replacement = rf'\1 {third_form}'
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def add_perspective_guidance_to_prompt(
    prompt: str,
    target_perspective: Perspective,
) -> str:
    """Add perspective guidance to a generation prompt.

    Args:
        prompt: Original prompt.
        target_perspective: Target perspective for generation.

    Returns:
        Prompt with perspective guidance added.
    """
    if target_perspective == Perspective.PRESERVE:
        return prompt

    guidance = {
        Perspective.FIRST_PERSON_SINGULAR: (
            "Write in FIRST PERSON SINGULAR (I/me/my). "
            "The narrator speaks directly about their own experience."
        ),
        Perspective.FIRST_PERSON_PLURAL: (
            "Write in FIRST PERSON PLURAL (we/us/our). "
            "The narrator speaks on behalf of a group."
        ),
        Perspective.THIRD_PERSON: (
            "Write in THIRD PERSON (they/them/their or he/she). "
            "The narrator describes others from outside."
        ),
    }

    perspective_hint = guidance.get(target_perspective, "")

    if perspective_hint:
        return f"{prompt}\n\nPERSPECTIVE: {perspective_hint}"

    return prompt


class PerspectiveTransformer:
    """Handles perspective transformation for style transfer pipeline."""

    def __init__(self, target_perspective: str = "preserve"):
        """Initialize the transformer.

        Args:
            target_perspective: Target perspective string from config.
        """
        try:
            self.target = Perspective(target_perspective)
        except ValueError:
            logger.warning(f"Invalid perspective '{target_perspective}', using 'preserve'")
            self.target = Perspective.PRESERVE

        self.source = None  # Auto-detected from input

    def detect_source(self, text: str) -> None:
        """Detect and cache the source perspective.

        Args:
            text: Source text to analyze.
        """
        self.source = detect_perspective(text)
        logger.info(f"Detected source perspective: {self.source.value}")

    def transform(self, text: str) -> str:
        """Transform text to target perspective.

        Args:
            text: Input text.

        Returns:
            Transformed text.
        """
        return transform_perspective(text, self.target, self.source)

    def enhance_prompt(self, prompt: str) -> str:
        """Add perspective guidance to generation prompt.

        Args:
            prompt: Original prompt.

        Returns:
            Enhanced prompt with perspective guidance.
        """
        return add_perspective_guidance_to_prompt(prompt, self.target)

    @property
    def is_active(self) -> bool:
        """Check if perspective transformation is active."""
        return self.target != Perspective.PRESERVE
