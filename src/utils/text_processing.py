"""Text processing utilities for style transfer pipeline."""

import re
from typing import List

def check_zipper_merge(prev_sent: str, new_sent: str) -> bool:
    """
    Returns True if a 'Stitch Glitch' (overlap/repetition) is detected.

    Detects three types of echo:
    - Full Echo: New sentence starts with entire previous sentence
    - Head Echo: Both sentences start with same 3+ words
    - Tail Echo: End of previous sentence matches start of new sentence

    Args:
        prev_sent: Previous sentence (or empty string)
        new_sent: Newly generated sentence

    Returns:
        True if echo/repetition detected, False otherwise
    """
    if not prev_sent or not new_sent:
        return False

    # Normalize to lowercase for comparison
    p_clean = prev_sent.strip().lower()
    n_clean = new_sent.strip().lower()

    if not p_clean or not n_clean:
        return False

    # 1. Full Echo (New starts with Prev)
    if n_clean.startswith(p_clean):
        return True

    # 2. Head Echo (Both start with same 3+ words)
    p_words = p_clean.split()
    n_words = n_clean.split()

    if len(p_words) >= 3 and len(n_words) >= 3:
        if p_words[:3] == n_words[:3]:
            return True

    # 3. Tail Echo (End of Prev matches Start of New)
    # Check if the first 3 words of New appear at the end of Prev
    if len(n_words) >= 3:
        start_trigram = " ".join(n_words[:3])
        # Check last 6 words of previous sentence
        prev_tail = " ".join(p_words[-6:]) if len(p_words) >= 6 else " ".join(p_words)
    if start_trigram in prev_tail:
        return True

    return False


def parse_variants_from_response(response: str, verbose: bool = False) -> List[str]:
    """
    Parse variants from LLM response using robust multi-format regex.

    Handles multiple output formats:
    - VAR: prefix (case insensitive)
    - Numbered lists: 1. or 1)
    - Bullet points: - or *
    - Sentence-like lines (fallback)
    - Entire response (final fallback)

    Args:
        response: LLM response text containing variants
        verbose: Enable verbose output

    Returns:
        List of parsed variant strings
    """
    if not response or not response.strip():
        return []

    variants = []
    lines = response.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Check for VAR: prefix (case insensitive) - handle with or without space
        line_upper = line.upper()
        if line_upper.startswith("VAR:"):
            # Strip "VAR:" prefix (4 characters) and any following whitespace
            clean_text = line[4:].lstrip()
            if clean_text:
                variants.append(clean_text)
            continue

        # 2. Check for numbered lists: 1. or 1)
        numbered_match = re.match(r'^(\d+[\.\)])\s+(.*)', line)
        if numbered_match:
            clean_text = numbered_match.group(2).strip()
            if clean_text:
                variants.append(clean_text)
            continue

        # 3. Check for bullet points: - or *
        if line.startswith("- ") or line.startswith("* "):
            clean_text = line[2:].strip()
            if clean_text:
                variants.append(clean_text)
            continue

        # Secondary parsing: Check for sentence-like lines (fallback)
        # Skip chatter lines
        line_lower = line.lower()
        if (line_lower.startswith("here is") or
            line_lower.startswith("generating") or
            line_lower.startswith("variant") or
            line_lower.startswith("option")):
            continue

        # If line looks like a sentence (starts with capital, ends with punctuation)
        # and isn't chatter, treat as variant
        if (len(line) > 10 and  # Reasonable length
            line[0].isupper() and  # Starts with capital
            line[-1] in ".!?\"'"):  # Ends with punctuation
            variants.append(line)

    # Tertiary fallback: If no variants found at all, treat entire response as single variant
    if not variants and response.strip():
        # Strip quotes and clean up
        clean_response = response.strip().strip('"').strip("'")
        if clean_response:
            variants.append(clean_response)

    # Final safety check: Strip any remaining "VAR:" prefixes that might have slipped through
    cleaned_variants = []
    for variant in variants:
        # Remove "VAR:" prefix if present (case insensitive, with or without space)
        variant_upper = variant.upper()
        if variant_upper.startswith("VAR:"):
            variant = variant[4:].lstrip()
        cleaned_variants.append(variant)

    return cleaned_variants

