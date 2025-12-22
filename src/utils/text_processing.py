"""Text processing utilities for style transfer pipeline."""

import re
from typing import List, Dict, Optional

def check_zipper_merge(prev_sent: str, new_sent: str) -> bool:
    """
    Returns True if a 'Stitch Glitch' (overlap/repetition) is detected.

    Detects three types of echo:
    - Full Echo: New sentence starts with entire previous sentence
    - Head Echo: Both sentences start with same 2+ words
    - Tail Echo: End of previous sentence matches start of new sentence (word overlap)

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

    # Strip punctuation from words for better matching
    def strip_punctuation(word: str) -> str:
        """Remove punctuation from word."""
        return re.sub(r'[^\w\s]', '', word)

    # 1. Full Echo (New starts with Prev)
    if n_clean.startswith(p_clean):
        return True

    # 2. Head Echo (Both start with same 2+ words, ignoring punctuation)
    p_words = [strip_punctuation(w) for w in p_clean.split()]
    n_words = [strip_punctuation(w) for w in n_clean.split()]

    # Remove empty strings from punctuation stripping
    p_words = [w for w in p_words if w]
    n_words = [w for w in n_words if w]

    if len(p_words) >= 2 and len(n_words) >= 2:
        # Check if first 2 words match
        if p_words[:2] == n_words[:2]:
            return True
        # Also check first 3 words if both sentences have at least 3 words
        if len(p_words) >= 3 and len(n_words) >= 3:
            if p_words[:3] == n_words[:3]:
                return True

    # 3. Tail Echo (End of Prev matches Start of New)
    # Check if the start of new sentence overlaps with the end of previous sentence
    if len(n_words) >= 2 and len(p_words) >= 2:
        # Get first 2-3 words of new sentence (normalized, no punctuation)
        new_start_words = n_words[:min(3, len(n_words))]
        # Get last 4-6 words of previous sentence (normalized, no punctuation)
        prev_tail_words = p_words[-min(6, len(p_words)):]

        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "was", "are", "were", "it", "this", "that", "these", "those"}

        # Strong signal: First word of new appears as last word of prev (and is not common)
        # This catches cases like "The doors opened." -> "Opened, the room..."
        if len(p_words) >= 1 and n_words[0] == p_words[-1] and n_words[0] not in common_words:
            return True

        # Strong signal: First word of new appears in last 2 words of prev (and is not common)
        if n_words[0] in prev_tail_words[-2:] and n_words[0] not in common_words:
            # Additional check: make sure it's not just a coincidence
            # Require that at least one more word from new_start_words appears in prev_tail_words
            if len(new_start_words) >= 2:
                overlap_count = sum(1 for w in new_start_words if w in prev_tail_words)
                if overlap_count >= 2:
                    return True

        # Check for consecutive word overlap at the END of previous sentence
        # Look for 2+ consecutive words from start of new appearing at the end of prev
        # This catches cases where the end of prev sentence is repeated at start of new
        for i in range(len(new_start_words) - 1):
            bigram = (new_start_words[i], new_start_words[i + 1])
            # Check if this bigram appears at the END of prev_tail_words (last 3 words)
            # This ensures we're detecting tail echo, not just general similarity
            tail_end = prev_tail_words[-3:] if len(prev_tail_words) >= 3 else prev_tail_words
            for j in range(len(tail_end) - 1):
                if (tail_end[j], tail_end[j + 1]) == bigram:
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


def get_semantic_replacement(forbidden_word: str, author_palette: Dict[str, List[str]], nlp_model) -> Optional[str]:
    """
    Finds a replacement for 'forbidden_word' using the author's vocabulary palette.
    Enforces POS matching to preserve grammar.

    Args:
        forbidden_word: Word to replace (case-insensitive)
        author_palette: Dictionary with categorized vocabulary lists:
            - "general": List of general vocabulary words
            - "sensory_verbs": List of sensory verb lemmas
            - "connectives": List of connective words
            - "intensifiers": List of intensifier adverbs
        nlp_model: spaCy language model for POS tagging and vector similarity

    Returns:
        Best matching replacement word (as lemma/base form), or None if no good match found
    """
    if not forbidden_word or not author_palette or not nlp_model:
        return None

    # Parse forbidden word with spaCy to get POS tag
    try:
        target_doc = nlp_model(forbidden_word)
        if not target_doc:
            return None
        target_token = target_doc[0]
        target_pos = target_token.pos_
    except Exception:
        return None

    # 1. Determine which palette category to check
    candidates = []

    if target_pos == "VERB":
        # Note: verbs will be skipped in cleanup, but we still provide candidates
        candidates = author_palette.get("sensory_verbs", []) + author_palette.get("general", [])
    elif target_pos in ["ADV", "SCONJ", "CCONJ"]:
        # Check if it's an intensifier (advmod dependency on ADJ head)
        if target_token.dep_ == "advmod":
            candidates = author_palette.get("intensifiers", []) + author_palette.get("connectives", [])
        else:
            candidates = author_palette.get("connectives", [])
    else:
        # Default: general palette
        candidates = author_palette.get("general", [])

    # Fallback if author palette is empty for this category
    if not candidates:
        # Generic safe fallbacks by POS
        fallbacks = {
            "ADJ": ["clear", "simple", "real", "human", "dark", "light", "heavy"],
            "NOUN": ["thing", "part", "system", "way", "fact", "world"],
            "VERB": ["is", "make", "go", "see", "know", "use"],
            "ADV": ["now", "then", "just", "well"]
        }
        candidates = fallbacks.get(target_pos, ["thing"])

    # 2. Vector Search
    best_word = None
    best_score = -1.0

    for cand_str in candidates:
        if not cand_str:
            continue

        try:
            cand_doc = nlp_model(cand_str)
            if not cand_doc:
                continue
            cand_token = cand_doc[0]

            # STRICT Constraint: Must match Part of Speech (prevent "vast" -> "ocean")
            if cand_token.pos_ != target_pos:
                continue

            # Don't suggest the forbidden word itself
            if cand_str.lower() == forbidden_word.lower():
                continue

            # Calculate similarity using spaCy vectors
            # Both tokens must have vectors for similarity to work
            if target_token.has_vector and cand_token.has_vector:
                similarity = target_token.similarity(cand_token)

                # We want "similar meaning" but not "identical synonym" if the original was bad.
                # But here we just want the closest valid author word.
                if similarity > best_score:
                    best_score = similarity
                    best_word = cand_str
        except Exception:
            # Skip candidate if processing fails
            continue

    # 3. Threshold Check (Don't pick random junk)
    if best_score < 0.3:
        # If no author word is close enough, use a very neutral stopgap
        if target_pos == "ADV":
            return None  # Return None for adverbs to trigger deletion
        elif target_pos == "NOUN":
            return "structure"
        elif target_pos == "ADJ":
            return "large"
        else:
            return None

    return best_word

