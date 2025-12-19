"""Structural skeleton extractor for JIT structural templating.

This module extracts sentence skeletons from RAG samples by replacing
nouns, verbs, and adjectives with placeholders while preserving the
exact syntactic structure.

Also provides paragraph rhythm extraction for structural cloning.
"""

import re
from typing import Optional, List, Dict
from src.generator.llm_provider import LLMProvider

# Rhetorical connectors that define sentence flow (case-insensitive matching)
RHETORICAL_OPENERS = {
    'but', 'however', 'thus', 'therefore', 'yet', 'conversely', 'although',
    'in contrast', 'nevertheless', 'nonetheless', 'moreover', 'furthermore',
    'consequently', 'hence', 'accordingly', 'indeed', 'in fact', 'specifically',
    'notably', 'significantly', 'importantly', 'crucially', 'meanwhile',
    'alternatively', 'additionally', 'similarly', 'likewise', 'instead',
    'rather', 'still', 'though', 'despite', 'regardless', 'nonetheless'
}


class Structuralizer:
    """Extracts structural skeletons from example sentences."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the structuralizer.

        Args:
            config_path: Path to configuration file.
        """
        self.llm_provider = LLMProvider(config_path=config_path)
        self.config_path = config_path

    def extract_skeleton(self, text: str, input_text: Optional[str] = None) -> str:
        """Extract structural skeleton from a sentence.

        Replaces Nouns with [NP], Verbs with [VP], Adjectives with [ADJ].
        Keeps all prepositions, conjunctions, and punctuation.

        Args:
            text: Input sentence to extract skeleton from.
            input_text: Optional original input text for pruning (if single sentence, truncate skeleton to first sentence).

        Returns:
            Skeleton template with placeholders.
        """
        if not text or not text.strip():
            return ""

        system_prompt = """You are a linguistic structure analyzer. Your task is to extract the syntactic skeleton of a sentence by replacing ALL content words with placeholders while preserving ONLY functional grammar words."""

        user_prompt = f"""Extract the structural skeleton by replacing ALL specific nouns, verbs, and adjectives with generic placeholders.

**CRITICAL RULES (GHOST WORD BAN):**
- You MUST replace ALL specific nouns, verbs, and adjectives with placeholders ([NP], [VP], [ADJ])
- Do NOT leave any specific content words like 'theory', 'knowledge', 'standpoint', 'practice', 'ideas', 'skies', 'mind', etc.
- Replace EVERY content word, no exceptions
- If you see words like 'ideas', 'skies', 'mind', 'theory', 'knowledge', 'correct', 'innate', they MUST become [NP] or [ADJ] or [VP]
- Only keep strictly rhetorical connectors (but, however, thus, therefore, yet, etc.) and functional grammar words

**KEEP (Functional Grammar Only):**
- Prepositions: of, in, to, for, with, by, from, at, on, etc.
- Conjunctions: and, but, or, if, when, while, etc.
- Determiners: the, a, an, this, that, these, those
- Auxiliary verbs: is, are, was, were, has, have, had, will, would, could, should, may, might, must, can
- Rhetorical connectors: but, however, thus, therefore, yet, conversely, nevertheless, nonetheless, moreover, furthermore, consequently, hence, accordingly, indeed, in fact, specifically, notably, significantly, importantly, crucially, meanwhile, alternatively, additionally, similarly, likewise, instead

**REPLACE (ALL Content Words - NO EXCEPTIONS):**
- ALL Nouns → [NP] (e.g., 'theory', 'knowledge', 'standpoint', 'practice', 'ideas', 'skies', 'mind' → [NP])
- ALL Verbs → [VP] (e.g., 'reinforce', 'affirm', 'serve', 'come', 'drop' → [VP])
- ALL Adjectives → [ADJ] (e.g., 'primary', 'objective', 'dialectical', 'correct', 'innate' → [ADJ])

**Example:**
Input: "The standpoint of practice is the primary standpoint."
Output: "The [NP] of [NP] is the [ADJ] [NP]."
(NOT "The [NP] of practice..." - you must replace ALL content words)

Input: "Where do correct ideas come from? Do they drop from the skies? Are they innate in the mind?"
Output: "Where [VP] [ADJ] [NP] [VP] from? [VP] [NP] [VP] from the [NP]? [VP] [NP] [ADJ] in the [NP]?"
(NOT "Where do correct ideas come from? Do they drop from the skies?" - replace ALL content words including 'ideas', 'skies', 'mind', 'correct', 'innate')

Input: "{text}"

Output ONLY the skeleton with placeholders, no explanations:"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,  # Low temperature for consistent extraction
                max_tokens=200
            )

            # Clean response
            skeleton = response.strip()

            # Remove quotes if present
            skeleton = re.sub(r'^["\']|["\']$', '', skeleton)
            skeleton = skeleton.strip()

            # SKELETON PRUNING: If input is single-sentence, truncate skeleton to first sentence delimiter
            if input_text:
                input_sentences = re.split(r'[.!?]+', input_text.strip())
                input_sentences = [s.strip() for s in input_sentences if s.strip()]
                if len(input_sentences) == 1:
                    # Input is single sentence - truncate skeleton to first sentence delimiter
                    # Find first sentence-ending punctuation in skeleton
                    for delimiter in ['.', '!', '?']:
                        if delimiter in skeleton:
                            # Truncate at first sentence delimiter
                            delimiter_pos = skeleton.find(delimiter)
                            skeleton = skeleton[:delimiter_pos + 1].strip()
                            break

            # Validate: should contain at least one placeholder
            if '[NP]' not in skeleton and '[VP]' not in skeleton and '[ADJ]' not in skeleton:
                # Fallback: return empty to indicate failure
                return ""

            # Ghost Word Ban: Check if skeleton contains semantic words that should be slots
            # Common semantic words that should NOT appear in skeletons
            semantic_words = [
                'ideas', 'skies', 'mind', 'theory', 'knowledge', 'practice', 'standpoint',
                'correct', 'innate', 'drop', 'come', 'reinforce', 'affirm', 'serve',
                'primary', 'objective', 'dialectical', 'raw', 'clear', 'good', 'bad'
            ]
            skeleton_lower = skeleton.lower()
            found_semantic_words = [word for word in semantic_words if word in skeleton_lower]

            if found_semantic_words:
                # Reject skeleton with ghost words - return empty to trigger fallback
                return ""

            return skeleton

        except Exception as e:
            # On error, return empty skeleton (will trigger fallback)
            return ""

    def count_skeleton_slots(self, skeleton: str) -> int:
        """Count the number of placeholder slots in a skeleton.

        Args:
            skeleton: Skeleton template string.

        Returns:
            Number of placeholder slots ([NP], [VP], [ADJ]).
        """
        if not skeleton:
            return 0

        # Count all placeholders
        np_count = len(re.findall(r'\[NP\]', skeleton))
        vp_count = len(re.findall(r'\[VP\]', skeleton))
        adj_count = len(re.findall(r'\[ADJ\]', skeleton))

        return np_count + vp_count + adj_count

    def adapt_skeleton(self, skeleton: str, target_word_count: int) -> str:
        """Adapt skeleton to target word count by compressing or expanding.

        If skeleton has too many slots, simplify it. If too few, expand it.
        Preserves the author's voice and connectors.

        Args:
            skeleton: Original skeleton template.
            target_word_count: Target number of slots (word count proxy).

        Returns:
            Adapted skeleton template, or original if adaptation fails.
        """
        if not skeleton:
            return skeleton

        current_slots = self.count_skeleton_slots(skeleton)

        # If skeleton is within acceptable range, return as-is
        if 0.5 * target_word_count <= current_slots <= 2.0 * target_word_count:
            return skeleton

        system_prompt = """You are a linguistic structure adapter. Your task is to modify sentence skeletons to match target complexity while preserving the author's distinctive voice and structural connectors."""

        if current_slots > target_word_count * 2:
            # Too long: compress
            user_prompt = f"""This sentence structure is too long ({current_slots} slots). Simplify it to approximately {target_word_count} slots while keeping the author's voice and connectors.

**Original Skeleton:** "{skeleton}"

**Instructions:**
- Reduce the number of [NP], [VP], and [ADJ] placeholders to approximately {target_word_count}
- Keep ALL prepositions, conjunctions, articles, and structural words exactly as they are
- Preserve the author's distinctive voice and connector style
- Simplify complex clauses but maintain the core structure

**Output:** Return ONLY the simplified skeleton with placeholders, no explanations:"""
        else:
            # Too short: expand
            user_prompt = f"""This sentence structure is too short ({current_slots} slots). Expand it to approximately {target_word_count} slots using the author's typical elaboration style.

**Original Skeleton:** "{skeleton}"

**Instructions:**
- Increase the number of [NP], [VP], and [ADJ] placeholders to approximately {target_word_count}
- Keep ALL existing prepositions, conjunctions, articles, and structural words
- Add elaboration in the author's typical style (e.g., additional clauses, descriptive phrases)
- Maintain the core structure while expanding complexity

**Output:** Return ONLY the expanded skeleton with placeholders, no explanations:"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Low temperature for consistent adaptation
                max_tokens=300
            )

            # Clean response
            adapted = response.strip()
            adapted = re.sub(r'^["\']|["\']$', '', adapted)
            adapted = adapted.strip()

            # Validate: should contain at least one placeholder
            if '[NP]' not in adapted and '[VP]' not in adapted and '[ADJ]' not in adapted:
                # Adaptation failed, return original
                return skeleton

            return adapted

        except Exception:
            # On error, return original skeleton
            return skeleton

def extract_paragraph_rhythm(text: str) -> List[Dict]:
        """Extract paragraph rhythm map from a text example.

        Analyzes sentence structure (length, type, opener) to create a rhythm
        template that can be used to force generated text to match human patterns.

        Args:
            text: Input paragraph text to analyze.

        Returns:
            List of dictionaries, each representing a sentence specification:
            [{'length': 'short', 'type': 'question', 'opener': None}, ...]
        """
        if not text or not text.strip():
            return []

        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = sent_tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]

        rhythm_map = []

        for sentence in sentences:
            if not sentence:
                continue

            # Count words (simple split)
            words = sentence.split()
            word_count = len(words)

            # Determine length class
            if word_count < 10:
                length = 'short'
            elif word_count <= 25:
                length = 'medium'
            else:
                length = 'long'

            # Determine type
            sentence_lower = sentence.lower()
            if sentence.rstrip().endswith('?'):
                sent_type = 'question'
            elif 'if' in sentence_lower and ('then' in sentence_lower or ',' in sentence):
                # Check for conditional structure (if...then or if...,)
                sent_type = 'conditional'
            else:
                sent_type = 'standard'

            # Determine opener (only if it's a rhetorical connector)
            opener = None
            if words:
                first_word = words[0].rstrip('.,!?;:').lower()
                if first_word in RHETORICAL_OPENERS:
                    # Preserve original capitalization from sentence
                    opener = words[0].rstrip('.,!?;:')

            rhythm_map.append({
                'length': length,
                'type': sent_type,
                'opener': opener
            })

        return rhythm_map


def generate_structure_signature(rhythm_map: List[Dict]) -> str:
    """Generate normalized signature for a rhythm map.

    Creates a string representation that captures:
    - Sentence count
    - First sentence opener (if any)
    - Pattern type (conditional, declarative, question, mixed)
    - Length distribution pattern (short-heavy, long-heavy, balanced)

    Args:
        rhythm_map: List of sentence specifications from extract_paragraph_rhythm()

    Returns:
        Normalized signature string, e.g., "3_sent_if_conditional_long-heavy"
    """
    if not rhythm_map:
        return "0_sent_none_none_none"

    count = len(rhythm_map)
    opener = rhythm_map[0].get('opener', 'none') or 'none'
    opener = opener.lower() if opener else 'none'

    # Determine pattern type
    conditional_count = sum(1 for r in rhythm_map if r.get('type') == 'conditional')
    question_count = sum(1 for r in rhythm_map if r.get('type') == 'question')

    if conditional_count > len(rhythm_map) * 0.5:
        pattern_type = "conditional"
    elif question_count > 0:
        pattern_type = "question"
    elif all(r.get('type') == 'standard' for r in rhythm_map):
        pattern_type = "declarative"
    else:
        pattern_type = "mixed"

    # Determine length pattern
    lengths = [r.get('length', 'medium') for r in rhythm_map]
    short_count = lengths.count('short')
    long_count = lengths.count('long')

    if short_count > len(rhythm_map) * 0.5:
        length_pattern = "short-heavy"
    elif long_count > len(rhythm_map) * 0.5:
        length_pattern = "long-heavy"
    else:
        length_pattern = "balanced"

    return f"{count}_sent_{opener}_{pattern_type}_{length_pattern}"

