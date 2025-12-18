"""Structural skeleton extractor for JIT structural templating.

This module extracts sentence skeletons from RAG samples by replacing
nouns, verbs, and adjectives with placeholders while preserving the
exact syntactic structure.
"""

import re
from typing import Optional
from src.generator.llm_provider import LLMProvider


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

**CRITICAL RULES:**
- You MUST replace ALL specific nouns, verbs, and adjectives with placeholders ([NP], [VP], [ADJ])
- Do NOT leave any specific content words like 'theory', 'knowledge', 'standpoint', 'practice', etc.
- Replace EVERY content word, no exceptions

**KEEP (Functional Grammar Only):**
- Prepositions: of, in, to, for, with, by, from, at, on, etc.
- Conjunctions: and, but, or, if, when, while, etc.
- Determiners: the, a, an, this, that, these, those
- Auxiliary verbs: is, are, was, were, has, have, had, will, would, could, should, may, might, must, can

**REPLACE (ALL Content Words):**
- ALL Nouns → [NP] (e.g., 'theory', 'knowledge', 'standpoint', 'practice' → [NP])
- ALL Verbs → [VP] (e.g., 'reinforce', 'affirm', 'serve' → [VP])
- ALL Adjectives → [ADJ] (e.g., 'primary', 'objective', 'dialectical' → [ADJ])

**Example:**
Input: "The standpoint of practice is the primary standpoint."
Output: "The [NP] of [NP] is the [ADJ] [NP]."
(NOT "The [NP] of practice..." - you must replace ALL content words)

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

