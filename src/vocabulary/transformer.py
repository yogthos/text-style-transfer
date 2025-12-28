"""Vocabulary transformer for post-processing generated text.

Applies light vocabulary transformations:
- Replaces LLM-speak with author-preferred words
- Tracks word usage globally to prevent clustering
- Preserves entities and content words
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import re

from ..style.profile import VocabularyPalette
from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransformationStats:
    """Stats from vocabulary transformation."""
    replacements_made: int = 0
    words_checked: int = 0
    llm_speak_found: List[str] = field(default_factory=list)
    replacements_detail: Dict[str, str] = field(default_factory=dict)


class VocabularyTransformer:
    """Transform vocabulary in generated text.

    Light post-processing approach:
    - Only replace obvious LLM-speak, not content words
    - Preserve entities and proper nouns
    - Track global usage to avoid word clustering
    """

    def __init__(self, palette: VocabularyPalette):
        """Initialize transformer with vocabulary palette.

        Args:
            palette: Author's vocabulary palette with replacements.
        """
        self.palette = palette
        self._nlp = None

        # Track words used globally across the document
        self.used_words: Set[str] = set()

        # Build replacement map (lowercased for matching)
        self.replacements = {k.lower(): v for k, v in palette.llm_replacements.items()}

        logger.info(f"Initialized VocabularyTransformer with {len(self.replacements)} replacements")

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def transform(self, text: str) -> Tuple[str, TransformationStats]:
        """Transform vocabulary in text.

        Args:
            text: Generated text to transform.

        Returns:
            Tuple of (transformed_text, stats).
        """
        stats = TransformationStats()
        doc = self.nlp(text)

        # Collect replacements (position, original, replacement)
        replacements = []

        for token in doc:
            stats.words_checked += 1

            # Skip: entities, proper nouns, punctuation, short words
            if token.ent_type_ or token.pos_ == 'PROPN' or not token.is_alpha:
                continue
            if len(token.text) < 3:
                continue

            word_lower = token.text.lower()

            # Check if this is LLM-speak we can replace
            if word_lower in self.replacements:
                replacement = self.replacements[word_lower]

                # Skip if replacement was recently used (prevent clustering)
                if replacement in self.used_words:
                    # Try to find alternative from semantic cluster
                    alt = self._find_alternative(replacement)
                    if alt:
                        replacement = alt

                # Match case of original
                if token.text[0].isupper():
                    replacement = replacement.capitalize()

                replacements.append((token.idx, token.text, replacement))
                stats.llm_speak_found.append(word_lower)
                stats.replacements_detail[token.text] = replacement

                # Track usage
                self.used_words.add(replacement.lower())

        # Apply replacements in reverse order (to preserve positions)
        result = text
        for pos, original, replacement in sorted(replacements, key=lambda x: -x[0]):
            result = result[:pos] + replacement + result[pos + len(original):]
            stats.replacements_made += 1

        if stats.replacements_made > 0:
            logger.debug(f"Transformed {stats.replacements_made} words: {stats.replacements_detail}")

        return result, stats

    def _find_alternative(self, word: str) -> Optional[str]:
        """Find alternative word from semantic clusters."""
        # Check if word is in any cluster
        for cluster_key, cluster_words in self.palette.semantic_clusters.items():
            if word == cluster_key or word in cluster_words:
                # Find unused word in cluster
                all_cluster = [cluster_key] + cluster_words
                for alt in all_cluster:
                    if alt not in self.used_words:
                        return alt
        return None

    def reset_tracking(self):
        """Reset word usage tracking (call between documents)."""
        self.used_words.clear()

    def get_stats(self) -> Dict:
        """Get transformer statistics."""
        return {
            "total_replacements": len(self.replacements),
            "words_tracked": len(self.used_words),
            "tracked_words": list(self.used_words)[:20],  # Sample
        }


def create_vocabulary_transformer(palette: VocabularyPalette) -> VocabularyTransformer:
    """Create vocabulary transformer from palette."""
    return VocabularyTransformer(palette)
