"""Vocabulary analyzer for style transfer.

Analyzes the difference between author corpus vocabulary and common LLM output
to create boost/penalty lists for vocabulary control.
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional, Tuple
from collections import Counter

from .semantic_filter import SemanticFilter, WordClassification
from ..style.profile import AuthorStyleProfile
from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VocabularyAnalysis:
    """Result of vocabulary analysis for an author."""

    author_name: str

    # Words to BOOST (from author corpus, stylistically distinctive)
    boost_words: Dict[str, float] = field(default_factory=dict)
    # word -> boost strength (0.0 to 1.0)

    # Words to PENALIZE (LLM-speak not in author corpus)
    penalty_words: Dict[str, float] = field(default_factory=dict)
    # word -> penalty strength (0.0 to 1.0)

    # Statistics
    corpus_stylistic_words: int = 0
    llm_speak_penalized: int = 0
    content_words_preserved: int = 0

    def to_dict(self) -> Dict:
        return {
            "author_name": self.author_name,
            "boost_words": self.boost_words,
            "penalty_words": self.penalty_words,
            "corpus_stylistic_words": self.corpus_stylistic_words,
            "llm_speak_penalized": self.llm_speak_penalized,
            "content_words_preserved": self.content_words_preserved,
        }


class VocabularyAnalyzer:
    """Analyze vocabulary for style transfer control.

    Key responsibilities:
    1. Extract stylistic vocabulary from author corpus
    2. Identify LLM-speak words not in author's vocabulary
    3. Filter out semantic content words (nouns, verbs)
    4. Generate boost/penalty lists safe for manipulation
    """

    def __init__(self):
        self._nlp = None
        self.semantic_filter = SemanticFilter()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def analyze_profile(
        self,
        profile: AuthorStyleProfile,
        input_text: Optional[str] = None,
        boost_strength: float = 0.5,
        penalty_strength: float = 0.7,
    ) -> VocabularyAnalysis:
        """Analyze author profile to generate vocabulary control lists.

        Args:
            profile: Author's style profile with MFW frequencies.
            input_text: Input text being transferred (to preserve its content words).
            boost_strength: Default boost strength for author words (0-1).
            penalty_strength: Default penalty for LLM-speak words (0-1).

        Returns:
            VocabularyAnalysis with boost and penalty word lists.
        """
        analysis = VocabularyAnalysis(author_name=profile.author_name)

        # Get author's vocabulary from MFW
        mfw = profile.delta_profile.mfw_frequencies
        author_words = set(mfw.keys())

        # Get LLM-speak words to potentially penalize
        llm_speak = self.semantic_filter.get_llm_speak_penalty_words()

        # Find LLM-speak words NOT in author corpus - these get penalized
        llm_speak_not_in_corpus = llm_speak - author_words

        # Filter to only stylistically safe words
        safe_llm_words = self.semantic_filter.get_safe_words(
            llm_speak_not_in_corpus,
            input_text
        )

        # Add penalty words
        for word in safe_llm_words:
            analysis.penalty_words[word] = penalty_strength

        analysis.llm_speak_penalized = len(safe_llm_words)

        # Extract stylistic words from author corpus to boost
        author_stylistic = self._extract_stylistic_vocabulary(mfw, input_text)
        analysis.corpus_stylistic_words = len(author_stylistic)

        # Add boost words (scale by frequency in corpus)
        for word, freq in author_stylistic.items():
            # Higher frequency = stronger boost
            strength = min(boost_strength * (1 + freq * 10), 1.0)
            analysis.boost_words[word] = strength

        # Count content words preserved
        if input_text:
            doc = self.nlp(input_text.lower())
            content_words = {
                t.text for t in doc
                if t.pos_ in ("NOUN", "VERB", "PROPN") and not t.is_stop
            }
            analysis.content_words_preserved = len(content_words)

        logger.info(
            f"Vocabulary analysis for {profile.author_name}: "
            f"boost={len(analysis.boost_words)}, "
            f"penalty={len(analysis.penalty_words)}, "
            f"preserved={analysis.content_words_preserved}"
        )

        return analysis

    def _extract_stylistic_vocabulary(
        self,
        mfw: Dict[str, float],
        input_text: Optional[str] = None
    ) -> Dict[str, float]:
        """Extract stylistic words from author's MFW.

        Only returns words that are:
        - Stylistically distinctive (adverbs, connectors, etc.)
        - NOT in the input text (to avoid content leakage)
        - Relatively frequent in the corpus
        """
        # Filter MFW through semantic filter
        all_words = set(mfw.keys())
        safe_words = self.semantic_filter.get_safe_words(all_words, input_text)

        # Return with frequencies
        return {
            word: mfw[word]
            for word in safe_words
            if mfw.get(word, 0) > 0.001  # Only reasonably frequent words
        }

    def get_common_llm_patterns(self) -> Set[str]:
        """Get common LLM output patterns to avoid.

        These are phrases and words that make text sound AI-generated.
        """
        return self.semantic_filter.get_llm_speak_penalty_words()

    def analyze_text_vocabulary(self, text: str) -> Dict[str, int]:
        """Analyze vocabulary of a text for debugging/inspection.

        Returns word frequency counts categorized by safety.
        """
        doc = self.nlp(text.lower())

        # Count words by category
        safe_words = Counter()
        content_words = Counter()

        for token in doc:
            if not token.is_alpha or len(token.text) < 2:
                continue

            classification = self.semantic_filter.classify_word(token.text)
            if classification.is_safe_to_manipulate:
                safe_words[token.text] += 1
            else:
                content_words[token.text] += 1

        return {
            "safe_stylistic": dict(safe_words.most_common(20)),
            "content_preserved": dict(content_words.most_common(20)),
        }
