"""Extract human writing patterns from author corpus.

Identifies patterns that signal human authorship:
- Sentence fragments
- Rhetorical questions
- Parenthetical asides (with dashes or parentheses)
- Colloquial phrases
- Extreme sentence length variation
- Creative punctuation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import re
from collections import Counter

from ..utils.nlp import get_nlp, split_into_sentences


@dataclass
class HumanPatterns:
    """Human writing patterns extracted from corpus."""

    # Sentence fragments (incomplete sentences)
    fragments: List[str] = field(default_factory=list)

    # Rhetorical questions
    questions: List[str] = field(default_factory=list)

    # Parenthetical asides (with dashes or parens)
    asides: List[str] = field(default_factory=list)

    # Em-dash patterns: "X—Y" or "X — Y"
    dash_patterns: List[str] = field(default_factory=list)

    # Colloquial/informal phrases
    colloquialisms: List[str] = field(default_factory=list)

    # Very short sentences (< 8 words)
    short_sentences: List[str] = field(default_factory=list)

    # Very long sentences (> 40 words)
    long_sentences: List[str] = field(default_factory=list)

    # Sentence-initial patterns (non-standard openings)
    unconventional_openers: List[str] = field(default_factory=list)

    # Statistics
    fragment_ratio: float = 0.0
    question_ratio: float = 0.0
    dash_ratio: float = 0.0
    short_ratio: float = 0.0
    long_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "fragments": self.fragments[:20],
            "questions": self.questions[:20],
            "asides": self.asides[:20],
            "dash_patterns": self.dash_patterns[:20],
            "colloquialisms": self.colloquialisms[:20],
            "short_sentences": self.short_sentences[:20],
            "long_sentences": self.long_sentences[:10],
            "unconventional_openers": self.unconventional_openers[:20],
            "fragment_ratio": self.fragment_ratio,
            "question_ratio": self.question_ratio,
            "dash_ratio": self.dash_ratio,
            "short_ratio": self.short_ratio,
            "long_ratio": self.long_ratio,
        }


class CorpusPatternExtractor:
    """Extract human writing patterns from author corpus."""

    def __init__(self):
        self._nlp = None

        # Patterns for detecting fragments
        self.fragment_indicators = {
            # No main verb patterns
            "noun_phrase_only",
            "prepositional_start",
            "participial_only",
        }

        # Common colloquial markers
        self.colloquial_markers = {
            "you know", "of course", "after all", "in fact",
            "to be sure", "indeed", "certainly", "surely",
            "well,", "so,", "now,", "look,", "see,",
            "crumbs", "throw", "bribe", "graft",
        }

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, paragraphs: List[str]) -> HumanPatterns:
        """Extract human writing patterns from corpus paragraphs.

        Args:
            paragraphs: List of paragraphs from author corpus.

        Returns:
            HumanPatterns with extracted patterns.
        """
        patterns = HumanPatterns()

        all_sentences = []
        for para in paragraphs:
            sentences = split_into_sentences(para)
            all_sentences.extend(sentences)

        total = len(all_sentences)
        if total == 0:
            return patterns

        for sentence in all_sentences:
            word_count = len(sentence.split())

            # Track sentence length extremes
            if word_count < 8:
                patterns.short_sentences.append(sentence)
            elif word_count > 40:
                patterns.long_sentences.append(sentence)

            # Detect questions
            if sentence.strip().endswith("?"):
                patterns.questions.append(sentence)

            # Detect fragments (no main verb)
            if self._is_fragment(sentence):
                patterns.fragments.append(sentence)

            # Detect dash patterns
            if "—" in sentence or " - " in sentence:
                patterns.dash_patterns.append(sentence)
                # Extract the aside part
                aside = self._extract_aside(sentence)
                if aside:
                    patterns.asides.append(aside)

            # Detect parenthetical asides
            if "(" in sentence and ")" in sentence:
                aside = self._extract_parenthetical(sentence)
                if aside:
                    patterns.asides.append(aside)

            # Detect colloquialisms
            lower = sentence.lower()
            for marker in self.colloquial_markers:
                if marker in lower:
                    patterns.colloquialisms.append(sentence)
                    break

            # Detect unconventional openers
            opener = self._get_opener(sentence)
            if opener and self._is_unconventional_opener(opener):
                patterns.unconventional_openers.append(sentence)

        # Calculate ratios
        patterns.fragment_ratio = len(patterns.fragments) / total
        patterns.question_ratio = len(patterns.questions) / total
        patterns.dash_ratio = len(patterns.dash_patterns) / total
        patterns.short_ratio = len(patterns.short_sentences) / total
        patterns.long_ratio = len(patterns.long_sentences) / total

        return patterns

    def _is_fragment(self, sentence: str) -> bool:
        """Check if sentence is a fragment (no main verb)."""
        # Quick heuristics before expensive NLP
        words = sentence.split()
        if len(words) < 3:
            return True  # Very short = likely fragment

        # Check for verb presence with spaCy
        doc = self.nlp(sentence)

        # Look for a main verb (ROOT)
        has_root_verb = any(
            token.dep_ == "ROOT" and token.pos_ == "VERB"
            for token in doc
        )

        # Also check for aux-only patterns like "What about X?"
        has_any_verb = any(token.pos_ in ("VERB", "AUX") for token in doc)

        # Fragment if no main verb and short
        if not has_root_verb and len(words) < 10:
            return True

        # Check for noun-phrase only patterns
        if not has_any_verb and len(words) < 15:
            return True

        return False

    def _extract_aside(self, sentence: str) -> Optional[str]:
        """Extract em-dash aside from sentence."""
        # Pattern: "main text—aside—more text" or "main text—aside."
        if "—" in sentence:
            parts = sentence.split("—")
            if len(parts) >= 2:
                # Return the middle part if exists
                return parts[1].strip()
        return None

    def _extract_parenthetical(self, sentence: str) -> Optional[str]:
        """Extract parenthetical aside from sentence."""
        match = re.search(r'\(([^)]+)\)', sentence)
        if match:
            return match.group(1)
        return None

    def _get_opener(self, sentence: str) -> Optional[str]:
        """Get the opening word/phrase of a sentence."""
        words = sentence.split()
        if words:
            return words[0].lower().rstrip(",")
        return None

    def _is_unconventional_opener(self, opener: str) -> bool:
        """Check if opener is unconventional (not standard subject)."""
        # Unconventional: starts with conjunction, adverb, interjection
        unconventional = {
            "and", "but", "or", "so", "yet", "for", "nor",
            "well", "now", "look", "see", "indeed", "certainly",
            "of", "in", "to", "from", "with", "by", "at",
            "what", "how", "why", "when", "where",
        }
        return opener in unconventional

    def get_burstiness_target(self, patterns: HumanPatterns) -> Tuple[float, float]:
        """Get target burstiness from patterns.

        Returns:
            Tuple of (short_ratio, long_ratio) to aim for.
        """
        # Aim for the author's natural ratios
        return (patterns.short_ratio, patterns.long_ratio)

    def get_pattern_samples(
        self,
        patterns: HumanPatterns,
        pattern_type: str,
        count: int = 5
    ) -> List[str]:
        """Get sample patterns of a specific type.

        Args:
            patterns: Extracted patterns.
            pattern_type: Type of pattern to get.
            count: Number of samples.

        Returns:
            List of sample sentences.
        """
        import random

        source = {
            "fragment": patterns.fragments,
            "question": patterns.questions,
            "aside": patterns.asides,
            "dash": patterns.dash_patterns,
            "short": patterns.short_sentences,
            "long": patterns.long_sentences,
            "opener": patterns.unconventional_openers,
        }.get(pattern_type, [])

        if not source:
            return []

        return random.sample(source, min(count, len(source)))
