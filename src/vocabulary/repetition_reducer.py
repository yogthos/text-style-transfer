"""Repetition reducer for post-processing generated text.

Lightweight post-processing that:
- Tracks word usage across the document
- Replaces overused words with synonyms
- Preserves entities and proper nouns

This is a simplified replacement for VocabularyTransformer that doesn't
require a style profile, making it compatible with the LoRA pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Common LLM-speak words to watch for
LLM_SPEAK = {
    "utilize": "use",
    "utilization": "use",
    "implementation": "setup",
    "functionality": "feature",
    "leverage": "use",
    "facilitate": "help",
    "comprehensive": "full",
    "robust": "strong",
    "scalable": "flexible",
    "streamline": "simplify",
    "optimize": "improve",
    "enhance": "improve",
    "innovative": "new",
    "cutting-edge": "modern",
    "state-of-the-art": "latest",
    "paradigm": "model",
    "synergy": "cooperation",
    "holistic": "complete",
    "proactive": "active",
    "impactful": "effective",
    "actionable": "practical",
    "deliverables": "results",
    "stakeholders": "people involved",
    "bandwidth": "capacity",
    "circle back": "return to",
    "deep dive": "detailed look",
    "moving forward": "next",
    # Fix weird Qwen vocabulary substitutions
    "ticker": "watch",
    "vigil": "watch",
    "lookout": "watch",
    "sentinel": "watch",
    "sentry": "watch",
    "picket": "watch",
    "spotter": "watch",
    "scout": "watch",
    "timekeeper": "watch",
    "cogwheel": "gear",
    "gearing": "gear",
    "geartrain": "gears",
    "paraphernalia": "gear",
    "appurtenance": "part",
    "unmarried": "single",
    "undivided": "single",
    "exclusive": "single",
    "unharmed": "whole",
    "unhurt": "whole",
    "unscathed": "whole",
    "peregrine": "mobile",
    "roving": "mobile",
    "wandering": "mobile",
    "nomadic": "mobile",
    "earphone": "phone",
    "earpiece": "phone",
    "headphone": "phone",
    "telephony": "phone",
    "macrocosm": "world",
    "cosmos": "universe",
    "creation": "world",
    "existence": "world",
    "domain": "world",
    "corporeal": "physical",
    "at the end of the day": "ultimately",
    "think outside the box": "be creative",
}


@dataclass
class ReductionStats:
    """Statistics from repetition reduction."""
    words_checked: int = 0
    replacements_made: int = 0
    overused_words: List[str] = field(default_factory=list)
    replacements_detail: Dict[str, str] = field(default_factory=dict)


class RepetitionReducer:
    """Reduce word repetition in generated text.

    Tracks word usage across the document and replaces words that
    appear too frequently with synonyms from WordNet or a simple
    synonym list.

    Usage:
        reducer = RepetitionReducer(threshold=3)

        # Process each paragraph
        for para in paragraphs:
            para, stats = reducer.reduce(para)

        # Reset between documents
        reducer.reset()
    """

    def __init__(
        self,
        threshold: int = 3,
        use_wordnet: bool = True,
        synonym_replacement: bool = False,  # Disable by default - causes problems
    ):
        """Initialize the reducer.

        Args:
            threshold: Number of uses before a word is considered overused.
            use_wordnet: Whether to use WordNet for synonyms.
            synonym_replacement: Whether to replace overused words with synonyms.
                               Defaults to False as this often introduces weird vocabulary.
        """
        self.threshold = threshold
        self.use_wordnet = use_wordnet
        self.synonym_replacement = synonym_replacement
        self._nlp = None
        self._wordnet = None

        # Track word usage across document
        self.word_counts: Counter = Counter()
        self.used_replacements: Set[str] = set()

        # Simple synonym cache
        self._synonym_cache: Dict[str, List[str]] = {}

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    @property
    def wordnet(self):
        """Lazy load WordNet."""
        if self._wordnet is None and self.use_wordnet:
            try:
                from nltk.corpus import wordnet
                self._wordnet = wordnet
            except ImportError:
                logger.warning("WordNet not available, using simple synonyms")
                self.use_wordnet = False
        return self._wordnet

    def reduce(self, text: str) -> Tuple[str, ReductionStats]:
        """Reduce repetition in text.

        Args:
            text: Text to process.

        Returns:
            Tuple of (processed_text, stats).
        """
        stats = ReductionStats()
        doc = self.nlp(text)

        # First pass: count words
        for token in doc:
            if self._should_track(token):
                word_lower = token.lemma_.lower()
                self.word_counts[word_lower] += 1

        # Second pass: collect replacements
        replacements = []

        for token in doc:
            stats.words_checked += 1

            # Check for LLM-speak FIRST - bypass _should_track for these
            if token.text.lower() in LLM_SPEAK:
                replacement = LLM_SPEAK[token.text.lower()]
                # Always fix vocabulary issues
                replacements.append((token.idx, token.text, self._match_case(replacement, token.text)))
                stats.replacements_detail[token.text] = replacement
                continue

            if not self._should_track(token):
                continue

            # Only do synonym replacement if enabled (disabled by default)
            if not self.synonym_replacement:
                continue

            word_lower = token.lemma_.lower()

            # Check if overused
            if self.word_counts[word_lower] >= self.threshold:
                synonym = self._get_synonym(token)
                if synonym and synonym not in self.used_replacements:
                    replacements.append((token.idx, token.text, self._match_case(synonym, token.text)))
                    self.used_replacements.add(synonym)
                    stats.overused_words.append(token.text)
                    stats.replacements_detail[token.text] = synonym

        # Apply replacements in reverse order
        result = text
        for pos, original, replacement in sorted(replacements, key=lambda x: -x[0]):
            result = result[:pos] + replacement + result[pos + len(original):]
            stats.replacements_made += 1

        if stats.replacements_made > 0:
            logger.debug(f"Reduced {stats.replacements_made} repetitions: {stats.replacements_detail}")
            # Also print to stdout for debugging
            print(f"DEBUG VOCAB: {stats.replacements_detail}")

        return result, stats

    def _should_track(self, token) -> bool:
        """Check if token should be tracked for repetition."""
        # Skip entities, proper nouns, punctuation, short words
        if token.ent_type_:
            return False
        if token.pos_ in ('PROPN', 'PUNCT', 'SPACE', 'NUM'):
            return False
        if not token.is_alpha or len(token.text) < 4:
            return False
        # Only track content words (nouns, verbs, adjectives, adverbs)
        if token.pos_ not in ('NOUN', 'VERB', 'ADJ', 'ADV'):
            return False
        return True

    def _get_synonym(self, token) -> Optional[str]:
        """Get a synonym for the token."""
        word = token.lemma_.lower()

        # Check cache first
        if word in self._synonym_cache:
            synonyms = self._synonym_cache[word]
            for syn in synonyms:
                # Skip LLM_SPEAK keys (problematic words we're trying to fix)
                if syn.lower() in LLM_SPEAK:
                    continue
                if syn not in self.used_replacements and syn != word:
                    return syn
            return None

        # Get synonyms from WordNet
        synonyms = []
        if self.wordnet:
            try:
                pos_map = {
                    'NOUN': 'n',
                    'VERB': 'v',
                    'ADJ': 'a',
                    'ADV': 'r',
                }
                wn_pos = pos_map.get(token.pos_)

                if wn_pos:
                    for synset in self.wordnet.synsets(word, pos=wn_pos):
                        for lemma in synset.lemmas():
                            name = lemma.name().replace('_', ' ')
                            if name != word and name.isalpha():
                                # Skip synonyms that are LLM_SPEAK keys (problematic words)
                                if name.lower() not in LLM_SPEAK:
                                    synonyms.append(name)
            except Exception as e:
                logger.debug(f"WordNet lookup failed for {word}: {e}")

        # Cache and return
        self._synonym_cache[word] = synonyms[:5]  # Keep top 5

        for syn in synonyms:
            # Skip LLM_SPEAK keys (problematic words we're trying to fix)
            if syn.lower() in LLM_SPEAK:
                continue
            if syn not in self.used_replacements:
                return syn
        return None

    def _match_case(self, replacement: str, original: str) -> str:
        """Match the case of the replacement to the original."""
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement.capitalize()
        return replacement.lower()

    def reset(self):
        """Reset word tracking between documents."""
        self.word_counts.clear()
        self.used_replacements.clear()

    def get_overused_words(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most overused words."""
        return [
            (word, count)
            for word, count in self.word_counts.most_common(limit)
            if count >= self.threshold
        ]
