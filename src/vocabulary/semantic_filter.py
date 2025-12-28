"""Semantic filtering to distinguish stylistic vs content words.

The key insight: we only want to manipulate STYLISTIC words (adjectives, adverbs,
function words) not SEMANTIC content words (nouns, verbs with specific meaning).

Example: Mao corpus contains "Marxism", "proletariat", "bourgeoisie"
- These are CONTENT words - they carry specific semantic meaning
- We should NOT boost/penalize these for text about cosmology
- We SHOULD boost/penalize stylistic words like "however", "indeed", "merely"
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional
from enum import Enum

from ..utils.nlp import get_nlp


class WordCategory(Enum):
    """Categories for vocabulary control."""

    # Safe to manipulate - stylistic, not semantic
    STYLISTIC_ADV = "stylistic_adverb"      # however, indeed, merely, quite
    STYLISTIC_ADJ = "stylistic_adjective"   # mere, utter, sheer, very
    CONNECTOR = "connector"                  # but, and, yet, so, thus
    HEDGE = "hedge"                          # perhaps, might, somewhat
    INTENSIFIER = "intensifier"              # very, extremely, absolutely
    DISCOURSE_MARKER = "discourse_marker"    # firstly, moreover, nevertheless

    # Do NOT manipulate - semantic content
    CONTENT_NOUN = "content_noun"            # Marxism, universe, particle
    CONTENT_VERB = "content_verb"            # analyze, collapse, expand
    PROPER_NOUN = "proper_noun"              # Einstein, Moscow, NASA
    TECHNICAL_TERM = "technical_term"        # entropy, proletariat, quantum

    # Neutral - context dependent
    NEUTRAL = "neutral"


@dataclass
class WordClassification:
    """Classification result for a word."""

    word: str
    category: WordCategory
    is_safe_to_manipulate: bool
    pos_tag: str = ""
    reason: str = ""


class SemanticFilter:
    """Filter words to distinguish stylistic from semantic content.

    Uses spaCy for POS tagging and semantic analysis to determine
    which words are safe to boost/penalize without affecting content.
    """

    def __init__(self):
        self._nlp = None

        # Common "LLM-speak" words that are safe to penalize
        # These are stylistic, not semantic
        self.llm_speak_words: Set[str] = {
            # Flowery/overwrought adjectives
            "profound", "intricate", "nuanced", "multifaceted", "holistic",
            "pivotal", "paramount", "quintessential", "myriad", "plethora",
            "meticulous", "robust", "seamless", "dynamic", "innovative",
            "compelling", "transformative", "groundbreaking", "unprecedented",

            # Overwrought adverbs
            "profoundly", "intrinsically", "fundamentally", "inherently",
            "meticulously", "seamlessly", "ultimately", "essentially",
            "remarkably", "undeniably", "unequivocally", "inexorably",

            # Cliche connectors and phrases
            "delve", "delving", "tapestry", "testament", "beacon",
            "landscape", "paradigm", "synergy", "leverage", "ecosystem",
            "resonate", "resonates", "resonating", "bustling",
            "embark", "embarking", "navigate", "navigating", "journey",
            "foster", "fostering", "cultivate", "cultivating",
            "underscore", "underscores", "underscoring",

            # Hedge words
            "perhaps", "somewhat", "arguably", "potentially", "presumably",

            # Intensifiers often overused
            "truly", "incredibly", "absolutely", "utterly", "tremendously",
        }

        # Stylistic adverbs that modify how something is said, not what
        self.stylistic_adverbs: Set[str] = {
            "however", "moreover", "furthermore", "nevertheless", "nonetheless",
            "consequently", "accordingly", "hence", "thus", "therefore",
            "indeed", "certainly", "surely", "clearly", "obviously",
            "merely", "simply", "just", "only", "even",
            "already", "still", "yet", "always", "never",
            "often", "sometimes", "usually", "rarely", "seldom",
        }

        # Discourse markers - safe to manipulate
        self.discourse_markers: Set[str] = {
            "firstly", "secondly", "thirdly", "finally", "lastly",
            "meanwhile", "subsequently", "previously", "initially",
            "alternatively", "conversely", "similarly", "likewise",
            "specifically", "particularly", "especially", "notably",
            "importantly", "significantly", "interestingly", "surprisingly",
        }

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def classify_word(self, word: str) -> WordClassification:
        """Classify a single word for vocabulary control.

        Returns whether it's safe to manipulate (stylistic) or not (semantic).
        """
        word_lower = word.lower()

        # Check known categories first
        if word_lower in self.llm_speak_words:
            return WordClassification(
                word=word,
                category=WordCategory.STYLISTIC_ADJ,
                is_safe_to_manipulate=True,
                reason="Known LLM-speak word"
            )

        if word_lower in self.stylistic_adverbs:
            return WordClassification(
                word=word,
                category=WordCategory.STYLISTIC_ADV,
                is_safe_to_manipulate=True,
                reason="Stylistic adverb"
            )

        if word_lower in self.discourse_markers:
            return WordClassification(
                word=word,
                category=WordCategory.DISCOURSE_MARKER,
                is_safe_to_manipulate=True,
                reason="Discourse marker"
            )

        # Use spaCy for unknown words
        doc = self.nlp(word)
        if not doc or len(doc) == 0:
            return WordClassification(
                word=word,
                category=WordCategory.NEUTRAL,
                is_safe_to_manipulate=False,
                reason="Could not analyze"
            )

        token = doc[0]
        pos = token.pos_

        # Proper nouns - never manipulate
        if pos == "PROPN":
            return WordClassification(
                word=word,
                category=WordCategory.PROPER_NOUN,
                is_safe_to_manipulate=False,
                pos_tag=pos,
                reason="Proper noun - semantic content"
            )

        # Nouns - generally don't manipulate (semantic content)
        if pos == "NOUN":
            return WordClassification(
                word=word,
                category=WordCategory.CONTENT_NOUN,
                is_safe_to_manipulate=False,
                pos_tag=pos,
                reason="Noun - semantic content"
            )

        # Verbs - generally don't manipulate (semantic content)
        if pos == "VERB":
            return WordClassification(
                word=word,
                category=WordCategory.CONTENT_VERB,
                is_safe_to_manipulate=False,
                pos_tag=pos,
                reason="Verb - semantic content"
            )

        # Adjectives - only manipulate if clearly stylistic
        if pos == "ADJ":
            # Check if it's a "quality" adjective (stylistic) vs descriptive
            if token.is_stop or word_lower in self.llm_speak_words:
                return WordClassification(
                    word=word,
                    category=WordCategory.STYLISTIC_ADJ,
                    is_safe_to_manipulate=True,
                    pos_tag=pos,
                    reason="Stylistic adjective"
                )
            # Default: don't manipulate adjectives as they may be descriptive
            return WordClassification(
                word=word,
                category=WordCategory.NEUTRAL,
                is_safe_to_manipulate=False,
                pos_tag=pos,
                reason="Adjective - may be descriptive"
            )

        # Adverbs - generally safe to manipulate
        if pos == "ADV":
            return WordClassification(
                word=word,
                category=WordCategory.STYLISTIC_ADV,
                is_safe_to_manipulate=True,
                pos_tag=pos,
                reason="Adverb - stylistic"
            )

        # Conjunctions and connectors - safe to manipulate
        if pos in ("CCONJ", "SCONJ"):
            return WordClassification(
                word=word,
                category=WordCategory.CONNECTOR,
                is_safe_to_manipulate=True,
                pos_tag=pos,
                reason="Connector - stylistic"
            )

        # Particles, interjections - safe to manipulate
        if pos in ("PART", "INTJ"):
            return WordClassification(
                word=word,
                category=WordCategory.STYLISTIC_ADV,
                is_safe_to_manipulate=True,
                pos_tag=pos,
                reason="Particle/interjection - stylistic"
            )

        # Default: don't manipulate
        return WordClassification(
            word=word,
            category=WordCategory.NEUTRAL,
            is_safe_to_manipulate=False,
            pos_tag=pos,
            reason=f"Unknown POS {pos} - conservative default"
        )

    def filter_vocabulary(
        self,
        words: Set[str],
        input_text: Optional[str] = None
    ) -> Dict[str, WordClassification]:
        """Classify a set of words for vocabulary control.

        Args:
            words: Set of words to classify.
            input_text: If provided, words appearing in input are marked unsafe
                       (they're part of the content, not style).

        Returns:
            Dictionary mapping words to their classifications.
        """
        # Extract content words from input to preserve
        input_content_words: Set[str] = set()
        if input_text:
            doc = self.nlp(input_text.lower())
            for token in doc:
                if token.pos_ in ("NOUN", "VERB", "PROPN") and not token.is_stop:
                    input_content_words.add(token.lemma_)
                    input_content_words.add(token.text)

        classifications = {}
        for word in words:
            classification = self.classify_word(word)

            # Override if word appears in input - it's content, not style
            if word.lower() in input_content_words:
                classification = WordClassification(
                    word=word,
                    category=WordCategory.CONTENT_NOUN,
                    is_safe_to_manipulate=False,
                    reason="Appears in input text - content word"
                )

            classifications[word] = classification

        return classifications

    def get_safe_words(
        self,
        words: Set[str],
        input_text: Optional[str] = None
    ) -> Set[str]:
        """Get only the words that are safe to manipulate.

        Args:
            words: Set of words to filter.
            input_text: Input text to exclude content words from.

        Returns:
            Set of words safe to boost/penalize.
        """
        classifications = self.filter_vocabulary(words, input_text)
        return {
            word for word, cls in classifications.items()
            if cls.is_safe_to_manipulate
        }

    def get_llm_speak_penalty_words(self) -> Set[str]:
        """Get the set of LLM-speak words to penalize.

        These are common overused words in LLM output that make
        text sound artificial.
        """
        return self.llm_speak_words.copy()
