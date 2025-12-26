"""Enhanced style feature extraction from corpus text."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter

from ..utils.nlp import get_nlp, split_into_sentences, split_into_paragraphs
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Transition word categories
TRANSITION_PATTERNS = {
    "causal": [
        "therefore", "thus", "hence", "consequently", "as a result",
        "for this reason", "accordingly", "because of this", "so",
        "it follows that", "owing to this"
    ],
    "adversative": [
        "however", "but", "yet", "nevertheless", "nonetheless",
        "on the other hand", "conversely", "in contrast", "although",
        "despite this", "even so", "still", "whereas", "while"
    ],
    "additive": [
        "moreover", "furthermore", "additionally", "also", "besides",
        "in addition", "what is more", "equally", "likewise",
        "similarly", "not only", "as well as"
    ],
    "temporal": [
        "then", "next", "subsequently", "afterward", "afterwards",
        "finally", "meanwhile", "previously", "before", "after",
        "simultaneously", "at the same time", "eventually"
    ],
    "exemplifying": [
        "for example", "for instance", "such as", "namely",
        "specifically", "to illustrate", "in particular", "as shown by"
    ],
    "concluding": [
        "in conclusion", "to conclude", "in summary", "to summarize",
        "in short", "ultimately", "all in all", "on the whole"
    ]
}


class TransitionExtractor:
    """Extracts transition words and connectors from text."""

    def __init__(self):
        self.patterns = TRANSITION_PATTERNS
        # Pre-compile regex patterns for multi-word transitions
        self._compiled_patterns = {}
        for category, words in self.patterns.items():
            self._compiled_patterns[category] = [
                (word, re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE))
                for word in sorted(words, key=len, reverse=True)  # Longer first
            ]

    def extract(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """Extract transition words with frequencies.

        Args:
            text: Text to analyze.

        Returns:
            Dict mapping category to list of (word, frequency) tuples.
        """
        if not text or not text.strip():
            return {cat: [] for cat in self.patterns}

        result = {}
        found_positions = set()  # Track positions to avoid double-counting

        for category, patterns in self._compiled_patterns.items():
            counts = Counter()

            for word, pattern in patterns:
                for match in pattern.finditer(text):
                    # Check if this position was already matched by a longer phrase
                    pos = match.start()
                    if pos not in found_positions:
                        counts[match.group()] += 1
                        # Mark all positions in this match as used
                        for i in range(match.start(), match.end()):
                            found_positions.add(i)

            # Sort by frequency, then alphabetically
            sorted_counts = sorted(
                counts.items(),
                key=lambda x: (-x[1], x[0].lower())
            )
            result[category] = sorted_counts

        return result

    def get_author_preferences(
        self,
        text: str
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Get transition preferences as normalized frequencies.

        Args:
            text: Text to analyze.

        Returns:
            Dict mapping category to list of (word, normalized_freq) tuples.
        """
        raw = self.extract(text)
        total_words = len(text.split())

        result = {}
        for category, counts in raw.items():
            # Normalize by total words (per 1000 words)
            normalized = [
                (word, count * 1000 / total_words if total_words > 0 else 0)
                for word, count in counts
            ]
            result[category] = normalized

        return result


class VoiceAnalyzer:
    """Analyzes active vs passive voice usage."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze voice distribution in text.

        Args:
            text: Text to analyze.

        Returns:
            Dict with active_ratio, passive_ratio, and counts.
        """
        if not text or not text.strip():
            return {
                "active_ratio": 0.5,
                "passive_ratio": 0.5,
                "active_count": 0,
                "passive_count": 0,
                "total_sentences": 0
            }

        sentences = split_into_sentences(text)
        active_count = 0
        passive_count = 0

        for sentence in sentences:
            voice = self._detect_voice(sentence)
            if voice == "active":
                active_count += 1
            elif voice == "passive":
                passive_count += 1

        total = active_count + passive_count
        if total == 0:
            return {
                "active_ratio": 0.5,
                "passive_ratio": 0.5,
                "active_count": 0,
                "passive_count": 0,
                "total_sentences": len(sentences)
            }

        return {
            "active_ratio": active_count / total,
            "passive_ratio": passive_count / total,
            "active_count": active_count,
            "passive_count": passive_count,
            "total_sentences": len(sentences)
        }

    def _detect_voice(self, sentence: str) -> str:
        """Detect voice of a single sentence.

        Args:
            sentence: Sentence to analyze.

        Returns:
            "active", "passive", or "unclear"
        """
        doc = self.nlp(sentence)

        # Look for passive voice markers
        for token in doc:
            # Check for passive auxiliary + past participle
            if token.dep_ == "auxpass":
                return "passive"

            # Check for "nsubjpass" (passive subject)
            if token.dep_ == "nsubjpass":
                return "passive"

        # Check for active subject-verb pattern
        has_nsubj = any(token.dep_ == "nsubj" for token in doc)
        has_verb = any(token.pos_ == "VERB" for token in doc)

        if has_nsubj and has_verb:
            return "active"

        return "unclear"


class OpenerExtractor:
    """Extracts sentence opener patterns."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """Extract sentence openers with frequencies.

        Args:
            text: Text to analyze.

        Returns:
            Dict with opener patterns by position.
        """
        if not text or not text.strip():
            return {
                "openers": [],
                "first_sentence": [],
                "middle_sentence": [],
                "last_sentence": []
            }

        paragraphs = split_into_paragraphs(text)

        all_openers = Counter()
        first_openers = Counter()
        middle_openers = Counter()
        last_openers = Counter()

        for para in paragraphs:
            sentences = split_into_sentences(para)
            if not sentences:
                continue

            for i, sentence in enumerate(sentences):
                opener = self._get_opener(sentence)
                if opener:
                    all_openers[opener] += 1

                    if i == 0:
                        first_openers[opener] += 1
                    elif i == len(sentences) - 1:
                        last_openers[opener] += 1
                    else:
                        middle_openers[opener] += 1

        return {
            "openers": sorted(all_openers.items(), key=lambda x: (-x[1], x[0])),
            "first_sentence": sorted(first_openers.items(), key=lambda x: (-x[1], x[0])),
            "middle_sentence": sorted(middle_openers.items(), key=lambda x: (-x[1], x[0])),
            "last_sentence": sorted(last_openers.items(), key=lambda x: (-x[1], x[0]))
        }

    def _get_opener(self, sentence: str, max_words: int = 3) -> Optional[str]:
        """Get the opening word(s) of a sentence.

        Args:
            sentence: Sentence to analyze.
            max_words: Maximum words to consider as opener.

        Returns:
            Opener string or None.
        """
        words = sentence.split()
        if not words:
            return None

        # Get first word (most common case)
        first_word = words[0]

        # Clean punctuation
        first_word = re.sub(r'["\'\(\[]', '', first_word)

        if not first_word:
            return None

        return first_word


class PhraseExtractor:
    """Extracts signature phrases and n-grams."""

    # Common stopwords to filter
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "it", "its", "as", "if", "not", "no", "so"
    }

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """Extract phrases and n-grams.

        Args:
            text: Text to analyze.

        Returns:
            Dict with bigrams, trigrams, and signature_phrases.
        """
        if not text or not text.strip():
            return {
                "bigrams": [],
                "trigrams": [],
                "signature_phrases": []
            }

        # Tokenize and get lemmas
        doc = self.nlp(text.lower())
        tokens = [token.text for token in doc if token.is_alpha]

        # Extract n-grams
        bigrams = self._extract_ngrams(tokens, 2)
        trigrams = self._extract_ngrams(tokens, 3)

        # Filter and count
        bigram_counts = self._filter_and_count(bigrams)
        trigram_counts = self._filter_and_count(trigrams)

        # Identify signature phrases (high frequency, not just stopwords)
        signature_phrases = self._identify_signatures(bigram_counts, trigram_counts)

        return {
            "bigrams": sorted(bigram_counts.items(), key=lambda x: (-x[1], x[0])),
            "trigrams": sorted(trigram_counts.items(), key=lambda x: (-x[1], x[0])),
            "signature_phrases": sorted(signature_phrases.items(), key=lambda x: (-x[1], x[0]))
        }

    def _extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Extract n-grams from token list."""
        return [
            " ".join(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        ]

    def _filter_and_count(self, ngrams: List[str]) -> Counter:
        """Filter out stopword-only n-grams and count."""
        counts = Counter()

        for ngram in ngrams:
            words = ngram.split()
            # Skip if all words are stopwords
            if all(w in self.STOPWORDS for w in words):
                continue
            # Skip if first and last are both stopwords
            if words[0] in self.STOPWORDS and words[-1] in self.STOPWORDS:
                continue

            counts[ngram] += 1

        # Filter by minimum frequency
        return Counter({k: v for k, v in counts.items() if v >= self.min_freq})

    def _identify_signatures(
        self,
        bigrams: Counter,
        trigrams: Counter
    ) -> Counter:
        """Identify signature phrases (distinctive, frequent)."""
        signatures = Counter()

        # Add high-frequency bigrams with content words
        for phrase, count in bigrams.items():
            words = phrase.split()
            content_words = [w for w in words if w not in self.STOPWORDS]
            if len(content_words) >= 1 and count >= self.min_freq:
                signatures[phrase] = count

        # Add high-frequency trigrams
        for phrase, count in trigrams.items():
            words = phrase.split()
            content_words = [w for w in words if w not in self.STOPWORDS]
            if len(content_words) >= 2 and count >= self.min_freq:
                signatures[phrase] = count

        return signatures


class PunctuationAnalyzer:
    """Analyzes punctuation usage patterns."""

    def analyze(self, text: str) -> Dict[str, Dict[str, float]]:
        """Analyze punctuation usage.

        Args:
            text: Text to analyze.

        Returns:
            Dict with punctuation stats.
        """
        if not text or not text.strip():
            return self._empty_result()

        sentences = split_into_sentences(text)
        num_sentences = len(sentences) if sentences else 1

        # Count various punctuation
        semicolons = text.count(';')
        colons = text.count(':')
        em_dashes = text.count('—') + text.count('--')
        parentheticals = len(re.findall(r'\([^)]+\)', text))
        questions = text.count('?')
        exclamations = text.count('!')

        return {
            "semicolon": {
                "count": semicolons,
                "per_sentence": semicolons / num_sentences
            },
            "colon": {
                "count": colons,
                "per_sentence": colons / num_sentences
            },
            "em_dash": {
                "count": em_dashes,
                "per_sentence": em_dashes / num_sentences
            },
            "parenthetical": {
                "count": parentheticals,
                "per_sentence": parentheticals / num_sentences
            },
            "question": {
                "count": questions,
                "per_sentence": questions / num_sentences
            },
            "exclamation": {
                "count": exclamations,
                "per_sentence": exclamations / num_sentences
            }
        }

    def _empty_result(self) -> Dict[str, Dict[str, float]]:
        """Return empty result structure."""
        return {
            "semicolon": {"count": 0, "per_sentence": 0.0},
            "colon": {"count": 0, "per_sentence": 0.0},
            "em_dash": {"count": 0, "per_sentence": 0.0},
            "parenthetical": {"count": 0, "per_sentence": 0.0},
            "question": {"count": 0, "per_sentence": 0.0},
            "exclamation": {"count": 0, "per_sentence": 0.0}
        }


@dataclass
class StyleFeatures:
    """Combined style features extracted from text."""

    transitions: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    voice_ratio: float = 0.5
    voice_details: Dict[str, float] = field(default_factory=dict)
    openers: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    phrases: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    punctuation: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def extract_from_text(cls, text: str) -> "StyleFeatures":
        """Extract all style features from text.

        Args:
            text: Text to analyze.

        Returns:
            StyleFeatures instance.
        """
        transition_extractor = TransitionExtractor()
        voice_analyzer = VoiceAnalyzer()
        opener_extractor = OpenerExtractor()
        phrase_extractor = PhraseExtractor()
        punctuation_analyzer = PunctuationAnalyzer()

        transitions = transition_extractor.extract(text)
        voice = voice_analyzer.analyze(text)
        openers = opener_extractor.extract(text)
        phrases = phrase_extractor.extract(text)
        punctuation = punctuation_analyzer.analyze(text)

        return cls(
            transitions=transitions,
            voice_ratio=voice["active_ratio"],
            voice_details=voice,
            openers=openers,
            phrases=phrases,
            punctuation=punctuation
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "transitions": {
                cat: [(w, f) for w, f in words]
                for cat, words in self.transitions.items()
            },
            "voice_ratio": self.voice_ratio,
            "voice_details": self.voice_details,
            "openers": {
                cat: [(w, f) for w, f in words]
                for cat, words in self.openers.items()
            },
            "phrases": {
                cat: [(w, f) for w, f in words]
                for cat, words in self.phrases.items()
            },
            "punctuation": self.punctuation
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StyleFeatures":
        """Create from dictionary."""
        return cls(
            transitions=data.get("transitions", {}),
            voice_ratio=data.get("voice_ratio", 0.5),
            voice_details=data.get("voice_details", {}),
            openers=data.get("openers", {}),
            phrases=data.get("phrases", {}),
            punctuation=data.get("punctuation", {})
        )

    @classmethod
    def merge(cls, features_list: List["StyleFeatures"]) -> "StyleFeatures":
        """Merge multiple StyleFeatures into one.

        Args:
            features_list: List of StyleFeatures to merge.

        Returns:
            Merged StyleFeatures.
        """
        if not features_list:
            return cls()

        if len(features_list) == 1:
            return features_list[0]

        # Merge transitions
        merged_transitions = {}
        for category in TRANSITION_PATTERNS.keys():
            combined = Counter()
            for features in features_list:
                for word, count in features.transitions.get(category, []):
                    combined[word] += count
            merged_transitions[category] = sorted(
                combined.items(), key=lambda x: (-x[1], x[0])
            )

        # Average voice ratio
        avg_voice = sum(f.voice_ratio for f in features_list) / len(features_list)

        # Merge openers
        merged_openers = {}
        for key in ["openers", "first_sentence", "middle_sentence", "last_sentence"]:
            combined = Counter()
            for features in features_list:
                for word, count in features.openers.get(key, []):
                    combined[word] += count
            merged_openers[key] = sorted(
                combined.items(), key=lambda x: (-x[1], x[0])
            )

        # Merge phrases
        merged_phrases = {}
        for key in ["bigrams", "trigrams", "signature_phrases"]:
            combined = Counter()
            for features in features_list:
                for phrase, count in features.phrases.get(key, []):
                    combined[phrase] += count
            merged_phrases[key] = sorted(
                combined.items(), key=lambda x: (-x[1], x[0])
            )

        # Average punctuation
        merged_punctuation = {}
        punct_keys = ["semicolon", "colon", "em_dash", "parenthetical", "question", "exclamation"]
        for key in punct_keys:
            total_count = sum(
                f.punctuation.get(key, {}).get("count", 0)
                for f in features_list
            )
            total_per_sent = sum(
                f.punctuation.get(key, {}).get("per_sentence", 0)
                for f in features_list
            ) / len(features_list)
            merged_punctuation[key] = {
                "count": total_count,
                "per_sentence": total_per_sent
            }

        return cls(
            transitions=merged_transitions,
            voice_ratio=avg_voice,
            voice_details={},  # Skip merging details
            openers=merged_openers,
            phrases=merged_phrases,
            punctuation=merged_punctuation
        )
