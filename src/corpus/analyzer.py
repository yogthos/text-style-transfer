"""Statistical analysis of text for style profiling."""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.nlp import (
    split_into_sentences,
    count_words,
    calculate_burstiness,
    get_pos_distribution,
    get_dependency_depth,
    detect_perspective,
    extract_keywords,
)
from ..utils.logging import get_logger
from .preprocessor import ProcessedDocument, ProcessedParagraph

logger = get_logger(__name__)


@dataclass
class FeatureVector:
    """Statistical features of a text segment."""
    # Sentence length statistics
    avg_sentence_length: float = 0.0
    min_sentence_length: int = 0
    max_sentence_length: int = 0
    sentence_count: int = 0

    # Rhythm/variation
    burstiness: float = 0.0  # Coefficient of variation

    # Complexity
    avg_dependency_depth: float = 0.0

    # Punctuation patterns
    punctuation_freq: Dict[str, float] = field(default_factory=dict)

    # POS distribution
    pos_distribution: Dict[str, float] = field(default_factory=dict)

    # Perspective
    perspective: str = "third_person"

    # Top vocabulary
    top_words: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "avg_sentence_length": self.avg_sentence_length,
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "sentence_count": self.sentence_count,
            "burstiness": self.burstiness,
            "avg_dependency_depth": self.avg_dependency_depth,
            "punctuation_freq": self.punctuation_freq,
            "pos_distribution": self.pos_distribution,
            "perspective": self.perspective,
            "top_words": self.top_words,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureVector":
        """Create from dictionary."""
        return cls(
            avg_sentence_length=data.get("avg_sentence_length", 0.0),
            min_sentence_length=data.get("min_sentence_length", 0),
            max_sentence_length=data.get("max_sentence_length", 0),
            sentence_count=data.get("sentence_count", 0),
            burstiness=data.get("burstiness", 0.0),
            avg_dependency_depth=data.get("avg_dependency_depth", 0.0),
            punctuation_freq=data.get("punctuation_freq", {}),
            pos_distribution=data.get("pos_distribution", {}),
            perspective=data.get("perspective", "third_person"),
            top_words=data.get("top_words", []),
        )


class StatisticalAnalyzer:
    """Analyzes text to extract statistical style features."""

    # Punctuation marks to track
    TRACKED_PUNCTUATION = {
        'em_dash': '—',
        'en_dash': '–',
        'semicolon': ';',
        'colon': ':',
        'ellipsis': '...',
        'exclamation': '!',
        'question': '?',
        'comma': ',',
        'parenthesis': '(',
    }

    def __init__(self, top_n_words: int = 50):
        """Initialize analyzer.

        Args:
            top_n_words: Number of top words to extract.
        """
        self.top_n_words = top_n_words

    def analyze_document(self, doc: ProcessedDocument) -> FeatureVector:
        """Analyze an entire processed document.

        Args:
            doc: Processed document.

        Returns:
            FeatureVector with statistical features.
        """
        # Collect all sentences
        all_sentences = []
        for para in doc.paragraphs:
            all_sentences.extend(para.sentences)

        return self.analyze_sentences(all_sentences, doc.cleaned_text)

    def analyze_paragraph(self, para: ProcessedParagraph) -> FeatureVector:
        """Analyze a single paragraph.

        Args:
            para: Processed paragraph.

        Returns:
            FeatureVector for the paragraph.
        """
        return self.analyze_sentences(para.sentences, para.text)

    def analyze_sentences(
        self,
        sentences: List[str],
        full_text: Optional[str] = None
    ) -> FeatureVector:
        """Analyze a list of sentences.

        Args:
            sentences: List of sentences to analyze.
            full_text: Optional full text for context analysis.

        Returns:
            FeatureVector with statistical features.
        """
        if not sentences:
            return FeatureVector()

        # Use full_text or reconstruct from sentences
        if full_text is None:
            full_text = " ".join(sentences)

        # Calculate sentence lengths
        lengths = [count_words(s) for s in sentences]

        # Sentence statistics
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0

        # Burstiness
        burstiness = calculate_burstiness(sentences)

        # Dependency depth (can be slow for large texts)
        avg_depth = get_dependency_depth(full_text)

        # Punctuation frequencies
        punct_freq = self._calculate_punctuation_freq(full_text)

        # POS distribution
        pos_dist = self._normalize_pos_distribution(get_pos_distribution(full_text))

        # Perspective
        perspective = detect_perspective(full_text)

        # Top vocabulary
        top_words = extract_keywords(full_text, top_n=self.top_n_words)

        return FeatureVector(
            avg_sentence_length=avg_length,
            min_sentence_length=min_length,
            max_sentence_length=max_length,
            sentence_count=len(sentences),
            burstiness=burstiness,
            avg_dependency_depth=avg_depth,
            punctuation_freq=punct_freq,
            pos_distribution=pos_dist,
            perspective=perspective,
            top_words=top_words,
        )

    def analyze_text(self, text: str) -> FeatureVector:
        """Analyze raw text.

        Args:
            text: Text to analyze.

        Returns:
            FeatureVector with statistical features.
        """
        sentences = split_into_sentences(text)
        return self.analyze_sentences(sentences, text)

    def _calculate_punctuation_freq(self, text: str) -> Dict[str, float]:
        """Calculate punctuation frequencies per 1000 characters.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary of punctuation mark to frequency.
        """
        if not text:
            return {}

        text_length = len(text)
        if text_length == 0:
            return {}

        frequencies = {}
        for name, char in self.TRACKED_PUNCTUATION.items():
            count = text.count(char)
            # Frequency per 1000 characters
            frequencies[name] = (count / text_length) * 1000

        return frequencies

    def _normalize_pos_distribution(self, pos_counts: Dict[str, int]) -> Dict[str, float]:
        """Normalize POS counts to proportions.

        Args:
            pos_counts: Dictionary of POS tag to count.

        Returns:
            Dictionary of POS tag to proportion.
        """
        total = sum(pos_counts.values())
        if total == 0:
            return {}

        return {pos: count / total for pos, count in pos_counts.items()}

    def compute_similarity(
        self,
        features1: FeatureVector,
        features2: FeatureVector
    ) -> float:
        """Compute similarity between two feature vectors.

        Uses weighted combination of:
        - Sentence length similarity
        - Burstiness similarity
        - Punctuation pattern similarity
        - POS distribution similarity

        Args:
            features1: First feature vector.
            features2: Second feature vector.

        Returns:
            Similarity score between 0 and 1.
        """
        scores = []
        weights = []

        # Sentence length similarity (lower weight for exact match)
        len_diff = abs(features1.avg_sentence_length - features2.avg_sentence_length)
        len_sim = max(0, 1 - len_diff / 20)  # Normalize by ~20 words
        scores.append(len_sim)
        weights.append(0.2)

        # Burstiness similarity
        burst_diff = abs(features1.burstiness - features2.burstiness)
        burst_sim = max(0, 1 - burst_diff / 0.5)  # Normalize by 0.5
        scores.append(burst_sim)
        weights.append(0.2)

        # Punctuation similarity (cosine similarity)
        punct_sim = self._cosine_similarity(
            features1.punctuation_freq,
            features2.punctuation_freq
        )
        scores.append(punct_sim)
        weights.append(0.3)

        # POS similarity (cosine similarity)
        pos_sim = self._cosine_similarity(
            features1.pos_distribution,
            features2.pos_distribution
        )
        scores.append(pos_sim)
        weights.append(0.3)

        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float]
    ) -> float:
        """Compute cosine similarity between two sparse vectors.

        Args:
            vec1: First vector as dict.
            vec2: Second vector as dict.

        Returns:
            Cosine similarity between 0 and 1.
        """
        if not vec1 or not vec2:
            return 0.0

        # Get all keys
        all_keys = set(vec1.keys()) | set(vec2.keys())

        # Compute dot product and magnitudes
        dot_product = 0.0
        mag1 = 0.0
        mag2 = 0.0

        for key in all_keys:
            v1 = vec1.get(key, 0.0)
            v2 = vec2.get(key, 0.0)
            dot_product += v1 * v2
            mag1 += v1 * v1
            mag2 += v2 * v2

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / ((mag1 ** 0.5) * (mag2 ** 0.5))
