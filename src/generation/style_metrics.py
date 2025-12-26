"""Style metrics for evaluating generated text against target style."""

from typing import List, Set, Optional, Dict
from dataclasses import dataclass

from ..models.style import StyleProfile
from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StyleScore:
    """Combined style score with individual metrics."""
    vocabulary_overlap: float  # 0-1, how much vocab matches author
    voice_match: float  # 0-1, how close voice ratio is to target
    sentence_length_match: float  # 0-1, how close avg length is
    punctuation_match: float = 0.5  # 0-1, how close punctuation is
    overall: float = 0.0  # Weighted average

    def passes_threshold(self, min_score: float = 0.6) -> bool:
        """Check if overall score passes threshold."""
        return self.overall >= min_score


class VocabularyScorer:
    """Scores text based on vocabulary overlap with author's style."""

    def __init__(self, style_profile: StyleProfile):
        """Initialize scorer.

        Args:
            style_profile: Target author's style profile.
        """
        self.style_profile = style_profile
        self._nlp = None
        self._vocab_set: Optional[Set[str]] = None

    @property
    def nlp(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    @property
    def vocab_set(self) -> Set[str]:
        """Get author's vocabulary as lowercase set."""
        if self._vocab_set is None:
            vocab = self.style_profile.get_effective_vocab()
            self._vocab_set = {w.lower() for w in vocab}
        return self._vocab_set

    def score(self, text: str) -> float:
        """Score text for vocabulary overlap.

        Args:
            text: Generated text to score.

        Returns:
            Score from 0.0 to 1.0.
        """
        if not text or not self.vocab_set:
            return 0.0

        doc = self.nlp(text)

        # Get content words (nouns, verbs, adjectives, adverbs)
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        content_words = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in content_pos and not token.is_stop and len(token.text) > 2
        ]

        if not content_words:
            return 0.0

        # Count matches
        matches = sum(1 for w in content_words if w in self.vocab_set)
        overlap = matches / len(content_words)

        return min(1.0, overlap)

    def get_matching_words(self, text: str) -> List[str]:
        """Get list of words in text that match author's vocabulary.

        Args:
            text: Text to analyze.

        Returns:
            List of matching words.
        """
        if not text or not self.vocab_set:
            return []

        doc = self.nlp(text)
        matches = []

        for token in doc:
            if token.lemma_.lower() in self.vocab_set:
                matches.append(token.text)

        return matches


class VoiceScorer:
    """Scores text based on active/passive voice match to author."""

    def __init__(self, style_profile: StyleProfile):
        """Initialize scorer.

        Args:
            style_profile: Target author's style profile.
        """
        self.style_profile = style_profile
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    @property
    def target_voice_ratio(self) -> float:
        """Get target active voice ratio."""
        return self.style_profile.primary_author.voice_ratio

    def analyze_voice(self, text: str) -> Dict[str, float]:
        """Analyze voice in text.

        Args:
            text: Text to analyze.

        Returns:
            Dict with active_ratio, passive_ratio.
        """
        if not text:
            return {"active_ratio": 0.5, "passive_ratio": 0.5}

        doc = self.nlp(text)

        active_count = 0
        passive_count = 0

        for sent in doc.sents:
            has_passive = False
            for token in sent:
                # Check for passive auxiliary
                if token.dep_ == "auxpass":
                    has_passive = True
                    break
                # Check for passive subject
                if token.dep_ == "nsubjpass":
                    has_passive = True
                    break

            if has_passive:
                passive_count += 1
            else:
                active_count += 1

        total = active_count + passive_count
        if total == 0:
            return {"active_ratio": 0.5, "passive_ratio": 0.5}

        return {
            "active_ratio": active_count / total,
            "passive_ratio": passive_count / total
        }

    def score(self, text: str) -> float:
        """Score text for voice match.

        Args:
            text: Generated text to score.

        Returns:
            Score from 0.0 to 1.0 (1.0 = perfect match).
        """
        analysis = self.analyze_voice(text)
        actual_ratio = analysis["active_ratio"]
        target_ratio = self.target_voice_ratio

        # Calculate how close we are (1.0 - abs difference)
        diff = abs(actual_ratio - target_ratio)
        return max(0.0, 1.0 - diff)


class SentenceLengthScorer:
    """Scores text based on sentence length match to author."""

    def __init__(self, style_profile: StyleProfile):
        """Initialize scorer.

        Args:
            style_profile: Target author's style profile.
        """
        self.style_profile = style_profile
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    @property
    def target_avg_length(self) -> float:
        """Get target average sentence length."""
        return self.style_profile.get_effective_avg_sentence_length()

    def analyze_lengths(self, text: str) -> Dict[str, float]:
        """Analyze sentence lengths in text.

        Args:
            text: Text to analyze.

        Returns:
            Dict with avg_length, min_length, max_length.
        """
        if not text:
            return {"avg_length": 0, "min_length": 0, "max_length": 0}

        doc = self.nlp(text)
        lengths = []

        for sent in doc.sents:
            words = [t for t in sent if not t.is_punct and not t.is_space]
            lengths.append(len(words))

        if not lengths:
            return {"avg_length": 0, "min_length": 0, "max_length": 0}

        return {
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }

    def score(self, text: str) -> float:
        """Score text for sentence length match.

        Args:
            text: Generated text to score.

        Returns:
            Score from 0.0 to 1.0 (1.0 = perfect match).
        """
        analysis = self.analyze_lengths(text)
        actual_avg = analysis["avg_length"]
        target_avg = self.target_avg_length

        if target_avg == 0:
            return 0.5

        # Calculate relative difference
        diff = abs(actual_avg - target_avg) / target_avg
        # Score: 1.0 if exact, 0.0 if 100%+ difference
        return max(0.0, 1.0 - diff)


class PunctuationScorer:
    """Scores text based on punctuation pattern match to author."""

    def __init__(self, style_profile: StyleProfile):
        """Initialize scorer.

        Args:
            style_profile: Target author's style profile.
        """
        self.style_profile = style_profile

    @property
    def target_patterns(self) -> Dict[str, Dict[str, float]]:
        """Get target punctuation patterns."""
        return self.style_profile.primary_author.punctuation_patterns

    def analyze_punctuation(self, text: str) -> Dict[str, float]:
        """Analyze punctuation usage in text.

        Args:
            text: Text to analyze.

        Returns:
            Dict with punctuation counts per sentence.
        """
        if not text:
            return {}

        # Count sentences (rough)
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences == 0:
            sentences = 1

        return {
            "semicolon": text.count(';') / sentences,
            "colon": text.count(':') / sentences,
            "em_dash": (text.count('—') + text.count('--')) / sentences,
            "parenthetical": text.count('(') / sentences,
            "question": text.count('?') / sentences,
            "exclamation": text.count('!') / sentences,
        }

    def score(self, text: str) -> float:
        """Score text for punctuation match.

        Args:
            text: Generated text to score.

        Returns:
            Score from 0.0 to 1.0 (1.0 = perfect match).
        """
        if not self.target_patterns:
            return 0.5  # No target patterns = neutral score

        actual = self.analyze_punctuation(text)
        if not actual:
            return 0.5

        # Compare each punctuation type
        scores = []
        for punc_type, target_data in self.target_patterns.items():
            target_freq = target_data.get("per_sentence", 0)
            actual_freq = actual.get(punc_type, 0)

            if target_freq == 0 and actual_freq == 0:
                scores.append(1.0)  # Both zero = match
            elif target_freq == 0:
                # Target doesn't use this, penalize if we do
                scores.append(max(0, 1.0 - actual_freq))
            else:
                # Compare frequencies
                diff = abs(target_freq - actual_freq) / max(target_freq, 0.01)
                scores.append(max(0, 1.0 - diff))

        return sum(scores) / len(scores) if scores else 0.5


class StyleScorer:
    """Combined style scorer that evaluates multiple metrics."""

    def __init__(
        self,
        style_profile: StyleProfile,
        vocab_weight: float = 0.25,
        voice_weight: float = 0.25,
        length_weight: float = 0.35,
        punctuation_weight: float = 0.15
    ):
        """Initialize combined scorer.

        Args:
            style_profile: Target author's style profile.
            vocab_weight: Weight for vocabulary score.
            voice_weight: Weight for voice score.
            length_weight: Weight for sentence length score.
            punctuation_weight: Weight for punctuation score.
        """
        self.style_profile = style_profile
        self.vocab_scorer = VocabularyScorer(style_profile)
        self.voice_scorer = VoiceScorer(style_profile)
        self.length_scorer = SentenceLengthScorer(style_profile)
        self.punctuation_scorer = PunctuationScorer(style_profile)

        # Normalize weights
        total = vocab_weight + voice_weight + length_weight + punctuation_weight
        self.vocab_weight = vocab_weight / total
        self.voice_weight = voice_weight / total
        self.length_weight = length_weight / total
        self.punctuation_weight = punctuation_weight / total

    def score(self, text: str) -> StyleScore:
        """Score text against target style.

        Args:
            text: Generated text to score.

        Returns:
            StyleScore with individual and overall scores.
        """
        vocab_score = self.vocab_scorer.score(text)
        voice_score = self.voice_scorer.score(text)
        length_score = self.length_scorer.score(text)
        punctuation_score = self.punctuation_scorer.score(text)

        overall = (
            vocab_score * self.vocab_weight +
            voice_score * self.voice_weight +
            length_score * self.length_weight +
            punctuation_score * self.punctuation_weight
        )

        return StyleScore(
            vocabulary_overlap=vocab_score,
            voice_match=voice_score,
            sentence_length_match=length_score,
            punctuation_match=punctuation_score,
            overall=overall
        )

    def get_feedback(self, text: str) -> str:
        """Get feedback for improving style match.

        Args:
            text: Generated text.

        Returns:
            Feedback string for revision.
        """
        score = self.score(text)
        feedback_parts = []

        if score.vocabulary_overlap < 0.3:
            vocab = self.style_profile.get_effective_vocab()[:5]
            feedback_parts.append(
                f"Use more characteristic vocabulary: {', '.join(vocab)}"
            )

        if score.voice_match < 0.6:
            target_ratio = self.voice_scorer.target_voice_ratio
            if target_ratio > 0.6:
                feedback_parts.append("Use more active voice constructions")
            elif target_ratio < 0.4:
                feedback_parts.append("Use more passive voice constructions")

        if score.sentence_length_match < 0.6:
            target_len = self.length_scorer.target_avg_length
            actual = self.length_scorer.analyze_lengths(text)["avg_length"]
            if actual < target_len:
                feedback_parts.append(f"Use longer sentences (target: ~{target_len:.0f} words)")
            else:
                feedback_parts.append(f"Use shorter sentences (target: ~{target_len:.0f} words)")

        if score.punctuation_match < 0.5:
            target_patterns = self.punctuation_scorer.target_patterns
            if target_patterns:
                # Find most distinctive punctuation pattern
                high_usage = [p for p, data in target_patterns.items()
                              if data.get("per_sentence", 0) > 0.2]
                if high_usage:
                    feedback_parts.append(
                        f"Consider using more {', '.join(high_usage[:2])} punctuation"
                    )

        return "; ".join(feedback_parts) if feedback_parts else "Style looks good"
