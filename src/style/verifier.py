"""Verify text against extracted style profile.

Uses statistical measures to determine if text matches author's style.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

from ..utils.nlp import split_into_sentences
from ..utils.logging import get_logger
from .profile import AuthorStyleProfile

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """Result of style verification."""

    is_acceptable: bool
    overall_score: float  # 0.0 = no match, 1.0 = perfect match

    # Component scores
    length_score: float = 0.0
    burstiness_score: float = 0.0
    transition_score: float = 0.0
    delta_score: float = 0.0

    # Issues found
    issues: List[str] = field(default_factory=list)

    # Repair suggestions
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SentenceVerification:
    """Verification result for a single sentence."""

    is_acceptable: bool
    actual_length: int
    target_length: int
    length_deviation: float  # as percentage
    issues: List[str] = field(default_factory=list)


class StyleVerifier:
    """Verify text against author's style profile."""

    def __init__(
        self,
        profile: AuthorStyleProfile,
        length_tolerance: float = 0.25,  # 25% deviation allowed
        burstiness_tolerance: float = 0.20,  # 20% burstiness difference
        transition_tolerance: float = 0.30,  # 30% transition frequency diff
        delta_threshold: float = 1.5,  # Burrows' Delta threshold
    ):
        """Initialize verifier.

        Args:
            profile: Author's style profile.
            length_tolerance: Allowed deviation from target length.
            burstiness_tolerance: Allowed burstiness difference.
            transition_tolerance: Allowed transition frequency difference.
            delta_threshold: Maximum Burrows' Delta for acceptance.
        """
        self.profile = profile
        self.length_tolerance = length_tolerance
        self.burstiness_tolerance = burstiness_tolerance
        self.transition_tolerance = transition_tolerance
        self.delta_threshold = delta_threshold

    def verify_sentence(
        self,
        sentence: str,
        target_length: int,
    ) -> SentenceVerification:
        """Verify a single sentence against profile.

        Args:
            sentence: The sentence to verify.
            target_length: Target word count.

        Returns:
            SentenceVerification result.
        """
        actual_length = len(sentence.split())
        deviation = abs(actual_length - target_length) / target_length if target_length > 0 else 0.0

        issues = []

        # Check length
        if deviation > self.length_tolerance:
            issues.append(
                f"Length {actual_length} deviates {deviation*100:.0f}% from target {target_length}"
            )

        # Check punctuation against profile
        register = self.profile.register_profile

        if ";" in sentence and register.semicolon_per_sentence < 0.05:
            issues.append("Semicolon used but author rarely uses them")

        if sentence.count("(") > 0 and register.parenthetical_per_sentence < 0.05:
            issues.append("Parenthetical used but author rarely uses them")

        return SentenceVerification(
            is_acceptable=len(issues) == 0,
            actual_length=actual_length,
            target_length=target_length,
            length_deviation=deviation,
            issues=issues,
        )

    def verify_paragraph(self, paragraph: str) -> VerificationResult:
        """Verify a paragraph against author's style profile.

        Args:
            paragraph: The paragraph to verify.

        Returns:
            VerificationResult with scores and issues.
        """
        sentences = split_into_sentences(paragraph)
        if not sentences:
            return VerificationResult(
                is_acceptable=False,
                overall_score=0.0,
                issues=["No sentences found"],
            )

        issues = []
        suggestions = []

        # 1. Length distribution check
        length_score, length_issues = self._check_lengths(sentences)
        issues.extend(length_issues)

        # 2. Burstiness check
        burstiness_score, burst_issues = self._check_burstiness(sentences)
        issues.extend(burst_issues)

        # 3. Transition frequency check
        transition_score, trans_issues = self._check_transitions(sentences)
        issues.extend(trans_issues)

        # 4. Burrows' Delta check
        delta_score, delta_issues = self._check_delta(paragraph)
        issues.extend(delta_issues)

        # Overall score (weighted average)
        overall_score = (
            length_score * 0.25 +
            burstiness_score * 0.25 +
            transition_score * 0.25 +
            delta_score * 0.25
        )

        # Generate suggestions
        if length_score < 0.7:
            suggestions.append(
                f"Adjust sentence lengths toward mean={self.profile.length_profile.mean:.0f}"
            )
        if burstiness_score < 0.7:
            target_burst = self.profile.length_profile.burstiness
            suggestions.append(
                f"Vary sentence lengths more (target burstiness={target_burst:.2f})"
            )
        if transition_score < 0.7:
            no_trans = self.profile.transition_profile.no_transition_ratio
            suggestions.append(
                f"Adjust transition usage ({no_trans*100:.0f}% sentences should have no transition)"
            )

        is_acceptable = overall_score >= 0.6 and len([i for i in issues if "critical" in i.lower()]) == 0

        return VerificationResult(
            is_acceptable=is_acceptable,
            overall_score=overall_score,
            length_score=length_score,
            burstiness_score=burstiness_score,
            transition_score=transition_score,
            delta_score=delta_score,
            issues=issues,
            suggestions=suggestions,
        )

    def _check_lengths(self, sentences: List[str]) -> Tuple[float, List[str]]:
        """Check sentence length distribution."""
        lengths = [len(s.split()) for s in sentences]
        issues = []

        if not lengths:
            return 0.0, ["No sentence lengths"]

        actual_mean = np.mean(lengths)
        target_mean = self.profile.length_profile.mean

        # Score based on mean deviation
        mean_deviation = abs(actual_mean - target_mean) / target_mean
        score = max(0.0, 1.0 - mean_deviation)

        if mean_deviation > 0.3:
            issues.append(
                f"Mean length {actual_mean:.1f} differs from target {target_mean:.1f}"
            )

        return score, issues

    def _check_burstiness(self, sentences: List[str]) -> Tuple[float, List[str]]:
        """Check sentence length burstiness."""
        lengths = [len(s.split()) for s in sentences]
        issues = []

        if len(lengths) < 2:
            return 0.5, []  # Can't check burstiness with 1 sentence

        actual_mean = np.mean(lengths)
        actual_std = np.std(lengths)
        actual_burstiness = actual_std / actual_mean if actual_mean > 0 else 0.0

        target_burstiness = self.profile.length_profile.burstiness
        burstiness_diff = abs(actual_burstiness - target_burstiness)

        score = max(0.0, 1.0 - burstiness_diff / max(target_burstiness, 0.1))

        if burstiness_diff > self.burstiness_tolerance:
            if actual_burstiness < target_burstiness:
                issues.append(
                    f"Burstiness too low ({actual_burstiness:.2f} vs target {target_burstiness:.2f}) - "
                    "add more sentence length variation"
                )
            else:
                issues.append(
                    f"Burstiness too high ({actual_burstiness:.2f} vs target {target_burstiness:.2f}) - "
                    "reduce sentence length variation"
                )

        return score, issues

    def _check_transitions(self, sentences: List[str]) -> Tuple[float, List[str]]:
        """Check transition word usage at sentence START only."""
        issues = []

        # Count sentences that START with transitions
        transition_words = self.profile.transition_profile.get_all_transitions()
        if not transition_words:
            return 0.5, []  # No transition profile

        sentences_with_trans = 0
        for sentence in sentences:
            # Check only the first 1-3 words for transition markers
            words = sentence.lower().split()[:3]
            first_word = words[0] if words else ""
            first_two = " ".join(words[:2]) if len(words) >= 2 else ""
            first_three = " ".join(words[:3]) if len(words) >= 3 else ""

            # Check if sentence STARTS with a transition (multi-word first)
            if any(trans == first_three for trans in transition_words if " " in trans and len(trans.split()) == 3):
                sentences_with_trans += 1
            elif any(trans == first_two for trans in transition_words if " " in trans and len(trans.split()) == 2):
                sentences_with_trans += 1
            elif any(trans == first_word for trans in transition_words if " " not in trans):
                sentences_with_trans += 1

        actual_trans_ratio = sentences_with_trans / len(sentences) if sentences else 0
        target_trans_ratio = 1.0 - self.profile.transition_profile.no_transition_ratio

        ratio_diff = abs(actual_trans_ratio - target_trans_ratio)
        score = max(0.0, 1.0 - ratio_diff / max(target_trans_ratio, 0.1))

        if ratio_diff > self.transition_tolerance:
            if actual_trans_ratio > target_trans_ratio:
                issues.append(
                    f"Too many transitions ({actual_trans_ratio*100:.0f}% vs target {target_trans_ratio*100:.0f}%)"
                )
            else:
                issues.append(
                    f"Too few transitions ({actual_trans_ratio*100:.0f}% vs target {target_trans_ratio*100:.0f}%)"
                )

        return score, issues

    def _check_delta(self, text: str) -> Tuple[float, List[str]]:
        """Check Burrows' Delta distance.

        Uses raw frequency comparison for short texts (< 200 words) since
        z-score normalization is unstable for small samples.
        """
        issues = []

        # Calculate word frequencies
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words:
            return 0.0, ["No words found"]

        total = len(words)
        word_counts = Counter(words)
        frequencies = {word: count / total for word, count in word_counts.items()}

        # For short texts, use raw frequency Delta (more stable)
        # Z-score Delta needs very large samples (1000+ words) to be stable
        if total < 1000:
            delta = self._calculate_raw_delta(frequencies)
            # Adjust threshold for short texts (raw delta is typically smaller)
            adjusted_threshold = 0.3  # Raw frequency delta threshold
            score = max(0.0, 1.0 - delta / adjusted_threshold)

            # Only flag if significantly off
            if delta > adjusted_threshold:
                issues.append(
                    f"Vocabulary divergence {delta:.2f} (short text mode)"
                )
        else:
            # Use standard z-score Delta for longer texts
            delta = self.profile.delta_profile.calculate_delta(frequencies)
            score = max(0.0, 1.0 - delta / self.delta_threshold)

            if delta > self.delta_threshold:
                issues.append(
                    f"Burrows' Delta {delta:.2f} exceeds threshold {self.delta_threshold}"
                )

        return score, issues

    def _calculate_raw_delta(self, text_frequencies: Dict[str, float]) -> float:
        """Calculate raw frequency Delta (no z-scores).

        More stable for short texts. Compares absolute frequency differences
        for top function words only (content-independent).
        """
        # Focus on function words (content-independent)
        function_words = {
            'the', 'of', 'and', 'a', 'to', 'in', 'is', 'that', 'it', 'for',
            'was', 'on', 'are', 'as', 'with', 'be', 'at', 'this', 'have', 'from',
            'or', 'by', 'not', 'but', 'what', 'all', 'were', 'we', 'when', 'there',
            'can', 'an', 'which', 'their', 'if', 'has', 'will', 'one', 'each', 'about',
        }

        delta_sum = 0.0
        count = 0

        mfw = self.profile.delta_profile.mfw_frequencies

        for word in function_words:
            if word in mfw:
                corpus_freq = mfw[word]
                text_freq = text_frequencies.get(word, 0.0)
                delta_sum += abs(corpus_freq - text_freq)
                count += 1

        return delta_sum / count if count > 0 else 0.0


def verify_against_profile(
    text: str,
    profile: AuthorStyleProfile,
) -> VerificationResult:
    """Convenience function to verify text against profile."""
    verifier = StyleVerifier(profile)
    return verifier.verify_paragraph(text)
