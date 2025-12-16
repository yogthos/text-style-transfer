"""
Phrase Distribution Analyzer Module

Analyzes paragraph opener phrases from sample text to build statistical
distributions. Ensures output matches sample's phrase frequency distribution
rather than just avoiding recently used phrases.
"""

import spacy
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path


@dataclass
class PhraseDistribution:
    """Statistical distribution of paragraph opener phrases."""
    phrase_frequencies: Dict[str, float]  # Normalized frequencies (0.0-1.0)
    phrase_counts: Dict[str, int]  # Raw counts
    total_paragraphs: int
    role_based_distributions: Dict[str, Dict[str, float]]  # By structural role


class PhraseDistributionAnalyzer:
    """
    Analyzes paragraph opener phrase distributions from sample text.

    Extracts first 2-3 words of each paragraph, counts frequencies,
    and builds normalized distributions for matching output to sample.
    """

    def __init__(self, sample_text: str, phrase_length: int = 3):
        """
        Initialize phrase distribution analyzer.

        Args:
            sample_text: The sample text to analyze
            phrase_length: Number of words to extract as opener phrase (2-3)
        """
        self.sample_text = sample_text
        self.phrase_length = max(2, min(3, phrase_length))  # Clamp to 2-3

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Analyze sample
        self.distribution: Optional[PhraseDistribution] = None
        self._analyze_sample()

    def _analyze_sample(self):
        """Extract and analyze phrase distributions from sample text."""
        paragraphs = [p.strip() for p in self.sample_text.split('\n\n') if p.strip()]

        if not paragraphs:
            self.distribution = PhraseDistribution(
                phrase_frequencies={},
                phrase_counts={},
                total_paragraphs=0,
                role_based_distributions={}
            )
            return

        # Extract opener phrases
        phrase_counts = Counter()
        role_phrase_counts = defaultdict(Counter)

        for i, para in enumerate(paragraphs):
            # Skip very short paragraphs (likely headers)
            if len(para.split()) < 10:
                continue

            # Extract opener phrase
            opener_phrase = self._extract_opener_phrase(para)
            if not opener_phrase:
                continue

            phrase_counts[opener_phrase] += 1

            # Determine role for role-based distribution
            role = self._determine_role(i, len(paragraphs), para)
            role_phrase_counts[role][opener_phrase] += 1

        # Normalize frequencies
        total = sum(phrase_counts.values())
        phrase_frequencies = {
            phrase: count / total if total > 0 else 0.0
            for phrase, count in phrase_counts.items()
        }

        # Normalize role-based distributions
        role_based_distributions = {}
        for role, role_counts in role_phrase_counts.items():
            role_total = sum(role_counts.values())
            role_based_distributions[role] = {
                phrase: count / role_total if role_total > 0 else 0.0
                for phrase, count in role_counts.items()
            }

        self.distribution = PhraseDistribution(
            phrase_frequencies=phrase_frequencies,
            phrase_counts=dict(phrase_counts),
            total_paragraphs=len(paragraphs),
            role_based_distributions=role_based_distributions
        )

    def _extract_opener_phrase(self, text: str) -> str:
        """
        Extract opener phrase from paragraph (first 2-3 words).

        Returns normalized phrase (lowercase, trimmed).
        """
        if not text:
            return ""

        # Get first sentence
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return ""

        first_sent = sentences[0]
        tokens = [t for t in first_sent if not t.is_space and not t.is_punct]

        # Extract first N words (normalized)
        if len(tokens) >= self.phrase_length:
            phrase_tokens = tokens[:self.phrase_length]
        elif len(tokens) >= 2:
            phrase_tokens = tokens[:2]
        elif tokens:
            phrase_tokens = tokens[:1]
        else:
            return ""

        # Join and normalize
        phrase = ' '.join([t.text.lower().strip() for t in phrase_tokens])
        phrase = ' '.join(phrase.split())  # Normalize whitespace

        # Filter out very generic phrases (only stop words)
        words = phrase.split()
        if all(self._is_stop_word(w) for w in words):
            return ""

        return phrase

    def _is_stop_word(self, word: str) -> bool:
        """Check if word is a stop word."""
        return word.lower() in {
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by',
            'from', 'as', 'it', 'its', 'it\'s'
        }

    def _determine_role(self, index: int, total: int, para: str) -> str:
        """Determine structural role of a paragraph."""
        # Check if section start
        first_line = para.split('\n')[0].strip()
        section_patterns = [
            r'^[0-9]+\)',
            r'^[a-z]\)',
            r'^\d+\.',
            r'^[IVX]+\.',
            r'^#+\s',
        ]
        if any(re.match(p, first_line) for p in section_patterns):
            return 'section_opener'

        # Check position
        if index == 0:
            return 'section_opener'
        elif index >= total - 2:
            return 'closer'
        elif index < 3:
            return 'paragraph_opener'
        else:
            return 'body'

    def get_distribution(self) -> PhraseDistribution:
        """Get the phrase distribution."""
        return self.distribution

    def get_underrepresented_phrases(self,
                                     output_phrase_counts: Dict[str, int],
                                     output_total: int,
                                     role: Optional[str] = None) -> List[tuple]:
        """
        Get phrases that are underrepresented in output compared to sample.

        Args:
            output_phrase_counts: Current phrase usage counts in output
            output_total: Total paragraphs in output so far
            role: Optional role to filter by

        Returns:
            List of (phrase, underrepresentation_score) tuples, sorted by score
        """
        if not self.distribution or output_total == 0:
            return []

        # Get appropriate distribution
        if role and role in self.distribution.role_based_distributions:
            sample_dist = self.distribution.role_based_distributions[role]
        else:
            sample_dist = self.distribution.phrase_frequencies

        # Calculate output distribution
        output_dist = {
            phrase: count / output_total
            for phrase, count in output_phrase_counts.items()
        }

        # Find underrepresented phrases
        underrepresented = []
        for phrase, sample_freq in sample_dist.items():
            output_freq = output_dist.get(phrase, 0.0)

            # If phrase is used less in output than in sample, it's underrepresented
            if output_freq < sample_freq:
                underrep_score = sample_freq - output_freq
                underrepresented.append((phrase, underrep_score))

        # Sort by underrepresentation score (highest first)
        underrepresented.sort(key=lambda x: -x[1])
        return underrepresented

    def score_phrase_distribution_match(self,
                                       phrase: str,
                                       output_phrase_counts: Dict[str, int],
                                       output_total: int,
                                       role: Optional[str] = None) -> float:
        """
        Score how well using this phrase would match sample distribution.

        Returns higher score for underrepresented phrases, lower for overused.

        Args:
            phrase: The phrase to score
            output_phrase_counts: Current phrase usage counts
            output_total: Total paragraphs in output
            role: Optional role to filter by

        Returns:
            Score from 0.0 to 2.0 (higher = better match to distribution)
        """
        if not self.distribution or output_total == 0:
            return 1.0  # Neutral if no data

        # Get appropriate distribution
        if role and role in self.distribution.role_based_distributions:
            sample_freq = self.distribution.role_based_distributions[role].get(phrase, 0.0)
        else:
            sample_freq = self.distribution.phrase_frequencies.get(phrase, 0.0)

        # Calculate current output frequency
        output_freq = output_phrase_counts.get(phrase, 0) / output_total if output_total > 0 else 0.0

        # Score based on underrepresentation
        if output_freq < sample_freq:
            # Underrepresented: boost score
            boost = (sample_freq - output_freq) * 2.0  # Scale boost
            return 1.0 + min(boost, 1.0)  # Cap at 2.0
        elif output_freq > sample_freq:
            # Overused: penalize score
            penalty = (output_freq - sample_freq) * 2.0  # Scale penalty
            return max(0.0, 1.0 - min(penalty, 1.0))  # Floor at 0.0
        else:
            # Matches distribution: neutral
            return 1.0


# Test function
if __name__ == '__main__':
    from pathlib import Path

    sample_path = Path(__file__).parent / "prompts" / "sample_mao.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("=== Phrase Distribution Analyzer Test ===\n")

        analyzer = PhraseDistributionAnalyzer(sample_text, phrase_length=3)
        dist = analyzer.get_distribution()

        print(f"Total paragraphs analyzed: {dist.total_paragraphs}")
        print(f"Unique opener phrases: {len(dist.phrase_frequencies)}")
        print(f"\nTop 20 most common opener phrases:")
        sorted_phrases = sorted(dist.phrase_frequencies.items(), key=lambda x: -x[1])
        for phrase, freq in sorted_phrases[:20]:
            count = dist.phrase_counts[phrase]
            print(f"  '{phrase}': {freq:.2%} ({count} times)")

        # Test underrepresentation
        print(f"\n\nTest underrepresentation:")
        output_counts = {'contrary to': 5, 'in this': 2}
        output_total = 8
        underrepresented = analyzer.get_underrepresented_phrases(output_counts, output_total)
        print(f"Underrepresented phrases (top 10):")
        for phrase, score in underrepresented[:10]:
            print(f"  '{phrase}': {score:.3f}")
    else:
        print("No sample file found.")

