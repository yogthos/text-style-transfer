"""Pattern injector for humanizing LLM output.

Injects human writing patterns into generation prompts:
- Request sentence fragments at corpus-appropriate frequency
- Mix rhetorical questions
- Include parenthetical asides with dashes
- Enforce extreme sentence length variation
- Use unconventional openers
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random

from .corpus_patterns import HumanPatterns, CorpusPatternExtractor
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HumanizationConfig:
    """Configuration for humanization."""

    # Probability of requesting a fragment
    fragment_probability: float = 0.05

    # Probability of requesting a question
    question_probability: float = 0.08

    # Probability of including a dash-aside
    dash_probability: float = 0.10

    # Probability of very short sentence (<8 words)
    short_probability: float = 0.15

    # Probability of very long sentence (>40 words)
    long_probability: float = 0.10

    # Probability of unconventional opener
    opener_probability: float = 0.20

    # Enable/disable humanization
    enabled: bool = True


class PatternInjector:
    """Inject human writing patterns into generation.

    Works by modifying generation prompts to request specific
    human-like patterns based on corpus analysis.
    """

    def __init__(
        self,
        patterns: Optional[HumanPatterns] = None,
        config: Optional[HumanizationConfig] = None,
    ):
        """Initialize pattern injector.

        Args:
            patterns: Human patterns extracted from corpus.
            config: Humanization configuration.
        """
        self.patterns = patterns
        self.config = config or HumanizationConfig()

        # Calibrate config from patterns if available
        if patterns:
            self._calibrate_from_patterns(patterns)

        # Track what we've injected for variety
        self._recent_patterns: List[str] = []
        self._sentence_count = 0

    def _calibrate_from_patterns(self, patterns: HumanPatterns):
        """Calibrate probabilities from corpus patterns."""
        # Scale probabilities based on what the author actually does
        self.config.fragment_probability = min(patterns.fragment_ratio * 1.5, 0.15)
        self.config.question_probability = min(patterns.question_ratio * 1.2, 0.15)
        self.config.dash_probability = min(patterns.dash_ratio * 1.2, 0.20)
        self.config.short_probability = min(patterns.short_ratio * 1.1, 0.25)
        self.config.long_probability = min(patterns.long_ratio * 1.1, 0.15)

        logger.info(
            f"Calibrated humanization: "
            f"fragment={self.config.fragment_probability:.2f}, "
            f"question={self.config.question_probability:.2f}, "
            f"dash={self.config.dash_probability:.2f}, "
            f"short={self.config.short_probability:.2f}, "
            f"long={self.config.long_probability:.2f}"
        )

    def get_pattern_request(self, position: str = "middle") -> Optional[str]:
        """Get a pattern request to inject into the prompt.

        Args:
            position: Position in paragraph ("start", "middle", "end").

        Returns:
            Pattern request string to add to prompt, or None.
        """
        if not self.config.enabled:
            return None

        self._sentence_count += 1

        # Don't repeat patterns too frequently
        if len(self._recent_patterns) >= 3:
            self._recent_patterns.pop(0)

        # Roll for each pattern type
        pattern = None

        # Fragments work best mid-paragraph for emphasis
        if (
            position == "middle"
            and "fragment" not in self._recent_patterns
            and random.random() < self.config.fragment_probability
        ):
            pattern = self._get_fragment_request()
            self._recent_patterns.append("fragment")

        # Questions work at any position
        elif (
            "question" not in self._recent_patterns
            and random.random() < self.config.question_probability
        ):
            pattern = self._get_question_request()
            self._recent_patterns.append("question")

        # Dash asides for mid-paragraph
        elif (
            position == "middle"
            and "dash" not in self._recent_patterns
            and random.random() < self.config.dash_probability
        ):
            pattern = self._get_dash_request()
            self._recent_patterns.append("dash")

        # Short sentences for punch
        elif (
            "short" not in self._recent_patterns
            and random.random() < self.config.short_probability
        ):
            pattern = self._get_short_request()
            self._recent_patterns.append("short")

        # Long sentences for complexity
        elif (
            "long" not in self._recent_patterns
            and random.random() < self.config.long_probability
        ):
            pattern = self._get_long_request()
            self._recent_patterns.append("long")

        # Unconventional openers
        elif (
            "opener" not in self._recent_patterns
            and random.random() < self.config.opener_probability
        ):
            pattern = self._get_opener_request()
            self._recent_patterns.append("opener")

        return pattern

    def _get_fragment_request(self) -> str:
        """Get request for a sentence fragment.

        IMPORTANT: Do NOT show corpus examples - they contaminate with author's topics.
        Only describe the STRUCTURAL pattern.
        """
        return "Write as a FRAGMENT (incomplete sentence, noun phrase only, no main verb)."

    def _get_question_request(self) -> str:
        """Get request for a rhetorical question.

        IMPORTANT: Do NOT show corpus examples - they contaminate with author's topics.
        Only describe the STRUCTURAL pattern.
        """
        return "Phrase this as a RHETORICAL QUESTION about the idea (ends with ?)."

    def _get_dash_request(self) -> str:
        """Get request for em-dash aside.

        IMPORTANT: Do NOT show corpus examples - they contaminate with author's topics.
        Only describe the STRUCTURAL pattern.
        """
        return "Include an EM-DASH aside (—a parenthetical comment—) mid-sentence."

    def _get_short_request(self) -> str:
        """Get request for very short sentence.

        IMPORTANT: Do NOT show corpus examples - they contaminate with author's topics.
        Only describe the STRUCTURAL pattern.
        """
        target = random.randint(4, 7)
        return f"Write VERY SHORT ({target} words max). Be punchy."

    def _get_long_request(self) -> str:
        """Get request for very long sentence."""
        target = random.randint(40, 55)
        return f"Write a LONG complex sentence ({target}+ words) with multiple clauses."

    def _get_opener_request(self) -> str:
        """Get request for unconventional opener.

        Extract opener WORDS only (not full sentences) to avoid topic contamination.
        """
        openers = ["And", "But", "So", "Yet", "Now", "Indeed"]
        if self.patterns and self.patterns.unconventional_openers:
            # Extract ONLY the opener words, not the full sentences
            for sent in self.patterns.unconventional_openers[:10]:
                word = sent.split()[0].rstrip(",")
                if word.capitalize() not in openers:
                    openers.append(word.capitalize())

        opener = random.choice(openers[:8])
        return f"Start with '{opener}' (unconventional opener)."

    def modify_length_target(
        self,
        base_target: int,
        position: str = "middle"
    ) -> int:
        """Modify sentence length target for burstiness.

        Args:
            base_target: Original length target from Markov chain.
            position: Position in paragraph.

        Returns:
            Modified target length.
        """
        if not self.config.enabled:
            return base_target

        # Add more variation than Markov provides
        if random.random() < self.config.short_probability:
            return random.randint(5, 10)
        elif random.random() < self.config.long_probability:
            return random.randint(40, 55)
        else:
            # Add noise to base target
            variation = random.randint(-5, 8)
            return max(8, base_target + variation)

    def get_stats(self) -> Dict:
        """Get humanization statistics."""
        return {
            "enabled": self.config.enabled,
            "sentences_processed": self._sentence_count,
            "patterns_injected": len(self._recent_patterns),
            "config": {
                "fragment_prob": self.config.fragment_probability,
                "question_prob": self.config.question_probability,
                "dash_prob": self.config.dash_probability,
                "short_prob": self.config.short_probability,
                "long_prob": self.config.long_probability,
            }
        }


def create_pattern_injector(paragraphs: List[str]) -> PatternInjector:
    """Create a pattern injector from corpus paragraphs.

    Args:
        paragraphs: Author's corpus paragraphs.

    Returns:
        Configured PatternInjector.
    """
    extractor = CorpusPatternExtractor()
    patterns = extractor.extract(paragraphs)

    logger.info(
        f"Extracted human patterns: "
        f"{len(patterns.fragments)} fragments, "
        f"{len(patterns.questions)} questions, "
        f"{len(patterns.dash_patterns)} dash asides, "
        f"{len(patterns.short_sentences)} short, "
        f"{len(patterns.long_sentences)} long"
    )

    return PatternInjector(patterns=patterns)
