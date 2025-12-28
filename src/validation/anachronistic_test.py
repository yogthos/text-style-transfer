"""Anachronistic testing for style transfer validation.

Tests that style transfers to modern contexts that couldn't exist in source material.
This validates style capture vs. content memorization.

Based on research from arXiv 2510.13939:
- Modern scenarios (smartphones, social media) test generalization
- If style transfers to new contexts, it's genuine style capture
- If output contains source phrases, it's memorization
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from collections import Counter

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Modern scenarios that couldn't exist in historical author corpora
# These test whether style generalizes beyond source material
ANACHRONISTIC_PROMPTS = [
    # Technology
    "A character checking their smartphone notifications after a long day",
    "The experience of scrolling through a social media feed late at night",
    "A person receiving a text message with unexpected news",
    "Two colleagues collaborating on a video call from different cities",
    "The frustration of waiting for a slow app to load",

    # Modern life
    "A first date at a trendy coffee shop with craft lattes",
    "The feeling of an empty inbox after clearing all emails",
    "A rideshare pickup in the rain, tracking the car on the app",
    "Ordering food delivery and watching the driver's location",
    "The glow of multiple screens in a dark room",

    # Contemporary themes
    "A conversation about work-life balance and burnout",
    "The anxiety of posting content and waiting for likes",
    "A person deciding whether to swipe left or right",
    "The loneliness of being connected but feeling disconnected",
    "A moment of digital detox, phone left behind",
]


@dataclass
class AnachronisticTestResult:
    """Result of testing style transfer on a modern scenario."""

    prompt: str
    output: str
    word_count: int = 0

    # Style metrics
    style_score: float = 0.0  # How well it matches target style

    # Novelty metrics (higher = better generalization)
    novelty_ratio: float = 0.0  # % of n-grams not in source
    source_overlap: List[str] = field(default_factory=list)  # Copied phrases

    # Quality checks
    is_fluent: bool = True
    is_on_topic: bool = True
    contains_modern_elements: bool = True  # Did it address the modern context?

    @property
    def passes_novelty_check(self) -> bool:
        """True if output is sufficiently novel (>95% novel n-grams)."""
        return self.novelty_ratio > 0.95 and len(self.source_overlap) == 0


@dataclass
class AnachronisticTestSuite:
    """Complete test suite for anachronistic validation."""

    results: List[AnachronisticTestResult] = field(default_factory=list)
    corpus_ngrams: set = field(default_factory=set)  # For novelty checking

    @property
    def pass_rate(self) -> float:
        """Percentage of tests that pass novelty check."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.passes_novelty_check)
        return passed / len(self.results)

    @property
    def average_novelty(self) -> float:
        """Average novelty ratio across all tests."""
        if not self.results:
            return 0.0
        return sum(r.novelty_ratio for r in self.results) / len(self.results)

    @property
    def average_style_score(self) -> float:
        """Average style score across all tests."""
        if not self.results:
            return 0.0
        return sum(r.style_score for r in self.results) / len(self.results)

    def get_copied_phrases(self) -> List[str]:
        """Get all phrases that were copied from source."""
        phrases = []
        for r in self.results:
            phrases.extend(r.source_overlap)
        return list(set(phrases))


def get_ngrams(text: str, n: int = 5) -> set:
    """Extract n-grams from text for novelty checking.

    Args:
        text: Input text.
        n: N-gram size (default 5 words).

    Returns:
        Set of n-gram tuples.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return set()

    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        ngrams.add(ngram)

    return ngrams


def calculate_novelty(
    output: str,
    corpus_ngrams: set,
    ngram_size: int = 5
) -> tuple:
    """Calculate novelty ratio and find overlapping phrases.

    Args:
        output: Generated output text.
        corpus_ngrams: Set of n-grams from training corpus.
        ngram_size: Size of n-grams to check.

    Returns:
        Tuple of (novelty_ratio, overlapping_phrases).
    """
    output_ngrams = get_ngrams(output, ngram_size)

    if not output_ngrams:
        return 1.0, []

    overlap = output_ngrams & corpus_ngrams
    novelty_ratio = 1 - (len(overlap) / len(output_ngrams))

    # Convert overlapping n-grams back to phrases
    overlapping_phrases = [" ".join(ng) for ng in overlap]

    return novelty_ratio, overlapping_phrases


def build_corpus_ngrams(corpus_text: str, ngram_size: int = 5) -> set:
    """Build set of n-grams from corpus for novelty checking.

    Args:
        corpus_text: Full corpus text.
        ngram_size: Size of n-grams to extract.

    Returns:
        Set of n-gram tuples.
    """
    return get_ngrams(corpus_text, ngram_size)


def run_anachronistic_tests(
    generator_fn: Callable[[str], str],
    corpus_text: str,
    style_scorer: Optional[Callable[[str], float]] = None,
    prompts: Optional[List[str]] = None,
    ngram_size: int = 5,
) -> AnachronisticTestSuite:
    """Run anachronistic tests on a style transfer generator.

    Args:
        generator_fn: Function that takes a prompt and returns styled text.
        corpus_text: Original corpus text for novelty checking.
        style_scorer: Optional function to score style match (0-1).
        prompts: Optional custom prompts (defaults to ANACHRONISTIC_PROMPTS).
        ngram_size: N-gram size for novelty checking.

    Returns:
        AnachronisticTestSuite with all test results.
    """
    test_prompts = prompts or ANACHRONISTIC_PROMPTS
    corpus_ngrams = build_corpus_ngrams(corpus_text, ngram_size)

    suite = AnachronisticTestSuite(corpus_ngrams=corpus_ngrams)

    logger.info(f"Running {len(test_prompts)} anachronistic tests...")

    for i, prompt in enumerate(test_prompts):
        logger.debug(f"Test {i+1}/{len(test_prompts)}: {prompt[:50]}...")

        try:
            # Generate styled output
            output = generator_fn(prompt)

            # Calculate novelty
            novelty_ratio, overlap = calculate_novelty(
                output, corpus_ngrams, ngram_size
            )

            # Score style if scorer provided
            style_score = style_scorer(output) if style_scorer else 0.5

            # Check if output addresses modern elements
            modern_keywords = {
                'phone', 'app', 'screen', 'text', 'message', 'social',
                'media', 'email', 'video', 'call', 'online', 'digital',
                'coffee', 'latte', 'uber', 'lyft', 'rideshare', 'delivery',
            }
            output_words = set(output.lower().split())
            contains_modern = bool(output_words & modern_keywords)

            result = AnachronisticTestResult(
                prompt=prompt,
                output=output,
                word_count=len(output.split()),
                style_score=style_score,
                novelty_ratio=novelty_ratio,
                source_overlap=overlap,
                contains_modern_elements=contains_modern,
            )

        except Exception as e:
            logger.warning(f"Test failed: {e}")
            result = AnachronisticTestResult(
                prompt=prompt,
                output=f"Error: {e}",
                is_fluent=False,
            )

        suite.results.append(result)

    logger.info(f"Anachronistic test results:")
    logger.info(f"  Pass rate: {suite.pass_rate:.1%}")
    logger.info(f"  Average novelty: {suite.average_novelty:.1%}")
    logger.info(f"  Average style score: {suite.average_style_score:.2f}")

    if suite.get_copied_phrases():
        logger.warning(f"  Copied phrases found: {suite.get_copied_phrases()[:5]}")

    return suite


def validate_style_generalization(
    suite: AnachronisticTestSuite,
    min_pass_rate: float = 0.9,
    min_novelty: float = 0.95,
) -> bool:
    """Validate that style transfer is generalizing, not memorizing.

    Args:
        suite: Test suite with results.
        min_pass_rate: Minimum required pass rate (default 90%).
        min_novelty: Minimum required average novelty (default 95%).

    Returns:
        True if validation passes.
    """
    if suite.pass_rate < min_pass_rate:
        logger.warning(f"Pass rate {suite.pass_rate:.1%} below threshold {min_pass_rate:.1%}")
        return False

    if suite.average_novelty < min_novelty:
        logger.warning(f"Novelty {suite.average_novelty:.1%} below threshold {min_novelty:.1%}")
        return False

    logger.info("Style generalization validation PASSED")
    return True
