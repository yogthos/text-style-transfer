"""Humanization module for making LLM output less detectable.

Addresses key AI-detection signals:
- Mechanical precision (too polished)
- Predictable syntax (uniform sentence structure)
- Low burstiness (uniform sentence length)
- Lack of creative grammar (no fragments, asides)
- Mechanical transitions (formulaic connectors)
"""

from .pattern_injector import PatternInjector, HumanizationConfig
from .corpus_patterns import CorpusPatternExtractor

__all__ = [
    "PatternInjector",
    "HumanizationConfig",
    "CorpusPatternExtractor",
]
