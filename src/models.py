from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class StyleProfile:
    """Represents the style characteristics extracted from a sample text."""

    vocab_map: Dict[str, List[str]]
    """Dictionary mapping generic words to sample-specific synonyms."""

    pos_markov_chain: np.ndarray
    """2D NumPy array representing Part-of-Speech tag transition probabilities.

    Each row represents a source POS tag, and each column represents a target POS tag.
    Values are probabilities that sum to 1.0 for each row.
    """

    sentence_flow_markov: np.ndarray
    """2D NumPy array representing sentence type transition probabilities.

    Each row represents a source sentence type (e.g., Simple, Compound, Complex, Fragment),
    and each column represents a target sentence type.
    Values are probabilities that sum to 1.0 for each row.
    """


@dataclass
class ContentUnit:
    """Represents the semantic content extracted from an input text segment."""

    svo_triples: List[Tuple[str, str, str]]
    """List of (Subject, Verb, Object) tuples extracted from the text."""

    entities: List[str]
    """List of named entities to preserve verbatim in the output."""

    original_text: str
    """The raw input segment from which this content was extracted."""

    content_words: List[str] = None
    """List of all important content words (nouns, verbs, adjectives) to preserve."""

    paragraph_idx: int = 0
    """Index of paragraph (0-based) in the overall text."""

    sentence_idx: int = 0
    """Index of sentence within paragraph (0-based)."""

    is_first_paragraph: bool = False
    """Whether this is the first paragraph in the text."""

    is_last_paragraph: bool = False
    """Whether this is the last paragraph in the text."""

    is_first_sentence: bool = False
    """Whether this is the first sentence in the paragraph."""

    is_last_sentence: bool = False
    """Whether this is the last sentence in the paragraph."""

    total_paragraphs: int = 1
    """Total number of paragraphs in the input text."""

    paragraph_length: int = 1
    """Number of sentences in the current paragraph."""

