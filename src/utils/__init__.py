"""Utility modules for the style transfer pipeline."""

from .logging import (
    get_logger,
    setup_logging,
    set_request_id,
    get_request_id,
    log_llm_call,
)
from .nlp import (
    get_nlp,
    split_into_sentences,
    split_into_paragraphs,
    extract_citations,
    remove_citations,
    extract_entities,
    extract_keywords,
    count_words,
    calculate_burstiness,
    get_pos_distribution,
    get_dependency_depth,
    detect_perspective,
    setup_nltk,
)

__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    "set_request_id",
    "get_request_id",
    "log_llm_call",
    # NLP
    "get_nlp",
    "split_into_sentences",
    "split_into_paragraphs",
    "extract_citations",
    "remove_citations",
    "extract_entities",
    "extract_keywords",
    "count_words",
    "calculate_burstiness",
    "get_pos_distribution",
    "get_dependency_depth",
    "detect_perspective",
    "setup_nltk",
]
