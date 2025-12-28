"""Ingestion module for text processing."""

from .proposition_extractor import (
    PropositionExtractor,
    SVOTriple,
    PropositionNode,
    # New semantic fidelity dataclasses
    EpistemicStance,
    LogicalRelation,
    ContentAnchor,
)

__all__ = [
    "PropositionExtractor",
    "PropositionNode",
    "SVOTriple",
    # New semantic fidelity dataclasses
    "EpistemicStance",
    "LogicalRelation",
    "ContentAnchor",
]
