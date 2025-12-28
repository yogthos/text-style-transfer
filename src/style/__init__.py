"""Data-driven style extraction and verification."""

from .profile import (
    SentenceLengthProfile,
    TransitionProfile,
    RegisterProfile,
    DeltaProfile,
    AuthorStyleProfile,
)
from .extractor import StyleProfileExtractor
from .verifier import StyleVerifier
from .blender import (
    AuthorCentroid,
    GhostVector,
    StyleBlender,
    compute_author_centroid,
    create_ghost_vector,
    blend_authors,
    slerp,
)
from .perspective import (
    Perspective,
    PerspectiveTransformer,
    detect_perspective,
    transform_perspective,
)

__all__ = [
    "SentenceLengthProfile",
    "TransitionProfile",
    "RegisterProfile",
    "DeltaProfile",
    "AuthorStyleProfile",
    "StyleProfileExtractor",
    "StyleVerifier",
    # Style blending
    "AuthorCentroid",
    "GhostVector",
    "StyleBlender",
    "compute_author_centroid",
    "create_ghost_vector",
    "blend_authors",
    "slerp",
    # Perspective transformation
    "Perspective",
    "PerspectiveTransformer",
    "detect_perspective",
    "transform_perspective",
]
