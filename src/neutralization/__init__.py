"""Neutralization module for style-free text generation.

This module provides tools for converting stylized text into neutral,
fact-preserving text suitable for style transfer training.

Uses OpenIE-based extraction to preserve all facts while creating
significant structural difference from the original styled prose.
"""

from .openie_flatten import (
    flatten_text,
    flatten_text_simple,
    FlattenedResult,
)

__all__ = ["flatten_text", "flatten_text_simple", "FlattenedResult"]
