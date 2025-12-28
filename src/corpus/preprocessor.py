"""Text preprocessing utilities for corpus ingestion."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

from ..utils.nlp import split_into_sentences, split_into_paragraphs
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedParagraph:
    """A processed paragraph with its sentences."""
    text: str
    sentences: List[str]
    index: int  # Position in document
    role: str = "BODY"  # INTRO, BODY, CONCLUSION


@dataclass
class ProcessedDocument:
    """A fully processed document."""
    original_text: str
    cleaned_text: str
    paragraphs: List[ProcessedParagraph]
    citations: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def total_sentences(self) -> int:
        return sum(len(p.sentences) for p in self.paragraphs)

    @property
    def total_paragraphs(self) -> int:
        return len(self.paragraphs)


class TextPreprocessor:
    """Preprocesses text for corpus ingestion.

    Handles:
    - Unicode normalization
    - Whitespace cleanup
    - Citation extraction and preservation
    - Paragraph and sentence splitting
    - Document role assignment (intro/body/conclusion)
    """

    # Pattern for footnote-style citations [^1], [^2], etc.
    CITATION_PATTERN = re.compile(r'\[\^\d+\]')

    # Pattern for bracketed references [1], [2], etc.
    BRACKET_REF_PATTERN = re.compile(r'\[\d+\]')

    def __init__(self, preserve_citations: bool = True):
        """Initialize preprocessor.

        Args:
            preserve_citations: Whether to preserve citation markers in output.
        """
        self.preserve_citations = preserve_citations

    def process(self, text: str) -> ProcessedDocument:
        """Process text into a structured document.

        Args:
            text: Raw input text.

        Returns:
            ProcessedDocument with paragraphs and sentences.
        """
        if not text or not text.strip():
            return ProcessedDocument(
                original_text=text,
                cleaned_text="",
                paragraphs=[],
                citations=[]
            )

        # Step 1: Normalize unicode
        cleaned = self._normalize_unicode(text)

        # Step 2: Extract citations
        citations = self._extract_citations(cleaned)

        # Step 3: Clean whitespace
        cleaned = self._normalize_whitespace(cleaned)

        # Step 4: Split into paragraphs
        raw_paragraphs = split_into_paragraphs(cleaned)

        # Step 5: Process each paragraph
        paragraphs = []
        for i, para_text in enumerate(raw_paragraphs):
            sentences = split_into_sentences(para_text)
            if sentences:  # Only include non-empty paragraphs
                role = self._determine_role(i, len(raw_paragraphs))
                paragraphs.append(ProcessedParagraph(
                    text=para_text,
                    sentences=sentences,
                    index=i,
                    role=role
                ))

        logger.debug(
            f"Processed document: {len(paragraphs)} paragraphs, "
            f"{sum(len(p.sentences) for p in paragraphs)} sentences"
        )

        return ProcessedDocument(
            original_text=text,
            cleaned_text=cleaned,
            paragraphs=paragraphs,
            citations=citations
        )

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)

        # Replace common unicode variants with ASCII equivalents
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '—',  # Em dash (keep as em dash)
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',  # Non-breaking space
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Clean up whitespace while preserving paragraph breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Collapse multiple newlines to double newlines (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _extract_citations(self, text: str) -> List[Tuple[str, int]]:
        """Extract all citations with their positions.

        Args:
            text: Text to extract citations from.

        Returns:
            List of (citation_text, position) tuples.
        """
        citations = []

        # Extract footnote-style citations
        for match in self.CITATION_PATTERN.finditer(text):
            citations.append((match.group(), match.start()))

        # Extract bracketed references
        for match in self.BRACKET_REF_PATTERN.finditer(text):
            citations.append((match.group(), match.start()))

        # Sort by position
        citations.sort(key=lambda x: x[1])

        return citations

    def _determine_role(self, index: int, total: int) -> str:
        """Determine the role of a paragraph based on its position.

        Args:
            index: Paragraph index (0-based).
            total: Total number of paragraphs.

        Returns:
            Role string: INTRO, BODY, or CONCLUSION.
        """
        if total <= 2:
            # Short documents: first is intro, last is conclusion
            if index == 0:
                return "INTRO"
            return "CONCLUSION" if index == total - 1 else "BODY"

        # Longer documents
        if index == 0:
            return "INTRO"
        elif index == total - 1:
            return "CONCLUSION"
        else:
            return "BODY"

    def remove_citations(self, text: str) -> str:
        """Remove all citation markers from text.

        Args:
            text: Text with citations.

        Returns:
            Text with citations removed.
        """
        text = self.CITATION_PATTERN.sub('', text)
        text = self.BRACKET_REF_PATTERN.sub('', text)
        return text
