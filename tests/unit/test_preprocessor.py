"""Unit tests for text preprocessing."""

import pytest
from src.corpus.preprocessor import TextPreprocessor, ProcessedDocument, ProcessedParagraph


class TestTextPreprocessor:
    """Test TextPreprocessor functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return TextPreprocessor()

    def test_empty_input(self, preprocessor):
        """Test handling of empty input."""
        result = preprocessor.process("")
        assert result.paragraphs == []
        assert result.cleaned_text == ""

    def test_whitespace_only(self, preprocessor):
        """Test handling of whitespace-only input."""
        result = preprocessor.process("   \n\n   ")
        assert result.paragraphs == []

    def test_basic_paragraph_split(self, preprocessor):
        """Test splitting text into paragraphs."""
        text = "First paragraph here.\n\nSecond paragraph here."
        result = preprocessor.process(text)

        assert len(result.paragraphs) == 2
        assert result.paragraphs[0].text == "First paragraph here."
        assert result.paragraphs[1].text == "Second paragraph here."

    def test_sentence_splitting(self, preprocessor):
        """Test splitting paragraphs into sentences."""
        text = "First sentence. Second sentence. Third sentence."
        result = preprocessor.process(text)

        assert len(result.paragraphs) == 1
        assert len(result.paragraphs[0].sentences) == 3

    def test_citation_extraction(self, preprocessor):
        """Test extraction of citations."""
        text = "Some text[^1] and more text[^2] here."
        result = preprocessor.process(text)

        assert len(result.citations) == 2
        assert result.citations[0][0] == "[^1]"
        assert result.citations[1][0] == "[^2]"

    def test_bracket_reference_extraction(self, preprocessor):
        """Test extraction of bracketed references."""
        text = "Research shows[1] that this is true[2]."
        result = preprocessor.process(text)

        assert len(result.citations) == 2
        assert "[1]" in result.citations[0][0]
        assert "[2]" in result.citations[1][0]

    def test_citation_preservation_in_text(self, preprocessor):
        """Test that citations are preserved in cleaned text."""
        text = "Some text[^1] here."
        result = preprocessor.process(text)

        assert "[^1]" in result.cleaned_text

    def test_unicode_normalization(self, preprocessor):
        """Test unicode character normalization."""
        text = "\u201cQuoted text\u201d with \u2018apostrophe\u2019"
        result = preprocessor.process(text)

        # Should normalize fancy quotes to standard ones
        assert '"' in result.cleaned_text or '\u201c' not in result.cleaned_text

    def test_whitespace_normalization(self, preprocessor):
        """Test whitespace cleanup."""
        text = "Multiple    spaces   here.\n\n\n\nMany newlines."
        result = preprocessor.process(text)

        # Multiple spaces should be collapsed
        assert "    " not in result.cleaned_text
        # Multiple newlines should be normalized
        assert "\n\n\n\n" not in result.cleaned_text

    def test_paragraph_role_assignment(self, preprocessor):
        """Test that paragraph roles are assigned correctly."""
        text = "Intro paragraph.\n\nBody paragraph one.\n\nBody paragraph two.\n\nConclusion."
        result = preprocessor.process(text)

        assert len(result.paragraphs) == 4
        assert result.paragraphs[0].role == "INTRO"
        assert result.paragraphs[1].role == "BODY"
        assert result.paragraphs[2].role == "BODY"
        assert result.paragraphs[3].role == "CONCLUSION"

    def test_short_document_roles(self, preprocessor):
        """Test role assignment for short documents."""
        text = "Only one paragraph here."
        result = preprocessor.process(text)

        assert len(result.paragraphs) == 1
        assert result.paragraphs[0].role == "INTRO"

    def test_two_paragraph_roles(self, preprocessor):
        """Test role assignment for two-paragraph documents."""
        text = "First paragraph.\n\nSecond paragraph."
        result = preprocessor.process(text)

        assert len(result.paragraphs) == 2
        assert result.paragraphs[0].role == "INTRO"
        assert result.paragraphs[1].role == "CONCLUSION"

    def test_paragraph_index(self, preprocessor):
        """Test that paragraph indices are assigned correctly."""
        text = "Para one.\n\nPara two.\n\nPara three."
        result = preprocessor.process(text)

        assert result.paragraphs[0].index == 0
        assert result.paragraphs[1].index == 1
        assert result.paragraphs[2].index == 2

    def test_total_sentences_property(self, preprocessor):
        """Test total_sentences property."""
        text = "Sentence one. Sentence two.\n\nSentence three."
        result = preprocessor.process(text)

        assert result.total_sentences == 3

    def test_total_paragraphs_property(self, preprocessor):
        """Test total_paragraphs property."""
        text = "Para one.\n\nPara two.\n\nPara three."
        result = preprocessor.process(text)

        assert result.total_paragraphs == 3

    def test_remove_citations(self, preprocessor):
        """Test citation removal method."""
        text = "Text with[^1] citations[^2] and[3] references."
        clean = preprocessor.remove_citations(text)

        assert "[^1]" not in clean
        assert "[^2]" not in clean
        assert "[3]" not in clean
        assert "Text with citations and references." == clean

    def test_single_newline_fallback(self, preprocessor):
        """Test fallback to single newlines when no double newlines."""
        text = "Line one.\nLine two.\nLine three."
        result = preprocessor.process(text)

        assert len(result.paragraphs) == 3

    def test_complex_text(self, preprocessor):
        """Test processing of complex text with multiple features."""
        text = """Introduction paragraph with some text.

        This is the body paragraph. It has multiple sentences. Including this one.

        Another body paragraph here[^1]. With a citation.

        And the conclusion wraps things up."""

        result = preprocessor.process(text)

        assert result.total_paragraphs == 4
        assert result.total_sentences >= 6
        assert len(result.citations) == 1
        assert result.paragraphs[0].role == "INTRO"
        assert result.paragraphs[-1].role == "CONCLUSION"
