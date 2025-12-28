"""Unit tests for corpus loading."""

import os
import tempfile
import pytest
from pathlib import Path

from src.corpus.loader import CorpusLoader, CorpusDocument, Corpus


class TestCorpusLoader:
    """Test CorpusLoader functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample files
            (Path(tmpdir) / "sample_author1.txt").write_text(
                "First paragraph by author one.\n\nSecond paragraph here."
            )
            (Path(tmpdir) / "sample_author2.txt").write_text(
                "Content by author two. Multiple sentences here."
            )
            (Path(tmpdir) / "sample_author1_clean.txt").write_text(
                "Additional content by author one."
            )
            yield tmpdir

    @pytest.fixture
    def loader(self, temp_dir):
        """Create loader with temp directory."""
        return CorpusLoader(base_path=temp_dir)

    def test_load_author(self, loader):
        """Test loading corpus for specific author."""
        corpus = loader.load_author("author1")

        assert len(corpus.documents) == 2
        assert corpus.author == "author1"

    def test_load_author_not_found(self, loader):
        """Test loading non-existent author returns empty corpus."""
        corpus = loader.load_author("nonexistent")

        assert len(corpus.documents) == 0

    def test_load_file(self, temp_dir):
        """Test loading a single file."""
        loader = CorpusLoader(base_path=temp_dir)
        file_path = Path(temp_dir) / "sample_author1.txt"

        doc = loader.load_file(str(file_path))

        assert doc is not None
        assert doc.author == "author1"
        assert doc.processed.total_paragraphs == 2

    def test_load_file_with_author_override(self, temp_dir):
        """Test loading file with author name override."""
        loader = CorpusLoader(base_path=temp_dir)
        file_path = Path(temp_dir) / "sample_author1.txt"

        doc = loader.load_file(str(file_path), author_name="custom_author")

        assert doc.author == "custom_author"

    def test_load_text(self):
        """Test loading text directly."""
        loader = CorpusLoader()
        text = "Some sample text. With two sentences."

        doc = loader.load_text(text, author_name="test_author")

        assert doc.author == "test_author"
        assert doc.filename == "<direct_input>"
        assert doc.processed.total_sentences == 2

    def test_load_all(self, loader):
        """Test loading all files from directory."""
        corpus = loader.load_all()

        assert len(corpus.documents) == 3  # 3 .txt files

    def test_corpus_iteration(self, loader):
        """Test that corpus is iterable."""
        corpus = loader.load_author("author1")

        docs = list(corpus)
        assert len(docs) == 2

    def test_corpus_len(self, loader):
        """Test corpus length."""
        corpus = loader.load_author("author1")

        assert len(corpus) == 2

    def test_corpus_total_paragraphs(self, loader):
        """Test total paragraphs across corpus."""
        corpus = loader.load_author("author1")

        assert corpus.total_paragraphs >= 2

    def test_corpus_total_sentences(self, loader):
        """Test total sentences across corpus."""
        corpus = loader.load_author("author1")

        assert corpus.total_sentences >= 3

    def test_document_id(self, loader):
        """Test that document IDs are generated."""
        corpus = loader.load_author("author1")

        for doc in corpus:
            assert doc.id is not None
            assert doc.author in doc.id

    def test_document_content_hash(self, loader):
        """Test that content hash is computed."""
        corpus = loader.load_author("author1")

        for doc in corpus:
            assert doc.content_hash is not None
            assert len(doc.content_hash) == 64  # SHA-256 hex

    def test_extract_author_name(self):
        """Test author name extraction from filename."""
        loader = CorpusLoader()

        assert loader._extract_author_name("sample_dawkins.txt") == "dawkins"
        assert loader._extract_author_name("sample_author_clean.txt") == "author"
        assert loader._extract_author_name("corpus_test.txt") == "test"
        assert loader._extract_author_name("regular_name.txt") == "regular_name"

    def test_empty_file_handling(self, temp_dir):
        """Test handling of empty files."""
        empty_file = Path(temp_dir) / "empty.txt"
        empty_file.write_text("")

        loader = CorpusLoader(base_path=temp_dir)
        doc = loader.load_file(str(empty_file))

        assert doc is None  # Empty files should be skipped

    def test_supported_extensions(self, temp_dir):
        """Test that only supported extensions are loaded."""
        # Create a non-txt file
        (Path(temp_dir) / "data.json").write_text('{"key": "value"}')

        loader = CorpusLoader(base_path=temp_dir)
        corpus = loader.load_all()

        # Should only load .txt and .md files
        filenames = [doc.filename for doc in corpus]
        assert not any(f.endswith('.json') for f in filenames)

    def test_md_extension_support(self, temp_dir):
        """Test that .md files are supported."""
        (Path(temp_dir) / "sample.md").write_text("# Markdown content\n\nParagraph here.")

        loader = CorpusLoader(base_path=temp_dir)
        corpus = loader.load_all()

        filenames = [doc.filename for doc in corpus]
        assert any(f.endswith('.md') for f in filenames)


class TestCorpusDocument:
    """Test CorpusDocument functionality."""

    def test_document_properties(self):
        """Test basic document properties."""
        loader = CorpusLoader()
        doc = loader.load_text("Test content here.", author_name="test")

        assert doc.author == "test"
        assert doc.content == "Test content here."
        assert doc.processed is not None


class TestCorpus:
    """Test Corpus functionality."""

    def test_empty_corpus(self):
        """Test empty corpus properties."""
        corpus = Corpus()

        assert corpus.total_documents == 0
        assert corpus.total_paragraphs == 0
        assert corpus.total_sentences == 0
        assert len(list(corpus)) == 0
