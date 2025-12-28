"""Corpus loader for reading author text files."""

import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Iterator

from .preprocessor import TextPreprocessor, ProcessedDocument
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CorpusDocument:
    """A document from the corpus with metadata."""
    author: str
    filename: str
    content: str
    processed: ProcessedDocument
    content_hash: str
    file_path: Path

    @property
    def id(self) -> str:
        """Unique identifier for this document."""
        return f"{self.author}_{self.content_hash[:12]}"


@dataclass
class Corpus:
    """Collection of documents from one or more authors."""
    documents: List[CorpusDocument] = field(default_factory=list)
    author: Optional[str] = None

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    @property
    def total_paragraphs(self) -> int:
        return sum(d.processed.total_paragraphs for d in self.documents)

    @property
    def total_sentences(self) -> int:
        return sum(d.processed.total_sentences for d in self.documents)

    def __iter__(self) -> Iterator[CorpusDocument]:
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)


class CorpusLoader:
    """Loads and processes text files from author corpus directories.

    Expects a directory structure like:
        styles/
            sample_author1.txt
            sample_author2.txt
    """

    SUPPORTED_EXTENSIONS = {'.txt', '.md'}

    def __init__(
        self,
        base_path: str = "styles",
        preprocessor: Optional[TextPreprocessor] = None
    ):
        """Initialize corpus loader.

        Args:
            base_path: Path to the styles directory.
            preprocessor: Optional preprocessor instance.
        """
        self.base_path = Path(base_path)
        self.preprocessor = preprocessor or TextPreprocessor()

    def load_author(self, author_name: str) -> Corpus:
        """Load corpus for a specific author.

        Args:
            author_name: Name of the author (matches filename pattern).

        Returns:
            Corpus containing all documents for the author.
        """
        corpus = Corpus(author=author_name)

        # Look for files matching the author name
        pattern = f"*{author_name}*"
        files = list(self.base_path.glob(pattern))

        if not files:
            logger.warning(f"No files found for author: {author_name}")
            return corpus

        for file_path in files:
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self._load_file(file_path, author_name)
                if doc:
                    corpus.documents.append(doc)

        logger.info(
            f"Loaded {len(corpus.documents)} documents for {author_name}: "
            f"{corpus.total_paragraphs} paragraphs, {corpus.total_sentences} sentences"
        )

        return corpus

    def load_file(self, file_path: str, author_name: Optional[str] = None) -> Optional[CorpusDocument]:
        """Load a single file.

        Args:
            file_path: Path to the file.
            author_name: Optional author name override.

        Returns:
            CorpusDocument or None if loading fails.
        """
        path = Path(file_path)
        if author_name is None:
            # Extract author from filename
            author_name = self._extract_author_name(path.name)

        return self._load_file(path, author_name)

    def load_text(self, text: str, author_name: str = "unknown") -> CorpusDocument:
        """Load text directly (not from file).

        Args:
            text: Text content.
            author_name: Author name.

        Returns:
            CorpusDocument.
        """
        processed = self.preprocessor.process(text)
        content_hash = self._compute_hash(text)

        return CorpusDocument(
            author=author_name,
            filename="<direct_input>",
            content=text,
            processed=processed,
            content_hash=content_hash,
            file_path=Path(".")
        )

    def load_all(self) -> Corpus:
        """Load all documents from the base path.

        Returns:
            Corpus containing all documents.
        """
        corpus = Corpus()

        if not self.base_path.exists():
            logger.warning(f"Base path does not exist: {self.base_path}")
            return corpus

        for file_path in self.base_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                author_name = self._extract_author_name(file_path.name)
                doc = self._load_file(file_path, author_name)
                if doc:
                    corpus.documents.append(doc)

        logger.info(f"Loaded {len(corpus.documents)} documents from {self.base_path}")

        return corpus

    def _load_file(self, path: Path, author_name: str) -> Optional[CorpusDocument]:
        """Load and process a single file.

        Args:
            path: File path.
            author_name: Author name.

        Returns:
            CorpusDocument or None if loading fails.
        """
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            try:
                content = path.read_text(encoding='latin-1')
            except Exception as e:
                logger.error(f"Failed to read file {path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return None

        if not content.strip():
            logger.warning(f"Empty file: {path}")
            return None

        processed = self.preprocessor.process(content)
        content_hash = self._compute_hash(content)

        doc = CorpusDocument(
            author=author_name,
            filename=path.name,
            content=content,
            processed=processed,
            content_hash=content_hash,
            file_path=path
        )

        logger.debug(
            f"Loaded {path.name}: {processed.total_paragraphs} paragraphs, "
            f"{processed.total_sentences} sentences"
        )

        return doc

    def _extract_author_name(self, filename: str) -> str:
        """Extract author name from filename.

        Expected patterns:
        - sample_author.txt -> author
        - sample_author_clean.txt -> author

        Args:
            filename: Name of the file.

        Returns:
            Extracted author name.
        """
        name = Path(filename).stem  # Remove extension

        # Remove common prefixes
        for prefix in ['sample_', 'corpus_', 'text_']:
            if name.startswith(prefix):
                name = name[len(prefix):]

        # Remove common suffixes
        for suffix in ['_clean', '_raw', '_processed']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        return name

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Text content.

        Returns:
            Hex digest of hash.
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
