"""NLP utilities using spaCy and NLTK."""

import re
from typing import List, Optional, Tuple

from .logging import get_logger

logger = get_logger(__name__)

# Lazy-loaded spaCy model
_nlp = None


def get_nlp():
    """Get the spaCy NLP model, loading it if necessary.

    Returns:
        spaCy Language model.

    Raises:
        RuntimeError: If spaCy model cannot be loaded.
    """
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Prefer large model with word vectors, fall back to smaller models
            models = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
            for model_name in models:
                try:
                    _nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            else:
                # No model found, download and use large model
                logger.info("Downloading spaCy model en_core_web_lg...")
                from spacy.cli import download
                download("en_core_web_lg")
                _nlp = spacy.load("en_core_web_lg")
                logger.info("Downloaded and loaded spaCy model: en_core_web_lg")
        except ImportError:
            raise RuntimeError("spaCy is required. Install with: pip install spacy")
    return _nlp


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy.

    Preserves citations like [^1] and handles edge cases.

    Args:
        text: Input text.

    Returns:
        List of sentence strings.
    """
    if not text or not text.strip():
        return []

    nlp = get_nlp()
    doc = nlp(text)

    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:
            sentences.append(sent_text)

    return sentences


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.

    Args:
        text: Input text.

    Returns:
        List of paragraph strings (non-empty).
    """
    if not text or not text.strip():
        return []

    # Try double newlines first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # If no double newlines, try single newlines
    if len(paragraphs) == 1 and '\n' in text:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    return paragraphs


def extract_citations(text: str) -> List[Tuple[str, int]]:
    """Extract citations from text.

    Args:
        text: Input text.

    Returns:
        List of (citation, position) tuples.
    """
    pattern = r'\[\^\d+\]'
    citations = []
    for match in re.finditer(pattern, text):
        citations.append((match.group(), match.start()))
    return citations


def remove_citations(text: str) -> str:
    """Remove citations from text.

    Args:
        text: Input text.

    Returns:
        Text with citations removed.
    """
    return re.sub(r'\[\^\d+\]', '', text)


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from text.

    Args:
        text: Input text.

    Returns:
        List of (entity_text, entity_label) tuples.
    """
    nlp = get_nlp()
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords (nouns and verbs) from text.

    Args:
        text: Input text.
        top_n: Maximum number of keywords to return.

    Returns:
        List of lemmatized keywords.
    """
    nlp = get_nlp()
    doc = nlp(text)

    # Extract nouns and verbs, lemmatize them
    keywords = []
    for token in doc:
        if token.pos_ in ("NOUN", "VERB", "PROPN") and not token.is_stop:
            lemma = token.lemma_.lower()
            if lemma not in keywords and len(lemma) > 2:
                keywords.append(lemma)

    return keywords[:top_n]


def count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Input text.

    Returns:
        Word count.
    """
    if not text:
        return 0
    return len(text.split())


def calculate_burstiness(sentences: List[str]) -> float:
    """Calculate burstiness (coefficient of variation of sentence lengths).

    Args:
        sentences: List of sentences.

    Returns:
        Burstiness value (0 = uniform, higher = more variable).
    """
    if len(sentences) < 2:
        return 0.0

    lengths = [count_words(s) for s in sentences]
    mean_length = sum(lengths) / len(lengths)

    if mean_length == 0:
        return 0.0

    variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5

    return std_dev / mean_length


def get_pos_distribution(text: str) -> dict:
    """Get POS tag distribution for text.

    Args:
        text: Input text.

    Returns:
        Dictionary of POS tag to count.
    """
    nlp = get_nlp()
    doc = nlp(text)

    distribution = {}
    for token in doc:
        pos = token.pos_
        distribution[pos] = distribution.get(pos, 0) + 1

    return distribution


def get_dependency_depth(text: str) -> float:
    """Get average dependency tree depth for sentences.

    Higher values indicate more complex sentence structures.

    Args:
        text: Input text.

    Returns:
        Average dependency depth.
    """
    nlp = get_nlp()
    doc = nlp(text)

    depths = []
    for sent in doc.sents:
        # Find max depth in this sentence
        max_depth = 0
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            max_depth = max(max_depth, depth)
        depths.append(max_depth)

    if not depths:
        return 0.0

    return sum(depths) / len(depths)


def detect_perspective(text: str) -> str:
    """Detect the perspective (first/third person) of text.

    Args:
        text: Input text.

    Returns:
        One of: "first_person_singular", "first_person_plural", "third_person"
    """
    text_lower = text.lower()

    # Count first-person pronouns
    first_singular = len(re.findall(r'\b(i|me|my|mine|myself)\b', text_lower))
    first_plural = len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text_lower))
    third = len(re.findall(r'\b(he|she|it|they|him|her|them|his|hers|its|their)\b', text_lower))

    if first_singular > first_plural and first_singular > third:
        return "first_person_singular"
    elif first_plural > first_singular and first_plural > third:
        return "first_person_plural"
    else:
        return "third_person"


def setup_nltk() -> None:
    """Download required NLTK data if not present."""
    try:
        import nltk

        required_packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'wordnet',
            'vader_lexicon'
        ]

        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt'
                              else f'taggers/{package}' if 'tagger' in package
                              else f'corpora/{package}' if package == 'wordnet'
                              else f'sentiment/{package}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)

    except ImportError:
        logger.warning("NLTK not installed, some features may be limited")


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts using spaCy word vectors.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity score between 0 and 1.
    """
    nlp = get_nlp()
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # If either doc has no vector, return 0
    if not doc1.has_vector or not doc2.has_vector:
        return 0.0

    return doc1.similarity(doc2)


def classify_by_similarity(
    text: str,
    prototypes: dict,
    threshold: float = 0.3
) -> Tuple[str, float]:
    """Classify text by computing similarity against prototype phrases.

    Uses spaCy word vectors to find the most similar prototype category.

    Args:
        text: Text to classify.
        prototypes: Dict mapping category names to list of prototype phrases.
        threshold: Minimum similarity to consider a match.

    Returns:
        Tuple of (best_category, similarity_score).
    """
    nlp = get_nlp()
    doc = nlp(text)

    if not doc.has_vector:
        # Fall back to first category if no vectors available
        return list(prototypes.keys())[0], 0.0

    best_category = list(prototypes.keys())[0]  # Default
    best_score = 0.0

    for category, phrases in prototypes.items():
        for phrase in phrases:
            phrase_doc = nlp(phrase)
            if phrase_doc.has_vector:
                score = doc.similarity(phrase_doc)
                if score > best_score:
                    best_score = score
                    best_category = category

    # Only return match if above threshold
    if best_score < threshold:
        return list(prototypes.keys())[0], best_score

    return best_category, best_score


class NLPManager:
    """Manager class for NLP operations.

    Provides a unified interface for NLP processing using spaCy.
    """

    def __init__(self):
        """Initialize NLP manager."""
        self._nlp = None

    @property
    def nlp(self):
        """Get the spaCy model, lazy-loading if needed."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def process(self, text: str):
        """Process text with spaCy.

        Args:
            text: Text to process.

        Returns:
            spaCy Doc object.
        """
        return self.nlp(text)

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentences.
        """
        return split_into_sentences(text)

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text.

        Args:
            text: Input text.

        Returns:
            List of (entity_text, entity_label) tuples.
        """
        return extract_entities(text)

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Input text.
            top_n: Maximum keywords to return.

        Returns:
            List of keywords.
        """
        return extract_keywords(text, top_n)

    def get_pos_distribution(self, text: str) -> dict:
        """Get POS tag distribution.

        Args:
            text: Input text.

        Returns:
            Dictionary of POS tag to count.
        """
        return get_pos_distribution(text)

    def get_dependency_depth(self, text: str) -> float:
        """Get average dependency depth.

        Args:
            text: Input text.

        Returns:
            Average depth.
        """
        return get_dependency_depth(text)

    def calculate_burstiness(self, sentences: List[str]) -> float:
        """Calculate burstiness of sentence lengths.

        Args:
            sentences: List of sentences.

        Returns:
            Burstiness value.
        """
        return calculate_burstiness(sentences)

    def detect_perspective(self, text: str) -> str:
        """Detect perspective of text.

        Args:
            text: Input text.

        Returns:
            Perspective string.
        """
        return detect_perspective(text)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between 0 and 1.
        """
        return compute_semantic_similarity(text1, text2)

    def classify_by_similarity(
        self,
        text: str,
        prototypes: dict,
        threshold: float = 0.3
    ) -> Tuple[str, float]:
        """Classify text by semantic similarity to prototype phrases.

        Args:
            text: Text to classify.
            prototypes: Dict mapping category names to prototype phrases.
            threshold: Minimum similarity threshold.

        Returns:
            Tuple of (category, score).
        """
        return classify_by_similarity(text, prototypes, threshold)


def is_heading(line: str) -> bool:
    """Detect if a line is likely a heading.

    Headings are identified by:
    - All uppercase (or mostly uppercase)
    - Markdown heading markers (#, ##, etc.)
    - Short lines without punctuation
    - Numbered section markers (1., 2.1, etc.)

    Args:
        line: A single line of text.

    Returns:
        True if the line appears to be a heading.
    """
    line = line.strip()

    if not line:
        return False

    # Markdown headings
    if line.startswith('#'):
        return True

    # Check for all caps (allowing numbers and punctuation)
    alpha_chars = [c for c in line if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        # Line is mostly uppercase (>80%) and short
        if upper_ratio > 0.8 and len(line.split()) <= 8:
            return True

    # Numbered section markers (e.g., "1.", "2.1", "Chapter 3")
    if re.match(r'^(\d+\.)+\s*\w', line):
        return True
    if re.match(r'^(chapter|section|part)\s+\d+', line.lower()):
        return True

    # Short lines without sentence-ending punctuation are likely headings
    if len(line.split()) <= 5 and not line.endswith(('.', '!', '?', ':')):
        # But not if it starts with common sentence starters
        first_word = line.split()[0].lower() if line.split() else ""
        sentence_starters = {'the', 'a', 'an', 'this', 'that', 'it', 'he', 'she', 'they', 'we', 'i'}
        if first_word not in sentence_starters:
            return True

    return False


def filter_headings(paragraphs: List[str]) -> List[str]:
    """Filter out heading paragraphs from a list.

    Args:
        paragraphs: List of paragraph texts.

    Returns:
        Filtered list with headings removed.
    """
    result = []
    for para in paragraphs:
        # Check if entire paragraph is a heading (single line)
        lines = para.strip().split('\n')
        if len(lines) == 1 and is_heading(lines[0]):
            logger.debug(f"Skipping heading: {para[:50]}...")
            continue

        # For multi-line paragraphs, filter out heading lines at the start
        filtered_lines = []
        for i, line in enumerate(lines):
            if i == 0 and is_heading(line):
                logger.debug(f"Skipping heading line: {line[:50]}...")
                continue
            filtered_lines.append(line)

        if filtered_lines:
            result.append('\n'.join(filtered_lines))

    return result


def split_paragraphs_preserving_headings(
    text: str
) -> Tuple[List[str], List[Tuple[int, str]]]:
    """Split text into paragraphs while tracking heading positions.

    Args:
        text: Input text.

    Returns:
        Tuple of (content_paragraphs, heading_info).
        heading_info is list of (original_index, heading_text).
    """
    all_paragraphs = split_into_paragraphs(text)
    content_paragraphs = []
    heading_info = []

    for i, para in enumerate(all_paragraphs):
        lines = para.strip().split('\n')
        if len(lines) == 1 and is_heading(lines[0]):
            heading_info.append((i, para))
        else:
            content_paragraphs.append(para)

    return content_paragraphs, heading_info
