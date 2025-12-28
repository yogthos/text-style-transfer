"""Vocabulary palette extraction for style transfer.

Extracts author-specific vocabulary organized by:
- POS categories (nouns, verbs, adjectives, adverbs)
- Semantic clusters (groups of related words)
- Synonym preferences (generic -> author-preferred mappings)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Common LLM-speak words to replace (extracted from observation)
LLM_SPEAK_WORDS = {
    # Overused intensifiers
    "crucial", "significant", "profound", "intricate", "nuanced",
    "compelling", "pivotal", "paramount", "imperative", "vital",
    # Overused connectives
    "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "nonetheless", "henceforth", "thereby",
    # Overused verbs
    "delve", "underscore", "highlight", "emphasize", "navigate",
    "leverage", "utilize", "facilitate", "encompass", "embody",
    # Overused adjectives
    "myriad", "plethora", "multifaceted", "holistic", "robust",
    "seamless", "innovative", "cutting-edge", "groundbreaking",
    # Overused abstract nouns
    "tapestry", "paradigm", "landscape", "realm", "sphere",
    "dimension", "framework", "cornerstone", "foundation",
}


@dataclass
class VocabularyPalette:
    """Author's vocabulary organized for style transfer.

    Like the example: general words, sensory verbs, connectives, intensifiers.
    Plus synonym clusters for smart replacement.
    """

    # Words by POS with frequencies
    nouns: Dict[str, float] = field(default_factory=dict)
    verbs: Dict[str, float] = field(default_factory=dict)
    adjectives: Dict[str, float] = field(default_factory=dict)
    adverbs: Dict[str, float] = field(default_factory=dict)

    # Top keywords (content words with high frequency)
    keywords: List[str] = field(default_factory=list)
    keyword_frequencies: Dict[str, float] = field(default_factory=dict)

    # Common sentence openers
    common_openers: List[str] = field(default_factory=list)

    # Semantic clusters: concept -> author's preferred words
    # e.g., "conflict" -> ["struggle", "opposition", "clash"]
    semantic_clusters: Dict[str, List[str]] = field(default_factory=dict)

    # LLM-speak replacements: generic word -> author's alternative
    llm_replacements: Dict[str, str] = field(default_factory=dict)

    # Intensifier preferences (sorted by author's usage)
    intensifiers: List[str] = field(default_factory=list)

    # Connective preferences
    connectives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "nouns": self.nouns,
            "verbs": self.verbs,
            "adjectives": self.adjectives,
            "adverbs": self.adverbs,
            "keywords": self.keywords,
            "keyword_frequencies": self.keyword_frequencies,
            "common_openers": self.common_openers,
            "semantic_clusters": self.semantic_clusters,
            "llm_replacements": self.llm_replacements,
            "intensifiers": self.intensifiers,
            "connectives": self.connectives,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VocabularyPalette":
        return cls(**data)


class VocabularyPaletteExtractor:
    """Extract vocabulary palette from author's corpus."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, paragraphs: List[str]) -> VocabularyPalette:
        """Extract vocabulary palette from corpus paragraphs.

        Args:
            paragraphs: List of author's paragraphs.

        Returns:
            VocabularyPalette with organized vocabulary.
        """
        logger.info(f"Extracting vocabulary palette from {len(paragraphs)} paragraphs")

        # Combine all text
        full_text = " ".join(paragraphs)
        doc = self.nlp(full_text)

        # Extract by POS
        nouns = self._extract_by_pos(doc, {'NOUN'})
        verbs = self._extract_by_pos(doc, {'VERB'})
        adjectives = self._extract_by_pos(doc, {'ADJ'})
        adverbs = self._extract_by_pos(doc, {'ADV'})

        # Extract keywords (top content words)
        keywords, keyword_freqs = self._extract_keywords(doc, limit=50)

        # Extract sentence openers
        openers = self._extract_openers(paragraphs)

        # Build semantic clusters using word vectors
        semantic_clusters = self._build_semantic_clusters(doc, keywords)

        # Build LLM-speak replacements
        llm_replacements = self._build_llm_replacements(
            nouns, verbs, adjectives, adverbs, semantic_clusters
        )

        # Extract intensifiers and connectives
        intensifiers = self._extract_intensifiers(doc)
        connectives = self._extract_connectives(doc)

        palette = VocabularyPalette(
            nouns=nouns,
            verbs=verbs,
            adjectives=adjectives,
            adverbs=adverbs,
            keywords=keywords,
            keyword_frequencies=keyword_freqs,
            common_openers=openers,
            semantic_clusters=semantic_clusters,
            llm_replacements=llm_replacements,
            intensifiers=intensifiers,
            connectives=connectives,
        )

        logger.info(
            f"Extracted palette: {len(nouns)} nouns, {len(verbs)} verbs, "
            f"{len(adjectives)} adj, {len(adverbs)} adv, "
            f"{len(semantic_clusters)} clusters, {len(llm_replacements)} replacements"
        )

        return palette

    def _extract_by_pos(self, doc, pos_tags: Set[str]) -> Dict[str, float]:
        """Extract words by POS with normalized frequencies."""
        counts = Counter()
        total = 0

        for token in doc:
            if token.pos_ in pos_tags and not token.is_stop and token.is_alpha:
                if len(token.text) > 2:  # Skip very short words
                    counts[token.lemma_.lower()] += 1
                    total += 1

        # Normalize to frequencies
        if total > 0:
            return {word: count / total for word, count in counts.most_common(100)}
        return {}

    def _extract_keywords(self, doc, limit: int = 50) -> Tuple[List[str], Dict[str, float]]:
        """Extract top content keywords."""
        counts = Counter()
        total = 0

        for token in doc:
            if token.pos_ in {'NOUN', 'VERB', 'ADJ'} and not token.is_stop and token.is_alpha:
                if len(token.text) > 3:
                    counts[token.lemma_.lower()] += 1
                    total += 1

        top_words = counts.most_common(limit)
        keywords = [word for word, _ in top_words]
        frequencies = {word: count / total for word, count in top_words} if total > 0 else {}

        return keywords, frequencies

    def _extract_openers(self, paragraphs: List[str], limit: int = 20) -> List[str]:
        """Extract common sentence opening words."""
        opener_counts = Counter()

        for para in paragraphs:
            doc = self.nlp(para)
            for sent in doc.sents:
                tokens = [t for t in sent if t.is_alpha]
                if tokens:
                    opener = tokens[0].text.lower()
                    # Skip very common words
                    if opener not in {'the', 'a', 'an', 'this', 'that', 'it', 'i', 'we', 'they', 'he', 'she'}:
                        opener_counts[opener] += 1

        return [word for word, _ in opener_counts.most_common(limit)]

    def _build_semantic_clusters(
        self,
        doc,
        keywords: List[str],
        similarity_threshold: float = 0.5
    ) -> Dict[str, List[str]]:
        """Build semantic clusters using spaCy word vectors.

        Groups words that are semantically similar, with the most
        frequent word in each group as the cluster key.
        """
        clusters = defaultdict(list)

        # Get word vectors for keywords
        word_vectors = {}
        for word in keywords[:30]:  # Limit for performance
            token = self.nlp(word)
            if token and token[0].has_vector:
                word_vectors[word] = token[0].vector

        if not word_vectors:
            logger.warning("No word vectors available for clustering")
            return {}

        # Simple clustering: group by similarity
        used = set()
        words = list(word_vectors.keys())

        for i, word1 in enumerate(words):
            if word1 in used:
                continue

            cluster = [word1]
            used.add(word1)

            for j, word2 in enumerate(words):
                if word2 in used or i == j:
                    continue

                # Calculate cosine similarity
                vec1 = word_vectors[word1]
                vec2 = word_vectors[word2]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

                if similarity > similarity_threshold:
                    cluster.append(word2)
                    used.add(word2)

            if len(cluster) > 1:
                clusters[word1] = cluster[1:]  # Key is first word, value is related words

        return dict(clusters)

    def _build_llm_replacements(
        self,
        nouns: Dict[str, float],
        verbs: Dict[str, float],
        adjectives: Dict[str, float],
        adverbs: Dict[str, float],
        semantic_clusters: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """Build LLM-speak to author-word replacement map.

        For each LLM-speak word, find the most similar word the author uses.
        """
        replacements = {}
        author_words = set(nouns.keys()) | set(verbs.keys()) | set(adjectives.keys()) | set(adverbs.keys())

        for llm_word in LLM_SPEAK_WORDS:
            # Skip if author actually uses this word
            if llm_word in author_words:
                continue

            # Find best replacement using word vectors
            best_match = self._find_best_replacement(llm_word, author_words)
            if best_match:
                replacements[llm_word] = best_match

        logger.info(f"Built {len(replacements)} LLM-speak replacements")
        return replacements

    def _find_best_replacement(
        self,
        word: str,
        candidates: Set[str],
        min_similarity: float = 0.3
    ) -> Optional[str]:
        """Find the best author-word replacement for an LLM-speak word."""
        token = self.nlp(word)
        if not token or not token[0].has_vector:
            return None

        word_vec = token[0].vector
        best_match = None
        best_sim = min_similarity

        for candidate in list(candidates)[:100]:  # Limit for performance
            cand_token = self.nlp(candidate)
            if not cand_token or not cand_token[0].has_vector:
                continue

            cand_vec = cand_token[0].vector
            similarity = np.dot(word_vec, cand_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(cand_vec) + 1e-8)

            if similarity > best_sim:
                best_sim = similarity
                best_match = candidate

        return best_match

    def _extract_intensifiers(self, doc) -> List[str]:
        """Extract intensifier adverbs used by author."""
        # Intensifiers typically modify adjectives or other adverbs
        intensifier_counts = Counter()

        for token in doc:
            if token.pos_ == 'ADV' and token.dep_ == 'advmod':
                # Check if modifying an adjective
                if token.head.pos_ in {'ADJ', 'ADV'}:
                    intensifier_counts[token.text.lower()] += 1

        return [word for word, _ in intensifier_counts.most_common(20)]

    def _extract_connectives(self, doc) -> List[str]:
        """Extract connective words/phrases used by author."""
        connective_counts = Counter()

        for token in doc:
            # Conjunctions and certain adverbs
            if token.pos_ in {'CCONJ', 'SCONJ'}:
                connective_counts[token.text.lower()] += 1
            elif token.pos_ == 'ADV' and token.dep_ in {'advmod', 'cc'}:
                connective_counts[token.text.lower()] += 1

        return [word for word, _ in connective_counts.most_common(30)]


def extract_vocabulary_palette(paragraphs: List[str]) -> VocabularyPalette:
    """Convenience function to extract vocabulary palette."""
    extractor = VocabularyPaletteExtractor()
    return extractor.extract(paragraphs)
