"""Vocabulary mapping module for style transfer.

This module builds a vocabulary map that maps generic/common words to
sample-specific synonyms found in the style sample text.
"""

import nltk
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Set

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def build_vocab_map(sample_text: str, similarity_threshold: float = 0.7) -> Dict[str, List[str]]:
    """Build a vocabulary map from sample text.

    Extracts content words (nouns, verbs, adjectives) from the sample text,
    clusters them by semantic similarity using word embeddings, and creates
    a mapping from generic/common words to sample-specific synonyms.

    Args:
        sample_text: The sample text to analyze for style-specific vocabulary.
        similarity_threshold: Minimum cosine similarity for words to be considered
            synonyms (default: 0.7).

    Returns:
        A dictionary mapping generic words to lists of sample-specific synonyms.
        Format: {generic_word: [sample_word_1, sample_word_2, ...]}
    """
    # Load sentence transformer model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Tokenize using NLTK
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(sample_text.lower())
    pos_tags = pos_tag(tokens)

    # Filter for nouns, verbs, and adjectives (excluding stop words and punctuation)
    content_words: List[str] = []
    word_set: Set[str] = set()

    for word, pos in pos_tags:
        # Map NLTK POS tags to categories
        is_content = (
            pos.startswith('NN') or  # Nouns
            pos.startswith('VB') or  # Verbs
            pos.startswith('JJ')     # Adjectives
        )

        if (is_content and
            word not in stop_words and
            word.isalpha() and
            len(word) > 1 and
            word not in word_set):
            content_words.append(word)
            word_set.add(word)

    if not content_words:
        return {}

    # Get embeddings for all content words
    # Use sentences (single words) for embedding
    word_embeddings = model.encode(content_words, convert_to_numpy=True)

    # Build similarity matrix and cluster words
    vocab_map: Dict[str, List[str]] = {}
    processed: Set[int] = set()

    # Group words by semantic similarity
    for i, word1 in enumerate(content_words):
        if i in processed:
            continue

        # Find all words similar to word1
        cluster = [i]
        embedding1 = word_embeddings[i]

        for j, word2 in enumerate(content_words[i+1:], start=i+1):
            if j in processed:
                continue

            embedding2 = word_embeddings[j]

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
            )

            if similarity >= similarity_threshold:
                cluster.append(j)
                processed.add(j)

        if len(cluster) > 1:
            # Use the most frequent word in the sample as the "generic" key
            word_counts = Counter(content_words)
            cluster_words = [content_words[idx] for idx in cluster]
            cluster_sorted = sorted(cluster_words, key=lambda w: word_counts.get(w, 0), reverse=True)
            generic_word = cluster_sorted[0]
            synonyms = [w for w in cluster_sorted[1:] if w != generic_word]

            if synonyms:
                vocab_map[generic_word] = synonyms
                processed.add(i)

    return vocab_map

