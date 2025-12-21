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


def extract_global_vocabulary(sample_text: str, top_n: int = 200) -> Dict[str, List[str]]:
    """Extract global vocabulary from sample text, clustered by sentiment.

    Extracts top adjectives and verbs from the sample text and organizes them
    by sentiment (Positive, Negative, Neutral) for vocabulary injection.

    Args:
        sample_text: The sample text to analyze.
        top_n: Maximum number of words to extract per sentiment (default: 200).

    Returns:
        Dictionary with keys 'positive', 'negative', 'neutral', each containing
        a list of words sorted by frequency.
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from collections import Counter

    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        use_vader = True
    except (LookupError, ImportError):
        use_vader = False

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(sample_text.lower())
    pos_tags = pos_tag(tokens)

    # Extract adjectives and verbs
    content_words = []
    word_counts = Counter()

    for word, pos in pos_tags:
        # Filter for adjectives and verbs
        is_content = (
            pos.startswith('JJ') or  # Adjectives
            pos.startswith('VB')    # Verbs
        )

        if (is_content and
            word not in stop_words and
            word.isalpha() and
            len(word) > 2):
            content_words.append(word)
            word_counts[word] += 1

    # Cluster by sentiment
    positive_words = []
    negative_words = []
    neutral_words = []

    # Get unique words sorted by frequency
    unique_words = sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)

    for word in unique_words[:top_n * 3]:  # Get more than needed, then filter
        if use_vader:
            # Use VADER sentiment analyzer
            scores = sia.polarity_scores(word)
            compound = scores['compound']

            if compound >= 0.05:
                positive_words.append(word)
            elif compound <= -0.05:
                negative_words.append(word)
            else:
                neutral_words.append(word)
        else:
            # Fallback: use spaCy semantic similarity to determine sentiment
            # Use seed words and semantic similarity instead of hardcoded lists
            try:
                from src.utils.nlp_manager import NLPManager
                nlp = NLPManager.get_nlp()

                word_token = nlp.vocab[word] if word in nlp.vocab else None
                if word_token and word_token.has_vector:
                    # Use seed words for sentiment
                    positive_seed = nlp.vocab.get("good")
                    negative_seed = nlp.vocab.get("bad")

                    positive_score = 0.0
                    negative_score = 0.0

                    if positive_seed and positive_seed.has_vector:
                        positive_score = word_token.similarity(positive_seed)

                    if negative_seed and negative_seed.has_vector:
                        negative_score = word_token.similarity(negative_seed)

                    if positive_score > 0.4 and positive_score > negative_score:
                        positive_words.append(word)
                    elif negative_score > 0.4 and negative_score > positive_score:
                        negative_words.append(word)
                    else:
                        neutral_words.append(word)
                else:
                    neutral_words.append(word)
            except (ImportError, OSError, KeyError, AttributeError):
                # If spaCy unavailable, treat as neutral
                neutral_words.append(word)

    # Limit to top_n per category
    return {
        'positive': positive_words[:top_n],
        'negative': negative_words[:top_n],
        'neutral': neutral_words[:top_n]
    }

