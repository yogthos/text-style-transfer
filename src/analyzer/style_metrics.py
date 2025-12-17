"""Style metrics extraction module.

This module extracts deterministic style vectors from text that capture
writing style characteristics independent of semantic content.
"""

import nltk
import numpy as np
from typing import List, Optional
from collections import Counter

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


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _get_dependency_depth_nltk(text: str) -> int:
    """Estimate dependency tree depth using NLTK (fallback method).

    Uses a simple heuristic: count nested phrases and clauses.
    """
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag

    sentences = sent_tokenize(text)
    if not sentences:
        return 0

    max_depth = 0
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        if not tokens:
            continue

        pos_tags = pos_tag(tokens)
        # Count subordinating conjunctions and relative pronouns (indicate depth)
        depth = 0
        for word, tag in pos_tags:
            if tag in ['IN', 'WDT', 'WP', 'WRB']:  # Prepositions, relative pronouns
                depth += 1
            elif tag == 'CC':  # Coordinating conjunctions (compound sentences)
                depth += 0.5

        max_depth = max(max_depth, int(depth))

    return max_depth


def _get_dependency_depth_spacy(text: str) -> Optional[int]:
    """Get dependency tree depth using spaCy."""
    try:
        import spacy

        # Try to load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed, return None to use NLTK fallback
            return None

        doc = nlp(text)
        if not doc.sents:
            return 0

        max_depth = 0
        for sent in doc.sents:
            # Calculate max depth in dependency tree
            depths = {}
            for token in sent:
                depth = 0
                head = token.head
                while head != token and head != head.head:
                    depth += 1
                    head = head.head
                    if depth > 100:  # Safety limit
                        break
                depths[token] = depth

            if depths:
                max_depth = max(max_depth, max(depths.values()))

        return max_depth
    except ImportError:
        return None


def _detect_passive_voice(text: str) -> float:
    """Detect passive voice frequency in text.

    Returns ratio of passive voice constructions to total verbs.
    """
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag

    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0

    passive_indicators = 0
    total_verbs = 0

    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        pos_tags = pos_tag(tokens)

        # Look for passive voice patterns: "be" + past participle
        for i in range(len(pos_tags) - 1):
            word, tag = pos_tags[i]
            if tag.startswith('VB'):  # Any verb
                total_verbs += 1
                # Check if followed by past participle (VBN) or if it's a form of "be"
                if word in ['is', 'are', 'was', 'were', 'be', 'been', 'being']:
                    if i + 1 < len(pos_tags):
                        next_word, next_tag = pos_tags[i + 1]
                        if next_tag == 'VBN':  # Past participle
                            passive_indicators += 1

    if total_verbs == 0:
        return 0.0

    return passive_indicators / total_verbs


def get_style_vector(text: str) -> np.ndarray:
    """Extract a normalized style vector from text.

    The style vector captures writing style characteristics independent
    of semantic content. All values are normalized to 0-1 range.

    Dimensions:
    1. Average sentence length (normalized)
    2. Punctuation density (commas, semicolons per 100 words)
    3. Stopword ratio
    4. Adjective/Verb ratio
    5. Dependency tree depth (normalized)
    6. Passive voice frequency
    7. Average word length

    Args:
        text: The text to analyze.

    Returns:
        A numpy array of 7 normalized float values (0-1 range).
    """
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
    from nltk.corpus import stopwords

    if not text or not text.strip():
        return np.zeros(7, dtype=np.float32)

    # Tokenize
    sentences = sent_tokenize(text)
    if not sentences:
        return np.zeros(7, dtype=np.float32)

    all_tokens = word_tokenize(text.lower())
    all_words = [w for w in all_tokens if w.isalnum()]

    if not all_words:
        return np.zeros(7, dtype=np.float32)

    # 1. Average sentence length (in words)
    sentence_lengths = []
    for sent in sentences:
        words = word_tokenize(sent)
        words = [w for w in words if w.isalnum()]
        if words:
            sentence_lengths.append(len(words))

    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0.0
    # Normalize: assume range 5-50 words per sentence
    norm_sentence_length = _normalize(avg_sentence_length, 5.0, 50.0)

    # 2. Punctuation density (commas, semicolons per 100 words)
    punctuation_count = text.count(',') + text.count(';') + text.count(':')
    punctuation_density = (punctuation_count / len(all_words)) * 100 if all_words else 0.0
    # Normalize: assume range 0-20 per 100 words
    norm_punctuation_density = _normalize(punctuation_density, 0.0, 20.0)

    # 3. Stopword ratio
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set()

    stopword_count = sum(1 for w in all_words if w in stop_words)
    stopword_ratio = stopword_count / len(all_words) if all_words else 0.0
    # Already in 0-1 range

    # 4. Adjective/Verb ratio
    pos_tags = pos_tag(all_tokens)
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))

    if verb_count > 0:
        adj_verb_ratio = adj_count / verb_count
    else:
        adj_verb_ratio = 0.0

    # Normalize: assume range 0-3
    norm_adj_verb_ratio = _normalize(adj_verb_ratio, 0.0, 3.0)

    # 5. Dependency tree depth
    depth = _get_dependency_depth_spacy(text)
    if depth is None:
        depth = _get_dependency_depth_nltk(text)

    # Normalize: assume range 0-10
    norm_depth = _normalize(float(depth), 0.0, 10.0)

    # 6. Passive voice frequency
    passive_freq = _detect_passive_voice(text)
    # Already in 0-1 range

    # 7. Average word length
    word_lengths = [len(w) for w in all_words]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0.0
    # Normalize: assume range 3-8 characters
    norm_word_length = _normalize(avg_word_length, 3.0, 8.0)

    # Combine into vector
    style_vector = np.array([
        norm_sentence_length,
        norm_punctuation_density,
        stopword_ratio,
        norm_adj_verb_ratio,
        norm_depth,
        passive_freq,
        norm_word_length
    ], dtype=np.float32)

    return style_vector

