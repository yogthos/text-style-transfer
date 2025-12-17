"""Structural Markov models for style transfer.

This module builds Markov chain transition matrices for:
1. Part-of-Speech (POS) tag sequences
2. Sentence type/flow sequences
"""

import nltk
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def classify_sentence_type(sentence: str) -> str:
    """Classify a sentence by its structural type.

    Args:
        sentence: The sentence text to classify.

    Returns:
        One of: 'Simple', 'Compound', 'Complex', 'Fragment'
    """
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag

    tokens = word_tokenize(sentence)
    if not tokens:
        return 'Fragment'

    pos_tags = pos_tag(tokens)
    pos_sequence = [tag for _, tag in pos_tags]

    # Count coordinating conjunctions (CC) - indicates compound
    cc_count = pos_sequence.count('CC')

    # Count subordinating conjunctions (IN starting clauses, WDT, etc.)
    subordinating = sum(1 for tag in pos_sequence if tag in ['IN', 'WDT', 'WP', 'WRB'])

    # Count verbs - helps determine completeness
    verb_count = sum(1 for tag in pos_sequence if tag.startswith('VB'))

    # Simple heuristics for classification
    if verb_count == 0 or len(tokens) < 3:
        return 'Fragment'
    elif cc_count >= 2 or (cc_count >= 1 and len(tokens) > 15):
        return 'Compound'
    elif subordinating >= 2:
        return 'Complex'
    else:
        return 'Simple'


def build_pos_markov(sample_text: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build a POS tag transition matrix from sample text.

    Creates a Markov chain where each state is a POS tag, and transitions
    represent the probability of moving from one POS tag to another.

    Args:
        sample_text: The sample text to analyze.

    Returns:
        A tuple of:
        - transition_matrix: 2D numpy array where [i, j] is the probability
          of transitioning from POS tag i to POS tag j
        - tag_to_index: Dictionary mapping POS tags to matrix indices
    """
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag

    # Tokenize into sentences
    sentences = sent_tokenize(sample_text)

    # Collect all POS tag sequences
    all_pos_sequences: List[List[str]] = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        if not tokens:
            continue
        pos_tags = pos_tag(tokens)
        pos_sequence = [tag for _, tag in pos_tags]
        if pos_sequence:
            all_pos_sequences.append(pos_sequence)

    if not all_pos_sequences:
        # Return empty matrix if no sequences found
        return np.array([]), {}

    # Build transition counts
    transitions: Dict[Tuple[str, str], int] = defaultdict(int)
    all_tags: set = set()

    for pos_sequence in all_pos_sequences:
        for i in range(len(pos_sequence) - 1):
            from_tag = pos_sequence[i]
            to_tag = pos_sequence[i + 1]
            transitions[(from_tag, to_tag)] += 1
            all_tags.add(from_tag)
            all_tags.add(to_tag)

    # Create tag to index mapping
    sorted_tags = sorted(all_tags)
    tag_to_index = {tag: idx for idx, tag in enumerate(sorted_tags)}
    num_tags = len(sorted_tags)

    # Build transition matrix
    transition_matrix = np.zeros((num_tags, num_tags))

    for (from_tag, to_tag), count in transitions.items():
        from_idx = tag_to_index[from_tag]
        to_idx = tag_to_index[to_tag]
        transition_matrix[from_idx, to_idx] = count

    # Normalize rows to get probabilities (each row sums to 1.0)
    # For rows with no transitions, set uniform distribution
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.flatten() == 0)
    if np.any(zero_rows):
        # Set uniform distribution for rows with no transitions
        uniform_prob = 1.0 / num_tags
        for idx in np.where(zero_rows)[0]:
            transition_matrix[idx, :] = uniform_prob
        row_sums = transition_matrix.sum(axis=1, keepdims=True)

    transition_matrix = transition_matrix / row_sums

    return transition_matrix, tag_to_index


def build_flow_markov(sample_text: str) -> Tuple[np.ndarray, Dict[str, int], Dict[str, np.ndarray]]:
    """Build a sentence flow transition matrix from sample text.

    Creates a Markov chain where each state is a sentence type (Simple, Compound,
    Complex, Fragment), and transitions represent the probability of moving from
    one sentence type to another.

    Also tracks paragraph start and end probabilities separately.

    Args:
        sample_text: The sample text to analyze.

    Returns:
        A tuple of:
        - transition_matrix: 2D numpy array where [i, j] is the probability
          of transitioning from sentence type i to sentence type j
        - type_to_index: Dictionary mapping sentence types to matrix indices
        - context_matrices: Dictionary with 'paragraph_start' and 'paragraph_end'
          transition matrices
    """
    from nltk.tokenize import sent_tokenize

    # Split into paragraphs (double newlines)
    paragraphs = [p.strip() for p in sample_text.split('\n\n') if p.strip()]

    if not paragraphs:
        # If no paragraphs, treat entire text as one paragraph
        paragraphs = [sample_text]

    # Classify all sentences and track paragraph boundaries
    all_sentences: List[Tuple[str, str, bool, bool]] = []  # (type, text, is_start, is_end)
    sentence_types: List[str] = []

    for para_idx, paragraph in enumerate(paragraphs):
        sentences = sent_tokenize(paragraph)
        if not sentences:
            continue

        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_type = classify_sentence_type(sentence)
            is_start = (sent_idx == 0)
            is_end = (sent_idx == len(sentences) - 1)

            all_sentences.append((sent_type, sentence, is_start, is_end))
            sentence_types.append(sent_type)

    if not sentence_types:
        # Return empty matrices if no sentences found
        return np.array([]), {}, {'paragraph_start': np.array([]), 'paragraph_end': np.array([])}

    # Get unique sentence types
    unique_types = sorted(set(sentence_types))
    type_to_index = {stype: idx for idx, stype in enumerate(unique_types)}
    num_types = len(unique_types)

    # Build main transition matrix
    transitions = defaultdict(int)
    for i in range(len(sentence_types) - 1):
        from_type = sentence_types[i]
        to_type = sentence_types[i + 1]
        transitions[(from_type, to_type)] += 1

    transition_matrix = np.zeros((num_types, num_types))
    for (from_type, to_type), count in transitions.items():
        from_idx = type_to_index[from_type]
        to_idx = type_to_index[to_type]
        transition_matrix[from_idx, to_idx] = count

    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums

    # Build paragraph start transitions
    para_start_transitions = defaultdict(int)
    for sent_type, _, is_start, _ in all_sentences:
        if is_start:
            para_start_transitions[sent_type] += 1

    para_start_matrix = np.zeros((1, num_types))
    total_starts = sum(para_start_transitions.values())
    if total_starts > 0:
        for sent_type, count in para_start_transitions.items():
            idx = type_to_index[sent_type]
            para_start_matrix[0, idx] = count / total_starts

    # Build paragraph end transitions
    para_end_transitions = defaultdict(int)
    for sent_type, _, _, is_end in all_sentences:
        if is_end:
            para_end_transitions[sent_type] += 1

    para_end_matrix = np.zeros((num_types, 1))
    total_ends = sum(para_end_transitions.values())
    if total_ends > 0:
        for sent_type, count in para_end_transitions.items():
            idx = type_to_index[sent_type]
            para_end_matrix[idx, 0] = count / total_ends

    context_matrices = {
        'paragraph_start': para_start_matrix,
        'paragraph_end': para_end_matrix
    }

    return transition_matrix, type_to_index, context_matrices

