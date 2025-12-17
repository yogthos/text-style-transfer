"""Style Navigator for predicting and retrieving style references.

This module provides functions to:
1. Build Markov chain from style clusters
2. Predict next style cluster
3. Retrieve style reference paragraphs from ChromaDB
"""

import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict

from src.atlas.builder import StyleAtlas


def build_cluster_markov(atlas: StyleAtlas) -> Tuple[np.ndarray, Dict[int, int]]:
    """Build Markov transition matrix from cluster sequence.

    Analyzes the sequence of clusters in the sample text to build
    a transition probability matrix.

    Args:
        atlas: StyleAtlas containing cluster assignments.

    Returns:
        Tuple of:
        - transition_matrix: 2D numpy array where [i, j] is probability
          of transitioning from cluster i to cluster j
        - cluster_to_index: Dictionary mapping cluster ID to matrix index
    """
    # Extract cluster sequence from paragraph order
    # Paragraphs are stored in order: para_0, para_1, ...
    cluster_sequence = []
    for idx in range(len(atlas.cluster_ids)):
        para_id = f"para_{idx}"
        if para_id in atlas.cluster_ids:
            cluster_sequence.append(atlas.cluster_ids[para_id])

    if len(cluster_sequence) < 2:
        # Not enough data for transitions
        unique_clusters = sorted(set(cluster_sequence)) if cluster_sequence else [0]
        num_clusters = len(unique_clusters)
        cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
        transition_matrix = np.ones((num_clusters, num_clusters)) / num_clusters
        return transition_matrix, cluster_to_index

    # Get unique clusters
    unique_clusters = sorted(set(cluster_sequence))
    num_clusters = len(unique_clusters)
    cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    index_to_cluster = {idx: cluster for cluster, idx in cluster_to_index.items()}

    # Build transition counts
    transitions = defaultdict(int)
    for i in range(len(cluster_sequence) - 1):
        from_cluster = cluster_sequence[i]
        to_cluster = cluster_sequence[i + 1]
        from_idx = cluster_to_index[from_cluster]
        to_idx = cluster_to_index[to_cluster]
        transitions[(from_idx, to_idx)] += 1

    # Build transition matrix
    transition_matrix = np.zeros((num_clusters, num_clusters))
    for (from_idx, to_idx), count in transitions.items():
        transition_matrix[from_idx, to_idx] = count

    # Normalize rows (each row sums to 1.0)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_matrix / row_sums

    # If any row has no transitions, use uniform distribution
    zero_rows = (transition_matrix.sum(axis=1) == 0)
    if zero_rows.any():
        uniform_prob = 1.0 / num_clusters
        for idx in np.where(zero_rows)[0]:
            transition_matrix[idx, :] = uniform_prob

    return transition_matrix, cluster_to_index


def predict_next_cluster(
    current_cluster: int,
    cluster_markov: Tuple[np.ndarray, Dict[int, int]]
) -> int:
    """Predict the most likely next cluster given current cluster.

    Args:
        current_cluster: Current cluster ID.
        cluster_markov: Tuple of (transition_matrix, cluster_to_index) from build_cluster_markov.

    Returns:
        Predicted next cluster ID.
    """
    transition_matrix, cluster_to_index = cluster_markov

    if current_cluster not in cluster_to_index:
        # Unknown cluster, return first cluster
        return list(cluster_to_index.keys())[0] if cluster_to_index else 0

    current_idx = cluster_to_index[current_cluster]

    if transition_matrix.size == 0 or current_idx >= transition_matrix.shape[0]:
        return current_cluster

    # Get transition probabilities from current cluster
    transition_probs = transition_matrix[current_idx, :]

    # Find most likely next cluster
    next_idx = np.argmax(transition_probs)

    # Map back to cluster ID
    index_to_cluster = {idx: cluster for cluster, idx in cluster_to_index.items()}
    return index_to_cluster.get(next_idx, current_cluster)


def find_situation_match(
    atlas: StyleAtlas,
    input_text: str,
    similarity_threshold: float = 0.3,
    top_k: int = 1
) -> Optional[str]:
    """Find a semantically similar paragraph for vocabulary grounding.

    Queries ChromaDB by semantic similarity only (ignores cluster).
    Returns a paragraph if similarity is above threshold, else None.

    Args:
        atlas: StyleAtlas containing ChromaDB collection.
        input_text: Input text to find similar paragraphs for.
        similarity_threshold: Minimum similarity score (0-1, default: 0.3).
        top_k: Number of results to return (default: 1).

    Returns:
        Most similar paragraph text, or None if no match above threshold.
    """
    if not hasattr(atlas, '_collection'):
        try:
            if hasattr(atlas, '_client'):
                atlas._collection = atlas._client.get_collection(name=atlas.collection_name)
            else:
                return None
        except:
            return None

    collection = atlas._collection

    # Query by semantic similarity
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(input_text, normalize_embeddings=True)

    # Query ChromaDB for similar paragraphs
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    if not results['ids'] or len(results['ids'][0]) == 0:
        return None

    # Check similarity (ChromaDB returns distances, convert to similarity)
    # Distance is typically 1 - cosine_similarity, so similarity = 1 - distance
    if results['distances'] and len(results['distances'][0]) > 0:
        # ChromaDB may return distances or similarities depending on configuration
        # For normalized embeddings, distance = 1 - similarity
        distance = results['distances'][0][0]
        # If distance is in [0, 2] range, convert: similarity = 1 - distance/2
        # If distance is already similarity-like, use directly
        if distance <= 1.0:
            similarity = 1.0 - distance
        else:
            # Distance might be squared or in different range
            similarity = max(0.0, 1.0 - (distance / 2.0))

        if similarity < similarity_threshold:
            return None

    # Return the most similar document
    if results['documents'] and len(results['documents'][0]) > 0:
        return results['documents'][0][0]

    return None


def find_structure_match(
    atlas: StyleAtlas,
    target_cluster_id: int,
    input_text: str,
    length_tolerance: float = 0.3,
    top_k: int = 1
) -> Optional[str]:
    """Find a paragraph matching the target style cluster for rhythm/structure.

    Queries ChromaDB by cluster_id and filters by length ratio to prevent expansion.
    Returns a paragraph from the target cluster that matches input length.

    Args:
        atlas: StyleAtlas containing ChromaDB collection.
        target_cluster_id: Target cluster ID to retrieve from.
        input_text: Input text to match length against (required for length filtering).
        length_tolerance: Tolerance for length matching (0.3 = 30%, default: 0.3).
        top_k: Number of results to return (default: 1).

    Returns:
        A paragraph from the target cluster matching input length, or None if not found.
    """
    if not hasattr(atlas, '_collection'):
        try:
            if hasattr(atlas, '_client'):
                atlas._collection = atlas._client.get_collection(name=atlas.collection_name)
            else:
                return None
        except:
            return None

    collection = atlas._collection

    # Calculate input length metrics
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        nltk_available = True
    except ImportError:
        nltk_available = False

    if nltk_available:
        input_word_count = len(word_tokenize(input_text))
        input_sent_count = len(sent_tokenize(input_text))
    else:
        # Fallback to simple counting
        input_word_count = len(input_text.split())
        input_sent_count = input_text.count('.') + input_text.count('!') + input_text.count('?')
        if input_sent_count == 0:
            input_sent_count = 1  # At least one sentence

    # Get all documents and filter by cluster_id
    all_results = collection.get()

    if not all_results['ids']:
        return None

    # Filter by cluster_id and length
    candidates = []
    for idx, para_id in enumerate(all_results['ids']):
        metadata = all_results['metadatas'][idx] if all_results['metadatas'] else {}
        cluster_id = metadata.get('cluster_id')

        if cluster_id != target_cluster_id:
            continue

        doc = all_results['documents'][idx] if all_results['documents'] else None
        if not doc:
            continue

        # Get length metadata
        cand_word_count = metadata.get('word_count', len(doc.split()))
        cand_sent_count = metadata.get('sentence_count', doc.count('.') + doc.count('!') + doc.count('?') or 1)

        # Calculate length ratios
        if input_word_count > 0:
            len_ratio = cand_word_count / input_word_count
        else:
            len_ratio = 1.0  # If input is empty, accept any length

        if input_sent_count > 0:
            sent_ratio = cand_sent_count / input_sent_count
        else:
            sent_ratio = 1.0

        # Accept candidates within tolerance (0.7x to 1.5x)
        min_ratio = 1.0 - length_tolerance
        max_ratio = 1.0 + length_tolerance

        if min_ratio <= len_ratio <= max_ratio:
            candidates.append((doc, len_ratio, cand_word_count))

    # If we have length-matched candidates, return the one closest to 1.0 ratio
    if candidates:
        best_match = min(candidates, key=lambda x: abs(x[1] - 1.0))
        return best_match[0]

    # Fallback: If no length match, return the shortest candidate to avoid expansion
    # Re-filter to get all cluster matches
    matching_paragraphs = []
    matching_metadata = []
    for idx, para_id in enumerate(all_results['ids']):
        metadata = all_results['metadatas'][idx] if all_results['metadatas'] else {}
        cluster_id = metadata.get('cluster_id')
        if cluster_id == target_cluster_id:
            doc = all_results['documents'][idx] if all_results['documents'] else None
            if doc:
                matching_paragraphs.append(doc)
                matching_metadata.append(metadata)

    if matching_paragraphs:
        # Find shortest by word count
        shortest_idx = 0
        shortest_word_count = float('inf')
        for idx, meta in enumerate(matching_metadata):
            word_count = meta.get('word_count', len(matching_paragraphs[idx].split()))
            if word_count < shortest_word_count:
                shortest_word_count = word_count
                shortest_idx = idx
        return matching_paragraphs[shortest_idx]

    return None



