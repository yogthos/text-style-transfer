"""Style Navigator for predicting and retrieving style references.

This module provides functions to:
1. Build Markov chain from style clusters
2. Predict next style cluster
3. Retrieve style reference paragraphs from ChromaDB
4. Stochastic selection with history tracking to prevent repetition
5. Sliding window retrieval for contextual rhythm matching
"""

import re
import random
import math
import json
import numpy as np
from typing import Dict, Optional, Tuple, List, Set
from collections import defaultdict

from src.atlas.builder import StyleAtlas
from src.analyzer.style_metrics import get_style_vector


def is_valid_structural_template(text: str) -> bool:
    """Filter out titles, headers, fragments, and other invalid templates.

    Centralized validation logic - "Single Source of Truth" for structural templates.
    Returns False if the text is likely a header, title, citation garbage, or fragment.
    Uses spaCy for verb checking if available, with graceful fallback.

    Args:
        text: Text to validate as a structural template.

    Returns:
        True if text is a valid structural template, False otherwise.
    """
    if not text:
        return False

    words = text.split()

    # CRITICAL: Reject tiny fragments (e.g., "If 4^...", "Page 12")
    # Increased from 3 to 4 to catch more fragments
    if len(words) < 4:
        return False

    # Check for non-sentence junk (no vowels, mostly numbers/symbols)
    # Rejects things like "4^", "123", "---", etc.
    if not any(c.lower() in 'aeiouy' for c in text):
        return False

    # Navigation Artifacts Killer
    # Rejects: "Return to Table of Contents", "Back to page 5", "See page 12", etc.
    lower_text = text.lower().strip()
    navigation_patterns = [
        "return to", "back to", "continued on", "see page", "table of contents",
        "see figure", "see chapter", "see section", "refer to", "see above",
        "see below", "continued from", "go to", "jump to"
    ]
    if any(lower_text.startswith(pattern) for pattern in navigation_patterns):
        return False

    # Try to load spaCy (optional)
    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        pass  # Graceful fallback - spaCy not available

    # 1. Citation Killer (enhanced regex)
    # Rejects: "15. Charles Higham...", "[1] See logic...", "Page 45"
    if re.match(r'^\[?\d+\]?\.?\s*[A-Z]', text.strip()):
        return False

    # 2. Header Killer
    # Rejects: "CHAPTER ONE", "THE FINITUDE RULE" (all caps with < 10 words)
    if text.isupper() and len(words) < 10:
        return False

    # 3. Verb Check (if spaCy available)
    # A valid sentence MUST have at least one verb. Citations usually don't.
    if nlp:
        try:
            doc = nlp(text)
            has_verb = any(token.pos_ == "VERB" for token in doc)
            if not has_verb:
                return False

            # 4. Imperative Fragment Check
            # Rejects imperative-only phrases that lack subjects (e.g., "See Figure 1", "Return to page")
            # Check if first token is a verb (imperative) and sentence is short
            if len(doc) > 0:
                first_token = doc[0]
                if first_token.pos_ == "VERB" and len(doc) <= 6:
                    # Likely an imperative fragment without a subject
                    # Check if there's a subject later in the sentence
                    has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] for token in doc)
                    if not has_subject:
                        return False

            # 5. Junk Filter (using spaCy tokens)
            if len(doc) < 4:  # Very short fragments
                return False
        except Exception:
            # If spaCy processing fails, fall through to basic checks
            pass
    else:
        # Fallback: basic length check (no spaCy available)
        # Increased from 5 to 6 to be more strict
        if len(words) < 6:
            return False

    return True


def _calculate_structural_distance(candidate: Dict, reference: str) -> float:
    """Calculate structural distance between candidate and reference.

    Returns a distance score (0.0 = identical, higher = more different).
    Prefers candidates that are structurally different from failed ones.

    Args:
        candidate: Candidate dict with 'text' and 'word_count' keys.
        reference: Reference text string to compare against.

    Returns:
        Distance score (0.0 to 1.0, where 1.0 is maximum difference).
    """
    # Calculate word count difference
    ref_word_count = len(reference.split())
    cand_word_count = candidate.get('word_count', len(candidate.get('text', '').split()))
    word_diff = abs(cand_word_count - ref_word_count)

    # Normalize to 0-1 range (assuming max difference of 50 words)
    word_distance = min(word_diff / 50.0, 1.0)

    # Could add sentence count difference, punctuation pattern, etc.
    return word_distance


class StructureNavigator:
    """Manages structure template selection with history tracking and weighted sampling.

    Prevents repetitive selection by tracking recently used templates and using
    weighted random selection from top candidates to create natural variety.
    """

    def __init__(self, history_limit: int = 3):
        """Initialize the structure navigator.

        Args:
            history_limit: Number of recent templates to track (default: 3).
        """
        self.history_buffer: List[str] = []
        self.history_limit = history_limit

    def select_template(
        self,
        candidates: List[Dict],
        input_length: int
    ) -> Optional[Dict]:
        """Select a structural template using weighted random sampling + history filtering.

        Args:
            candidates: List of candidate dicts with keys: 'id', 'text', 'word_count'
            input_length: Target word count for the input text.

        Returns:
            Selected candidate dict, or None if no candidates available.
        """
        if not candidates:
            return None

        # 1. Filter by History (Anti-Repetition)
        available_candidates = [
            c for c in candidates
            if c.get('id') not in self.history_buffer
        ]

        # Fallback: If we filtered everything (rare), clear buffer
        if not available_candidates:
            available_candidates = candidates
            self.history_buffer = []

        # 2. Score Candidates (Distance from ideal length)
        # We prefer candidates closer to input_length, but allow variance
        # Use quality_score if available, otherwise calculate from length difference
        scored_candidates = []
        for cand in available_candidates:
            # Prefer quality_score if available (from structure match analysis)
            if 'quality_score' in cand:
                score = cand['quality_score']
            else:
                word_count = cand.get('word_count', len(cand.get('text', '').split()))
                len_diff = abs(word_count - input_length)
                # Inverse score: lower diff = higher score
                score = 1.0 / (len_diff + 1.0)
            scored_candidates.append((cand, score))

        # 3. Weighted Random Selection (The "Temperature")
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top 5
        top_n = scored_candidates[:5]

        if not top_n:
            return None

        # Create probabilities (favor top matches, but allow others)
        total_score = sum(x[1] for x in top_n)
        if total_score > 0:
            probs = [x[1] / total_score for x in top_n]
        else:
            # Uniform if all scores are zero
            probs = [1.0 / len(top_n)] * len(top_n)

        selected_tuple = random.choices(top_n, weights=probs, k=1)[0]
        selected_template = selected_tuple[0]

        # 4. Update History
        template_id = selected_template.get('id')
        if template_id:
            self.history_buffer.append(template_id)
            if len(self.history_buffer) > self.history_limit:
                self.history_buffer.pop(0)

        return selected_template


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


def retrieve_window_match(
    atlas: StyleAtlas,
    input_sentences: List[str],
    target_cluster_id: Optional[int] = None,
    similarity_threshold: float = 0.3
) -> Optional[List[Tuple[str, str]]]:
    """Retrieve a matching window and "zip" skeletons onto input sentences.

    This implements the "Zipper" method: retrieves a window of 3 sentences
    from the sample that matches the flow of the input, then maps each
    skeleton (sentence template) to each input sentence.

    Args:
        atlas: StyleAtlas containing ChromaDB collection.
        input_sentences: List of input sentence strings (typically 3).
        target_cluster_id: Optional target cluster ID to filter by.
        similarity_threshold: Minimum similarity threshold for matching.

    Returns:
        List of tuples (input_sentence, skeleton_template) if match found,
        or None if no suitable window found.
    """
    if not input_sentences:
        return None

    if not hasattr(atlas, '_collection'):
        try:
            if hasattr(atlas, '_client'):
                atlas._collection = atlas._client.get_collection(name=atlas.collection_name)
            else:
                return None
        except:
            return None

    collection = atlas._collection

    # Combine input into a block to find a "Flow Match"
    input_block = " ".join(input_sentences)
    input_word_count = len(input_block.split())

    # Generate style vector for the input block
    input_style_vec = get_style_vector(input_block)

    # Query for matching windows by semantic similarity
    try:
        # Get all documents first
        all_results = collection.get()

        if not all_results['ids'] or len(all_results['ids']) == 0:
            return None

        # Filter and score candidates
        candidates = []
        for idx, window_id in enumerate(all_results['ids']):
            metadata = all_results['metadatas'][idx] if all_results['metadatas'] else {}

            # Filter by cluster if specified
            if target_cluster_id is not None:
                cluster_id = metadata.get('cluster_id')
                if cluster_id != target_cluster_id:
                    continue

            # Get window text and skeletons
            window_text = all_results['documents'][idx] if all_results['documents'] else None
            if not window_text:
                continue

            # Parse skeletons from metadata (stored as JSON string)
            skeletons_json = metadata.get('skeletons', '[]')
            try:
                skeletons = json.loads(skeletons_json)
            except (json.JSONDecodeError, TypeError):
                # Fallback: if skeletons not available, skip this window
                continue

            if not skeletons or len(skeletons) == 0:
                continue

            # Calculate length similarity
            window_word_count = metadata.get('word_count', len(window_text.split()))
            if input_word_count > 0:
                len_ratio = window_word_count / input_word_count
            else:
                len_ratio = 1.0

            # Apply hard cap: only consider windows within configured length ratio bounds
            # Get thresholds from config (with fallback defaults)
            try:
                with open("config.json", 'r') as f:
                    config = json.load(f)
                atlas_config = config.get("atlas", {})
                min_length_ratio = atlas_config.get("min_length_ratio", 0.5)
                max_length_ratio = atlas_config.get("max_length_ratio", 2.0)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                min_length_ratio = 0.5
                max_length_ratio = 2.0

            if len_ratio < min_length_ratio or len_ratio > max_length_ratio:
                continue

            # Calculate Gaussian length penalty
            if len_ratio > 0:
                length_penalty = 1.0 / (1.0 + abs(math.log(len_ratio)))
            else:
                length_penalty = 0.0

            # For now, use length penalty as quality score
            # (Could enhance with semantic similarity if needed)
            quality_score = length_penalty

            candidates.append({
                'id': window_id,
                'text': window_text,
                'skeletons': skeletons,
                'word_count': window_word_count,
                'len_ratio': len_ratio,
                'quality_score': quality_score,
                'metadata': metadata
            })

        if not candidates:
            return None

        # Sort by quality score (Gaussian length penalty)
        candidates.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)

        # Select best matching window
        best_window = candidates[0]
        raw_skeletons = best_window['skeletons']

        # FILTER: Sanitize the skeletons using the central validation logic
        # This prevents fragments like "If 4^..." from being used as structure matches
        valid_skeletons = [
            s for s in raw_skeletons
            if is_valid_structural_template(s)
        ]

        # If the window contains ONLY garbage, discard the whole window
        # This triggers fallback to find_structure_match
        if not valid_skeletons:
            return None

        # The Zipper: Map valid skeletons to input sentences
        # If input has N sentences and window has M valid skeletons, map 1:1 up to min(N, M)
        mapped_pairs = []
        for i, input_sent in enumerate(input_sentences):
            if i < len(valid_skeletons):
                mapped_pairs.append((input_sent, valid_skeletons[i]))
            else:
                # If we run out of valid skeletons, reuse the last valid one
                mapped_pairs.append((input_sent, valid_skeletons[-1]))

        return mapped_pairs

    except Exception as e:
        # Collection might not exist or be empty
        return None


def find_structure_match(
    atlas: StyleAtlas,
    target_cluster_id: int,
    input_text: str,
    length_tolerance: float = 0.3,
    top_k: int = 1,
    navigator: Optional[StructureNavigator] = None,
    exclude_texts: Optional[Set[str]] = None,
    prefer_different_from: Optional[str] = None
) -> Optional[str]:
    """Find a paragraph matching the target style cluster for rhythm/structure.

    Queries ChromaDB by cluster_id and filters by length ratio to prevent expansion.
    Returns a paragraph from the target cluster that matches input length.
    If navigator is provided, uses weighted random selection from top candidates.

    Args:
        atlas: StyleAtlas containing ChromaDB collection.
        target_cluster_id: Target cluster ID to retrieve from.
        input_text: Input text to match length against (required for length filtering).
        length_tolerance: Tolerance for length matching (0.3 = 30%, default: 0.3).
        top_k: Number of candidates to retrieve (default: 1, but will use top 10 if navigator provided).
        navigator: Optional StructureNavigator for stochastic selection (default: None).

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
    try:
        all_results = collection.get()
    except Exception as e:
        # Collection might not exist or be empty
        return None

    if not all_results['ids'] or len(all_results['ids']) == 0:
        return None

    # Filter by cluster_id and length
    candidates = []
    candidate_dicts = []
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

        # Hard Cap: Only consider candidates within configured length ratio bounds
        # This prevents "Procrustean Bed" problem where we try to force incompatible lengths
        # Get thresholds from config (with fallback defaults)
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
            atlas_config = config.get("atlas", {})
            min_length_ratio = atlas_config.get("min_length_ratio", 0.5)
            max_length_ratio = atlas_config.get("max_length_ratio", 2.0)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            min_length_ratio = 0.5
            max_length_ratio = 2.0

        if len_ratio < min_length_ratio or len_ratio > max_length_ratio:
            continue

        # FILTER: Reject tiny fragments
        # If a template is below minimum word count, it's likely a page number, header, or OCR artifact.
        # Get threshold from config (with fallback default)
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
            atlas_config = config.get("atlas", {})
            min_structure_words = atlas_config.get("min_structure_words", 3)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            min_structure_words = 3

        if len(doc.split()) < min_structure_words:
            continue

        # Filter out invalid structural templates (titles, headers, fragments)
        if not is_valid_structural_template(doc):
            continue

        # Exclude tried matches (check early to avoid unnecessary computation)
        if exclude_texts and doc in exclude_texts:
            continue

        # Gaussian Length Penalty: Score candidates based on length similarity
        # score = 1.0 / (1.0 + abs(log(len_ratio)))
        # This gives a huge boost to exact matches and massive penalty to mismatches
        if len_ratio > 0:
            length_penalty = 1.0 / (1.0 + abs(math.log(len_ratio)))
        else:
            length_penalty = 0.0

        # For now, we use length_penalty as the quality score
        # (In future, could multiply by vector_similarity if we have it)
        quality_score = length_penalty

        # Distance weighting - prefer structurally different candidates
        if prefer_different_from:
            structural_distance = _calculate_structural_distance(
                {'text': doc, 'word_count': cand_word_count},
                prefer_different_from
            )
            # Boost candidates that are different (distance 0.5-1.0 get bonus)
            # This helps when the original match was too short/long
            if structural_distance > 0.3:
                quality_score *= (1.0 + structural_distance * 0.5)  # Up to 50% bonus

        candidates.append((doc, len_ratio, cand_word_count))
        candidate_dicts.append({
            'id': para_id,
            'text': doc,
            'word_count': cand_word_count,
            'len_ratio': len_ratio,
            'quality_score': quality_score
        })

    # Hard Cap: If no candidates exist within configured length ratio bounds, return None
    # Do not force a bad match
    if not candidate_dicts:
        return None

    # If navigator is provided, use stochastic selection
    if navigator:
        # Sort by quality_score (Gaussian length penalty) - higher is better
        candidate_dicts.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)
        top_candidates = candidate_dicts[:10]

        if top_candidates:
            selected = navigator.select_template(top_candidates, input_word_count)
            if selected:
                return selected.get('text')
            # Fallback if navigator returns None - use best quality score
            return top_candidates[0].get('text')

    # If we have candidates (and no navigator or navigator failed), return the one with best quality score
    if candidate_dicts:
        # Sort by quality_score (Gaussian length penalty) - higher is better
        candidate_dicts.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)
        return candidate_dicts[0].get('text')

    # No candidates found within acceptable length range
    return None



