"""Style blending module for mixing multiple author styles.

This module implements vector space operations to blend styles from multiple
authors by finding "bridge texts" that naturally connect two author styles.
"""

import random
from typing import List, Optional, Dict
import numpy as np

from src.atlas.builder import StyleAtlas
from src.analyzer.vocabulary import extract_global_vocabulary
from src.analyzer.style_metrics import get_style_vector


class StyleBlender:
    """Handles style blending using vector interpolation to find bridge texts."""

    def __init__(self, atlas: StyleAtlas):
        """Initialize StyleBlender with a StyleAtlas.

        Args:
            atlas: StyleAtlas containing ChromaDB collection with author-tagged paragraphs.
        """
        self.atlas = atlas
        if not hasattr(atlas, '_collection'):
            # Try to get collection from client
            if hasattr(atlas, '_client'):
                try:
                    atlas._collection = atlas._client.get_collection(name=atlas.collection_name)
                except:
                    raise ValueError("Could not access ChromaDB collection from atlas")

    def get_author_centroid(self, author_id: str) -> np.ndarray:
        """Calculate the average style vector for a specific author.

        Args:
            author_id: Author identifier to calculate centroid for.

        Returns:
            Mean style vector (numpy array) representing the author's average style.

        Raises:
            ValueError: If author_id not found in collection.
        """
        collection = self.atlas._collection

        # Query ChromaDB for all paragraphs with this author_id
        # Include documents so we can recompute style vectors
        try:
            results = collection.get(
                where={"author_id": author_id},
                include=["metadatas", "documents"]
            )
        except Exception as e:
            # Fallback: query all and filter
            all_results = collection.get(include=["metadatas", "documents"])
            results = {
                'metadatas': [],
                'documents': []
            }
            for idx, meta in enumerate(all_results.get('metadatas', [])):
                if meta.get('author_id') == author_id:
                    results['metadatas'].append(meta)
                    if all_results.get('documents'):
                        results['documents'].append(all_results['documents'][idx])

        if not results.get('metadatas'):
            raise ValueError(f"Author '{author_id}' not found in collection")

        # Extract style vectors by recomputing from document text
        # (style_vec is not stored in metadata because ChromaDB doesn't allow lists)
        style_vectors = []
        for idx, meta in enumerate(results['metadatas']):
            # Recompute style vector from document text
            if results.get('documents') and idx < len(results['documents']):
                doc_text = results['documents'][idx]
                style_vec = get_style_vector(doc_text)
                style_vectors.append(style_vec)

        if not style_vectors:
            raise ValueError(f"No style vectors found for author '{author_id}'")

        # Calculate mean across all dimensions
        return np.mean(style_vectors, axis=0)

    def retrieve_blended_template(
        self,
        author_a: str,
        author_b: str,
        blend_ratio: float = 0.5,
        n_results: int = 5
    ) -> Optional[str]:
        """Find 'Bridge Texts' that represent a mix of two authors.

        Uses vector interpolation to find paragraphs that naturally blend
        the styles of two authors.

        Args:
            author_a: First author identifier.
            author_b: Second author identifier.
            blend_ratio: Blend ratio (0.0 = All Author A, 1.0 = All Author B, default: 0.5).
            n_results: Number of results to return (default: 5).

        Returns:
            Best matching bridge text paragraph, or None if no match found.
        """
        # 1. Get author centroids
        try:
            vec_a = self.get_author_centroid(author_a)
            vec_b = self.get_author_centroid(author_b)
        except ValueError as e:
            print(f"  ⚠ Warning: {e}")
            return None

        # 2. Calculate interpolated vector (Linear Interpolation)
        target_vec = (vec_a * (1 - blend_ratio)) + (vec_b * blend_ratio)

        # 3. Query ChromaDB for paragraphs matching the interpolated style
        # We need to query by style vector, but ChromaDB stores semantic embeddings
        # So we'll query all paragraphs from both authors and find closest style match
        collection = self.atlas._collection

        try:
            # Get all paragraphs from both authors
            results = collection.get(
                where={"author_id": {"$in": [author_a, author_b]}},
                include=["metadatas", "documents"]
            )
        except Exception as e:
            # Fallback: get all and filter
            all_results = collection.get(include=["metadatas", "documents"])
            results = {
                'metadatas': [],
                'documents': []
            }
            for idx, meta in enumerate(all_results.get('metadatas', [])):
                if meta.get('author_id') in [author_a, author_b]:
                    results['metadatas'].append(meta)
                    if all_results.get('documents'):
                        results['documents'].append(all_results['documents'][idx])

        if not results.get('metadatas'):
            return None

        # 4. Find closest style matches
        candidates = []
        for idx, meta in enumerate(results['metadatas']):
            # Recompute style vector from document text (not stored in metadata)
            if results.get('documents') and idx < len(results['documents']):
                para_style_vec = get_style_vector(results['documents'][idx])
            else:
                continue

            # Calculate distance to target vector
            distance = np.linalg.norm(para_style_vec - target_vec)
            doc_text = results['documents'][idx] if results.get('documents') else None
            if doc_text:
                candidates.append((distance, doc_text))

        if not candidates:
            return None

        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x[0])

        # Return the best match
        return candidates[0][1]

    def get_hybrid_vocab(
        self,
        author_a: str,
        author_b: str,
        ratio: float = 0.5,
        vocab_size: int = 10
    ) -> List[str]:
        """Create a vocabulary bag that mixes both authors.

        Args:
            author_a: First author identifier.
            author_b: Second author identifier.
            ratio: Blend ratio (0.0 = All Author A, 1.0 = All Author B, default: 0.5).
            vocab_size: Total number of words to return (default: 10).

        Returns:
            List of words sampled from both authors based on ratio.
        """
        collection = self.atlas._collection

        # Get sample texts for both authors
        def get_author_texts(author_id: str) -> str:
            try:
                results = collection.get(
                    where={"author_id": author_id},
                    include=["documents"]
                )
                if results.get('documents'):
                    return " ".join(results['documents'])
            except:
                pass
            return ""

        text_a = get_author_texts(author_a)
        text_b = get_author_texts(author_b)

        if not text_a and not text_b:
            return []

        # Extract vocabulary from both authors
        vocab_a_dict = extract_global_vocabulary(text_a, top_n=200) if text_a else {'positive': [], 'negative': [], 'neutral': []}
        vocab_b_dict = extract_global_vocabulary(text_b, top_n=200) if text_b else {'positive': [], 'negative': [], 'neutral': []}

        # Combine all words from both authors
        vocab_a_all = vocab_a_dict.get('positive', []) + vocab_a_dict.get('negative', []) + vocab_a_dict.get('neutral', [])
        vocab_b_all = vocab_b_dict.get('positive', []) + vocab_b_dict.get('negative', []) + vocab_b_dict.get('neutral', [])

        # Sample based on ratio
        count_b = int(vocab_size * ratio)
        count_a = vocab_size - count_b

        # Sample words (with replacement if needed)
        sampled_a = random.sample(vocab_a_all, min(count_a, len(vocab_a_all))) if vocab_a_all else []
        sampled_b = random.sample(vocab_b_all, min(count_b, len(vocab_b_all))) if vocab_b_all else []

        # If we need more words, sample with replacement
        while len(sampled_a) < count_a and vocab_a_all:
            sampled_a.append(random.choice(vocab_a_all))
        while len(sampled_b) < count_b and vocab_b_all:
            sampled_b.append(random.choice(vocab_b_all))

        return sampled_a + sampled_b

