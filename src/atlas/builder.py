"""Style Atlas builder for ChromaDB-based style retrieval.

This module builds a Style Atlas by:
1. Chunking sample text into paragraphs
2. Generating dual embeddings (semantic + style)
3. Storing in ChromaDB
4. Running K-means clustering on style vectors
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

from src.analyzer.style_metrics import get_style_vector
import math


def clear_chromadb_collection(
    collection_name: str = "style_atlas",
    persist_directory: Optional[str] = None
) -> bool:
    """Clear/delete a ChromaDB collection.

    Args:
        collection_name: Name of the collection to clear (default: "style_atlas").
        persist_directory: Directory where ChromaDB is persisted (if None, uses in-memory).

    Returns:
        True if collection was cleared, False if it didn't exist.
    """
    if not CHROMADB_AVAILABLE:
        return False

    try:
        # Initialize ChromaDB client
        if persist_directory:
            client = chromadb.PersistentClient(path=persist_directory)
        else:
            client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Try to get and delete the collection
        try:
            collection = client.get_collection(name=collection_name)
            client.delete_collection(name=collection_name)
            return True
        except:
            # Collection doesn't exist
            return False
    except Exception as e:
        print(f"Warning: Failed to clear ChromaDB collection: {e}")
        return False


@dataclass
class StyleAtlas:
    """Container for Style Atlas data."""

    collection_name: str
    """Name of the ChromaDB collection."""

    cluster_ids: Dict[str, int]
    """Mapping from paragraph ID to cluster ID."""

    cluster_centers: np.ndarray
    """K-means cluster centers (num_clusters x style_vector_dim)."""

    style_vectors: List[np.ndarray]
    """All style vectors for paragraphs."""

    num_clusters: int
    """Number of clusters used."""

    author_style_dna: Optional[Dict[str, str]] = None
    """Mapping from author name to Style DNA string."""

    top_vocab: Optional[Dict[str, List[str]]] = None
    """Mapping from author name to list of top 50 characteristic words."""

    def __post_init__(self):
        """Ensure numpy arrays are properly typed."""
        if isinstance(self.cluster_centers, list):
            self.cluster_centers = np.array(self.cluster_centers)
        if isinstance(self.style_vectors, list):
            self.style_vectors = [np.array(v) if not isinstance(v, np.ndarray) else v
                                 for v in self.style_vectors]

    def get_examples_by_rhetoric(
        self,
        rhetorical_type,
        top_k: int = 3,
        exclude: Optional[List[str]] = None,
        author_name: Optional[str] = None,
        query_text: Optional[str] = None
    ) -> List[str]:
        """Get examples from atlas using 4-Tier Fallback Strategy.

        Tier 1: Perfect Match (Rhetorical Type + Length ±30%)
        Tier 2: Length Match (Ignore Rhetorical Type, Match Length ±30%)
        Tier 3: Type Match (Ignore Length, Match Rhetorical Type)
        Tier 4: Desperate (Vector similarity, author only)

        Args:
            rhetorical_type: RhetoricalType enum value to filter by.
            top_k: Number of examples to return (default: 3).
            exclude: Optional list of text strings to exclude from results.
            author_name: Optional author name to filter examples by.
            query_text: Optional query text to calculate length window for filtering.

        Returns:
            List of example text strings. Only returns empty if database is completely empty.
        """
        if not CHROMADB_AVAILABLE:
            return []

        # Import here to avoid circular dependency
        from src.atlas.rhetoric import RhetoricalType

        # Handle string input (for flexibility)
        if isinstance(rhetorical_type, str):
            # Try to match to enum
            for rtype in RhetoricalType:
                if rtype.value == rhetorical_type:
                    rhetorical_type = rtype
                    break
            else:
                # Not found, try Tier 4 (desperate)
                return self._tier4_vector_search(query_text, author_name, top_k, exclude)

        try:
            # Get collection
            if not hasattr(self, '_collection'):
                if hasattr(self, '_client'):
                    self._collection = self._client.get_collection(name=self.collection_name)
                else:
                    return []

            collection = self._collection

            # Calculate length window if query_text provided
            min_length = None
            max_length = None
            if query_text:
                input_length = len(query_text)
                min_length = int(input_length * 0.7)
                max_length = int(input_length * 1.3)

            # TIER 1: Perfect Match - Rhetorical Type + Length ±30%
            if query_text and min_length and max_length:
                where_conditions = [
                    {"rhetorical_type": rhetorical_type.value},
                    {"text_length": {"$gte": min_length}},
                    {"text_length": {"$lte": max_length}}
                ]
                if author_name:
                    where_conditions.append({"author_id": author_name})

                where_clause = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]

                try:
                    results = collection.get(where=where_clause, limit=top_k * 2)
                    if results and results.get('documents'):
                        examples = self._filter_and_shuffle(results['documents'], exclude, top_k)
                        if examples:
                            return examples
                except Exception:
                    pass  # Fall through to Tier 2

            # TIER 2: Length Match - Ignore Rhetorical Type, Match Length ±30%
            # (Better to have the right rhythm than the right logic)
            if query_text and min_length and max_length:
                where_conditions = [
                    {"text_length": {"$gte": min_length}},
                    {"text_length": {"$lte": max_length}}
                ]
                if author_name:
                    where_conditions.append({"author_id": author_name})

                where_clause = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]

                try:
                    results = collection.get(where=where_clause, limit=top_k * 2)
                    if results and results.get('documents'):
                        examples = self._filter_and_shuffle(results['documents'], exclude, top_k)
                        if examples:
                            return examples
                except Exception:
                    pass  # Fall through to Tier 3

            # TIER 3: Type Match - Ignore Length, Match Rhetorical Type
            # (Better to have the right logic than the right rhythm)
            where_conditions = [{"rhetorical_type": rhetorical_type.value}]
            if author_name:
                where_conditions.append({"author_id": author_name})

            where_clause = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]

            try:
                results = collection.get(where=where_clause, limit=top_k * 2)
                if results and results.get('documents'):
                    examples = self._filter_and_shuffle(results['documents'], exclude, top_k)
                    if examples:
                        return examples
            except Exception:
                pass  # Fall through to Tier 4

            # TIER 4: Desperate - Vector similarity, author only
            # Ignore everything, return top vector matches restricted ONLY by author
            return self._tier4_vector_search(query_text, author_name, top_k, exclude)

        except Exception as e:
            # If anything fails, try Tier 4 as last resort
            return self._tier4_vector_search(query_text, author_name, top_k, exclude)

    def _filter_and_shuffle(self, documents: List[str], exclude: Optional[List[str]], top_k: int) -> List[str]:
        """Filter out excluded texts and shuffle for variety."""
        if exclude:
            documents = [ex for ex in documents if ex not in exclude]
        random.shuffle(documents)
        return documents[:top_k]

    def _tier4_vector_search(
        self,
        query_text: Optional[str],
        author_name: Optional[str],
        top_k: int,
        exclude: Optional[List[str]]
    ) -> List[str]:
        """Tier 4: Vector similarity search restricted by author only.

        Args:
            query_text: Optional query text for semantic similarity.
            author_name: Optional author name to filter by.
            top_k: Number of results to return.
            exclude: Optional list of texts to exclude.

        Returns:
            List of example text strings, or empty if database is empty.
        """
        try:
            collection = self._collection

            # Build where clause (author only)
            where_clause = None
            if author_name:
                where_clause = {"author_id": author_name}

            if query_text:
                # Use vector similarity search
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(query_text, normalize_embeddings=True)

                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k * 2,
                    where=where_clause
                )

                if results and results.get('documents') and len(results['documents'][0]) > 0:
                    examples = results['documents'][0]
                    return self._filter_and_shuffle(examples, exclude, top_k)
            else:
                # No query text, just get random examples by author
                results = collection.get(
                    where=where_clause,
                    limit=top_k * 2
                )
                if results and results.get('documents'):
                    return self._filter_and_shuffle(results['documents'], exclude, top_k)

            # If we get here, database might be empty
            return []

        except Exception:
            # Last resort: try to get ANY documents
            try:
                results = collection.get(limit=top_k)
                if results and results.get('documents'):
                    return self._filter_and_shuffle(results['documents'], exclude, top_k)
            except Exception:
                pass
            return []

    def get_author_style_vector(self, author_name: str) -> Optional[np.ndarray]:
        """Get the average style vector for a specific author.

        Uses StyleBlender internally to calculate the author centroid.

        Args:
            author_name: Author identifier to get style vector for.

        Returns:
            Mean style vector (numpy array) representing the author's average style,
            or None if author not found or StyleBlender unavailable.
        """
        try:
            from src.atlas.blender import StyleBlender
            blender = StyleBlender(self)
            return blender.get_author_centroid(author_name)
        except (ValueError, ImportError, Exception) as e:
            # Author not found or StyleBlender unavailable - return None
            return None


def _chunk_into_paragraphs(text: str) -> List[str]:
    """Chunk text into paragraphs.

    Args:
        text: Input text.

    Returns:
        List of paragraph strings (non-empty).
    """
    # Try double newlines first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # If no double newlines, try single newlines
    if len(paragraphs) == 1 and '\n' in text:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    # If still only one, treat entire text as one paragraph
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    return paragraphs


def _chunk_into_windows(sentences: List[str], window_size: int = 3, stride: int = 1) -> List[Dict]:
    """Create overlapping windows of sentences to capture flow.

    Creates sliding windows to preserve contextual rhythm and cadence.
    Each window contains multiple sentences that flow together.

    Args:
        sentences: List of sentence strings.
        window_size: Number of sentences per window (default: 3).
        stride: Step size for sliding window (default: 1 for overlapping).

    Returns:
        List of window dictionaries with:
        - "text": Combined text of all sentences in window
        - "skeletons": List of individual sentence texts (structural templates)
        - "avg_length": Average word count per sentence in window
    """
    windows = []

    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < window_size:
        # If we have fewer sentences than window size, create a single window with all sentences
        combined_text = " ".join(sentences)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        windows.append({
            "text": combined_text,
            "skeletons": sentences,  # Individual sentences as structural templates
            "avg_length": avg_length
        })
        return windows

    # Create overlapping windows
    for i in range(0, len(sentences) - window_size + 1, stride):
        window_sentences = sentences[i : i + window_size]
        combined_text = " ".join(window_sentences)
        avg_length = sum(len(s.split()) for s in window_sentences) / window_size

        windows.append({
            "text": combined_text,
            "skeletons": window_sentences,  # Individual sentences as structural templates
            "avg_length": avg_length
        })

    return windows


def extract_characteristic_vocabulary(
    paragraphs: List[str],
    author_name: str,
    top_k: int = 50
) -> List[str]:
    """Extract top characteristic words for an author using frequency analysis.

    Extracts nouns, verbs, and adjectives from paragraphs, calculates their
    frequency, and returns the top_k most characteristic words (lemmas).

    Args:
        paragraphs: List of paragraph texts from the author.
        author_name: Name of the author (for logging).
        top_k: Number of top words to return (default: 50).

    Returns:
        List of top characteristic words (lemmas, lowercase, no duplicates).
    """
    if not paragraphs:
        return []

    # Try to use spaCy for better lemmatization
    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        pass  # Fallback to NLTK

    word_frequencies = {}
    all_text = " ".join(paragraphs)

    if nlp:
        # Use spaCy for lemmatization
        try:
            doc = nlp(all_text)
            for token in doc:
                # Extract nouns, verbs, and adjectives (content words)
                if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
                    lemma = token.lemma_.lower()
                    if lemma and len(lemma) > 2:  # Filter very short words
                        word_frequencies[lemma] = word_frequencies.get(lemma, 0) + 1
        except Exception:
            # If spaCy processing fails, fall through to NLTK
            nlp = None

    if not nlp:
        # Fallback to NLTK
        try:
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import stopwords

            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                import nltk
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words('english'))

            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(all_text.lower())
            pos_tags = pos_tag(tokens)

            for word, pos in pos_tags:
                if pos.startswith(('NN', 'VB', 'JJ')):  # Nouns, verbs, adjectives
                    if word.lower() not in stop_words and len(word) > 2:
                        # Lemmatize based on POS
                        if pos.startswith('NN'):
                            lemma = lemmatizer.lemmatize(word, pos='n')
                        elif pos.startswith('VB'):
                            lemma = lemmatizer.lemmatize(word, pos='v')
                        else:
                            lemma = lemmatizer.lemmatize(word, pos='a')

                        if lemma and len(lemma) > 2:
                            word_frequencies[lemma] = word_frequencies.get(lemma, 0) + 1
        except (ImportError, LookupError):
            # If NLTK is not available, use simple word frequency
            words = all_text.lower().split()
            from collections import Counter
            word_counts = Counter(words)
            # Filter out very short words and common stop words using spaCy if available
            stop_words = set()
            try:
                # Try to use spaCy for stop words
                if nlp:
                    doc = nlp(all_text)
                    from src.utils.spacy_linguistics import get_stop_words
                    stop_words = get_stop_words(doc)
                else:
                    # Fallback to hardcoded list if spaCy not available
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
            except Exception:
                # Fallback to hardcoded list on error
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
            for word, count in word_counts.items():
                if word not in stop_words and len(word) > 2:
                    word_frequencies[word] = count

    # Sort by frequency and return top_k
    sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, _ in sorted_words[:top_k]]

    if top_words:
        print(f"  Extracted {len(top_words)} characteristic words for {author_name}")

    return top_words


def build_style_atlas(
    sample_text: str,
    num_clusters: int = 5,
    collection_name: str = "style_atlas",
    persist_directory: Optional[str] = None,
    author_id: Optional[str] = None
) -> StyleAtlas:
    """Build a Style Atlas from sample text.

    This function:
    1. Chunks sample text into paragraphs
    2. Generates semantic embeddings (sentence-transformers)
    3. Generates style vectors (deterministic metrics)
    4. Stores in ChromaDB with metadata (including author_id if provided)
    5. Runs K-means clustering on style vectors
    6. Assigns cluster IDs to paragraphs

    Args:
        sample_text: The sample text to analyze.
        num_clusters: Number of K-means clusters (default: 5).
        collection_name: Name for ChromaDB collection (default: "style_atlas").
        persist_directory: Optional directory to persist ChromaDB (default: in-memory).
        author_id: Optional author identifier to tag paragraphs with (for style blending).

    Returns:
        StyleAtlas object containing collection, cluster assignments, and metadata.
    """
    if not CHROMADB_AVAILABLE:
        raise ImportError(
            "ChromaDB is not available. Please install it with: pip install chromadb\n"
            "Note: ChromaDB may require additional dependencies like onnxruntime."
        )

    # Initialize ChromaDB client
    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client(Settings(anonymized_telemetry=False))

    # Get or create collection (don't delete if adding author to existing collection)
    try:
        collection = client.get_collection(name=collection_name)
        # If author_id is provided and collection exists, we'll add to it instead of clearing
        if author_id is None:
            # Clear existing collection only if not adding an author
            collection.delete()
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "Style Atlas for text style transfer"}
            )
        # If author_id is provided, keep existing collection to add to it
    except:
        # Collection doesn't exist, create it
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Style Atlas for text style transfer"}
        )

    # Chunk into paragraphs first, then extract sentences for windows
    paragraphs = _chunk_into_paragraphs(sample_text)

    if not paragraphs:
        raise ValueError("Sample text contains no paragraphs")

    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Import NLTK for accurate counting and sentence tokenization
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        nltk_available = True
    except ImportError:
        nltk_available = False

    # Extract all sentences from paragraphs and create sliding windows
    all_sentences = []
    for paragraph in paragraphs:
        if nltk_available:
            para_sentences = sent_tokenize(paragraph)
        else:
            # Fallback: split by sentence-ending punctuation
            para_sentences = []
            for sent in paragraph.split('.'):
                sent = sent.strip()
                if sent:
                    para_sentences.append(sent + '.')
            if not para_sentences:
                para_sentences = [paragraph]
        all_sentences.extend([s.strip() for s in para_sentences if s.strip()])

    # CLEAN DATA AT THE SOURCE: Filter out invalid sentences before creating windows
    # This prevents fragments like "If 4^..." from entering ChromaDB
    # Import here to avoid circular dependency (navigator imports StyleAtlas from builder)
    from src.atlas.navigator import is_valid_structural_template
    all_sentences = [s for s in all_sentences if is_valid_structural_template(s)]

    if not all_sentences:
        raise ValueError("Sample text contains no valid sentences after filtering")

    # Create sliding windows of 3 sentences
    windows = _chunk_into_windows(all_sentences, window_size=3, stride=1)

    if not windows:
        raise ValueError("Failed to create sliding windows from sentences")

    # Process each window
    window_ids = []
    semantic_embeddings = []
    style_vectors = []
    texts = []
    metadatas = []

    for idx, window in enumerate(windows):
        window_id = f"window_{idx}"
        window_ids.append(window_id)

        # Use combined text for embedding and style vector
        combined_text = window["text"]
        texts.append(combined_text)

        # Generate semantic embedding from combined window text
        semantic_emb = embedding_model.encode(combined_text, normalize_embeddings=True)
        semantic_embeddings.append(semantic_emb.tolist())

        # Generate style vector from combined window text
        style_vec = get_style_vector(combined_text)
        style_vectors.append(style_vec)

        # Calculate length metadata
        if nltk_available:
            word_count = len(word_tokenize(combined_text))
            sentence_count = len(window["skeletons"])
        else:
            word_count = len(combined_text.split())
            sentence_count = len(window["skeletons"])

        # Calculate text_length (character count) for length window filtering
        text_length = len(combined_text)

        # Classify rhetorical type for this window
        # Use the first sentence in the window for classification (representative)
        from src.atlas.rhetoric import RhetoricalClassifier
        classifier = RhetoricalClassifier()
        first_sentence = window["skeletons"][0] if window["skeletons"] else combined_text
        rhetorical_type = classifier.classify_heuristic(first_sentence)

        # Store skeletons (individual sentences) as JSON string in metadata
        # ChromaDB doesn't support lists, so we'll store as JSON string
        import json
        skeletons_json = json.dumps(window["skeletons"])

        metadata_entry = {
            "window_idx": idx,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "skeletons": skeletons_json,  # JSON string of sentence list
            "avg_length": int(window["avg_length"]),
            "text_length": text_length,  # Character count for length window filtering
            "rhetorical_type": rhetorical_type.value  # Rhetorical type classification
        }

        # Add author_id if provided
        if author_id:
            metadata_entry["author_id"] = author_id

        metadatas.append(metadata_entry)

    # Run K-means clustering on style vectors (before storing, so we can add cluster_id to metadata)
    style_matrix = np.array(style_vectors)

    if len(windows) < num_clusters:
        # Not enough windows for requested clusters
        num_clusters = len(windows)

    if num_clusters > 0:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(style_matrix)
        cluster_centers = kmeans.cluster_centers_
    else:
        cluster_labels = np.zeros(len(windows), dtype=int)
        cluster_centers = np.array([style_vectors[0]]) if style_vectors else np.array([])

    # Add cluster_id to metadata before storing
    for idx, meta in enumerate(metadatas):
        meta["cluster_id"] = int(cluster_labels[idx])

    # Store in ChromaDB
    # If collection already exists and we're adding an author, we need unique IDs
    # Use author_id prefix if provided to avoid conflicts
    if author_id:
        # Prefix window IDs with author_id to ensure uniqueness
        prefixed_ids = [f"{author_id}_{window_id}" for window_id in window_ids]
    else:
        prefixed_ids = window_ids

    # ChromaDB has a maximum batch size limit (typically around 5000-6000 items)
    # Split into batches to avoid "Batch size exceeds max batch size" error
    batch_size = 5000  # Safe batch size for ChromaDB
    total_items = len(prefixed_ids)

    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)
        batch_ids = prefixed_ids[batch_start:batch_end]
        batch_embeddings = semantic_embeddings[batch_start:batch_end]
        batch_documents = texts[batch_start:batch_end]
        batch_metadatas = metadatas[batch_start:batch_end]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

        if batch_end < total_items:
            print(f"  Processed {batch_end}/{total_items} windows...")

    # Create cluster_id mapping using prefixed IDs
    cluster_ids = {prefixed_id: int(cluster_labels[idx])
                  for idx, prefixed_id in enumerate(prefixed_ids)}

    # Centroid-based sample selection for Style DNA generation
    # Find the largest cluster and select the window closest to its center
    representative_sample = None
    if num_clusters > 0 and len(cluster_labels) > 0:
        from collections import Counter
        cluster_counts = Counter(cluster_labels)
        largest_cluster_id = cluster_counts.most_common(1)[0][0]

        # Find window closest to cluster center
        largest_cluster_center = cluster_centers[largest_cluster_id]
        min_distance = float('inf')
        best_sample_idx = None

        for idx, (label, style_vec) in enumerate(zip(cluster_labels, style_vectors)):
            if label == largest_cluster_id:
                distance = np.linalg.norm(style_vec - largest_cluster_center)
                if distance < min_distance:
                    min_distance = distance
                    best_sample_idx = idx

        # Extract sample text from the best window
        if best_sample_idx is not None and best_sample_idx < len(windows):
            sample_window = windows[best_sample_idx]
            # Use first 5 sentences from the window as representative sample
            sample_sentences = sample_window.get("skeletons", [])[:5]
            representative_sample = " ".join(sample_sentences)
            print(f"  Selected representative sample from cluster {largest_cluster_id} (closest to centroid)")

    # Create StyleAtlas object
    atlas = StyleAtlas(
        collection_name=collection_name,
        cluster_ids=cluster_ids,
        cluster_centers=cluster_centers,
        style_vectors=style_vectors,
        num_clusters=num_clusters,
        author_style_dna={}
    )

    # Store collection reference (for later use)
    atlas._client = client
    atlas._collection = collection

    # Generate and store Style DNA if author_id is provided
    if author_id and representative_sample:
        from src.atlas.style_registry import StyleRegistry
        from src.generator.prompt_builder import generate_author_style_dna

        # Determine config_path (use default, could be passed as parameter if needed)
        # For now, we'll use the default "config.json" in the project root
        config_path = "config.json"

        # Initialize Style Registry
        cache_dir = persist_directory if persist_directory else "atlas_cache"
        registry = StyleRegistry(cache_dir)

        # Check if DNA already exists
        existing_dna = registry.get_dna(author_id)
        if existing_dna:
            print(f"  Using existing Style DNA for {author_id}")
            atlas.author_style_dna[author_id] = existing_dna
        else:
            # Generate new Style DNA
            print(f"  Generating Style DNA for {author_id}...")
            try:
                dna = generate_author_style_dna(author_id, representative_sample, config_path)
                registry.set_dna(author_id, dna)
                atlas.author_style_dna[author_id] = dna
            except Exception as e:
                print(f"  ⚠ Warning: Failed to generate Style DNA for {author_id}: {e}")
                print(f"  Continuing without Style DNA injection.")

    # Extract characteristic vocabulary for the author
    if author_id:
        try:
            print(f"  Extracting characteristic vocabulary for {author_id}...")
            vocab = extract_characteristic_vocabulary(paragraphs, author_id, top_k=50)
            if vocab:
                if atlas.top_vocab is None:
                    atlas.top_vocab = {}
                atlas.top_vocab[author_id] = vocab
        except Exception as e:
            print(f"  ⚠ Warning: Failed to extract vocabulary for {author_id}: {e}")
            print(f"  Continuing without vocabulary injection.")

    return atlas


def save_atlas(atlas: StyleAtlas, filepath: str):
    """Save StyleAtlas to disk (metadata only, ChromaDB persists separately).

    Args:
        atlas: StyleAtlas to save.
        filepath: Path to save file.
    """
    # Convert to dict for JSON serialization
    data = {
        'collection_name': atlas.collection_name,
        'cluster_ids': atlas.cluster_ids,
        'cluster_centers': atlas.cluster_centers.tolist(),
        'style_vectors': [v.tolist() for v in atlas.style_vectors],
        'num_clusters': atlas.num_clusters,
        'author_style_dna': atlas.author_style_dna or {},
        'top_vocab': atlas.top_vocab or {}
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_atlas(
    filepath: str,
    persist_directory: Optional[str] = None
) -> StyleAtlas:
    """Load StyleAtlas from disk.

    Args:
        filepath: Path to saved atlas file.
        persist_directory: Directory where ChromaDB is persisted (if applicable).

    Returns:
        StyleAtlas object.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Reconstruct StyleAtlas
    atlas = StyleAtlas(
        collection_name=data['collection_name'],
        cluster_ids=data['cluster_ids'],
        cluster_centers=np.array(data['cluster_centers']),
        style_vectors=[np.array(v) for v in data['style_vectors']],
        num_clusters=data['num_clusters'],
        author_style_dna=data.get('author_style_dna', {}),
        top_vocab=data.get('top_vocab', {})
    )

    if not CHROMADB_AVAILABLE:
        raise ImportError("ChromaDB is not available. Cannot load atlas.")

    # Load Style DNA from StyleRegistry if available
    if persist_directory:
        from src.atlas.style_registry import StyleRegistry
        registry = StyleRegistry(persist_directory)

        # Load DNA for all authors found in registry
        # Note: We load all profiles, but the pipeline will determine which authors to use
        all_profiles = registry.get_all_profiles()
        for author_name, profile in all_profiles.items():
            dna = profile.get("style_dna", "")
            if dna:
                atlas.author_style_dna[author_name] = dna

    # Reconnect to ChromaDB collection
    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client(Settings(anonymized_telemetry=False))

    try:
        collection = client.get_collection(name=atlas.collection_name)
        atlas._client = client
        atlas._collection = collection
    except:
        # Collection doesn't exist, will need to rebuild
        pass

    return atlas

