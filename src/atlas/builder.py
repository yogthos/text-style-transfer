"""Style Atlas builder for ChromaDB-based style retrieval.

This module builds a Style Atlas by:
1. Chunking sample text into paragraphs
2. Generating dual embeddings (semantic + style)
3. Storing in ChromaDB
4. Running K-means clustering on style vectors
"""

import json
import pickle
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

    def __post_init__(self):
        """Ensure numpy arrays are properly typed."""
        if isinstance(self.cluster_centers, list):
            self.cluster_centers = np.array(self.cluster_centers)
        if isinstance(self.style_vectors, list):
            self.style_vectors = [np.array(v) if not isinstance(v, np.ndarray) else v
                                 for v in self.style_vectors]


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

        # Store skeletons (individual sentences) as JSON string in metadata
        # ChromaDB doesn't support lists, so we'll store as JSON string
        import json
        skeletons_json = json.dumps(window["skeletons"])

        metadata_entry = {
            "window_idx": idx,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "skeletons": skeletons_json,  # JSON string of sentence list
            "avg_length": int(window["avg_length"])
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
        'author_style_dna': atlas.author_style_dna or {}
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
        author_style_dna=data.get('author_style_dna', {})
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

