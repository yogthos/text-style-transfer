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

    # Chunk into paragraphs
    paragraphs = _chunk_into_paragraphs(sample_text)

    if not paragraphs:
        raise ValueError("Sample text contains no paragraphs")

    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process each paragraph
    paragraph_ids = []
    semantic_embeddings = []
    style_vectors = []
    texts = []
    metadatas = []

    # Import NLTK for accurate counting
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        nltk_available = True
    except ImportError:
        nltk_available = False

    for idx, paragraph in enumerate(paragraphs):
        para_id = f"para_{idx}"
        paragraph_ids.append(para_id)
        texts.append(paragraph)

        # Generate semantic embedding
        semantic_emb = embedding_model.encode(paragraph, normalize_embeddings=True)
        semantic_embeddings.append(semantic_emb.tolist())

        # Generate style vector
        style_vec = get_style_vector(paragraph)
        style_vectors.append(style_vec)

        # Calculate length metadata
        if nltk_available:
            word_count = len(word_tokenize(paragraph))
            sentence_count = len(sent_tokenize(paragraph))
        else:
            # Fallback to simple counting
            word_count = len(paragraph.split())
            sentence_count = paragraph.count('.') + paragraph.count('!') + paragraph.count('?')
            if sentence_count == 0:
                sentence_count = 1  # At least one sentence

        metadata_entry = {
            "paragraph_idx": idx,
            "word_count": word_count,
            "sentence_count": sentence_count
        }

        # Add author_id if provided
        if author_id:
            metadata_entry["author_id"] = author_id

        # Note: We don't store style_vec in metadata because ChromaDB doesn't allow lists
        # Style vectors will be recomputed from document text when needed (e.g., in StyleBlender)

        metadatas.append(metadata_entry)

    # Run K-means clustering on style vectors (before storing, so we can add cluster_id to metadata)
    style_matrix = np.array(style_vectors)

    if len(paragraphs) < num_clusters:
        # Not enough paragraphs for requested clusters
        num_clusters = len(paragraphs)

    if num_clusters > 0:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(style_matrix)
        cluster_centers = kmeans.cluster_centers_
    else:
        cluster_labels = np.zeros(len(paragraphs), dtype=int)
        cluster_centers = np.array([style_vectors[0]]) if style_vectors else np.array([])

    # Add cluster_id to metadata before storing
    for idx, meta in enumerate(metadatas):
        meta["cluster_id"] = int(cluster_labels[idx])

    # Store in ChromaDB
    # If collection already exists and we're adding an author, we need unique IDs
    # Use author_id prefix if provided to avoid conflicts
    if author_id:
        # Prefix paragraph IDs with author_id to ensure uniqueness
        prefixed_ids = [f"{author_id}_{para_id}" for para_id in paragraph_ids]
    else:
        prefixed_ids = paragraph_ids

    collection.add(
        ids=prefixed_ids,
        embeddings=semantic_embeddings,
        documents=texts,
        metadatas=metadatas
    )

    # Create cluster_id mapping using prefixed IDs
    cluster_ids = {prefixed_id: int(cluster_labels[idx])
                  for idx, prefixed_id in enumerate(prefixed_ids)}


    # Create StyleAtlas object
    atlas = StyleAtlas(
        collection_name=collection_name,
        cluster_ids=cluster_ids,
        cluster_centers=cluster_centers,
        style_vectors=style_vectors,
        num_clusters=num_clusters
    )

    # Store collection reference (for later use)
    atlas._client = client
    atlas._collection = collection

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
        'num_clusters': atlas.num_clusters
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
        num_clusters=data['num_clusters']
    )

    if not CHROMADB_AVAILABLE:
        raise ImportError("ChromaDB is not available. Cannot load atlas.")

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

