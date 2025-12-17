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
    persist_directory: Optional[str] = None
) -> StyleAtlas:
    """Build a Style Atlas from sample text.

    This function:
    1. Chunks sample text into paragraphs
    2. Generates semantic embeddings (sentence-transformers)
    3. Generates style vectors (deterministic metrics)
    4. Stores in ChromaDB with metadata
    5. Runs K-means clustering on style vectors
    6. Assigns cluster IDs to paragraphs

    Args:
        sample_text: The sample text to analyze.
        num_clusters: Number of K-means clusters (default: 5).
        collection_name: Name for ChromaDB collection (default: "style_atlas").
        persist_directory: Optional directory to persist ChromaDB (default: in-memory).

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

    # Get or create collection
    try:
        collection = client.get_collection(name=collection_name)
        # Clear existing collection
        collection.delete()
    except:
        pass

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

    # Store in ChromaDB
    collection.add(
        ids=paragraph_ids,
        embeddings=semantic_embeddings,
        documents=texts,
        metadatas=[{"paragraph_idx": idx} for idx in range(len(paragraphs))]
    )

    # Run K-means clustering on style vectors
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

    # Create cluster_id mapping
    cluster_ids = {para_id: int(label) for para_id, label in zip(paragraph_ids, cluster_labels)}

    # Update ChromaDB metadata with cluster IDs
    collection.update(
        ids=paragraph_ids,
        metadatas=[{"paragraph_idx": idx, "cluster_id": int(cluster_labels[idx])}
                  for idx in range(len(paragraphs))]
    )

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

