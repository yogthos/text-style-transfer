"""Style RAG retriever for fetching semantically relevant style fragments.

This module provides a lightweight class to retrieve style fragments from a
ChromaDB collection based on semantic similarity to a query text.
"""

import json
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions


class StyleRAG:
    """Retrieves style fragments based on semantic similarity using high-fidelity embeddings."""

    def __init__(self, atlas_dir: str, author: str, config_path: str = "config.json"):
        """Initialize the Style RAG retriever.

        Args:
            atlas_dir: Base directory for paragraph atlas (e.g., "atlas_cache/paragraph_atlas")
            author: Author name (e.g., "Mao" or "mao")
            config_path: Path to configuration file
        """
        self.atlas_dir = Path(atlas_dir)
        self.author = author
        author_lower = author.lower()
        self.author_dir = self.atlas_dir / author_lower

        # Load config to get embedding model name
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        style_rag_config = self.config.get("style_rag", {})
        embedding_model = style_rag_config.get("embedding_model", "all-mpnet-base-v2")

        # Initialize ChromaDB client with custom embedding function
        chroma_dir = self.author_dir / "style_fragments_chroma"
        try:
            self.client = chromadb.PersistentClient(path=str(chroma_dir))

            # CRITICAL: Use the SAME embedding function as indexer
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )

            collection_name = f"style_fragments_{author_lower}"
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_fn
                )
            except Exception:
                # Collection doesn't exist yet
                self.collection = None
                print(f"Warning: Style fragments collection '{collection_name}' not found for author '{author}'.")
                print(f"  Run: python tools/build_rag_index.py --author {author}")
        except Exception as e:
            print(f"Warning: Could not connect to ChromaDB for style fragments: {e}")
            self.client = None
            self.collection = None

    def retrieve_palette(self, query_text: str, n: int = 8) -> List[str]:
        """Retrieve style fragments semantically similar to the query text.

        Args:
            query_text: The neutral summary text to find similar fragments for
            n: Number of fragments to retrieve (default: 8)

        Returns:
            List of style fragment strings, or empty list if collection unavailable
        """
        if not self.collection:
            return []

        if not query_text or not query_text.strip():
            return []

        try:
            # Perform semantic search using ChromaDB's query method
            # The embedding function is automatically used for the query
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n
            )

            if results and results.get("documents") and len(results["documents"]) > 0:
                # Return the list of fragment texts
                return results["documents"][0]

            return []
        except Exception as e:
            print(f"Warning: Could not retrieve style palette: {e}")
            return []

