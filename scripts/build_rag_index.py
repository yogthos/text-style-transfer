#!/usr/bin/env python3
"""Build RAG index for style fragments.

This script creates a ChromaDB collection of 3-sentence sliding window fragments
from an author's corpus for semantic retrieval during style transfer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load spaCy with safety check
print("Loading spaCy model...")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    import spacy
    nlp = spacy.load("en_core_web_sm")


def split_into_sliding_windows(text: str, window_size: int = 3, overlap: int = 1) -> List[str]:
    """Split text into sliding windows of sentences.

    Args:
        text: Input text to split
        window_size: Number of sentences per window (default: 3)
        overlap: Number of sentences to overlap between windows (default: 1)

    Returns:
        List of text fragments (each containing window_size sentences)
    """
    # Process text with spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if len(sentences) < window_size:
        # If we have fewer sentences than window size, return all as one fragment
        return [" ".join(sentences)] if sentences else []

    fragments = []
    step = window_size - overlap  # How many sentences to advance each step

    for i in range(0, len(sentences) - window_size + 1, step):
        window = sentences[i:i + window_size]
        fragment = " ".join(window)
        fragments.append(fragment)

    # Handle remaining sentences if any
    if len(sentences) % step != 0:
        remaining_start = (len(sentences) // step) * step
        if remaining_start < len(sentences):
            remaining = sentences[remaining_start:]
            if len(remaining) >= window_size - overlap:  # Only add if substantial
                fragments.append(" ".join(remaining))

    return fragments


def build_fragment_index(corpus_file: str, author: str, config_path: str, output_dir: str = None):
    """Build fragment index and store in ChromaDB.

    Args:
        corpus_file: Path to corpus text file
        author: Author name
        config_path: Path to config.json
        output_dir: Optional override for output directory
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    style_rag_config = config.get("style_rag", {})
    window_size = style_rag_config.get("window_size", 3)
    overlap = style_rag_config.get("overlap", 1)
    embedding_model = style_rag_config.get("embedding_model", "all-mpnet-base-v2")

    # Determine output directory
    if output_dir:
        output_base = Path(output_dir)
    else:
        atlas_path = config.get("paragraph_atlas", {}).get("path", "atlas_cache/paragraph_atlas")
        output_base = Path(atlas_path)

    # Create author directory
    author_lower = "".join(c for c in author if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_').lower()
    author_dir = output_base / author_lower
    author_dir.mkdir(parents=True, exist_ok=True)

    # ChromaDB directory
    chroma_dir = author_dir / "style_fragments_chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print(f"Corpus: {corpus_file}")
    print(f"Author: {author} (Safe: {author_lower})")
    print(f"Output: {chroma_dir}")
    print(f"Embedding model: {embedding_model}")
    print(f"Window size: {window_size}, Overlap: {overlap}")

    # Read corpus
    print("\nReading corpus...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_text = f.read()

    # Split into fragments
    print("Splitting into sliding windows...")
    fragments = split_into_sliding_windows(corpus_text, window_size=window_size, overlap=overlap)
    print(f"Generated {len(fragments)} fragments")

    if len(fragments) == 0:
        print("Error: No fragments generated. Check corpus file format.")
        sys.exit(1)

    # Initialize ChromaDB
    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # CRITICAL: Use high-quality embedding model
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )

    collection_name = f"style_fragments_{author_lower}"

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass  # Collection doesn't exist, that's fine

    # Create collection with embedding function
    print(f"Creating collection: {collection_name}")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    # Add fragments to collection
    print(f"\nIndexing {len(fragments)} fragments...")
    batch_size = 100
    for i in range(0, len(fragments), batch_size):
        batch = fragments[i:i + batch_size]
        ids = [str(j) for j in range(i, i + len(batch))]
        metadatas = [{"fragment_index": j, "author": author} for j in range(i, i + len(batch))]

        collection.add(
            ids=ids,
            documents=batch,
            metadatas=metadatas
        )

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Indexed {i + len(batch)}/{len(fragments)} fragments...")

    print(f"\n✓ Done. Indexed {len(fragments)} fragments in collection '{collection_name}'")
    print(f"  Location: {chroma_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG index for style fragments from author corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--author",
        type=str,
        required=True,
        help="Author name (e.g., 'Mao')"
    )

    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Path to corpus file (default: data/corpus/{author}.txt)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json (default: config.json)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override default atlas path"
    )

    args = parser.parse_args()

    # Determine corpus file
    if args.corpus_file:
        corpus_file = args.corpus_file
    else:
        # Default to styles directory
        author_lower = args.author.lower()
        corpus_file = project_root / "styles" / f"sample_{author_lower}.txt"
        if not corpus_file.exists():
            # Try with original case
            corpus_file = project_root / "styles" / f"sample_{args.author}.txt"

    if not os.path.exists(corpus_file):
        print(f"Error: Corpus file not found: {corpus_file}")
        print(f"  Please provide --corpus-file or ensure file exists in styles/ directory")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        build_fragment_index(
            corpus_file=str(corpus_file),
            author=args.author,
            config_path=args.config,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

