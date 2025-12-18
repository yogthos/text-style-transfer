#!/usr/bin/env python3
"""Generate missing Style DNA for authors already loaded in ChromaDB.

This script retroactively generates Style DNA for authors that are already
in ChromaDB but don't have Style DNA in the sidecar registry. This is useful
if styles were loaded before the Style DNA feature was implemented, or if
Style DNA generation failed during initial loading.

Usage:
    python scripts/generate_style_dna.py
    python scripts/generate_style_dna.py --author "Hemingway"
    python scripts/generate_style_dna.py --force  # Regenerate all
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Error: ChromaDB is not installed.", file=sys.stderr)
    sys.exit(1)

from src.analyzer.style_metrics import get_style_vector
from src.atlas.style_registry import StyleRegistry
from src.generator.prompt_builder import generate_author_style_dna


def get_representative_sample_for_author(collection, author_id: str) -> str:
    """Get a representative sample text for an author using centroid method.

    Args:
        collection: ChromaDB collection object.
        author_id: Author identifier.

    Returns:
        Representative sample text (first 5 sentences from centroid document).
        Returns empty string if no sample can be found.
    """
    # Query for all documents with this author_id
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

    if not results.get('metadatas') or not results.get('documents'):
        return ""

    metadatas = results['metadatas']
    documents = results['documents']

    # OPTIMIZATION: Limit the number of documents processed to avoid slow computation
    # For large corpora, we sample a subset (max 100 documents) to find representative sample
    MAX_DOCS_TO_PROCESS = 100
    if len(documents) > MAX_DOCS_TO_PROCESS:
        # Sample documents evenly across the corpus
        import random
        sample_indices = sorted(random.sample(range(len(documents)), MAX_DOCS_TO_PROCESS))
        sampled_documents = [documents[i] for i in sample_indices]
        sampled_metadatas = [metadatas[i] for i in sample_indices]
    else:
        sample_indices = list(range(len(documents)))
        sampled_documents = documents
        sampled_metadatas = metadatas

    # Compute style vectors for sampled documents
    style_vectors = []
    valid_indices = []
    total_docs = len(sampled_documents)
    for local_idx, doc_text in enumerate(sampled_documents):
        if doc_text and len(doc_text.strip()) > 50:  # Minimum length check
            try:
                # Show progress for large batches
                if total_docs > 10 and (local_idx + 1) % max(1, total_docs // 10) == 0:
                    print(f"    Processing document {local_idx + 1}/{total_docs}...", end='\r')
                style_vec = get_style_vector(doc_text)
                style_vectors.append(style_vec)
                # Map back to original index
                original_idx = sample_indices[local_idx]
                valid_indices.append(original_idx)
            except Exception:
                continue
    if total_docs > 10:
        print()  # New line after progress

    if not style_vectors:
        return ""

    # Group by cluster_id if available
    cluster_groups = defaultdict(list)
    for vec_idx, valid_idx in enumerate(valid_indices):
        meta = metadatas[valid_idx]
        cluster_id = meta.get('cluster_id')
        if cluster_id is not None:
            cluster_groups[cluster_id].append((valid_idx, vec_idx))

    # Find largest cluster or use all documents
    if cluster_groups:
        cluster_counts = {cid: len(indices) for cid, indices in cluster_groups.items()}
        largest_cluster_id = max(cluster_counts.items(), key=lambda x: x[1])[0]
        cluster_items = cluster_groups[largest_cluster_id]

        # Get style vectors for this cluster (using vec_idx from tuple)
        cluster_style_vectors = [style_vectors[vec_idx] for _, vec_idx in cluster_items]
        cluster_center = np.mean(cluster_style_vectors, axis=0)

        # Find document closest to cluster center
        min_distance = float('inf')
        best_idx = None
        for orig_idx, vec_idx in cluster_items:
            distance = np.linalg.norm(style_vectors[vec_idx] - cluster_center)
            if distance < min_distance:
                min_distance = distance
                best_idx = orig_idx
    else:
        # No cluster info - use centroid of all documents
        all_center = np.mean(style_vectors, axis=0)
        min_distance = float('inf')
        best_idx = None
        for vec_idx, orig_idx in enumerate(valid_indices):
            distance = np.linalg.norm(style_vectors[vec_idx] - all_center)
            if distance < min_distance:
                min_distance = distance
                best_idx = orig_idx

    if best_idx is None:
        # Fallback: use first valid document
        best_idx = valid_indices[0]

    # Try to get skeletons from metadata first (more accurate)
    meta = metadatas[best_idx]
    skeletons_json = meta.get('skeletons')
    if skeletons_json:
        try:
            import json
            skeletons = json.loads(skeletons_json)
            if skeletons and isinstance(skeletons, list):
                # Use first 5 sentences from skeletons
                representative_sample = " ".join(skeletons[:5])
                if len(representative_sample.strip()) >= 50:
                    return representative_sample
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: extract sentences from document text
    sample_text = documents[best_idx]

    # Try to extract sentences
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        sentences = sent_tokenize(sample_text)
        if sentences:
            # Use first 5 sentences
            representative_sample = " ".join(sentences[:5])
            if len(representative_sample.strip()) >= 50:
                return representative_sample
    except Exception:
        pass

    # Final fallback: use first 500 characters
    return sample_text[:500]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate missing Style DNA for authors in ChromaDB.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Generate for all authors missing DNA
  %(prog)s --author "Hemingway"  # Generate for specific author
  %(prog)s --force            # Regenerate all (overwrite existing)
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )

    parser.add_argument(
        '--atlas-cache',
        type=str,
        default=None,
        help='Path to ChromaDB persistence directory (overrides config.json)'
    )

    parser.add_argument(
        '--collection-name',
        type=str,
        default=None,
        help='ChromaDB collection name (overrides config.json)'
    )

    parser.add_argument(
        '--author',
        type=str,
        default=None,
        help='Generate DNA for specific author only (default: all missing authors)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate Style DNA even if it already exists'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    atlas_config = config.get("atlas", {})
    atlas_cache_path = args.atlas_cache or atlas_config.get("persist_path")
    collection_name = args.collection_name or atlas_config.get("collection_name", "style_atlas")

    if args.verbose:
        print(f"Config: {args.config}")
        print(f"Atlas cache: {atlas_cache_path or '(in-memory)'}")
        print(f"Collection: {collection_name}")
        print()

    # Initialize ChromaDB client
    try:
        if atlas_cache_path:
            client = chromadb.PersistentClient(path=atlas_cache_path)
        else:
            client = chromadb.Client(Settings(anonymized_telemetry=False))
    except Exception as e:
        print(f"Error: Failed to initialize ChromaDB client: {e}", file=sys.stderr)
        sys.exit(1)

    # Get collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' does not exist or cannot be accessed: {e}", file=sys.stderr)
        print(f"Persist directory: {atlas_cache_path or '(in-memory)'}")
        sys.exit(1)

    # Get all documents with metadata
    try:
        all_results = collection.get(include=["metadatas", "documents"])
    except Exception as e:
        print(f"Error: Failed to query collection: {e}", file=sys.stderr)
        sys.exit(1)

    if not all_results.get('ids') or len(all_results['ids']) == 0:
        print(f"Collection '{collection_name}' is empty.")
        print(f"Persist directory: {atlas_cache_path or '(in-memory)'}")
        print(f"\nLoad styles using: python scripts/load_style.py --style-file <file> --author <name>")
        return 0

    # Get unique authors
    authors = set()
    for meta in all_results.get('metadatas', []):
        author_id = meta.get('author_id')
        if author_id:
            authors.add(author_id)

    if not authors:
        print("No authors found in collection (paragraphs may not have author_id tags).")
        return 0

    # Initialize Style Registry
    cache_dir = atlas_cache_path if atlas_cache_path else "atlas_cache"
    registry = StyleRegistry(cache_dir)

    # Filter authors based on arguments
    authors_to_process = []
    if args.author:
        if args.author not in authors:
            print(f"Error: Author '{args.author}' not found in collection.", file=sys.stderr)
            print(f"Available authors: {', '.join(sorted(authors))}")
            sys.exit(1)
        authors_to_process = [args.author]
    else:
        # Get all authors missing DNA (or all if --force)
        for author in sorted(authors):
            has_dna = registry.has_dna(author)
            if not has_dna or args.force:
                authors_to_process.append(author)

    if not authors_to_process:
        print("✓ All authors already have Style DNA.")
        if not args.force:
            print("Use --force to regenerate existing Style DNA.")
        return 0

    print(f"ChromaDB Collection: {collection_name}")
    print(f"Persist directory: {atlas_cache_path or '(in-memory)'}")
    print(f"Found {len(authors)} author(s) in collection")
    print(f"Will generate Style DNA for {len(authors_to_process)} author(s)")
    print()

    # Generate Style DNA for each author
    success_count = 0
    fail_count = 0

    for idx, author_id in enumerate(authors_to_process, 1):
        print(f"[{idx}/{len(authors_to_process)}] Processing {author_id}...")

        # Check if already exists (unless --force)
        if not args.force and registry.has_dna(author_id):
            existing_dna = registry.get_dna(author_id)
            print(f"  ✓ Style DNA already exists: {existing_dna[:60]}...")
            success_count += 1
            continue

        try:
            # Get representative sample
            if args.verbose:
                print(f"  Retrieving representative sample...")
            # Get document count first for progress indication
            try:
                doc_count = len(collection.get(where={"author_id": author_id}, include=["documents"]).get('documents', []))
                if doc_count > 100:
                    print(f"  Processing {doc_count} documents (sampling 100 for performance)...")
            except:
                pass
            representative_sample = get_representative_sample_for_author(collection, author_id)

            if not representative_sample or len(representative_sample.strip()) < 50:
                print(f"  ⚠ Warning: Could not get sufficient sample text for {author_id}")
                print(f"  Skipping...")
                fail_count += 1
                continue

            if args.verbose:
                print(f"  Sample length: {len(representative_sample)} characters")
                print(f"  Sample preview: {representative_sample[:100]}...")

            # Generate Style DNA
            print(f"  Generating Style DNA...")
            dna = generate_author_style_dna(author_id, representative_sample, str(config_path))

            # Save to registry
            registry.set_dna(author_id, dna)
            print(f"  ✓ Generated and saved Style DNA for {author_id}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Failed to generate Style DNA for {author_id}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            fail_count += 1

    print()
    print(f"✓ Successfully generated Style DNA for {success_count} author(s)")
    if fail_count > 0:
        print(f"⚠ Failed to generate Style DNA for {fail_count} author(s)")
    print(f"\nStyle DNA stored in: {registry.path}")


if __name__ == '__main__':
    sys.exit(main())

