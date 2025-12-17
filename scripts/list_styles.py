#!/usr/bin/env python3
"""List author styles loaded in ChromaDB.

This script shows which author styles are currently loaded in ChromaDB,
along with statistics for each author.

Usage:
    python scripts/list_styles.py
    python scripts/list_styles.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='List author styles loaded in ChromaDB.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --verbose
  %(prog)s --collection-name "style_atlas"
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
        help='Collection name to list (overrides config.json)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed statistics'
    )

    args = parser.parse_args()

    if not CHROMADB_AVAILABLE:
        print("Error: ChromaDB is not available. Please install it with: pip install chromadb", file=sys.stderr)
        sys.exit(1)

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

    # Group by author_id
    authors = defaultdict(lambda: {'count': 0, 'clusters': set(), 'word_counts': []})
    no_author_count = 0

    for idx, meta in enumerate(all_results.get('metadatas', [])):
        author_id = meta.get('author_id')
        if author_id:
            authors[author_id]['count'] += 1
            cluster_id = meta.get('cluster_id')
            if cluster_id is not None:
                authors[author_id]['clusters'].add(cluster_id)
            word_count = meta.get('word_count')
            if word_count:
                authors[author_id]['word_counts'].append(word_count)
        else:
            no_author_count += 1

    # Print summary
    print(f"ChromaDB Collection: {collection_name}")
    print(f"Persist directory: {atlas_cache_path or '(in-memory)'}")
    print(f"Total paragraphs: {len(all_results['ids'])}")
    print()

    if authors:
        print(f"Loaded Authors ({len(authors)}):")
        print()

        for author_id in sorted(authors.keys()):
            author_data = authors[author_id]
            paragraph_count = author_data['count']
            cluster_count = len(author_data['clusters'])
            word_counts = author_data['word_counts']
            avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
            total_words = sum(word_counts) if word_counts else 0

            print(f"  {author_id}:")
            print(f"    Paragraphs: {paragraph_count}")
            print(f"    Clusters: {cluster_count}")
            if word_counts:
                print(f"    Total words: {total_words:,}")
                print(f"    Avg words/paragraph: {avg_words:.1f}")

            if args.verbose:
                cluster_list = sorted(author_data['clusters'])
                print(f"    Cluster IDs: {cluster_list}")

            print()
    else:
        print("No authors found (paragraphs may not have author_id tags).")

    if no_author_count > 0:
        print(f"⚠ Note: {no_author_count} paragraph(s) without author_id tags")

    return 0


if __name__ == '__main__':
    sys.exit(main())

