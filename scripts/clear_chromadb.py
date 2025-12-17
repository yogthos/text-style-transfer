#!/usr/bin/env python3
"""Clear ChromaDB collections.

This script clears author styles from ChromaDB. You can clear the entire
collection or remove specific authors.

Usage:
    python scripts/clear_chromadb.py --all
    python scripts/clear_chromadb.py --author "Sagan"
"""

import argparse
import json
import sys
from pathlib import Path

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
        description='Clear ChromaDB collections or specific author data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all
  %(prog)s --author "Sagan"
  %(prog)s --author "Hemingway" --collection-name "style_atlas"
        """
    )

    parser.add_argument(
        '--collection-name',
        type=str,
        default=None,
        help='Collection name to clear (default: from config.json or "style_atlas")'
    )

    parser.add_argument(
        '--author',
        type=str,
        default=None,
        help='Author name to clear (clears only this author\'s data)'
    )

    parser.add_argument(
        '--atlas-cache',
        type=str,
        default=None,
        help='Path to ChromaDB persistence directory (overrides config.json)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Clear entire collection (default if no --author specified)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
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

    # Determine action: if --author specified, clear by author; otherwise clear all
    clear_all = args.all or (args.author is None)

    if args.verbose:
        print(f"Config: {args.config}")
        print(f"Atlas cache: {atlas_cache_path}")
        print(f"Collection: {collection_name}")
        if clear_all:
            print("Action: Clear entire collection")
        else:
            print(f"Action: Clear author '{args.author}'")
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
        sys.exit(1)

    if clear_all:
        # Clear entire collection
        try:
            # Get count before deletion
            all_results = collection.get(limit=1)
            total_count = collection.count() if hasattr(collection, 'count') else None

            # Delete collection
            client.delete_collection(name=collection_name)

            if total_count is not None:
                print(f"✓ Cleared entire collection '{collection_name}' ({total_count} paragraphs removed)")
            else:
                print(f"✓ Cleared entire collection '{collection_name}'")

        except Exception as e:
            print(f"Error: Failed to clear collection: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    else:
        # Clear by author
        try:
            # Query for paragraphs with this author_id
            results = collection.get(
                where={"author_id": args.author},
                include=["metadatas"]
            )

            if not results.get('ids') or len(results['ids']) == 0:
                print(f"⚠ No data found for author '{args.author}' in collection '{collection_name}'")
                return 0

            # Get IDs to delete
            ids_to_delete = results['ids']
            paragraph_count = len(ids_to_delete)

            if args.verbose:
                print(f"Found {paragraph_count} paragraphs for author '{args.author}'")

            # Delete by IDs
            collection.delete(ids=ids_to_delete)

            print(f"✓ Cleared {paragraph_count} paragraphs for author '{args.author}' from collection '{collection_name}'")

        except Exception as e:
            print(f"Error: Failed to clear author data: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    return 0


if __name__ == '__main__':
    sys.exit(main())

