#!/usr/bin/env python3
"""Load author styles into ChromaDB.

This script loads style text files into ChromaDB with author tags for use
in style blending and text restyling. This creates the style atlas used for
statistical/archetype-based generation (fallback mode).

For graph-based generation, you also need to run:
    python scripts/build_style_graph_index.py --corpus-file <file> --author <author>

Usage:
    python scripts/load_style.py --style-file data/corpus/sagan.txt --author "Sagan"
    python scripts/load_style.py --style-file file1.txt --author "Author1" --style-file file2.txt --author "Author2"
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.builder import build_style_atlas, save_atlas


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Load author styles into ChromaDB for text restyling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --style-file data/corpus/sagan.txt --author "Sagan"
  %(prog)s --style-file data/corpus/hemingway.txt --author "Hemingway" \\
           --style-file data/corpus/lovecraft.txt --author "Lovecraft"
        """
    )

    parser.add_argument(
        '--style-file',
        type=str,
        action='append',
        required=True,
        help='Path to style text file (can be specified multiple times for multiple authors)'
    )

    parser.add_argument(
        '--author',
        type=str,
        action='append',
        required=True,
        help='Author name to tag the style (must match number of --style-file arguments)'
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
        '--num-clusters',
        type=int,
        default=None,
        help='Number of K-means clusters (overrides config.json)'
    )

    parser.add_argument(
        '--collection-name',
        type=str,
        default=None,
        help='ChromaDB collection name (overrides config.json)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Validate that style-file and author lists match in length
    if len(args.style_file) != len(args.author):
        print("Error: Number of --style-file arguments must match number of --author arguments", file=sys.stderr)
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
    num_clusters = args.num_clusters or atlas_config.get("num_clusters", 5)
    collection_name = args.collection_name or atlas_config.get("collection_name", "style_atlas")

    # Validate style files exist
    for style_file in args.style_file:
        style_path = Path(style_file)
        if not style_path.exists():
            print(f"Error: Style file not found: {style_file}", file=sys.stderr)
            sys.exit(1)

    # Warn if no persist path (data will be in-memory)
    if not atlas_cache_path:
        print("⚠ Warning: No persist directory specified. Data will be stored in-memory and lost when process exits.")
        print("  Specify --atlas-cache or set 'atlas.persist_path' in config.json to persist data.")
        print()

    if args.verbose:
        print(f"Config: {args.config}")
        print(f"Atlas cache: {atlas_cache_path or '(in-memory)'}")
        print(f"Collection: {collection_name}")
        print(f"Num clusters: {num_clusters}")
        print(f"Loading {len(args.style_file)} author style(s)...")
        print()
    else:
        # Always print the path being used so users know where data is stored
        print(f"Using ChromaDB at: {atlas_cache_path or '(in-memory)'}")
        print(f"Collection: {collection_name}")
        print()

    # Load each style file
    for idx, (style_file, author_name) in enumerate(zip(args.style_file, args.author)):
        if args.verbose:
            print(f"Loading style {idx + 1}/{len(args.style_file)}: {author_name}")
            print(f"  Style file: {style_file}")

        try:
            # Read style file
            with open(style_file, 'r', encoding='utf-8') as f:
                style_text = f.read()

            # Build atlas with author_id
            atlas = build_style_atlas(
                sample_text=style_text,
                num_clusters=num_clusters,
                collection_name=collection_name,
                persist_directory=atlas_cache_path,
                author_id=author_name
            )

            paragraph_count = len(atlas.cluster_ids)
            print(f"  ✓ Loaded style for author '{author_name}': {paragraph_count} paragraphs, {atlas.num_clusters} clusters")

            # Save atlas.json file (needed by pipeline)
            if atlas_cache_path:
                atlas_file = Path(atlas_cache_path) / "atlas.json"
                save_atlas(atlas, str(atlas_file))
                if args.verbose:
                    print(f"  Saved atlas metadata to: {atlas_file}")

            if args.verbose:
                print(f"  Collection: {collection_name}")
                print(f"  Persist directory: {atlas_cache_path}")
                print()

        except Exception as e:
            print(f"  ✗ Failed to load style for author '{author_name}': {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    print(f"\n✓ Successfully loaded {len(args.style_file)} author style(s) into ChromaDB")
    print(f"  Collection: {collection_name}")
    print(f"  Persist directory: {atlas_cache_path or '(in-memory)'}")
    if not atlas_cache_path:
        print(f"\n⚠ Note: Data is in-memory. Use --atlas-cache or set 'atlas.persist_path' in config.json to persist.")

    print(f"\nNote: This script loads styles for statistical/archetype-based generation.")
    print(f"      For graph-based generation, also run:")
    print(f"        python scripts/build_style_graph_index.py --corpus-file <file> --author <author>")
    print(f"\nYou can now use restyle.py to transform text using these styles.")


if __name__ == '__main__':
    sys.exit(main())

