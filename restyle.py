#!/usr/bin/env python3
"""CLI entry point for text style transfer pipeline.

Usage:
    python restyle.py input/small.md -o output/small.md
    python restyle.py input/small.md -o output/small.md --config config.json
    python restyle.py input/small.md -o output/small.md --blend-ratio 0.7
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import run_pipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Transform text to match a target style while preserving meaning.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input/small.md -o output/small.md
  %(prog)s input/small.md -o output/small.md --max-retries 5
  %(prog)s input/small.md -o output/small.md --blend-ratio 0.7

Note: Author styles must be loaded into ChromaDB first using:
  python scripts/load_style.py --style-file <file> --author <name>
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input text file to transform'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output file path for generated text'
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.json',
        help='Configuration file path (default: config.json)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum number of retry attempts per sentence (default: 3)'
    )

    parser.add_argument(
        '--atlas-cache',
        type=str,
        default=None,
        help='Path to ChromaDB persistence directory (overrides config.json)'
    )

    parser.add_argument(
        '--blend-ratio',
        type=float,
        default=None,
        help='Blend ratio for style mixing (0.0 = All Author A, 1.0 = All Author B, default: 0.5). Overrides config.json'
    )

    parser.add_argument(
        '--perspective',
        type=str,
        default=None,
        choices=['first_person_singular', 'first_person_plural', 'third_person'],
        help='Force specific perspective (overrides author profile and input detection). Options: first_person_singular, first_person_plural, third_person'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--graph-mode',
        action='store_true',
        help='Enable graph-based generation mode (requires style graph index)'
    )

    args = parser.parse_args()

    # Validate blend_ratio if provided
    if args.blend_ratio is not None:
        if args.blend_ratio < 0.0 or args.blend_ratio > 1.0:
            print("Error: --blend-ratio must be between 0.0 and 1.0", file=sys.stderr)
            sys.exit(1)

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load config to get default atlas_cache_path
    import json
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Check for graph mode and verify index exists
    if args.graph_mode:
        import chromadb

        atlas_cache_path = args.atlas_cache or config.get("atlas", {}).get("persist_path", "atlas_cache")
        chroma_path = Path(atlas_cache_path) / "chroma"

        # Check if ChromaDB collection exists
        if not chroma_path.exists():
            print("⚠️  Style Graph Index missing.", file=sys.stderr)
            print(f"   Expected location: {chroma_path}", file=sys.stderr)
            print("   Please run 'python scripts/build_style_graph_index.py' first.", file=sys.stderr)
            sys.exit(1)

        # Try to connect and verify collection exists
        try:
            client = chromadb.PersistentClient(path=str(chroma_path))
            collection = client.get_collection(name="style_graphs")
            count = collection.count()
            if count == 0:
                print("⚠️  Style Graph Index is empty.", file=sys.stderr)
                print("   Please run 'python scripts/build_style_graph_index.py' first.", file=sys.stderr)
                sys.exit(1)
            if args.verbose:
                print(f"✓ Style Graph Index found ({count} graphs)")
        except Exception as e:
            print("⚠️  Style Graph Index missing or invalid.", file=sys.stderr)
            print(f"   Error: {e}", file=sys.stderr)
            print("   Please run 'python scripts/build_style_graph_index.py' first.", file=sys.stderr)
            sys.exit(1)

    # Resolve atlas_cache_path: CLI override > config.json > None
    atlas_cache_path = args.atlas_cache or config.get("atlas", {}).get("persist_path")

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print configuration
    if args.verbose:
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Config: {args.config}")
        print(f"Max retries: {args.max_retries}")
        print(f"Atlas cache: {atlas_cache_path or '(in-memory)'}")
        if args.blend_ratio is not None:
            print(f"Blend ratio: {args.blend_ratio}")
        print()

    try:
        # Run the pipeline
        output = run_pipeline(
            input_file=args.input,
            config_path=args.config,
            output_file=args.output,
            max_retries=args.max_retries,
            atlas_cache_path=atlas_cache_path,
            blend_ratio=args.blend_ratio,
            perspective=args.perspective,
            verbose=args.verbose
        )

        if args.verbose:
            print(f"\n✓ Successfully generated {len(output)} sentence(s)")
            print(f"✓ Output saved to {args.output}")
        else:
            print(f"✓ Generated {len(output)} sentence(s) -> {args.output}")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as e:
        # Display FileNotFoundError messages clearly (they already contain helpful instructions)
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())

