#!/usr/bin/env python3
"""Turnkey script to initialize an author for style transfer.

This script runs all required setup steps:
1. Load styles into ChromaDB
2. Build paragraph atlas
3. Build RAG index
4. Build style graph index (for graph-based generation)

Usage:
    python scripts/init_author.py --author "Mao" --style-file data/corpus/mao.txt
    python scripts/init_author.py --author "Sagan" --style-file data/corpus/sagan.txt --config config.json --verbose
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, verbose=False):
    """Run a command and handle errors.

    Args:
        cmd: List of command arguments
        description: Description of what the command does
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Step: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")
    else:
        print(f"\n→ {description}...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=not verbose,
            text=True
        )
        if verbose and result.stdout:
            print(result.stdout)
        if not verbose:
            print(f"  ✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {description} failed", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"  ✗ Error: Script not found: {cmd[0]}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Initialize an author for style transfer (loads styles, builds atlas and RAG)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --author "Mao" --style-file data/corpus/mao.txt
  %(prog)s --author "Sagan" --style-file data/corpus/sagan.txt --config config.json --verbose
  %(prog)s --author "Hemingway" --style-file styles/hemingway.txt --skip-rag
        """
    )

    # Required arguments
    parser.add_argument(
        '--author',
        type=str,
        required=True,
        help='Author name'
    )

    parser.add_argument(
        '--style-file',
        type=str,
        required=True,
        help='Path to style text file (corpus)'
    )

    # Optional arguments
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
        help='Number of K-means clusters for style atlas (overrides config.json)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for atlas and RAG (overrides config.json)'
    )

    parser.add_argument(
        '--skip-style-load',
        action='store_true',
        help='Skip loading styles into ChromaDB (if already loaded)'
    )

    parser.add_argument(
        '--skip-atlas',
        action='store_true',
        help='Skip building paragraph atlas (if already built)'
    )

    parser.add_argument(
        '--skip-rag',
        action='store_true',
        help='Skip building RAG index (if already built)'
    )

    parser.add_argument(
        '--skip-graph-index',
        action='store_true',
        help='Skip building style graph index (if already built)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output for all steps'
    )

    parser.add_argument(
        '--relaxed',
        action='store_true',
        help='Use relaxed filtering for paragraph atlas (min-sentences=1, min-style-score=3)'
    )

    parser.add_argument(
        '--min-sentences',
        type=int,
        default=None,
        help='Minimum sentences per paragraph (overrides --relaxed if set)'
    )

    parser.add_argument(
        '--min-style-score',
        type=int,
        default=None,
        help='Minimum style score 1-5 (overrides --relaxed if set)'
    )

    args = parser.parse_args()

    # Validate style file exists
    style_file_path = Path(args.style_file)
    if not style_file_path.exists():
        print(f"Error: Style file not found: {args.style_file}", file=sys.stderr)
        sys.exit(1)

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"\n{'='*60}")
    print(f"Initializing author: {args.author}")
    print(f"Style file: {args.style_file}")
    print(f"Config: {args.config}")
    print(f"{'='*60}\n")

    success = True

    # Step 1: Load styles into ChromaDB
    if not args.skip_style_load:
        load_cmd = [
            sys.executable,
            str(script_dir / "load_style.py"),
            "--style-file", args.style_file,
            "--author", args.author,
            "--config", args.config
        ]
        if args.atlas_cache:
            load_cmd.extend(["--atlas-cache", args.atlas_cache])
        if args.num_clusters:
            load_cmd.extend(["--num-clusters", str(args.num_clusters)])
        if args.verbose:
            load_cmd.append("--verbose")

        if not run_command(load_cmd, "Loading styles into ChromaDB", args.verbose):
            success = False
            if not args.verbose:
                print("  Continuing with remaining steps...")
    else:
        print("\n→ Skipping style load (--skip-style-load)")

    # Step 2: Build paragraph atlas
    if not args.skip_atlas:
        atlas_cmd = [
            sys.executable,
            str(script_dir / "build_paragraph_atlas.py"),
            args.style_file,
            "--author", args.author,
            "--config", args.config
        ]
        if args.output_dir:
            atlas_cmd.extend(["--output-dir", args.output_dir])
        if args.num_clusters:
            atlas_cmd.extend(["--clusters", str(args.num_clusters)])
        if args.relaxed:
            atlas_cmd.append("--relaxed")
        if args.min_sentences is not None:
            atlas_cmd.extend(["--min-sentences", str(args.min_sentences)])
        if args.min_style_score is not None:
            atlas_cmd.extend(["--min-style-score", str(args.min_style_score)])

        if not run_command(atlas_cmd, "Building paragraph atlas", args.verbose):
            success = False
            if not args.verbose:
                print("  Continuing with remaining steps...")
    else:
        print("\n→ Skipping paragraph atlas build (--skip-atlas)")

    # Step 3: Build RAG index
    if not args.skip_rag:
        rag_cmd = [
            sys.executable,
            str(script_dir / "build_rag_index.py"),
            "--author", args.author,
            "--corpus-file", args.style_file,
            "--config", args.config
        ]
        if args.output_dir:
            rag_cmd.extend(["--output-dir", args.output_dir])

        if not run_command(rag_cmd, "Building RAG index", args.verbose):
            success = False
    else:
        print("\n→ Skipping RAG index build (--skip-rag)")

    # Step 4: Build Style Graph Index
    if not args.skip_graph_index:
        graph_index_cmd = [
            sys.executable,
            str(script_dir / "build_style_graph_index.py"),
            "--corpus-file", args.style_file,
            "--author", args.author,
            "--config", args.config
        ]

        if not run_command(graph_index_cmd, "Building style graph index", args.verbose):
            success = False
            if not args.verbose:
                print("  Continuing...")
    else:
        print("\n→ Skipping style graph index build (--skip-graph-index)")

    # Summary
    print(f"\n{'='*60}")
    if success:
        print(f"✓ Author '{args.author}' initialized successfully!")
        print(f"\nYou can now use this author for style transfer:")
        print(f"  python3 restyle.py input.txt -o output.txt")
        print(f"\nMake sure '{args.author}' is listed in config.json under 'blend.authors'")
        print(f"\nNote: Graph-based generation requires the style graph index (built in Step 4).")
    else:
        print(f"✗ Initialization completed with errors")
        print(f"  Review the output above for details")
        sys.exit(1)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

