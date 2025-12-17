#!/usr/bin/env python3
"""CLI entry point for text style transfer pipeline.

Usage:
    python restyle.py input/small.md -o output/small.md
    python restyle.py input/small.md -o output/small.md --sample prompts/sample_mao.txt
    python restyle.py input/small.md -o output/small.md --config config.json
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
  %(prog)s input/small.md -o output/small.md --sample prompts/custom_sample.txt
  %(prog)s input/small.md -o output/small.md --max-retries 5
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
        '-s', '--sample',
        type=str,
        default=None,
        help='Sample text file defining target style (default: from config.json)'
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
        help='Path to cache Style Atlas (speeds up subsequent runs)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

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

    # Validate sample file if provided
    if args.sample:
        sample_path = Path(args.sample)
        if not sample_path.exists():
            print(f"Error: Sample file not found: {args.sample}", file=sys.stderr)
            sys.exit(1)

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print configuration
    if args.verbose:
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Config: {args.config}")
        if args.sample:
            print(f"Sample: {args.sample}")
        print(f"Max retries: {args.max_retries}")
        if args.atlas_cache:
            print(f"Atlas cache: {args.atlas_cache}")
        print()

    try:
        # Run the pipeline
        output = run_pipeline(
            input_file=args.input,
            sample_file=args.sample,
            config_path=args.config,
            output_file=args.output,
            max_retries=args.max_retries,
            atlas_cache_path=args.atlas_cache
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
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())

