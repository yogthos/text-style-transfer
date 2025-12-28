#!/usr/bin/env python3
"""Fast style transfer using LoRA-adapted models.

This is the simplified CLI for style transfer that uses pre-trained
LoRA adapters for fast, consistent output.

Usage:
    # Basic usage
    python scripts/fast_restyle.py input.md -o output.md \
        --adapter lora_adapters/sagan \
        --author "Carl Sagan"

    # With verbose output
    python scripts/fast_restyle.py input.md -o output.md \
        --adapter lora_adapters/sagan \
        --author "Carl Sagan" \
        --verbose

    # List available adapters
    python scripts/fast_restyle.py --list-adapters

Performance target: ~15-30 seconds per paragraph (vs 10+ minutes previously)
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def list_adapters(adapters_dir: str = "lora_adapters") -> None:
    """List available LoRA adapters."""
    adapters_path = Path(adapters_dir)

    if not adapters_path.exists():
        print(f"No adapters directory found at: {adapters_path}")
        print("\nTo train an adapter, run:")
        print("  python scripts/train_mlx_lora.py --prepare --train \\")
        print("      --corpus styles/sample_author.txt \\")
        print("      --author 'Author Name' \\")
        print("      --output lora_adapters/author")
        return

    adapters = []
    for item in adapters_path.iterdir():
        if item.is_dir():
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                adapters.append({
                    "path": str(item),
                    "author": metadata.get("author", "Unknown"),
                    "base_model": metadata.get("base_model", "Unknown"),
                    "rank": metadata.get("lora_rank", 16),
                    "examples": metadata.get("training_examples", 0),
                })

    if not adapters:
        print(f"No adapters found in: {adapters_path}")
        return

    print(f"\nAvailable LoRA adapters in {adapters_path}:\n")
    print(f"{'Author':<25} {'Path':<30} {'Rank':<6} {'Examples'}")
    print("-" * 75)

    for adapter in adapters:
        print(
            f"{adapter['author']:<25} "
            f"{Path(adapter['path']).name:<30} "
            f"{adapter['rank']:<6} "
            f"{adapter['examples']}"
        )

    print()


def transfer_file(
    input_path: str,
    output_path: str,
    adapter_path: str,
    author: str,
    temperature: float = 0.7,
    verify: bool = True,
    verbose: bool = False,
) -> None:
    """Transfer a file using LoRA adapter.

    Args:
        input_path: Path to input file.
        output_path: Path to output file.
        adapter_path: Path to LoRA adapter.
        author: Author name.
        temperature: Generation temperature.
        verify: Whether to verify entailment.
        verbose: Whether to print verbose output.
    """
    from src.generation.fast_transfer import FastStyleTransfer, TransferConfig

    # Load input
    print(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        input_text = f.read()

    word_count = len(input_text.split())
    print(f"Input: {word_count} words")

    # Configure transfer
    config = TransferConfig(
        temperature=temperature,
        verify_entailment=verify,
    )

    # Create transfer pipeline
    print(f"\nInitializing LoRA adapter: {adapter_path}")
    print(f"Author: {author}")

    transfer = FastStyleTransfer(
        adapter_path=adapter_path,
        author_name=author,
        config=config,
    )

    # Progress callback
    def on_progress(current: int, total: int, status: str):
        if verbose:
            print(f"  [{current}/{total}] {status}")
        else:
            # Simple progress bar
            pct = int(current / total * 50)
            bar = "=" * pct + "-" * (50 - pct)
            print(f"\r  [{bar}] {current}/{total}", end="", flush=True)

    # Run transfer
    print("\nTransferring...")
    start_time = time.time()

    output_text, stats = transfer.transfer_document(
        input_text,
        on_progress=on_progress,
    )

    if not verbose:
        print()  # New line after progress bar

    # Save output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(output_text)

    # Print stats
    elapsed = time.time() - start_time
    output_words = len(output_text.split())

    print(f"\nComplete!")
    print(f"  Output: {output_path}")
    print(f"  Words: {word_count} -> {output_words}")
    print(f"  Time: {elapsed:.1f}s ({stats.avg_time_per_paragraph:.1f}s/paragraph)")

    if stats.entailment_scores:
        avg_score = sum(stats.entailment_scores) / len(stats.entailment_scores)
        print(f"  Content preservation: {avg_score:.1%}")

    if stats.paragraphs_repaired > 0:
        print(f"  Paragraphs repaired: {stats.paragraphs_repaired}")


def main():
    parser = argparse.ArgumentParser(
        description="Fast style transfer using LoRA adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Positional arguments
    parser.add_argument(
        "input",
        nargs="?",
        help="Input file path",
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )

    # Adapter settings
    parser.add_argument(
        "--adapter",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--author",
        help="Author name (optional if adapter has metadata)",
    )

    # Generation settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable entailment verification",
    )

    # Utility options
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        help="List available LoRA adapters",
    )
    parser.add_argument(
        "--adapters-dir",
        default="lora_adapters",
        help="Directory containing adapters (default: lora_adapters)",
    )

    # Output options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # List adapters mode
    if args.list_adapters:
        list_adapters(args.adapters_dir)
        return

    # Validate required arguments for transfer
    if not args.input:
        parser.error("Input file is required (or use --list-adapters)")

    if not args.output:
        # Default output name
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix(".styled" + input_path.suffix))

    if not args.adapter:
        parser.error("--adapter is required")

    # Load author from metadata if not provided
    author = args.author
    if not author:
        metadata_path = Path(args.adapter) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            author = metadata.get("author")

    if not author:
        parser.error("--author is required (not found in adapter metadata)")

    # Run transfer
    transfer_file(
        input_path=args.input,
        output_path=args.output,
        adapter_path=args.adapter,
        author=author,
        temperature=args.temperature,
        verify=not args.no_verify,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
