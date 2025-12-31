#!/usr/bin/env python3
"""Style transfer using LoRA-adapted models.

Uses pre-trained LoRA adapters for fast, consistent style transfer with
a critic/repair loop to ensure content preservation and grammatical correctness.

Usage:
    # Basic usage
    python restyle.py input.md -o output.md \\
        --adapter lora_adapters/sagan \\
        --author "Carl Sagan"

    # With verbose output
    python restyle.py input.md -o output.md \\
        --adapter lora_adapters/sagan \\
        --author "Carl Sagan" \\
        --verbose

    # List available adapters
    python restyle.py --list-adapters

To train a LoRA adapter for a new author:
    # 1. Curate corpus (optional, for large corpuses)
    python scripts/curate_corpus.py --input corpus.txt --output curated.txt

    # 2. Generate training data
    python scripts/neutralize_corpus.py --input curated.txt \\
        --output data/neutralized/author.jsonl --author "Author"

    # 3. Train the adapter
    python scripts/train_mlx_lora.py --from-neutralized data/neutralized/author.jsonl \\
        --author "Author" --train --output lora_adapters/author
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
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
    config_path: str = "config.json",
    temperature: float = 0.7,
    perspective: str = None,
    verify: bool = True,
    verbose: bool = False,
) -> None:
    """Transfer a file using LoRA adapter.

    Args:
        input_path: Path to input file.
        output_path: Path to output file.
        adapter_path: Path to LoRA adapter.
        author: Author name.
        config_path: Path to config file.
        temperature: Generation temperature.
        perspective: Output perspective (None uses config default).
        verify: Whether to verify entailment.
        verbose: Whether to print verbose output.
    """
    from src.generation.transfer import StyleTransfer, TransferConfig
    from src.config import load_config
    from src.llm.deepseek import DeepSeekProvider

    # Load config
    try:
        app_config = load_config(config_path)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        app_config = None

    # Load input
    print(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        input_text = f.read()

    word_count = len(input_text.split())
    print(f"Input: {word_count} words")

    # Configure transfer from app config or defaults
    # Determine perspective: CLI overrides config
    effective_perspective = perspective
    if effective_perspective is None and app_config:
        effective_perspective = app_config.style.perspective
    if effective_perspective is None:
        effective_perspective = "preserve"

    if app_config:
        gen = app_config.generation
        config = TransferConfig(
            # Use CLI temperature if specified, otherwise config value
            temperature=temperature,
            verify_entailment=verify,
            # Perspective
            perspective=effective_perspective,
            # From config file
            max_repair_attempts=gen.max_repair_attempts,
            entailment_threshold=gen.entailment_threshold,
            max_expansion_ratio=gen.max_expansion_ratio,
            target_expansion_ratio=gen.target_expansion_ratio,
            truncate_over_expanded=gen.truncate_over_expanded,
            # LoRA influence
            lora_scale=gen.lora_scale,
            # Neutralization
            skip_neutralization=gen.skip_neutralization,
            # Post-processing
            reduce_repetition=gen.reduce_repetition,
            repetition_threshold=gen.repetition_threshold,
            use_document_context=gen.use_document_context,
            pass_headings_unchanged=gen.pass_headings_unchanged,
            min_paragraph_words=gen.min_paragraph_words,
        )
    else:
        config = TransferConfig(
            temperature=temperature,
            verify_entailment=verify,
            perspective=effective_perspective,
        )

    # Create critic provider for repairs
    if app_config and app_config.llm.providers.get("deepseek"):
        deepseek_config = app_config.llm.get_provider_config("deepseek")
        critic_provider = DeepSeekProvider(config=deepseek_config)
        print(f"Using DeepSeek for critic/repair")
    else:
        # Try to get API key from environment
        import os
        from src.config import LLMProviderConfig
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("Warning: No DeepSeek API key found. Repairs will be disabled.")
            print("Set DEEPSEEK_API_KEY or configure in config.json")
            critic_provider = None
        else:
            deepseek_config = LLMProviderConfig(
                api_key=api_key,
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            )
            critic_provider = DeepSeekProvider(config=deepseek_config)
            print(f"Using DeepSeek for critic/repair (from env)")

    # Create transfer pipeline
    print(f"\nInitializing LoRA adapter: {adapter_path}")
    print(f"Author: {author}")

    transfer = StyleTransfer(
        adapter_path=adapter_path,
        author_name=author,
        critic_provider=critic_provider,
        config=config,
    )

    # Set up output file for streaming
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Track paragraphs for streaming output
    output_paragraphs = []

    # Progress callback
    def on_progress(current: int, total: int, status: str):
        if verbose:
            print(f"  [{current}/{total}] {status}")
        else:
            # Simple progress bar
            pct = int(current / total * 50)
            bar = "=" * pct + "-" * (50 - pct)
            print(f"\r  [{bar}] {current}/{total}", end="", flush=True)

    # Paragraph callback - write to file as each paragraph completes
    def on_paragraph(index: int, paragraph: str):
        output_paragraphs.append(paragraph)
        # Write all paragraphs so far to file (overwrite for clean state)
        with open(output_file, 'w') as f:
            f.write("\n\n".join(output_paragraphs))
        if verbose:
            print(f"\n--- Paragraph {index + 1} written to {output_path} ---")

    # Run transfer
    print(f"\nTransferring... (streaming to {output_path})")
    start_time = time.time()

    try:
        output_text, stats = transfer.transfer_document(
            input_text,
            on_progress=on_progress,
            on_paragraph=on_paragraph,
        )

        if not verbose:
            print()  # New line after progress bar

        # Final save (ensures proper formatting)
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

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        # Get partial results
        partial_text, partial_stats = transfer.get_partial_results()
        elapsed = time.time() - start_time

        if output_paragraphs:
            # Save what we have
            with open(output_file, 'w') as f:
                f.write("\n\n".join(output_paragraphs))
            print(f"  Partial output saved: {output_path}")
            print(f"  Paragraphs completed: {len(output_paragraphs)}")
            print(f"  Time: {elapsed:.1f}s")
        else:
            print("  No paragraphs completed yet.")

        sys.exit(130)  # Standard exit code for Ctrl+C


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
        "--perspective",
        choices=["preserve", "first_person_singular", "first_person_plural",
                 "third_person", "author_voice_third_person"],
        default=None,
        help="Output perspective: preserve (default), first_person_singular, "
             "first_person_plural, third_person, or author_voice_third_person "
             "(writes AS author using third person)",
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

    # Config
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to config file (default: config.json)",
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
        config_path=args.config,
        temperature=args.temperature,
        perspective=args.perspective,
        verify=not args.no_verify,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)
