#!/usr/bin/env python3
"""Command-line interface for text style transfer."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def cmd_transfer(args):
    """Transfer text to target author's style using LoRA adapter."""
    from src.generation import FastStyleTransfer, TransferConfig

    # Load input text
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    input_text = input_path.read_text()
    paragraph_count = len([p for p in input_text.split('\n\n') if p.strip()])
    print(f"Loaded input: {paragraph_count} paragraphs")

    # Find adapter path (None if --no-adapter)
    adapter_path = None
    if not getattr(args, 'no_adapter', False):
        adapter_path = Path(args.adapter) if args.adapter else Path("lora_adapters") / args.author.lower()
        if not adapter_path.exists():
            print(f"Error: Adapter not found at {adapter_path}")
            print(f"Train one with: python cli.py train --corpus styles/sample_{args.author.lower()}.txt --author {args.author}")
            print(f"Or use --no-adapter to use base model with prompts only")
            sys.exit(1)
        print(f"Using LoRA adapter: {adapter_path}")
        adapter_path = str(adapter_path)
    else:
        print(f"Using base model (no adapter) for {args.author}'s style")

    # Configure transfer
    config = TransferConfig(
        temperature=args.temperature,
        verify_entailment=not args.no_verify,
        entailment_threshold=args.threshold,
        max_repair_attempts=1 if not args.no_repair else 0,
        reduce_repetition=not args.no_reduce_repetition,
    )

    # Create transfer pipeline
    transfer = FastStyleTransfer(
        adapter_path=adapter_path,
        author_name=args.author,
        config=config,
    )

    # Progress callback
    def on_progress(current, total, status):
        print(f"  [{current}/{total}] {status}")

    print(f"\nTransferring to {args.author}'s style...")

    # Run transfer
    output_text, stats = transfer.transfer_document(input_text, on_progress)

    # Show summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Paragraphs: {stats.paragraphs_processed}")
    print(f"  Time: {stats.total_time_seconds:.1f}s ({stats.avg_time_per_paragraph:.1f}s/para)")
    if stats.entailment_scores:
        avg_score = sum(stats.entailment_scores) / len(stats.entailment_scores)
        print(f"  Avg entailment: {avg_score:.2f}")
    if stats.paragraphs_repaired > 0:
        print(f"  Repaired: {stats.paragraphs_repaired} paragraphs")
    if stats.words_replaced > 0:
        print(f"  Repetition fixes: {stats.words_replaced} words")

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
        print(f"\nOutput written to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("TRANSFORMED TEXT:")
        print("=" * 60)
        print(output_text)

    print("\nDone!")


def cmd_train(args):
    """Train a LoRA adapter for an author's style."""
    import subprocess

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {args.corpus}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path("lora_adapters") / args.author.lower()

    cmd = [
        sys.executable,
        "scripts/train_mlx_lora.py",
        "--corpus", str(corpus_path),
        "--author", args.author,
        "--output", str(output_path),
        "--iters", str(args.iters),
    ]

    print(f"Training LoRA adapter for '{args.author}'...")
    print(f"  Corpus: {corpus_path}")
    print(f"  Output: {output_path}")
    print(f"  Iterations: {args.iters}")
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_analyze(args):
    """Analyze a text file for style metrics."""
    from src.utils.nlp import split_into_sentences, get_nlp
    import numpy as np

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    text = input_path.read_text()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Gather sentences
    all_sentences = []
    for para in paragraphs:
        all_sentences.extend(split_into_sentences(para))

    # Basic stats
    lengths = [len(s.split()) for s in all_sentences]
    total_words = sum(lengths)

    print(f"Analysis of: {args.input}")
    print(f"\nStructure:")
    print(f"  Paragraphs: {len(paragraphs)}")
    print(f"  Sentences: {len(all_sentences)}")
    print(f"  Words: {total_words}")

    if lengths:
        mean = np.mean(lengths)
        std = np.std(lengths)
        burstiness = std / mean if mean > 0 else 0

        print(f"\nSentence Length:")
        print(f"  Mean: {mean:.1f} words")
        print(f"  Std: {std:.1f}")
        print(f"  Range: {min(lengths)} - {max(lengths)} words")
        print(f"  Burstiness: {burstiness:.3f}")


def cmd_list_adapters(args):
    """List available LoRA adapters."""
    adapters_path = Path("lora_adapters")
    if not adapters_path.exists():
        print("No adapters found. Train one with 'python cli.py train'")
        return

    adapters = [d for d in adapters_path.iterdir() if d.is_dir() and (d / "adapters.safetensors").exists()]
    if not adapters:
        print("No trained adapters found.")
        return

    print("Available LoRA adapters:")
    for adapter in sorted(adapters):
        # Check for metadata
        metadata_file = adapter / "adapter_config.json"
        if metadata_file.exists():
            import json
            metadata = json.loads(metadata_file.read_text())
            rank = metadata.get("lora_rank", "?")
            print(f"  {adapter.name}: rank={rank}")
        else:
            print(f"  {adapter.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Text Style Transfer - Transform text to match an author's writing style"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Transfer command
    transfer_parser = subparsers.add_parser(
        "transfer",
        help="Transfer text to target author's style using LoRA"
    )
    transfer_parser.add_argument(
        "input",
        help="Input text file to transform"
    )
    transfer_parser.add_argument(
        "--author", "-a",
        required=True,
        help="Target author name"
    )
    transfer_parser.add_argument(
        "--adapter",
        help="Path to LoRA adapter (default: lora_adapters/<author>)"
    )
    transfer_parser.add_argument(
        "--output", "-o",
        help="Output file (prints to stdout if not specified)"
    )
    transfer_parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    transfer_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Entailment threshold for repair (default: 0.7)"
    )
    transfer_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable entailment verification"
    )
    transfer_parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Disable repair attempts"
    )
    transfer_parser.add_argument(
        "--no-reduce-repetition",
        action="store_true",
        help="Disable repetition reduction post-processing"
    )
    transfer_parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Use base model without LoRA adapter (prompted style transfer)"
    )
    transfer_parser.set_defaults(func=cmd_transfer)

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a LoRA adapter for an author"
    )
    train_parser.add_argument(
        "--corpus", "-c",
        required=True,
        help="Corpus file containing author's text"
    )
    train_parser.add_argument(
        "--author", "-a",
        required=True,
        help="Author name"
    )
    train_parser.add_argument(
        "--output", "-o",
        help="Output path for adapter (default: lora_adapters/<author>)"
    )
    train_parser.add_argument(
        "--iters", "-i",
        type=int,
        default=1000,
        help="Training iterations (default: 1000)"
    )
    train_parser.set_defaults(func=cmd_train)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze text for style metrics"
    )
    analyze_parser.add_argument(
        "input",
        help="Text file to analyze"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # List adapters command
    list_parser = subparsers.add_parser(
        "list",
        help="List available LoRA adapters"
    )
    list_parser.set_defaults(func=cmd_list_adapters)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
