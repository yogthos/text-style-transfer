#!/usr/bin/env python3
"""Train LoRA adapter for author style transfer using MLX.

This script trains a LoRA adapter on Apple Silicon that captures an author's
writing style. The trained adapter can then be used for fast style transfer.

Based on research showing ~0.9M tokens is optimal for style capture.

Usage:
    # Step 1: Curate corpus to optimal size (~0.9M tokens)
    python scripts/curate_corpus.py \
        --input data/corpus/sagan_full.txt \
        --output data/corpus/sagan.txt

    # Step 2: Generate content descriptions
    python scripts/neutralize_corpus.py \
        --input data/corpus/sagan.txt \
        --output data/neutralized/sagan.jsonl \
        --author "Carl Sagan"

    # Step 3: Train LoRA adapter (1 epoch is sufficient)
    python scripts/train_mlx_lora.py \
        --from-neutralized data/neutralized/sagan.jsonl \
        --author "Carl Sagan" \
        --train \
        --output lora_adapters/sagan
"""

import argparse
import json
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_default_model() -> str:
    """Get default model from config.json."""
    config_path = project_root / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            model = config.get("llm", {}).get("providers", {}).get("mlx", {}).get("model")
            if model:
                return model
        except Exception:
            pass
    return "mlx-community/Qwen3-8B-4bit"


DEFAULT_MODEL = get_default_model()


def check_mlx_available():
    """Check if MLX is available."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: ~4 chars per token)."""
    return len(text) // 4


def split_text_to_fit(text: str, max_tokens: int) -> list:
    """Split text into chunks that fit within token limit.

    Splits on sentence boundaries when possible.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Try to split on sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(sent)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def prepare_from_neutralized(
    neutralized_path: str,
    author: str,
    output_path: str,
    max_seq_length: int = 2048,
) -> dict:
    """Prepare training dataset from content descriptions JSONL.

    Uses the instruction back-translation format from the paper:
    "Write a {word_count} word excerpt about the content below emulating
    the style and voice of {author}\\n\\n{description}\\n\\n{original}"

    Args:
        neutralized_path: Path to JSONL file (from neutralize_corpus.py).
        author: Author name.
        output_path: Base path for output files.
        max_seq_length: Maximum sequence length in tokens.

    Returns:
        Dict with paths to train and validation files.
    """
    print(f"Loading training data from {neutralized_path}...")

    # Overhead for prompt structure
    overhead_tokens = 150
    content_budget = max_seq_length - overhead_tokens

    examples = []
    split_count = 0

    with open(neutralized_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Support both old format (neutral) and new format (description)
            description = data.get("description") or data.get("neutral", "")
            original = data["original"]
            word_count = data.get("word_count", len(original.split()))

            # Build the instruction prompt (following the paper's format)
            instruction = f"Write a {word_count} word excerpt about the content below emulating the style and voice of {author}"

            # Check if this example needs splitting
            total_content_tokens = estimate_tokens(description) + estimate_tokens(original)

            if total_content_tokens > content_budget:
                # Split the original into parts, keep description as-is
                part_budget = content_budget - estimate_tokens(description) - 50
                original_parts = split_text_to_fit(original, part_budget)

                for part in original_parts:
                    part_words = len(part.split())
                    part_instruction = f"Write a {part_words} word excerpt about the content below emulating the style and voice of {author}"
                    example = {
                        "text": f"{part_instruction}\n\n{description}\n\n{part}"
                    }
                    examples.append(example)
                split_count += 1
            else:
                # Text format matching the paper
                example = {
                    "text": f"{instruction}\n\n{description}\n\n{original}"
                }
                examples.append(example)

    print(f"Loaded {len(examples)} examples ({split_count} long examples were split)")

    return _save_train_val_split(examples, output_path)


def _save_train_val_split(examples: list, output_path: str) -> dict:
    """Save examples to train/validation split files."""
    import random
    from pathlib import Path

    random.seed(42)
    random.shuffle(examples)

    val_size = max(1, len(examples) // 10)
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    train_path = str(path.with_suffix('.train.jsonl'))
    val_path = str(path.with_suffix('.valid.jsonl'))

    with open(train_path, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')

    with open(val_path, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Dataset saved:")
    print(f"  Training: {train_path} ({len(train_examples)} examples)")
    print(f"  Validation: {val_path} ({len(val_examples)} examples)")

    return {'train': train_path, 'valid': val_path}


def train_lora(
    dataset_path: str,
    author: str,
    output_dir: str,
    base_model: str = None,
    epochs: int = 1,  # 1 epoch sufficient - overtraining causes hallucination
    batch_size: int = 1,  # Per paper: learns individual style nuances better
    learning_rate: float = 1e-5,  # Lower LR with high rank for stability
    lora_rank: int = 64,  # High rank for syntactic patterns (research: 64-128)
    lora_alpha: int = 128,  # Alpha = 2x rank for strong style signal
    validation_path: str = None,
    resume: bool = False,
):
    """Train LoRA adapter for author style.

    Uses the mlx_lm CLI for training, which is more stable than the Python API.

    Args:
        dataset_path: Path to training JSONL file.
        author: Author name (for metadata).
        output_dir: Directory to save adapter.
        base_model: MLX model to fine-tune.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        lora_rank: LoRA rank (higher = more capacity).
        lora_alpha: LoRA alpha (scaling factor).
        validation_path: Path to validation JSONL file.
        resume: Whether to resume from last checkpoint.
    """
    import subprocess
    import shutil

    # Use default model from config if not specified
    if base_model is None:
        base_model = DEFAULT_MODEL

    if not check_mlx_available():
        print("ERROR: MLX is not available. Install with: pip install mlx mlx-lm")
        print("Note: MLX only works on Apple Silicon Macs.")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # mlx_lm expects a data directory with train.jsonl and valid.jsonl
    # Set up the data directory
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Read all training examples
    with open(dataset_path, 'r') as f:
        all_lines = f.readlines()

    # Determine validation file
    valid_lines = []
    train_lines = all_lines

    if validation_path and Path(validation_path).exists():
        # Use provided validation file
        with open(validation_path, 'r') as f:
            valid_lines = f.readlines()
    else:
        # Try to find validation file automatically
        dataset_p = Path(dataset_path)
        auto_valid = dataset_p.with_suffix('.valid.jsonl')
        if not auto_valid.exists():
            # Try alternate naming
            auto_valid = dataset_p.parent / dataset_p.name.replace('.train.jsonl', '.valid.jsonl')

        if auto_valid.exists():
            with open(auto_valid, 'r') as f:
                valid_lines = f.readlines()
            print(f"Found validation set: {auto_valid}")
        else:
            # Split training data to create validation set (10%)
            import random
            random.seed(42)
            shuffled = all_lines.copy()
            random.shuffle(shuffled)

            val_size = max(1, len(shuffled) // 10)
            valid_lines = shuffled[:val_size]
            train_lines = shuffled[val_size:]

            print(f"Created validation set from training data: {val_size} examples")

    # Write train file
    train_target = data_dir / "train.jsonl"
    with open(train_target, 'w') as f:
        f.writelines(train_lines)

    # Write valid file
    valid_target = data_dir / "valid.jsonl"
    with open(valid_target, 'w') as f:
        f.writelines(valid_lines)

    num_train = len(train_lines)
    num_valid = len(valid_lines)
    print(f"Training examples: {num_train}, Validation examples: {num_valid}")

    # Calculate iterations based on actual training set size
    iters_per_epoch = max(1, num_train // batch_size)
    total_iters = epochs * iters_per_epoch

    print(f"\n{'='*60}")
    print(f"MLX LoRA Training")
    print(f"{'='*60}")
    print(f"Author: {author}")
    print(f"Base model: {base_model}")
    print(f"Training examples: {num_train} (validation: {num_valid})")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Total iterations: {total_iters}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Build CLI command
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(output_path),
        "--batch-size", str(batch_size),
        "--iters", str(total_iters),
        "--learning-rate", str(learning_rate),
        "--num-layers", "-1",  # Apply LoRA to ALL transformer layers
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--save-every", "100",
        # Note: --mask-prompt only works with chat format, not text format
        # For base models, we use text format which trains on all tokens
    ]

    # Create lora_config.yaml with rank and alpha settings
    lora_config = {
        "lora_parameters": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": 0.0,
            "scale": float(lora_alpha) / float(lora_rank),
        }
    }
    config_path = output_path / "lora_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(lora_config, f)

    cmd.extend(["-c", str(config_path)])

    # Handle resume from checkpoint
    if resume:
        # Find the latest checkpoint
        checkpoints = sorted(output_path.glob("adapters-*.safetensors"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            # Extract step number from filename (adapters-100.safetensors -> 100)
            checkpoint_step = int(latest_checkpoint.stem.split('-')[1])
            print(f"Resuming from checkpoint: {latest_checkpoint.name} (step {checkpoint_step})")
            cmd.extend(["--resume-adapter-file", str(latest_checkpoint)])
        else:
            print("No checkpoint found, starting from scratch")

    print("Starting training...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run training
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=str(project_root),
        )
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        raise
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    # Save metadata
    metadata = {
        "author": author,
        "base_model": base_model,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "training_examples": num_train,
        "validation_examples": num_valid,
        "training_data": dataset_path,
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Adapter saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*60}")

    return str(output_path)


def test_generation(
    adapter_path: str,
    author: str,
    test_prompt: str = None,
):
    """Test generation with trained adapter.

    Args:
        adapter_path: Path to LoRA adapter.
        author: Author name.
        test_prompt: Optional test prompt.
    """
    if not check_mlx_available():
        print("ERROR: MLX is not available.")
        sys.exit(1)

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    # Default test prompt
    if test_prompt is None:
        test_prompt = (
            "The stars above remind us that we are small, "
            "yet our curiosity reaches infinitely outward."
        )

    # Load model with adapter
    print(f"Loading model with adapter from {adapter_path}...")

    metadata_path = Path(adapter_path) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        base_model = metadata.get("base_model", DEFAULT_MODEL)
    else:
        base_model = DEFAULT_MODEL

    model, tokenizer = load(base_model, adapter_path=adapter_path)

    # Check if base model (no chat template)
    is_base_model = "instruct" not in base_model.lower() and "chat" not in base_model.lower()

    if is_base_model:
        # For base models, use simple prompt format
        prompt = f"Write in the style of {author}:\n\n{test_prompt}\n\n"
    else:
        # For instruct models, use chat template
        messages = [
            {"role": "system", "content": f"You write in the style of {author}. Render the given content in this voice."},
            {"role": "user", "content": test_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Generate
    print(f"\nTest prompt: {test_prompt}\n")
    print("Generating...")

    sampler = make_sampler(temp=0.7, top_p=0.9)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        sampler=sampler,
    )

    print(f"\n{'='*60}")
    print(f"Generated ({author}'s style):")
    print(f"{'='*60}")
    print(response)
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter for author style transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Actions
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train LoRA adapter",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test generation with trained adapter",
    )

    # Required arguments
    parser.add_argument(
        "--author",
        required=True,
        help="Author name",
    )

    # Input/output paths
    parser.add_argument(
        "--dataset",
        help="Path to training JSONL file (for --train)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for adapter directory",
    )

    # Training hyperparameters
    parser.add_argument(
        "--model",
        default=None,
        help=f"Base model for training (default: from config.json or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (default: 1 - sufficient with curated ~0.9M token corpus)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size (default: 1 per paper - learns individual examples better)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5, lower with high rank for stability)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="LoRA rank (default: 64 for syntactic patterns, research: 64-128)",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=128,
        help="LoRA alpha (default: 128 = 2x rank for strong style signal)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )

    # Optional
    parser.add_argument(
        "--from-neutralized",
        help="Path to pre-neutralized JSONL file (from neutralize_corpus.py)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--test-prompt",
        help="Test prompt for generation test",
    )

    args = parser.parse_args()

    # Validate arguments
    if not (args.train or args.test):
        parser.error("Must specify at least one action: --train or --test")

    dataset_path = args.dataset
    validation_path = None

    # Prepare dataset from pre-neutralized file
    if args.from_neutralized:
        paths = prepare_from_neutralized(
            neutralized_path=args.from_neutralized,
            author=args.author,
            output_path=args.output,
            max_seq_length=args.max_seq_length,
        )
        dataset_path = paths['train']
        validation_path = paths['valid']
    elif dataset_path:
        # Try to find validation file
        val_path = Path(dataset_path).with_suffix('.valid.jsonl')
        if val_path.exists():
            validation_path = str(val_path)

    # Train LoRA
    if args.train:
        if not dataset_path:
            parser.error("--dataset or --from-neutralized is required for --train")

        adapter_path = train_lora(
            dataset_path=dataset_path,
            author=args.author,
            output_dir=args.output,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.rank,
            lora_alpha=args.alpha,
            validation_path=validation_path,
            resume=args.resume,
        )
    else:
        adapter_path = args.output

    # Test generation
    if args.test:
        test_generation(
            adapter_path=adapter_path,
            author=args.author,
            test_prompt=args.test_prompt,
        )


if __name__ == "__main__":
    main()
