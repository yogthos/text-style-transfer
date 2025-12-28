#!/usr/bin/env python3
"""Train LoRA adapter for author style transfer using MLX.

This script trains a LoRA adapter on Apple Silicon that captures an author's
writing style. The trained adapter can then be used for fast style transfer.

Usage:
    # Generate dataset first (if not already done)
    python scripts/train_mlx_lora.py --prepare \
        --corpus styles/sample_sagan.txt \
        --author "Carl Sagan" \
        --output data/sft/sagan

    # Train LoRA adapter
    python scripts/train_mlx_lora.py --train \
        --dataset data/sft/sagan.train.jsonl \
        --author "Carl Sagan" \
        --output lora_adapters/sagan

    # Or do both in one command
    python scripts/train_mlx_lora.py --prepare --train \
        --corpus styles/sample_sagan.txt \
        --author "Carl Sagan" \
        --output lora_adapters/sagan
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_mlx_available():
    """Check if MLX is available."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def create_ollama_generator(model: str = "qwen2.5:7b"):
    """Create an Ollama generator function (fallback).

    Args:
        model: Ollama model name to use.

    Returns:
        Function that generates text from a prompt.
    """
    import requests

    def generate(prompt: str) -> str:
        """Generate text using Ollama."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=300,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Ollama error: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""

    return generate


def create_mlx_generator(model: str = None):
    """Create an MLX generator function (default, self-contained).

    Args:
        model: Model name (defaults to Qwen2.5-7B-Instruct-4bit).

    Returns:
        Function that generates text from a prompt.
    """
    from src.llm.mlx_provider import MLXGenerator

    generator = MLXGenerator(model_name=model, temperature=0.3)

    def generate(prompt: str) -> str:
        """Generate text using MLX."""
        try:
            return generator.generate(prompt, max_tokens=512)
        except Exception as e:
            print(f"MLX error: {e}")
            return ""

    return generate


def prepare_dataset(
    corpus_path: str,
    author: str,
    output_path: str,
    llm_provider: str = None,
) -> dict:
    """Prepare training dataset from corpus.

    Args:
        corpus_path: Path to corpus text file.
        author: Author name.
        output_path: Base path for output files.
        llm_provider: LLM provider for generating neutral paraphrases.

    Returns:
        Dict with paths to train and validation files.
    """
    from src.sft.mlx_dataset import generate_mlx_dataset

    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'r') as f:
        corpus_text = f.read()

    # Set up LLM for generating neutral paraphrases
    llm_generate = None
    if llm_provider:
        print(f"Using {llm_provider} for generating neutral paraphrases...")
        if llm_provider == "mlx" or llm_provider.startswith("mlx:"):
            # Use MLX (self-contained, no external services)
            parts = llm_provider.split(":", 1)
            model = parts[1] if len(parts) > 1 else None
            print("Using MLX generator (self-contained)")
            llm_generate = create_mlx_generator(model)
        elif llm_provider.startswith("ollama"):
            # Use Ollama (requires server)
            parts = llm_provider.split(":", 1)
            model = parts[1] if len(parts) > 1 else "qwen2.5:7b"
            print(f"Using Ollama: {model}")
            llm_generate = create_ollama_generator(model)
        else:
            # Default to MLX
            print(f"Unknown provider: {llm_provider}, using MLX")
            llm_generate = create_mlx_generator()

    print(f"Generating dataset for {author}...")
    if llm_generate:
        print("Note: Generating neutral paraphrases for each chunk (this will take a while)...")

    paths = generate_mlx_dataset(
        corpus_text=corpus_text,
        author=author,
        output_path=output_path,
        llm_generate=llm_generate,
        min_chunk_words=50,
        max_chunk_words=150,
    )

    print(f"Dataset saved:")
    print(f"  Training: {paths['train']}")
    print(f"  Validation: {paths['valid']}")

    return paths


def prepare_from_neutralized(
    neutralized_path: str,
    author: str,
    output_path: str,
) -> dict:
    """Prepare training dataset from pre-neutralized JSONL.

    Args:
        neutralized_path: Path to neutralized JSONL file (from neutralize_corpus.py).
        author: Author name.
        output_path: Base path for output files.

    Returns:
        Dict with paths to train and validation files.
    """
    import random
    from pathlib import Path

    print(f"Loading neutralized data from {neutralized_path}...")

    examples = []
    with open(neutralized_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Training format: neutral input -> author's original output
            example = {
                "messages": [
                    {"role": "system", "content": f"You write exactly like {author}. Transform the text into {author}'s distinctive voice and style."},
                    {"role": "user", "content": data["neutral"]},
                    {"role": "assistant", "content": data["original"]},
                ]
            }
            examples.append(example)

    print(f"Loaded {len(examples)} examples")

    # Shuffle and split
    random.seed(42)
    random.shuffle(examples)

    val_size = max(1, len(examples) // 10)
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]

    # Save
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
    base_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    validation_path: str = None,
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
    """
    import subprocess
    import shutil

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

    num_examples = len(all_lines)

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
        "--num-layers", str(lora_rank),  # num-layers controls LoRA application
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--save-every", "100",
    ]

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
        base_model = metadata.get("base_model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    else:
        base_model = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    model, tokenizer = load(base_model, adapter_path=adapter_path)

    # Format prompt
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

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        temp=0.7,
        top_p=0.9,
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
        "--prepare",
        action="store_true",
        help="Prepare training dataset from corpus",
    )
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
        "--corpus",
        help="Path to corpus text file (for --prepare)",
    )
    parser.add_argument(
        "--dataset",
        help="Path to training JSONL file (for --train)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (dataset dir for --prepare, adapter dir for --train)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="Base model for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )

    # Optional
    parser.add_argument(
        "--llm-provider",
        default="mlx",
        help="LLM provider for neutral paraphrases: 'mlx' (default, self-contained) or 'ollama:model'",
    )
    parser.add_argument(
        "--from-neutralized",
        help="Path to pre-neutralized JSONL file (from neutralize_corpus.py)",
    )
    parser.add_argument(
        "--test-prompt",
        help="Test prompt for generation test",
    )

    args = parser.parse_args()

    # Validate arguments
    if not (args.prepare or args.train or args.test):
        parser.error("Must specify at least one action: --prepare, --train, or --test")

    dataset_path = args.dataset
    validation_path = None

    # Prepare dataset from pre-neutralized file
    if args.from_neutralized:
        paths = prepare_from_neutralized(
            neutralized_path=args.from_neutralized,
            author=args.author,
            output_path=args.output,
        )
        dataset_path = paths['train']
        validation_path = paths['valid']

    # Prepare dataset from corpus (with live neutralization)
    elif args.prepare:
        if not args.corpus:
            parser.error("--corpus is required for --prepare")

        paths = prepare_dataset(
            corpus_path=args.corpus,
            author=args.author,
            output_path=args.output,
            llm_provider=args.llm_provider,
        )
        dataset_path = paths['train']
        validation_path = paths['valid']

    else:
        if dataset_path:
            # Try to find validation file
            val_path = Path(dataset_path).with_suffix('.valid.jsonl')
            if val_path.exists():
                validation_path = str(val_path)

    # Train LoRA
    if args.train:
        if not dataset_path:
            parser.error("--dataset is required for --train (unless using --prepare)")

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
