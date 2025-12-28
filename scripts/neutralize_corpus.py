#!/usr/bin/env python3
"""Convert author corpus to neutral paraphrases for LoRA training.

This script takes an author's distinctive text and generates neutral/generic
paraphrases. The output is used for training: neutral → author style pairs.

Uses MLX for local generation (self-contained, no external services needed).

Usage:
    python scripts/neutralize_corpus.py \
        --input styles/sample_mao.txt \
        --output data/neutralized/mao.jsonl \
        --author "Mao"

    # Or use Ollama (requires ollama server running):
    python scripts/neutralize_corpus.py \
        --input styles/sample_mao.txt \
        --output data/neutralized/mao.jsonl \
        --author "Mao" \
        --llm ollama:qwen3:8b
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_ollama_generator(model: str = "qwen3:8b", timeout: int = 300):
    """Create an Ollama generator function (fallback)."""
    import requests

    def generate(prompt: str) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""

    return generate


def create_mlx_generator(model: str = None):
    """Create an MLX generator function (default, self-contained)."""
    from src.llm.mlx_provider import MLXGenerator

    generator = MLXGenerator(model_name=model, temperature=0.3)

    def generate(prompt: str) -> str:
        try:
            return generator.generate(prompt, max_tokens=512)
        except Exception as e:
            print(f"MLX error: {e}")
            return ""

    return generate


NEUTRALIZE_PROMPT = """Rewrite in plain English, same length:

INPUT: "The cosmos is vast beyond imagining. We are tiny specks in an ocean of stars."
OUTPUT: "The universe is very large. We are small parts of a huge number of stars."

INPUT: "{chunk}"
OUTPUT: """


def neutralize_chunk(chunk: str, llm_generate) -> str:
    """Generate neutral version of a text chunk."""
    import re

    prompt = NEUTRALIZE_PROMPT.format(chunk=chunk[:2000])

    response = llm_generate(prompt)
    if not response:
        return chunk  # Fallback to original

    paraphrase = response.strip()

    # Strip any prefix like 'Neutral (47 words): "...'
    paraphrase = re.sub(r'^Neutral\s*\(\d+\s*words?\)\s*:\s*', '', paraphrase)

    # For base models: detect thinking/reasoning output and skip to quoted content
    # Common patterns: "Okay, let's...", "Let me...", "First, I need..."
    thinking_patterns = [
        r'^(Okay|Let me|First|I need|I\'ll|Sure|Alright|The user|This is)[^"]*"',
    ]
    for pattern in thinking_patterns:
        match = re.search(pattern, paraphrase, re.IGNORECASE)
        if match:
            # Skip to the quote
            paraphrase = paraphrase[match.end()-1:]  # Keep the opening quote
            break

    # Extract only the first quoted content
    if paraphrase.startswith('"'):
        # Find the closing quote - handle multi-sentence quotes
        # Look for quote followed by end or newline
        match = re.match(r'^"((?:[^"\\]|\\.)+)"', paraphrase)
        if match:
            paraphrase = match.group(1)
        else:
            # No closing quote found, take until first double-newline or thinking pattern
            paraphrase = paraphrase[1:]  # Remove opening quote
            for stop_pattern in ['\n\n', '\nOkay', '\nLet me', '\nI ', '\nNow', '\nFirst']:
                if stop_pattern in paraphrase:
                    paraphrase = paraphrase.split(stop_pattern)[0]
                    break
            # Also stop at closing quote if present
            if '"' in paraphrase:
                paraphrase = paraphrase.split('"')[0]

    # Clean up any trailing incomplete sentences
    paraphrase = paraphrase.strip()
    if paraphrase and paraphrase[-1] not in '.!?"\'':
        # Try to find the last complete sentence
        last_period = paraphrase.rfind('.')
        if last_period > len(paraphrase) * 0.5:  # Only trim if we keep most of it
            paraphrase = paraphrase[:last_period + 1]

    return paraphrase.strip()


def segment_corpus(text: str, min_words: int = 50, max_words: int = 150) -> list:
    """Segment corpus into chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_words + para_words > max_words and current_words >= min_words:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(para)
        current_words += para_words

    if current_chunk and current_words >= min_words:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Convert author corpus to neutral paraphrases"
    )
    parser.add_argument("--input", "-i", required=True, help="Input corpus file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--author", "-a", required=True, help="Author name")
    parser.add_argument(
        "--llm",
        default="mlx",
        help="LLM provider: 'mlx' (default, self-contained) or 'ollama:model'"
    )
    parser.add_argument("--min-words", type=int, default=50, help="Min chunk words")
    parser.add_argument("--max-words", type=int, default=150, help="Max chunk words")

    args = parser.parse_args()

    # Load corpus
    print(f"Loading corpus from {args.input}...")
    with open(args.input, 'r') as f:
        corpus = f.read()

    total_words = len(corpus.split())
    print(f"Corpus: {total_words} words")

    # Segment into chunks
    chunks = segment_corpus(corpus, args.min_words, args.max_words)
    print(f"Segmented into {len(chunks)} chunks")

    # Set up LLM
    if args.llm == "mlx" or args.llm.startswith("mlx:"):
        # Use MLX (self-contained, no external services)
        parts = args.llm.split(":", 1)
        model = parts[1] if len(parts) > 1 else None
        print(f"Using MLX generator (self-contained)")
        llm_generate = create_mlx_generator(model)
    elif args.llm.startswith("ollama"):
        # Use Ollama (requires server)
        parts = args.llm.split(":", 1)
        model = parts[1] if len(parts) > 1 else "qwen3:8b"
        print(f"Using Ollama: {model}")
        llm_generate = create_ollama_generator(model)
    else:
        print(f"Unknown LLM provider: {args.llm}")
        print("Use 'mlx' (default) or 'ollama:model_name'")
        sys.exit(1)

    # Process chunks
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] Neutralizing chunk ({len(chunk.split())} words)...")

        neutral = neutralize_chunk(chunk, llm_generate)
        neutral_words = len(neutral.split())
        original_words = len(chunk.split())

        if neutral_words < original_words * 0.3:
            print(f"  Warning: Neutral too short ({neutral_words} vs {original_words}), using original")
            neutral = chunk

        results.append({
            "author": args.author,
            "original": chunk,
            "neutral": neutral,
            "original_words": original_words,
            "neutral_words": len(neutral.split()),
        })

    # Save results
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\nSaved {len(results)} pairs to {output_path}")
    print(f"Train LoRA adapter with:")
    print(f"  python scripts/train_mlx_lora.py --from-neutralized {output_path} --author '{args.author}' --train --output lora_adapters/{args.author.lower()}")


if __name__ == "__main__":
    main()
