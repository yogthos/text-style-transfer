#!/usr/bin/env python3
"""Generate instruction descriptions for LoRA style training.

This script follows the methodology from muratcankoylan.com/projects/gertrude-stein-style-training
which achieves human-like output that fools AI detectors.

Key differences from neutralize_corpus.py:
1. Generates DESCRIPTIONS (what to write) instead of neutral paraphrases
2. Uses rotating templates to prevent memorization
3. Creates overlapping chunks to capture "transitions" where style emerges
4. Forces model to generate style from scratch, not just "un-neutralize"

Usage:
    python scripts/describe_corpus.py \
        --input styles/sample_mao.txt \
        --output data/described/mao.jsonl \
        --author "Mao"
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 15 rotating templates to prevent memorization
DESCRIPTION_TEMPLATES = [
    "Describe what this passage is about in 2-3 sentences. Focus on the main ideas and any emotions conveyed. Do not quote the text directly.\n\nPassage: {chunk}\n\nDescription:",
    "Summarize the key points of this text in 2-3 sentences. Mention any characters, events, or arguments. No direct quotes.\n\nText: {chunk}\n\nSummary:",
    "What is the author discussing here? Write 2-3 sentences describing the content and themes. Avoid copying phrases.\n\nContent: {chunk}\n\nDescription:",
    "In your own words, explain what this passage conveys in 2-3 sentences. Note any emotions or rhetorical devices.\n\nPassage: {chunk}\n\nExplanation:",
    "Provide a brief synopsis (2-3 sentences) of this text's meaning and intent. Do not use quotes.\n\nText: {chunk}\n\nSynopsis:",
    "Write a short description (2-3 sentences) of what the author is expressing here. Focus on substance, not style.\n\nContent: {chunk}\n\nDescription:",
    "Capture the essence of this passage in 2-3 sentences. What ideas or feelings does it communicate?\n\nPassage: {chunk}\n\nEssence:",
    "Briefly explain (2-3 sentences) what this text is trying to say. Describe the themes without quoting.\n\nText: {chunk}\n\nExplanation:",
    "What message or idea is conveyed in this passage? Describe in 2-3 sentences without direct quotation.\n\nPassage: {chunk}\n\nMessage:",
    "Summarize the content of this text in 2-3 plain sentences. Focus on what is being said, not how.\n\nText: {chunk}\n\nPlain summary:",
    "Describe the subject matter and emotional tone of this passage in 2-3 sentences. No quotes.\n\nPassage: {chunk}\n\nSubject and tone:",
    "In 2-3 sentences, explain what this author is arguing or describing. Paraphrase completely.\n\nContent: {chunk}\n\nParaphrase:",
    "What are the main points in this text? Describe them in 2-3 sentences without using the author's words.\n\nText: {chunk}\n\nMain points:",
    "Write a neutral description (2-3 sentences) of what this passage discusses. Focus on facts and ideas.\n\nPassage: {chunk}\n\nNeutral description:",
    "Condense this text into 2-3 sentences that capture its meaning. Use your own words entirely.\n\nText: {chunk}\n\nCondensed meaning:",
]


def create_ollama_generator(model: str = "qwen3:8b", timeout: int = 300):
    """Create an Ollama generator function."""
    import requests

    def generate(prompt: str) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.5},  # Higher for varied descriptions
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
    """Create an MLX generator function."""
    from src.llm.mlx_provider import MLXGenerator

    generator = MLXGenerator(model_name=model, temperature=0.5)

    def generate(prompt: str) -> str:
        try:
            return generator.generate(prompt, max_tokens=256)
        except Exception as e:
            print(f"MLX error: {e}")
            return ""

    return generate


def describe_chunk(chunk: str, llm_generate, template_idx: int) -> str:
    """Generate a description of a text chunk using rotating templates."""
    import re

    template = DESCRIPTION_TEMPLATES[template_idx % len(DESCRIPTION_TEMPLATES)]
    prompt = template.format(chunk=chunk[:2500])

    response = llm_generate(prompt)
    if not response:
        return ""

    description = response.strip()

    # Clean up base model artifacts
    # Remove thinking patterns
    thinking_patterns = [
        r'^(Okay|Let me|First|I need|I\'ll|Sure|Alright|The user|This is|Here)[^.]*\.\s*',
    ]
    for pattern in thinking_patterns:
        description = re.sub(pattern, '', description, flags=re.IGNORECASE)

    # Extract content before any "Input:" or continuation markers
    stop_patterns = ["\n\nPassage:", "\nPassage:", "\n\nText:", "\nText:", "\n\nContent:", "\nContent:"]
    for pattern in stop_patterns:
        if pattern in description:
            description = description.split(pattern)[0]

    # Clean up quotes if present
    description = description.strip('"').strip()

    # Ensure we have complete sentences
    if description and description[-1] not in '.!?':
        last_period = description.rfind('.')
        if last_period > len(description) * 0.5:
            description = description[:last_period + 1]

    return description.strip()


def segment_corpus_overlapping(
    text: str,
    min_words: int = 150,
    max_words: int = 400,
    overlap_sentences: int = 2,
) -> list:
    """Segment corpus into overlapping chunks.

    "Style lives in the transitions" - overlapping chunks capture
    boundary conditions where stylistic patterns emerge.
    """
    import re

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # First, create paragraph-level chunks
    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_words + para_words > max_words and current_words >= min_words:
            chunks.append("\n\n".join(current_chunk))
            # Keep last paragraph for overlap
            if overlap_sentences > 0 and current_chunk:
                # Extract last N sentences from the last paragraph
                last_para = current_chunk[-1]
                sentences = re.split(r'(?<=[.!?])\s+', last_para)
                overlap = ' '.join(sentences[-overlap_sentences:]) if len(sentences) >= overlap_sentences else last_para
                current_chunk = [overlap]
                current_words = len(overlap.split())
            else:
                current_chunk = []
                current_words = 0

        current_chunk.append(para)
        current_words += para_words

    if current_chunk and current_words >= min_words:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def process_chunk(args_tuple):
    """Process a single chunk (for parallel execution)."""
    idx, chunk, author, llm_generate, total = args_tuple
    template_idx = idx % len(DESCRIPTION_TEMPLATES)

    description = describe_chunk(chunk, llm_generate, template_idx)

    if not description or len(description.split()) < 10:
        # Retry with different template
        description = describe_chunk(chunk, llm_generate, (template_idx + 7) % len(DESCRIPTION_TEMPLATES))

    if description and len(description.split()) >= 10:
        return {
            "idx": idx,
            "author": author,
            "original": chunk,
            "instruction": description,
            "original_words": len(chunk.split()),
            "instruction_words": len(description.split()),
            "template_idx": template_idx,
        }
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate instruction descriptions for LoRA style training"
    )
    parser.add_argument("--input", "-i", required=True, help="Input corpus file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--author", "-a", required=True, help="Author name")
    parser.add_argument(
        "--llm",
        default="mlx",
        help="LLM provider: 'mlx' (default) or 'ollama:model'"
    )
    parser.add_argument("--min-words", type=int, default=150, help="Min chunk words")
    parser.add_argument("--max-words", type=int, default=400, help="Max chunk words")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap sentences between chunks")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")

    args = parser.parse_args()

    # Load corpus
    print(f"Loading corpus from {args.input}...")
    with open(args.input, 'r') as f:
        corpus = f.read()

    total_words = len(corpus.split())
    print(f"Corpus: {total_words} words")

    # Segment into overlapping chunks
    chunks = segment_corpus_overlapping(
        corpus,
        args.min_words,
        args.max_words,
        args.overlap,
    )
    print(f"Segmented into {len(chunks)} overlapping chunks")

    # Set up LLM
    if args.llm == "mlx" or args.llm.startswith("mlx:"):
        parts = args.llm.split(":", 1)
        model = parts[1] if len(parts) > 1 else None
        print("Using MLX generator (self-contained)")
        llm_generate = create_mlx_generator(model)
    elif args.llm.startswith("ollama"):
        parts = args.llm.split(":", 1)
        model = parts[1] if len(parts) > 1 else "qwen3:8b"
        print(f"Using Ollama: {model}")
        llm_generate = create_ollama_generator(model)
    else:
        print(f"Unknown LLM provider: {args.llm}")
        sys.exit(1)

    # Process chunks with rotating templates
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(chunks)

    if args.workers > 1:
        # Parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Processing {total} chunks with {args.workers} workers...")

        # Prepare arguments for each chunk
        work_items = [(i, chunk, args.author, llm_generate, total) for i, chunk in enumerate(chunks)]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_chunk, item): item[0] for item in work_items}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    results.append(result)
                    print(f"[{completed}/{total}] Completed chunk {result['idx']+1}")
                else:
                    idx = futures[future]
                    print(f"[{completed}/{total}] Skipped chunk {idx+1} - could not generate valid description")

        # Sort by original index to maintain order
        results.sort(key=lambda x: x["idx"])
        # Remove the idx field
        for r in results:
            del r["idx"]
    else:
        # Sequential processing (original behavior)
        for i, chunk in enumerate(chunks):
            template_idx = i % len(DESCRIPTION_TEMPLATES)
            print(f"[{i+1}/{total}] Describing chunk ({len(chunk.split())} words, template {template_idx + 1})...")

            description = describe_chunk(chunk, llm_generate, template_idx)

            if not description or len(description.split()) < 10:
                print(f"  Warning: Description too short, retrying with different template...")
                description = describe_chunk(chunk, llm_generate, (template_idx + 7) % len(DESCRIPTION_TEMPLATES))

            if description and len(description.split()) >= 10:
                results.append({
                    "author": args.author,
                    "original": chunk,
                    "instruction": description,
                    "original_words": len(chunk.split()),
                    "instruction_words": len(description.split()),
                    "template_idx": template_idx,
                })
            else:
                print(f"  Skipping chunk - could not generate valid description")

    # Save results
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\nSaved {len(results)} instruction pairs to {output_path}")
    print(f"\nTrain LoRA adapter with improved hyperparameters:")
    print(f"  python scripts/train_mlx_lora.py \\")
    print(f"      --from-described {output_path} \\")
    print(f"      --author '{args.author}' \\")
    print(f"      --train \\")
    print(f"      --output lora_adapters/{args.author.lower()} \\")
    print(f"      --rank 32 \\")
    print(f"      --learning-rate 5e-4")


if __name__ == "__main__":
    main()
