#!/usr/bin/env python3
"""Curate author corpus to optimal size for style transfer training.

Based on research findings that ~0.9M tokens (approx. 2 books) is sufficient
for expert-level style emulation. More data doesn't improve quality and
increases overfitting risk.

This script:
1. Filters out low-quality text (short paragraphs, OCR artifacts, fragments)
2. Selects diverse, representative passages using embedding clustering
3. Caps output at target token budget (~0.9M tokens by default)

Usage:
    python scripts/curate_corpus.py \
        --input styles/sample_author_full.txt \
        --output styles/sample_author.txt \
        --target-tokens 900000

The curated output can then be passed to neutralize_corpus.py for training.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def estimate_tokens(text: str) -> int:
    """Estimate token count (~1.3 tokens per word for English)."""
    words = len(text.split())
    return int(words * 1.3)


def estimate_words_from_tokens(tokens: int) -> int:
    """Convert token target to approximate word count."""
    return int(tokens / 1.3)


def is_quality_paragraph(para: str, min_words: int = 40) -> Tuple[bool, str]:
    """Check if paragraph meets quality standards.

    Returns:
        Tuple of (is_quality, reason_if_rejected)
    """
    words = para.split()
    word_count = len(words)

    # Too short - likely headers, captions, or fragments
    if word_count < min_words:
        return False, f"too short ({word_count} words)"

    # OCR/encoding artifacts
    if re.search(r'[^\x00-\x7F]{3,}', para):  # Multiple non-ASCII in sequence
        # Allow common punctuation like em-dashes, quotes
        cleaned = re.sub(r'[\u2014\u2013\u2018\u2019\u201c\u201d\u2026]', '', para)
        if re.search(r'[^\x00-\x7F]{3,}', cleaned):
            return False, "encoding artifacts"

    # Excessive special characters (likely corrupted)
    special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'"()-]', para)) / max(len(para), 1)
    if special_ratio > 0.1:
        return False, f"too many special chars ({special_ratio:.1%})"

    # Check for complete sentences (should end with proper punctuation)
    sentences = re.split(r'[.!?]+', para.strip())
    complete_sentences = [s for s in sentences if s.strip() and len(s.split()) >= 3]
    if len(complete_sentences) < 2:
        return False, "too few complete sentences"

    # Check for sentence fragments (very short "sentences")
    fragment_count = sum(1 for s in sentences if s.strip() and len(s.split()) < 3)
    if fragment_count > len(sentences) * 0.5:
        return False, "too many fragments"

    # Repetition check (same word repeated excessively)
    word_freq = {}
    for w in words:
        w_lower = w.lower().strip('.,!?;:\'"')
        if len(w_lower) > 3:  # Skip short words
            word_freq[w_lower] = word_freq.get(w_lower, 0) + 1

    max_freq = max(word_freq.values()) if word_freq else 0
    if max_freq > word_count * 0.15 and max_freq > 5:
        return False, f"excessive repetition"

    return True, "ok"


def compute_paragraph_embeddings(paragraphs: List[str]) -> List[List[float]]:
    """Compute embeddings for paragraphs using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer

        print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        print(f"Computing embeddings for {len(paragraphs)} paragraphs...")
        # Truncate long paragraphs for embedding (first 512 chars captures essence)
        truncated = [p[:512] for p in paragraphs]
        embeddings = model.encode(truncated, show_progress_bar=True)

        return embeddings.tolist()
    except ImportError:
        print("Warning: sentence-transformers not available, using random sampling")
        return None


def cluster_and_sample(
    paragraphs: List[str],
    embeddings: List[List[float]],
    target_words: int,
) -> List[int]:
    """Select diverse paragraphs using k-means clustering.

    Clusters paragraphs by semantic similarity, then samples proportionally
    from each cluster to ensure diverse coverage of the author's style.

    Returns:
        List of indices into the original paragraphs list.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    n_paragraphs = len(paragraphs)
    para_words = [len(p.split()) for p in paragraphs]
    total_words = sum(para_words)

    # Determine number of clusters (roughly 1 per 10K words of target)
    n_clusters = min(max(10, target_words // 10000), n_paragraphs // 5, 100)

    print(f"Clustering into {n_clusters} semantic groups...")

    embeddings_array = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)

    # Group paragraphs by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    # Calculate how many words to sample from each cluster (proportional to cluster size)
    cluster_sizes = {label: len(indices) for label, indices in clusters.items()}
    total_cluster_items = sum(cluster_sizes.values())

    selected_indices = []
    current_words = 0

    # Sample from each cluster proportionally
    for label, indices in sorted(clusters.items(), key=lambda x: -len(x[1])):
        # Target words for this cluster (proportional to size)
        cluster_target = int(target_words * len(indices) / total_cluster_items)

        # Sort indices by paragraph quality (longer paragraphs often better quality)
        indices_by_length = sorted(indices, key=lambda i: para_words[i], reverse=True)

        cluster_words = 0
        for idx in indices_by_length:
            if cluster_words >= cluster_target:
                break
            if current_words + para_words[idx] > target_words * 1.1:  # 10% buffer
                continue
            selected_indices.append(idx)
            cluster_words += para_words[idx]
            current_words += para_words[idx]

        if current_words >= target_words:
            break

    # If we haven't hit target, add more from largest clusters
    if current_words < target_words * 0.9:
        remaining = set(range(n_paragraphs)) - set(selected_indices)
        remaining_sorted = sorted(remaining, key=lambda i: para_words[i], reverse=True)

        for idx in remaining_sorted:
            if current_words >= target_words:
                break
            selected_indices.append(idx)
            current_words += para_words[idx]

    return sorted(selected_indices)  # Return in original order


def sequential_sample(
    paragraphs: List[str],
    target_words: int,
) -> List[int]:
    """Fallback: sample paragraphs sequentially with even distribution.

    Takes paragraphs from throughout the corpus to capture style evolution.
    """
    n_paragraphs = len(paragraphs)
    para_words = [len(p.split()) for p in paragraphs]
    total_words = sum(para_words)

    if total_words <= target_words:
        return list(range(n_paragraphs))

    # Calculate sampling ratio
    ratio = target_words / total_words

    # Sample evenly distributed indices
    step = max(1, int(1 / ratio))
    selected_indices = []
    current_words = 0

    for i in range(0, n_paragraphs, step):
        if current_words >= target_words:
            break
        selected_indices.append(i)
        current_words += para_words[i]

    return selected_indices


def main():
    parser = argparse.ArgumentParser(
        description="Curate corpus to optimal size for style transfer training"
    )
    parser.add_argument("--input", "-i", required=True, help="Input corpus file")
    parser.add_argument("--output", "-o", required=True, help="Output curated corpus file")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=900000,
        help="Target token count (default: 900000, ~2 books)"
    )
    parser.add_argument(
        "--min-para-words",
        type=int,
        default=40,
        help="Minimum words per paragraph (default: 40)"
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Skip clustering, use sequential sampling instead"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed filtering statistics"
    )

    args = parser.parse_args()

    # Load corpus
    print(f"Loading corpus from {args.input}...")
    with open(args.input, 'r') as f:
        corpus = f.read()

    # Split into paragraphs
    raw_paragraphs = [p.strip() for p in corpus.split("\n\n") if p.strip()]
    print(f"Found {len(raw_paragraphs)} paragraphs")

    original_words = sum(len(p.split()) for p in raw_paragraphs)
    original_tokens = estimate_tokens(corpus)
    print(f"Original size: {original_words:,} words (~{original_tokens:,} tokens)")

    target_words = estimate_words_from_tokens(args.target_tokens)
    print(f"Target size: {target_words:,} words (~{args.target_tokens:,} tokens)")

    # Quality filtering
    print("\nFiltering for quality...")
    quality_paragraphs = []
    rejection_reasons = {}

    for para in raw_paragraphs:
        is_quality, reason = is_quality_paragraph(para, args.min_para_words)
        if is_quality:
            quality_paragraphs.append(para)
        else:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    filtered_words = sum(len(p.split()) for p in quality_paragraphs)
    print(f"After quality filter: {len(quality_paragraphs)} paragraphs ({filtered_words:,} words)")

    if args.verbose and rejection_reasons:
        print("Rejection reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Check if we need to reduce further
    if filtered_words <= target_words:
        print(f"Corpus already within target ({filtered_words:,} <= {target_words:,}), using all quality paragraphs")
        selected_paragraphs = quality_paragraphs
    else:
        # Need to select subset
        print(f"\nSelecting representative subset ({filtered_words:,} -> {target_words:,} words)...")

        if args.no_cluster:
            selected_indices = sequential_sample(quality_paragraphs, target_words)
        else:
            # Try embedding-based clustering
            embeddings = compute_paragraph_embeddings(quality_paragraphs)
            if embeddings:
                selected_indices = cluster_and_sample(quality_paragraphs, embeddings, target_words)
            else:
                selected_indices = sequential_sample(quality_paragraphs, target_words)

        selected_paragraphs = [quality_paragraphs[i] for i in selected_indices]

    # Final stats
    final_words = sum(len(p.split()) for p in selected_paragraphs)
    final_tokens = int(final_words * 1.3)

    print(f"\nFinal corpus: {len(selected_paragraphs)} paragraphs")
    print(f"Final size: {final_words:,} words (~{final_tokens:,} tokens)")
    print(f"Reduction: {(1 - final_words/original_words)*100:.1f}%")

    # Save curated corpus
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("\n\n".join(selected_paragraphs))

    print(f"\nSaved curated corpus to {args.output}")
    print(f"\nNext step:")
    print(f"  python scripts/neutralize_corpus.py --input {args.output} --output data/neutralized/author.jsonl --author 'Author Name'")


if __name__ == "__main__":
    main()
