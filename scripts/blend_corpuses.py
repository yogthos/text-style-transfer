#!/usr/bin/env python3
"""Blend multiple author corpuses into a single stylistically coherent corpus.

This script creates a blended training corpus by:
1. Starting with a primary author's curated corpus as the style anchor
2. Finding stylistically compatible paragraphs from secondary authors
3. Building up a corpus that maintains stylistic coherence

Uses style embeddings (LUAR or fallback) to compute compatibility via
cosine similarity to a running centroid.

Usage:
    python scripts/blend_corpuses.py \
        --primary styles/primary_author.txt \
        --secondary styles/author2.txt styles/author3.txt \
        --output styles/blended_corpus.txt \
        --threshold 0.85
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.curate_corpus import (
    is_quality_paragraph,
    sequential_sample,
    estimate_tokens,
    estimate_words_from_tokens,
)


@dataclass
class BlendStats:
    """Statistics from the blending process."""
    primary_author: str = ""
    primary_paragraphs: int = 0
    secondary_authors: List[str] = field(default_factory=list)
    paragraphs_per_author: dict = field(default_factory=dict)
    total_paragraphs: int = 0
    total_tokens: int = 0
    threshold_used: float = 0.85

    def report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 60,
            "CORPUS BLENDING REPORT",
            "=" * 60,
            f"Primary author: {self.primary_author}",
            f"  Paragraphs: {self.primary_paragraphs}",
            "",
            "Secondary authors:",
        ]
        for author in self.secondary_authors:
            count = self.paragraphs_per_author.get(author, 0)
            lines.append(f"  {author}: {count} paragraphs")
        lines.extend([
            "",
            f"Similarity threshold: {self.threshold_used}",
            f"Total paragraphs: {self.total_paragraphs}",
            f"Estimated tokens: {self.total_tokens:,}",
            "=" * 60,
        ])
        return "\n".join(lines)


class StyleEmbedder:
    """Compute style embeddings for text passages."""

    def __init__(self, model_name: str = "rrivera1849/LUAR-MUD"):
        """Initialize the style embedder.

        Args:
            model_name: HuggingFace model for style embeddings.
                       Default is LUAR, designed for authorship attribution.
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None
        self._fallback = False

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            print(f"Loading style embedding model: {self.model_name}")
            # LUAR requires trust_remote_code=True for custom architecture
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Use MPS on Apple Silicon, CUDA if available, else CPU
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            self._model = self._model.to(self._device)
            self._model.eval()
            print(f"Model loaded on {self._device}")

        except Exception as e:
            print(f"Could not load LUAR model: {e}")
            print("Falling back to sentence-transformers...")
            self._load_fallback()

    def _load_fallback(self):
        """Load fallback sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a model that captures writing style reasonably well
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._fallback = True
            print("Using sentence-transformers fallback (less style-specific)")
        except ImportError:
            raise RuntimeError(
                "Neither transformers nor sentence-transformers available. "
                "Install with: pip install transformers sentence-transformers"
            )

    def embed(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Embed a list of texts into style vectors.

        Args:
            texts: List of text passages to embed.
            batch_size: Batch size for processing.

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        self._load_model()

        if self._fallback:
            return self._embed_fallback(texts, batch_size)
        return self._embed_luar(texts, batch_size)

    def _embed_luar(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed using LUAR model.

        LUAR expects 3D input: (batch_size, episode_length, sequence_length)
        For style comparison, we treat each paragraph as a single-document episode.
        """
        import torch

        all_embeddings = []
        max_length = 512

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            actual_batch_size = len(batch)

            # Tokenize each text
            inputs = self._tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # LUAR expects 3D input: (batch_size, episode_length, sequence_length)
            # For single paragraphs, episode_length = 1
            input_ids = inputs["input_ids"].unsqueeze(1)  # (batch, 1, seq_len)
            attention_mask = inputs["attention_mask"].unsqueeze(1)  # (batch, 1, seq_len)

            input_ids = input_ids.to(self._device)
            attention_mask = attention_mask.to(self._device)

            # Get embeddings
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # LUAR returns embeddings directly (batch_size, 512)
                embeddings = outputs.cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def _embed_fallback(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed using sentence-transformers fallback."""
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def cosine_similarities(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between multiple vectors and a centroid.

    Args:
        vectors: Array of shape (n, d) with n vectors of dimension d
        centroid: Array of shape (d,) with the centroid vector

    Returns:
        Array of shape (n,) with similarity scores
    """
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    normalized = vectors / norms

    # Normalize centroid
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

    # Compute dot products
    return np.dot(normalized, centroid_norm)


def split_into_paragraphs(
    text: str,
    min_words: int = 100,
    max_words: int = 300
) -> List[str]:
    """Split text into paragraphs suitable for style analysis.

    Args:
        text: Full text to split.
        min_words: Minimum words per paragraph.
        max_words: Maximum words per paragraph.

    Returns:
        List of paragraph strings.
    """
    # Split on double newlines
    raw_paragraphs = text.split("\n\n")

    paragraphs = []
    current = []
    current_words = 0

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        words = len(para.split())

        # If this paragraph alone is big enough
        if words >= min_words:
            # Flush any accumulated text first
            if current:
                combined = " ".join(current)
                if len(combined.split()) >= min_words:
                    paragraphs.append(combined)
                current = []
                current_words = 0

            # Add this paragraph (split if too long)
            if words <= max_words:
                paragraphs.append(para)
            else:
                # Split long paragraphs at sentence boundaries
                sentences = para.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
                chunk = []
                chunk_words = 0
                for sent in sentences:
                    sent_words = len(sent.split())
                    if chunk_words + sent_words > max_words and chunk:
                        paragraphs.append(" ".join(chunk))
                        chunk = [sent]
                        chunk_words = sent_words
                    else:
                        chunk.append(sent)
                        chunk_words += sent_words
                if chunk and chunk_words >= min_words:
                    paragraphs.append(" ".join(chunk))
        else:
            # Accumulate small paragraphs
            current.append(para)
            current_words += words

            if current_words >= min_words:
                combined = " ".join(current)
                if len(combined.split()) <= max_words:
                    paragraphs.append(combined)
                current = []
                current_words = 0

    # Handle remaining accumulated text
    if current and current_words >= min_words:
        paragraphs.append(" ".join(current))

    return paragraphs


def load_and_prepare_corpus(
    filepath: Path,
    min_words: int = 100,
    max_words: int = 300,
    curate: bool = False,
    target_tokens: Optional[int] = None
) -> List[str]:
    """Load a corpus file and prepare paragraphs.

    Args:
        filepath: Path to the corpus file.
        min_words: Minimum words per paragraph.
        max_words: Maximum words per paragraph.
        curate: Whether to apply quality filtering.
        target_tokens: Target token count (for primary corpus curation).

    Returns:
        List of paragraph strings.
    """
    print(f"Loading corpus: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    paragraphs = split_into_paragraphs(text, min_words, max_words)
    print(f"  Split into {len(paragraphs)} paragraphs")

    if curate:
        # Apply quality filtering (is_quality_paragraph returns (bool, reason) tuple)
        quality_paragraphs = [p for p in paragraphs if is_quality_paragraph(p)[0]]
        print(f"  Quality filter: {len(quality_paragraphs)} passed")

        if target_tokens and quality_paragraphs:
            # Sample to target token count - convert to words for sequential_sample
            target_words = estimate_words_from_tokens(target_tokens)
            indices = sequential_sample(quality_paragraphs, target_words)
            sampled = [quality_paragraphs[i] for i in indices]
            print(f"  Sampled to {len(sampled)} paragraphs (~{target_tokens:,} tokens)")
            return sampled

        return quality_paragraphs

    return paragraphs


def compute_weighted_centroid(
    embeddings: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute weighted centroid of embeddings.

    Args:
        embeddings: Array of shape (n, d)
        weights: Optional array of shape (n,) with weights per embedding.
                 If None, uses uniform weights.

    Returns:
        Centroid vector of shape (d,)
    """
    if weights is None:
        return embeddings.mean(axis=0)

    # Normalize weights
    weights = weights / weights.sum()

    # Weighted average
    return np.average(embeddings, axis=0, weights=weights)


def blend_corpuses(
    primary_path: Path,
    secondary_paths: List[Path],
    threshold: Optional[float] = None,
    top_percent: Optional[float] = None,
    primary_weight: float = 2.0,
    min_paragraphs: int = 10,
    target_primary_tokens: Optional[int] = None,
    min_words: int = 100,
    max_words: int = 300,
) -> Tuple[List[str], BlendStats]:
    """Blend multiple author corpuses into a stylistically coherent corpus.

    Args:
        primary_path: Path to primary author's corpus (style anchor).
        secondary_paths: Paths to secondary author corpuses.
        threshold: Cosine similarity threshold for compatibility (0.0-1.0).
                   If None, uses top_percent instead.
        top_percent: Take top N% most similar paragraphs from each author.
                     Useful when absolute similarities are low.
        primary_weight: Weight multiplier for primary corpus in centroid.
        min_paragraphs: Minimum compatible paragraphs to include an author.
        target_primary_tokens: Target tokens for primary corpus curation.
        min_words: Minimum words per paragraph.
        max_words: Maximum words per paragraph.

    Returns:
        Tuple of (blended paragraphs list, statistics).
    """
    # Default to top 20% if neither threshold nor top_percent specified
    if threshold is None and top_percent is None:
        top_percent = 20.0

    stats = BlendStats(threshold_used=threshold if threshold else f"top {top_percent}%")
    embedder = StyleEmbedder()

    # === Step 1: Load and curate primary corpus ===
    primary_name = primary_path.stem
    stats.primary_author = primary_name

    primary_paragraphs = load_and_prepare_corpus(
        primary_path,
        min_words=min_words,
        max_words=max_words,
        curate=True,
        target_tokens=target_primary_tokens
    )

    if not primary_paragraphs:
        raise ValueError(f"No quality paragraphs found in primary corpus: {primary_path}")

    stats.primary_paragraphs = len(primary_paragraphs)
    print(f"\nPrimary corpus ({primary_name}): {len(primary_paragraphs)} paragraphs")

    # === Step 2: Compute initial centroid ===
    print("\nComputing style embeddings for primary corpus...")
    primary_embeddings = embedder.embed(primary_paragraphs)

    # Initialize centroid with primary corpus
    # Use uniform weights initially
    centroid = compute_weighted_centroid(primary_embeddings)

    # Track all corpus embeddings with weights
    # Primary gets higher weight to anchor the style
    all_paragraphs = list(primary_paragraphs)
    all_embeddings = [primary_embeddings]
    all_weights = [np.ones(len(primary_paragraphs)) * primary_weight]

    # === Step 3: Process secondary authors ===
    for secondary_path in secondary_paths:
        secondary_name = secondary_path.stem
        stats.secondary_authors.append(secondary_name)

        print(f"\nProcessing secondary author: {secondary_name}")

        # Load secondary corpus (no curation - we'll filter by style)
        secondary_paragraphs = load_and_prepare_corpus(
            secondary_path,
            min_words=min_words,
            max_words=max_words,
            curate=False
        )

        if not secondary_paragraphs:
            print(f"  No paragraphs found, skipping")
            stats.paragraphs_per_author[secondary_name] = 0
            continue

        # Compute embeddings
        print(f"  Computing embeddings for {len(secondary_paragraphs)} paragraphs...")
        secondary_embeddings = embedder.embed(secondary_paragraphs)

        # Compute similarities to current centroid
        similarities = cosine_similarities(secondary_embeddings, centroid)

        print(f"  Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")

        # Filter by threshold or top percent
        if threshold is not None:
            # Absolute threshold
            compatible_mask = similarities >= threshold
            compatible_indices = np.where(compatible_mask)[0]
            print(f"  Compatible paragraphs (>= {threshold}): {len(compatible_indices)}")
        else:
            # Top percent - take the most similar paragraphs
            n_to_take = max(min_paragraphs, int(len(similarities) * top_percent / 100))
            # Get indices of top N most similar
            compatible_indices = np.argsort(similarities)[-n_to_take:]
            # Filter to only those with positive similarity
            compatible_indices = compatible_indices[similarities[compatible_indices] > 0]
            percentile_threshold = similarities[compatible_indices].min() if len(compatible_indices) > 0 else 0
            print(f"  Taking top {top_percent}%: {len(compatible_indices)} paragraphs (sim >= {percentile_threshold:.3f})")

        if len(compatible_indices) < min_paragraphs:
            print(f"  Too few compatible paragraphs (< {min_paragraphs}), skipping author")
            stats.paragraphs_per_author[secondary_name] = 0
            continue

        # Add compatible paragraphs
        compatible_paragraphs = [secondary_paragraphs[i] for i in compatible_indices]
        compatible_embeddings = secondary_embeddings[compatible_indices]

        all_paragraphs.extend(compatible_paragraphs)
        all_embeddings.append(compatible_embeddings)
        # Secondary authors get weight 1.0 (vs primary's higher weight)
        all_weights.append(np.ones(len(compatible_paragraphs)))

        stats.paragraphs_per_author[secondary_name] = len(compatible_paragraphs)

        # Update centroid
        combined_embeddings = np.vstack(all_embeddings)
        combined_weights = np.concatenate(all_weights)
        centroid = compute_weighted_centroid(combined_embeddings, combined_weights)

        print(f"  Added {len(compatible_paragraphs)} paragraphs, updated centroid")

    # === Finalize statistics ===
    stats.total_paragraphs = len(all_paragraphs)
    stats.total_tokens = estimate_tokens("\n\n".join(all_paragraphs))

    return all_paragraphs, stats


def main():
    parser = argparse.ArgumentParser(
        description="Blend multiple author corpuses into a stylistically coherent corpus"
    )
    parser.add_argument(
        "--primary",
        type=Path,
        required=True,
        help="Path to primary author corpus (style anchor)"
    )
    parser.add_argument(
        "--secondary",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to secondary author corpuses"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for blended corpus"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold (0.0-1.0). Mutually exclusive with --top-percent"
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=None,
        help="Take top N%% most similar paragraphs from each author (default: 20 if no threshold)"
    )
    parser.add_argument(
        "--primary-weight",
        type=float,
        default=2.0,
        help="Weight for primary corpus in centroid (default: 2.0)"
    )
    parser.add_argument(
        "--primary-tokens",
        type=int,
        default=500000,
        help="Target tokens for primary corpus curation (default: 500000)"
    )
    parser.add_argument(
        "--min-paragraphs",
        type=int,
        default=10,
        help="Minimum compatible paragraphs to include an author (default: 10)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=100,
        help="Minimum words per paragraph (default: 100)"
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=300,
        help="Maximum words per paragraph (default: 300)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.primary.exists():
        print(f"Error: Primary corpus not found: {args.primary}")
        sys.exit(1)

    for path in args.secondary:
        if not path.exists():
            print(f"Error: Secondary corpus not found: {path}")
            sys.exit(1)

    if args.threshold is not None and not 0.0 < args.threshold <= 1.0:
        print(f"Error: Threshold must be in (0.0, 1.0], got {args.threshold}")
        sys.exit(1)

    if args.top_percent is not None and not 0.0 < args.top_percent <= 100.0:
        print(f"Error: top-percent must be in (0.0, 100.0], got {args.top_percent}")
        sys.exit(1)

    # Run blending
    print("=" * 60)
    print("CORPUS BLENDING")
    print("=" * 60)
    print(f"Primary: {args.primary}")
    print(f"Secondary: {[str(p) for p in args.secondary]}")
    if args.threshold:
        print(f"Selection: threshold >= {args.threshold}")
    else:
        print(f"Selection: top {args.top_percent or 20}% most similar")
    print(f"Primary weight: {args.primary_weight}")
    print("=" * 60)

    try:
        paragraphs, stats = blend_corpuses(
            primary_path=args.primary,
            secondary_paths=args.secondary,
            threshold=args.threshold,
            top_percent=args.top_percent,
            primary_weight=args.primary_weight,
            min_paragraphs=args.min_paragraphs,
            target_primary_tokens=args.primary_tokens,
            min_words=args.min_words,
            max_words=args.max_words,
        )
    except Exception as e:
        print(f"\nError during blending: {e}")
        sys.exit(1)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(paragraphs))

    print(f"\nOutput written to: {args.output}")
    print()
    print(stats.report())


if __name__ == "__main__":
    main()
