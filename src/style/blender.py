"""Author style blending using SLERP in embedding space.

Implements spherical linear interpolation (SLERP) for blending author styles
in high-dimensional embedding space. The resulting "Ghost Vector" serves as
a synthetic target for the Critic to score candidate generations.

Key insight: In embedding space, style centroids reside on a hypersphere.
Linear interpolation shrinks magnitude; SLERP traces the geodesic.

$$\\vec{V}_{target} = \\frac{\\sin((1-\\lambda)\\theta)}{\\sin\\theta}\\vec{C}_A +
                      \\frac{\\sin(\\lambda\\theta)}{\\sin\\theta}\\vec{C}_B$$

where θ = arccos(C_A · C_B)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Lazy load sentence transformers to avoid import overhead
_sentence_model = None


def get_sentence_model():
    """Lazy load the sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a model that balances quality and speed
            # all-MiniLM-L6-v2 is fast; all-mpnet-base-v2 is more accurate
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed. Style blending disabled.")
            return None
    return _sentence_model


@dataclass
class AuthorCentroid:
    """Centroid representation of an author's style in embedding space."""

    author_name: str
    centroid: np.ndarray  # Unit vector on hypersphere
    sentence_count: int = 0
    variance: float = 0.0  # Spread of sentences around centroid

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "author_name": self.author_name,
            "centroid": self.centroid.tolist(),
            "sentence_count": self.sentence_count,
            "variance": self.variance,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AuthorCentroid":
        """Deserialize from dictionary."""
        return cls(
            author_name=data["author_name"],
            centroid=np.array(data["centroid"]),
            sentence_count=data.get("sentence_count", 0),
            variance=data.get("variance", 0.0),
        )

    def save(self, path: str) -> None:
        """Save centroid to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AuthorCentroid":
        """Load centroid from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class GhostVector:
    """Blended style target for Critic scoring."""

    vector: np.ndarray  # Unit vector representing blended style
    blend_config: Dict[str, float] = field(default_factory=dict)  # author -> weight

    def cosine_similarity(self, embedding: np.ndarray) -> float:
        """Compute cosine similarity to this ghost vector.

        Args:
            embedding: Candidate text embedding (will be normalized).

        Returns:
            Cosine similarity in [-1, 1], higher is better.
        """
        # Normalize the input embedding
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return 0.0
        normalized = embedding / norm

        # Cosine similarity (both vectors are unit vectors)
        return float(np.dot(self.vector, normalized))

    def score_text(self, text: str) -> float:
        """Score a text's similarity to the ghost vector.

        Args:
            text: Text to score.

        Returns:
            Similarity score in [0, 1].
        """
        model = get_sentence_model()
        if model is None:
            return 0.5  # Neutral score if model unavailable

        embedding = model.encode(text, convert_to_numpy=True)
        similarity = self.cosine_similarity(embedding)

        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2


def compute_author_centroid(
    sentences: List[str],
    author_name: str,
    batch_size: int = 64,
) -> Optional[AuthorCentroid]:
    """Compute the style centroid for an author from their sentences.

    Args:
        sentences: List of sentences from author's corpus.
        author_name: Name of the author.
        batch_size: Batch size for embedding computation.

    Returns:
        AuthorCentroid with normalized centroid vector.
    """
    model = get_sentence_model()
    if model is None:
        return None

    if not sentences:
        logger.warning(f"No sentences provided for {author_name}")
        return None

    logger.info(f"Computing centroid for {author_name} from {len(sentences)} sentences...")

    # Compute embeddings in batches
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=len(sentences) > 1000,
        convert_to_numpy=True,
    )

    # Compute mean (centroid)
    centroid = np.mean(embeddings, axis=0)

    # Normalize to unit vector (project onto hypersphere)
    norm = np.linalg.norm(centroid)
    if norm < 1e-8:
        logger.warning(f"Degenerate centroid for {author_name}")
        return None
    centroid = centroid / norm

    # Compute variance (average distance from centroid)
    # This measures how "tight" the author's style is
    distances = []
    for emb in embeddings:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        dist = 1 - np.dot(centroid, emb_norm)  # 1 - cosine similarity
        distances.append(dist)
    variance = float(np.mean(distances))

    logger.info(f"Computed centroid for {author_name}: variance={variance:.4f}")

    return AuthorCentroid(
        author_name=author_name,
        centroid=centroid,
        sentence_count=len(sentences),
        variance=variance,
    )


def slerp(
    v0: np.ndarray,
    v1: np.ndarray,
    t: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors.

    Traces the geodesic (great circle) on the hypersphere from v0 to v1.

    Args:
        v0: First unit vector (t=0).
        v1: Second unit vector (t=1).
        t: Interpolation parameter in [0, 1].
        eps: Small value to handle numerical edge cases.

    Returns:
        Interpolated unit vector.
    """
    # Ensure inputs are unit vectors
    v0 = v0 / (np.linalg.norm(v0) + eps)
    v1 = v1 / (np.linalg.norm(v1) + eps)

    # Compute angle between vectors
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)

    # Handle edge cases
    if theta < eps:
        # Vectors are nearly identical, linear interpolation is fine
        result = (1 - t) * v0 + t * v1
        return result / (np.linalg.norm(result) + eps)

    if theta > np.pi - eps:
        # Vectors are nearly opposite, SLERP is undefined
        # Use a perpendicular vector to define the path
        logger.warning("SLERP between nearly opposite vectors, using fallback")
        perp = np.zeros_like(v0)
        perp[0] = 1.0
        perp = perp - np.dot(perp, v0) * v0
        perp = perp / (np.linalg.norm(perp) + eps)
        # Interpolate through the perpendicular
        if t < 0.5:
            return slerp(v0, perp, t * 2, eps)
        else:
            return slerp(perp, v1, (t - 0.5) * 2, eps)

    # Standard SLERP formula
    sin_theta = np.sin(theta)
    s0 = np.sin((1 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta

    result = s0 * v0 + s1 * v1

    # Ensure result is normalized (should be, but numerical stability)
    return result / (np.linalg.norm(result) + eps)


def create_ghost_vector(
    centroids: List[AuthorCentroid],
    weights: List[float],
) -> GhostVector:
    """Create a ghost vector by blending multiple author centroids.

    For two authors, uses SLERP. For more, uses iterative pairwise SLERP.

    Args:
        centroids: List of author centroids to blend.
        weights: Blending weights (should sum to 1.0).

    Returns:
        GhostVector representing the blended style.
    """
    if len(centroids) != len(weights):
        raise ValueError("Number of centroids must match number of weights")

    if len(centroids) == 0:
        raise ValueError("At least one centroid required")

    # Normalize weights
    total = sum(weights)
    if total < 1e-8:
        raise ValueError("Weights sum to zero")
    weights = [w / total for w in weights]

    # Build blend config
    blend_config = {c.author_name: w for c, w in zip(centroids, weights)}

    if len(centroids) == 1:
        # Single author, no blending needed
        return GhostVector(
            vector=centroids[0].centroid.copy(),
            blend_config=blend_config,
        )

    if len(centroids) == 2:
        # Two authors: simple SLERP
        # weight[0] = 1 means 100% author 0, so t = 1 - weight[0]
        t = weights[1]  # Interpolate from centroid[0] to centroid[1]
        blended = slerp(centroids[0].centroid, centroids[1].centroid, t)

        logger.info(f"Created ghost vector: {blend_config}")
        return GhostVector(vector=blended, blend_config=blend_config)

    # Multiple authors: iterative pairwise SLERP
    # This is an approximation; true multi-point spherical averaging is complex
    logger.info(f"Blending {len(centroids)} authors with iterative SLERP")

    # Sort by weight (descending) for numerical stability
    sorted_pairs = sorted(zip(centroids, weights), key=lambda x: -x[1])

    # Start with highest-weighted author
    result = sorted_pairs[0][0].centroid.copy()
    accumulated_weight = sorted_pairs[0][1]

    for centroid, weight in sorted_pairs[1:]:
        # Compute relative weight for this step
        new_total = accumulated_weight + weight
        t = weight / new_total  # How much to move toward this centroid

        result = slerp(result, centroid.centroid, t)
        accumulated_weight = new_total

    logger.info(f"Created ghost vector: {blend_config}")
    return GhostVector(vector=result, blend_config=blend_config)


def blend_authors(
    author_configs: Dict[str, Tuple[List[str], float]],
) -> GhostVector:
    """Convenience function to blend authors from sentences and weights.

    Args:
        author_configs: Dict mapping author_name -> (sentences, weight).
            Example: {"Hemingway": (hemingway_sentences, 0.7),
                      "Woolf": (woolf_sentences, 0.3)}

    Returns:
        GhostVector for the blended style.
    """
    centroids = []
    weights = []

    for author_name, (sentences, weight) in author_configs.items():
        centroid = compute_author_centroid(sentences, author_name)
        if centroid is not None:
            centroids.append(centroid)
            weights.append(weight)

    if not centroids:
        raise ValueError("Failed to compute any author centroids")

    return create_ghost_vector(centroids, weights)


class StyleBlender:
    """Manages author centroids and ghost vector creation for style blending."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the style blender.

        Args:
            cache_dir: Optional directory to cache computed centroids.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.centroids: Dict[str, AuthorCentroid] = {}
        self.current_ghost: Optional[GhostVector] = None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def add_author(
        self,
        author_name: str,
        sentences: List[str],
        force_recompute: bool = False,
    ) -> Optional[AuthorCentroid]:
        """Add an author's centroid, computing if necessary.

        Args:
            author_name: Name of the author.
            sentences: Sentences from the author's corpus.
            force_recompute: If True, recompute even if cached.

        Returns:
            The computed or cached AuthorCentroid.
        """
        # Check cache first
        if not force_recompute:
            if author_name in self.centroids:
                return self.centroids[author_name]

            if self.cache_dir:
                cache_path = self.cache_dir / f"{author_name.lower().replace(' ', '_')}_centroid.json"
                if cache_path.exists():
                    try:
                        centroid = AuthorCentroid.load(str(cache_path))
                        self.centroids[author_name] = centroid
                        logger.info(f"Loaded cached centroid for {author_name}")
                        return centroid
                    except Exception as e:
                        logger.warning(f"Failed to load cached centroid: {e}")

        # Compute centroid
        centroid = compute_author_centroid(sentences, author_name)
        if centroid is None:
            return None

        self.centroids[author_name] = centroid

        # Save to cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{author_name.lower().replace(' ', '_')}_centroid.json"
            try:
                centroid.save(str(cache_path))
                logger.info(f"Cached centroid for {author_name}")
            except Exception as e:
                logger.warning(f"Failed to cache centroid: {e}")

        return centroid

    def create_blend(self, blend_config: Dict[str, float]) -> Optional[GhostVector]:
        """Create a ghost vector from a blend configuration.

        Args:
            blend_config: Dict mapping author_name -> weight.
                Example: {"Hemingway": 0.7, "Woolf": 0.3}

        Returns:
            GhostVector for the blended style, or None if authors not found.
        """
        centroids = []
        weights = []

        for author_name, weight in blend_config.items():
            if author_name not in self.centroids:
                logger.warning(f"Author {author_name} not found, skipping")
                continue
            centroids.append(self.centroids[author_name])
            weights.append(weight)

        if not centroids:
            logger.error("No valid authors for blending")
            return None

        self.current_ghost = create_ghost_vector(centroids, weights)
        return self.current_ghost

    def score_text(self, text: str) -> float:
        """Score text against the current ghost vector.

        Args:
            text: Text to score.

        Returns:
            Similarity score in [0, 1], or 0.5 if no ghost vector.
        """
        if self.current_ghost is None:
            return 0.5
        return self.current_ghost.score_text(text)
