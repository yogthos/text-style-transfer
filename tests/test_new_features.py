"""Tests for new style transfer features.

Tests cover:
- SLERP-based author style blending
- Perspective transformation
- Heading detection
- Config-driven thresholds
- Anachronistic testing
- SFT dataset generation
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

# Style blending tests
from src.style.blender import (
    slerp,
    compute_author_centroid,
    create_ghost_vector,
    AuthorCentroid,
    GhostVector,
    StyleBlender,
)

# Perspective tests
from src.style.perspective import (
    Perspective,
    detect_perspective,
    transform_perspective,
    PerspectiveTransformer,
)

# Heading detection tests
from src.utils.nlp import (
    is_heading,
    filter_headings,
    split_paragraphs_preserving_headings,
)

# Config tests
from src.config import (
    Config,
    FitnessWeightsConfig,
    ThresholdsConfig,
    BlendingConfig,
)

# SFT dataset generation
from src.sft import DatasetGenerator

# Anachronistic testing
from src.validation.anachronistic_test import (
    get_ngrams,
    calculate_novelty,
    ANACHRONISTIC_PROMPTS,
)


class TestSLERP:
    """Test SLERP (spherical linear interpolation) implementation."""

    def test_slerp_endpoints(self):
        """SLERP should return exact endpoints at t=0 and t=1."""
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])

        # At t=0, should return v0
        result = slerp(v0, v1, 0.0)
        np.testing.assert_array_almost_equal(result, v0)

        # At t=1, should return v1
        result = slerp(v0, v1, 1.0)
        np.testing.assert_array_almost_equal(result, v1)

    def test_slerp_midpoint(self):
        """SLERP midpoint should be on the hypersphere."""
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])

        result = slerp(v0, v1, 0.5)

        # Should have unit magnitude
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

        # Should be equidistant from both endpoints (by angle)
        angle_to_v0 = np.arccos(np.clip(np.dot(result, v0), -1, 1))
        angle_to_v1 = np.arccos(np.clip(np.dot(result, v1), -1, 1))
        assert abs(angle_to_v0 - angle_to_v1) < 1e-6

    def test_slerp_preserves_unit_norm(self):
        """SLERP should always produce unit vectors."""
        v0 = np.array([0.6, 0.8, 0.0])
        v1 = np.array([0.0, 0.6, 0.8])

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = slerp(v0, v1, t)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_slerp_identical_vectors(self):
        """SLERP between identical vectors should return that vector."""
        v = np.array([0.577, 0.577, 0.577])  # Approximately unit
        v = v / np.linalg.norm(v)

        result = slerp(v, v, 0.5)
        np.testing.assert_array_almost_equal(result, v, decimal=5)


class TestGhostVector:
    """Test ghost vector creation and scoring."""

    def test_create_ghost_vector_single_author(self):
        """Single author ghost vector should be the author's centroid."""
        centroid = AuthorCentroid(
            author_name="Test",
            centroid=np.array([1.0, 0.0, 0.0]),
            sentence_count=100,
        )

        ghost = create_ghost_vector([centroid], [1.0])

        np.testing.assert_array_almost_equal(ghost.vector, centroid.centroid)
        assert ghost.blend_config == {"Test": 1.0}

    def test_create_ghost_vector_two_authors(self):
        """Two author ghost vector should blend using SLERP."""
        c1 = AuthorCentroid("A", np.array([1.0, 0.0, 0.0]), 100)
        c2 = AuthorCentroid("B", np.array([0.0, 1.0, 0.0]), 100)

        ghost = create_ghost_vector([c1, c2], [0.5, 0.5])

        # Should be unit vector
        assert abs(np.linalg.norm(ghost.vector) - 1.0) < 1e-6

        # Should be between the two centroids
        assert ghost.vector[0] > 0
        assert ghost.vector[1] > 0

    def test_ghost_vector_cosine_similarity(self):
        """Ghost vector should correctly compute cosine similarity."""
        ghost = GhostVector(
            vector=np.array([1.0, 0.0, 0.0]),
            blend_config={"Test": 1.0},
        )

        # Same direction should be 1.0
        same = np.array([1.0, 0.0, 0.0])
        assert abs(ghost.cosine_similarity(same) - 1.0) < 1e-6

        # Opposite direction should be -1.0
        opposite = np.array([-1.0, 0.0, 0.0])
        assert abs(ghost.cosine_similarity(opposite) - (-1.0)) < 1e-6

        # Orthogonal should be 0.0
        orthogonal = np.array([0.0, 1.0, 0.0])
        assert abs(ghost.cosine_similarity(orthogonal)) < 1e-6


class TestPerspective:
    """Test perspective detection and transformation."""

    def test_detect_first_person_singular(self):
        """Should detect first person singular perspective."""
        text = "I think this is important. My view is clear."
        assert detect_perspective(text) == Perspective.FIRST_PERSON_SINGULAR

    def test_detect_first_person_plural(self):
        """Should detect first person plural perspective."""
        text = "We believe this is important. Our view is clear."
        assert detect_perspective(text) == Perspective.FIRST_PERSON_PLURAL

    def test_detect_third_person(self):
        """Should detect third person perspective."""
        text = "They think this is important. Their view is clear."
        assert detect_perspective(text) == Perspective.THIRD_PERSON

    def test_transform_first_to_third(self):
        """Should transform first person to third person."""
        text = "I think this is right."
        result = transform_perspective(text, Perspective.THIRD_PERSON)
        assert "they" in result.lower() or "he" in result.lower() or "she" in result.lower()
        assert "i" not in result.lower().split()

    def test_transform_preserves_content(self):
        """Transformation should preserve non-pronoun content."""
        text = "I love the beautiful sunset."
        result = transform_perspective(text, Perspective.THIRD_PERSON)
        assert "love" in result.lower()
        assert "beautiful" in result.lower()
        assert "sunset" in result.lower()

    def test_perspective_transformer_class(self):
        """PerspectiveTransformer should work correctly."""
        transformer = PerspectiveTransformer("first_person_plural")

        assert transformer.target == Perspective.FIRST_PERSON_PLURAL
        assert transformer.is_active

        result = transformer.transform("I think this is right.")
        assert "we" in result.lower()


class TestHeadingDetection:
    """Test heading detection and filtering."""

    @pytest.mark.parametrize("text,expected", [
        ("INTRODUCTION", True),
        ("Chapter 1", True),
        ("# Markdown Heading", True),
        ("## Another Heading", True),
        ("1.2 Section Title", True),
        ("PART ONE: THE BEGINNING", True),
        ("This is a normal sentence.", False),
        ("The quick brown fox jumps over the lazy dog.", False),
        ("A longer paragraph with punctuation and multiple clauses.", False),
    ])
    def test_is_heading(self, text, expected):
        """Test individual heading detection."""
        assert is_heading(text) == expected

    def test_filter_headings(self):
        """Test filtering headings from paragraph list."""
        paragraphs = [
            "CHAPTER ONE",
            "This is the first paragraph of actual content.",
            "## Section 1.1",
            "More content here with important information.",
            "CONCLUSION",
        ]

        filtered = filter_headings(paragraphs)

        assert len(filtered) == 2
        assert "actual content" in filtered[0]
        assert "important information" in filtered[1]

    def test_split_paragraphs_preserving_headings(self):
        """Test splitting with heading preservation."""
        text = """INTRODUCTION

This is the introduction paragraph.

## Methods

This describes the methods used."""

        content, headings = split_paragraphs_preserving_headings(text)

        assert len(content) == 2
        assert len(headings) == 2
        assert "INTRODUCTION" in headings[0][1]


class TestConfig:
    """Test configuration dataclasses."""

    def test_default_fitness_weights(self):
        """Default fitness weights should sum to approximately 1.0."""
        config = FitnessWeightsConfig()

        total = (
            config.content +
            config.length +
            config.transition +
            config.vocabulary +
            config.fluency
        )

        assert abs(total - 1.0) < 0.01

    def test_blending_weights_sum(self):
        """Blending weights should also sum to approximately 1.0."""
        config = FitnessWeightsConfig()
        blending = config.style_blending

        total = (
            blending.content_with_blending +
            blending.enabled_weight +
            blending.length_with_blending +
            blending.transition_with_blending +
            blending.vocabulary_with_blending +
            blending.fluency_with_blending
        )

        assert abs(total - 1.0) < 0.01

    def test_thresholds_defaults(self):
        """Thresholds should have reasonable defaults."""
        config = ThresholdsConfig()

        assert config.overuse_word_count > 0
        assert 0 < config.entailment_score < 1
        assert config.delta_score > 0
        assert 0 < config.novelty_min <= 1


class TestSFTDatasetGenerator:
    """Test SFT dataset generation."""

    def test_segment_corpus(self):
        """Test corpus segmentation into chunks."""
        generator = DatasetGenerator(min_chunk_words=50, max_chunk_words=100)

        paragraphs = [
            " ".join(["word"] * 40) for _ in range(5)
        ]

        chunks = generator.segment_corpus(paragraphs)

        # Should create multiple chunks
        assert len(chunks) >= 1

        # Each chunk should be within bounds
        for chunk in chunks:
            words = len(chunk.split())
            assert words >= 50 or words == len(" ".join(paragraphs).split())

    def test_create_training_example(self):
        """Test training example creation."""
        generator = DatasetGenerator()

        example = generator.create_training_example(
            chunk="This is sample text.",
            instruction="Write about a test.",
            author="Test Author",
            template_idx=0,
            system_idx=0,
        )

        assert example.system  # Has system prompt
        assert "Test Author" in example.user  # Author name in user message
        assert example.assistant == "This is sample text."  # Chunk becomes assistant

    def test_training_example_to_dict(self):
        """Test conversion to conversation format."""
        generator = DatasetGenerator()

        example = generator.create_training_example(
            chunk="Sample text.",
            instruction="Test",
            author="Author",
            template_idx=0,
            system_idx=0,
        )

        d = example.to_dict()

        assert "messages" in d
        assert len(d["messages"]) == 3
        assert d["messages"][0]["role"] == "system"
        assert d["messages"][1]["role"] == "user"
        assert d["messages"][2]["role"] == "assistant"


class TestAnachronisticTesting:
    """Test anachronistic validation utilities."""

    def test_get_ngrams(self):
        """Test n-gram extraction."""
        text = "the quick brown fox jumps over the lazy dog"
        ngrams = get_ngrams(text, n=3)

        assert len(ngrams) > 0
        assert ("the", "quick", "brown") in ngrams
        assert ("quick", "brown", "fox") in ngrams

    def test_calculate_novelty_no_overlap(self):
        """Test novelty calculation with no overlap."""
        output = "completely different words here"
        corpus_ngrams = get_ngrams("the quick brown fox jumps", n=5)

        novelty, overlap = calculate_novelty(output, corpus_ngrams, ngram_size=5)

        assert novelty == 1.0
        assert len(overlap) == 0

    def test_calculate_novelty_with_overlap(self):
        """Test novelty calculation with overlap."""
        corpus = "the quick brown fox jumps over the lazy dog"
        output = "the quick brown fox jumps in the forest"  # Some overlap

        corpus_ngrams = get_ngrams(corpus, n=5)
        novelty, overlap = calculate_novelty(output, corpus_ngrams, ngram_size=5)

        assert novelty < 1.0
        assert len(overlap) > 0

    def test_anachronistic_prompts_exist(self):
        """Test that anachronistic prompts are defined."""
        assert len(ANACHRONISTIC_PROMPTS) > 0
        # Should contain modern technology references
        all_prompts = " ".join(ANACHRONISTIC_PROMPTS).lower()
        assert "smartphone" in all_prompts or "phone" in all_prompts


class TestStyleBlender:
    """Test StyleBlender class."""

    def test_blender_initialization(self):
        """Test blender initialization."""
        blender = StyleBlender()

        assert len(blender.centroids) == 0
        assert blender.current_ghost is None

    def test_blender_create_blend_no_authors(self):
        """Test that blending with no authors returns None."""
        blender = StyleBlender()

        result = blender.create_blend({"Unknown": 1.0})

        assert result is None

    def test_blender_score_no_ghost(self):
        """Test scoring without ghost vector returns neutral score."""
        blender = StyleBlender()

        score = blender.score_text("Some text")

        assert score == 0.5  # Neutral


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
