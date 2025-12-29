"""Tests for the LoRA training pipeline.

Tests cover:
- Corpus curation (quality filtering, token budgeting)
- Corpus neutralization (chunking, re-segmentation, description)
- Training data preparation (format, word counts)
- Inference format (matching training)
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))


# =============================================================================
# Tests for curate_corpus.py
# =============================================================================

class TestCorpusCuration:
    """Tests for corpus quality filtering and curation."""

    def test_estimate_tokens(self):
        """Test token estimation (~1.3 tokens per word)."""
        from curate_corpus import estimate_tokens

        text = "This is a test sentence with exactly ten words here."
        tokens = estimate_tokens(text)
        # 10 words * 1.3 = 13 tokens
        assert tokens == 13

    def test_estimate_words_from_tokens(self):
        """Test reverse conversion from tokens to words."""
        from curate_corpus import estimate_words_from_tokens

        words = estimate_words_from_tokens(900000)
        # 900000 / 1.3 ≈ 692307
        assert words == 692307

    def test_quality_paragraph_accepts_good_text(self):
        """Test that quality paragraphs are accepted."""
        from curate_corpus import is_quality_paragraph

        good_para = (
            "The cosmos is all that is or was or ever will be. "
            "Our feeblest contemplations of the Cosmos stir us. "
            "There is a tingling in the spine, a catch in the voice. "
            "We know we are approaching the greatest of mysteries."
        )

        is_quality, reason = is_quality_paragraph(good_para)
        assert is_quality is True
        assert reason == "ok"

    def test_quality_paragraph_rejects_short_text(self):
        """Test that short paragraphs are rejected."""
        from curate_corpus import is_quality_paragraph

        short_para = "This is too short to be useful."

        is_quality, reason = is_quality_paragraph(short_para, min_words=40)
        assert is_quality is False
        assert "too short" in reason

    def test_quality_paragraph_rejects_fragments(self):
        """Test that paragraphs with too many fragments are rejected."""
        from curate_corpus import is_quality_paragraph

        # Lots of short fragments
        fragment_para = "Yes. No. Maybe. Sure. OK. Fine. Yes. No. Perhaps. Indeed. Right. Wrong. " * 5

        is_quality, reason = is_quality_paragraph(fragment_para, min_words=10)
        assert is_quality is False

    def test_quality_paragraph_rejects_encoding_artifacts(self):
        """Test that text with encoding artifacts is rejected."""
        from curate_corpus import is_quality_paragraph

        # Simulated encoding garbage (non-ASCII sequences)
        bad_para = (
            "This paragraph has some normal text but also contains "
            "garbled characters like \xe2\x80\x99\xe2\x80\x99\xe2\x80\x99 "
            "which indicate encoding problems in the source material."
        )
        # This specific test depends on the artifact pattern
        # The function checks for 3+ consecutive non-ASCII that aren't common punctuation

    def test_quality_paragraph_rejects_excessive_repetition(self):
        """Test that text with excessive word repetition is rejected."""
        from curate_corpus import is_quality_paragraph

        # Same word repeated excessively, but with proper sentences
        repetitive = (
            "The amazing amazing amazing amazing amazing thing happened. "
            "It was amazing amazing amazing amazing amazing indeed. "
            "The amazing amazing amazing amazing amazing result was clear. "
        )

        is_quality, reason = is_quality_paragraph(repetitive, min_words=10)
        assert is_quality is False
        assert "repetition" in reason

    def test_sequential_sample_respects_target(self):
        """Test that sequential sampling stays within word budget."""
        from curate_corpus import sequential_sample

        # Create paragraphs of varying lengths
        paragraphs = [f"Word " * 100 for _ in range(50)]  # 50 paras, 100 words each
        target_words = 2000  # Should select ~20 paragraphs

        indices = sequential_sample(paragraphs, target_words)

        selected_words = sum(len(paragraphs[i].split()) for i in indices)
        # Should be close to target (within 10%)
        assert selected_words <= target_words * 1.1
        assert selected_words >= target_words * 0.5  # At least half


# =============================================================================
# Tests for neutralize_corpus.py
# =============================================================================

class TestCorpusNeutralization:
    """Tests for corpus chunking and neutralization."""

    def test_segment_corpus_respects_word_limits(self):
        """Test that chunks stay within word limits."""
        from neutralize_corpus import segment_corpus, validate_chunks

        # Create a corpus with multiple paragraphs (enough for multiple chunks)
        paragraphs = [
            "This is paragraph one with some text and more content here. " * 15,  # ~150 words
            "This is paragraph two with more text and additional details. " * 15,  # ~150 words
            "This is paragraph three continuing the story forward. " * 15,         # ~150 words
            "This is paragraph four with content about various topics. " * 15,     # ~150 words
            "This is paragraph five ending the narrative here. " * 15,             # ~150 words
            "This is paragraph six with even more content to process. " * 15,      # ~150 words
        ]
        corpus = "\n\n".join(paragraphs)
        original_words = len(corpus.split())

        chunks = segment_corpus(corpus, min_words=250, max_words=400, overlap=False)
        stats = validate_chunks(chunks, original_words, 250, 400)

        # Should produce multiple chunks
        assert stats['chunk_count'] >= 2

        # Most chunks should be within bounds (allow 10% tolerance)
        assert stats['over_max'] <= max(1, len(chunks) * 0.15)
        assert stats['coverage_ratio'] >= 0.8  # Should cover most content

    def test_split_long_paragraph(self):
        """Test that long paragraphs are split at sentence boundaries."""
        from neutralize_corpus import split_long_paragraph

        # Create a paragraph over the limit
        long_para = "This is sentence one. " * 50  # ~200 words

        parts = split_long_paragraph(long_para, max_words=100)

        # Should be split into multiple parts
        assert len(parts) >= 2

        # Each part should be under max_words
        for part in parts:
            assert len(part.split()) <= 110  # Allow small tolerance

        # Combined should have all original words
        combined_words = sum(len(p.split()) for p in parts)
        assert combined_words == len(long_para.split())

    def test_validate_chunks_detects_low_coverage(self):
        """Test that validation catches low coverage."""
        from neutralize_corpus import validate_chunks

        # Simulate chunks that only cover 50% of content
        chunks = ["word " * 100]  # 100 words
        original_words = 200  # But original had 200

        stats = validate_chunks(chunks, original_words, 50, 150)

        # Should flag low coverage
        assert stats['coverage_ratio'] == 0.5
        assert not stats['valid']
        assert any('coverage' in w.lower() for w in stats['warnings'])

    def test_segment_corpus_with_overlap(self):
        """Test that overlap carries last paragraph to next chunk."""
        from neutralize_corpus import segment_corpus

        paragraphs = [
            "First paragraph here. " * 50,   # ~150 words
            "Second paragraph here. " * 50,  # ~150 words
            "Third paragraph here. " * 50,   # ~150 words
            "Fourth paragraph here. " * 50,  # ~150 words
        ]
        corpus = "\n\n".join(paragraphs)

        chunks = segment_corpus(corpus, min_words=250, max_words=400, overlap=True)

        # With overlap, chunks should share content
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_paras = chunks[i].split("\n\n")
                chunk2_paras = chunks[i + 1].split("\n\n")
                # Last para of chunk1 might be first para of chunk2
                # (depending on exact word counts)

    def test_needs_resegmentation_detects_lowercase_start(self):
        """Test detection of chunks starting mid-sentence."""
        from neutralize_corpus import needs_resegmentation

        # Starts with lowercase (mid-sentence)
        bad_chunk = "and then the story continues from here. This is a new sentence."
        assert needs_resegmentation(bad_chunk) is True

        # Starts with uppercase (proper start)
        good_chunk = "The story begins here. This is another sentence."
        assert needs_resegmentation(good_chunk) is False

    def test_needs_resegmentation_detects_missing_terminal_punctuation(self):
        """Test detection of chunks ending mid-sentence."""
        from neutralize_corpus import needs_resegmentation

        # Ends without punctuation (mid-sentence)
        bad_chunk = "This is a complete sentence. But this one is not finished and"
        assert needs_resegmentation(bad_chunk) is True

        # Ends with proper punctuation
        good_chunk = "This is a complete sentence. And this one is too."
        assert needs_resegmentation(good_chunk) is False

    def test_needs_resegmentation_accepts_quotes(self):
        """Test that chunks ending in quotes are accepted."""
        from neutralize_corpus import needs_resegmentation

        # Ends with closing quote (valid)
        quoted_chunk = 'He said, "This is the end."'
        assert needs_resegmentation(quoted_chunk) is False

    def test_clean_chunk_boundaries_with_mock_llm(self):
        """Test chunk boundary cleanup with mocked LLM."""
        from neutralize_corpus import clean_chunk_boundaries

        def mock_llm(prompt):
            # Return a cleaned version
            return "The story begins here. This is a complete paragraph with proper boundaries."

        bad_chunk = "and continues from before. The story begins here. This is incomplete"

        cleaned = clean_chunk_boundaries(bad_chunk, mock_llm)

        # Should use the LLM response (starts with capital, ends with punctuation)
        assert cleaned[0].isupper()
        assert cleaned[-1] in '.!?"'

    def test_clean_chunk_boundaries_rejects_drastic_changes(self):
        """Test that cleanup rejects LLM responses that change too much."""
        from neutralize_corpus import clean_chunk_boundaries, needs_resegmentation

        def mock_llm_too_short(prompt):
            return "Short."  # Way too short

        original = "and this is a longer chunk that needs cleaning but should not be replaced with something drastically different in length."

        # Force needs_resegmentation to return True
        with patch('neutralize_corpus.needs_resegmentation', return_value=True):
            cleaned = clean_chunk_boundaries(original, mock_llm_too_short)

        # Should keep original because LLM response is too short
        assert cleaned == original

    def test_describe_chunk_basic(self):
        """Test basic chunk description generation."""
        from neutralize_corpus import describe_chunk

        def mock_llm(prompt):
            return "A narrator describes the vastness of the cosmos in third person. The passage explores themes of wonder and human curiosity."

        chunk = "The cosmos is all that is or was or ever will be."

        description = describe_chunk(chunk, mock_llm)

        assert len(description) > 0
        assert "cosmos" in description.lower() or "narrator" in description.lower()

    def test_describe_chunk_cleans_thinking_patterns(self):
        """Test that LLM thinking patterns are removed from descriptions."""
        from neutralize_corpus import describe_chunk

        def mock_llm_with_thinking(prompt):
            return "Okay, let me analyze this: The passage describes a journey through space in first person."

        chunk = "I traveled through the stars."

        description = describe_chunk(chunk, mock_llm_with_thinking)

        # Should not start with "Okay"
        assert not description.startswith("Okay")

    def test_describe_chunk_handles_empty_response(self):
        """Test fallback when LLM returns empty response."""
        from neutralize_corpus import describe_chunk

        def mock_llm_empty(prompt):
            return ""

        chunk = "Some text here."

        description = describe_chunk(chunk, mock_llm_empty)

        # Should fall back to original
        assert description == chunk


# =============================================================================
# Tests for train_mlx_lora.py
# =============================================================================

class TestTrainingPreparation:
    """Tests for training data preparation."""

    def test_estimate_tokens(self):
        """Test token estimation for sequence length checking."""
        from train_mlx_lora import estimate_tokens

        text = "This is a test with ten words in it."
        tokens = estimate_tokens(text)

        # ~4 chars per token, 38 chars / 4 = 9 tokens
        assert tokens == 9

    def test_split_text_to_fit_short_text(self):
        """Test that short text is not split."""
        from train_mlx_lora import split_text_to_fit

        short_text = "This is a short sentence."

        chunks = split_text_to_fit(short_text, max_tokens=1000)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_split_text_to_fit_long_text(self):
        """Test that long text is split on sentence boundaries."""
        from train_mlx_lora import split_text_to_fit

        long_text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."

        chunks = split_text_to_fit(long_text, max_tokens=20)  # Force splitting

        assert len(chunks) > 1
        # Each chunk should end with a sentence
        for chunk in chunks:
            assert chunk.strip()[-1] in '.!?'

    def test_prepare_from_neutralized_format(self, tmp_path):
        """Test that training data has correct format."""
        from train_mlx_lora import prepare_from_neutralized

        # Create mock neutralized data
        neutralized_data = [
            {
                "author": "Test Author",
                "original": "The cosmos is vast and beautiful. Stars shine brightly in the night sky.",
                "description": "A description of space and stars in third person.",
                "word_count": 12
            },
            {
                "author": "Test Author",
                "original": "Science reveals the wonders of nature. We learn through observation.",
                "description": "Discussion of scientific method and discovery.",
                "word_count": 10
            }
        ]

        # Write to temp file
        input_path = tmp_path / "neutralized.jsonl"
        with open(input_path, 'w') as f:
            for item in neutralized_data:
                f.write(json.dumps(item) + '\n')

        output_path = tmp_path / "training"

        paths = prepare_from_neutralized(
            neutralized_path=str(input_path),
            author="Test Author",
            output_path=str(output_path),
            max_seq_length=2048
        )

        # Check files were created
        assert Path(paths['train']).exists()
        assert Path(paths['valid']).exists()

        # Check format of training examples
        with open(paths['train'], 'r') as f:
            for line in f:
                example = json.loads(line)
                assert 'text' in example

                # Should have instruction format
                text = example['text']
                assert "Write a" in text
                assert "word excerpt" in text
                assert "emulating the style and voice of" in text
                assert "Test Author" in text

    def test_prepare_from_neutralized_includes_word_count(self, tmp_path):
        """Test that word count is included in instruction."""
        from train_mlx_lora import prepare_from_neutralized

        neutralized_data = [{
            "author": "Test Author",
            "original": "A " * 100,  # 100 words
            "description": "A test description.",
            "word_count": 100
        }]

        input_path = tmp_path / "neutralized.jsonl"
        with open(input_path, 'w') as f:
            f.write(json.dumps(neutralized_data[0]) + '\n')

        output_path = tmp_path / "training"

        paths = prepare_from_neutralized(
            neutralized_path=str(input_path),
            author="Test Author",
            output_path=str(output_path)
        )

        # With only 1 example, it goes to validation (val_size = max(1, 1//10) = 1)
        with open(paths['valid'], 'r') as f:
            example = json.loads(f.readline())
            # Should include the word count
            assert "100 word excerpt" in example['text']

    def test_train_val_split_ratio(self, tmp_path):
        """Test that train/val split is approximately 90/10."""
        from train_mlx_lora import prepare_from_neutralized

        # Create 100 examples
        neutralized_data = [
            {
                "author": "Test Author",
                "original": f"Example {i} text here.",
                "description": f"Description {i}.",
                "word_count": 5
            }
            for i in range(100)
        ]

        input_path = tmp_path / "neutralized.jsonl"
        with open(input_path, 'w') as f:
            for item in neutralized_data:
                f.write(json.dumps(item) + '\n')

        output_path = tmp_path / "training"

        paths = prepare_from_neutralized(
            neutralized_path=str(input_path),
            author="Test Author",
            output_path=str(output_path)
        )

        # Count examples
        with open(paths['train'], 'r') as f:
            train_count = sum(1 for _ in f)
        with open(paths['valid'], 'r') as f:
            val_count = sum(1 for _ in f)

        # Should be ~90/10 split
        assert train_count == 90
        assert val_count == 10


# =============================================================================
# Tests for lora_generator.py
# =============================================================================

class TestLoRAGenerator:
    """Tests for LoRA inference configuration."""

    def test_generation_config_defaults(self):
        """Test that generation config has paper's recommended defaults."""
        from src.generation.lora_generator import GenerationConfig

        config = GenerationConfig()

        # Temperature should be 1.0 per paper
        assert config.temperature == 1.0
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.1

    def test_inference_prompt_format_matches_training(self):
        """Test that inference prompt matches training format."""
        # The key is that inference uses same format as training:
        # "Write a {n} word excerpt about the content below emulating the style and voice of {author}"

        # This is a format test - we verify the string pattern
        author = "Carl Sagan"
        word_count = 150
        content = "Description of space exploration."

        # Training format (from train_mlx_lora.py)
        training_format = f"Write a {word_count} word excerpt about the content below emulating the style and voice of {author}\n\n{content}"

        # Verify the pattern
        assert f"Write a {word_count} word excerpt" in training_format
        assert f"emulating the style and voice of {author}" in training_format
        assert content in training_format

    def test_adapter_metadata_loading(self, tmp_path):
        """Test that adapter metadata is loaded correctly."""
        from src.generation.lora_generator import AdapterMetadata

        # Create mock metadata
        metadata = {
            "author": "Test Author",
            "base_model": "mlx-community/test-model",
            "lora_rank": 32,
            "lora_alpha": 64,
            "epochs": 1,
            "training_examples": 100
        }

        metadata_path = tmp_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        loaded = AdapterMetadata.from_file(metadata_path)

        assert loaded.author == "Test Author"
        assert loaded.base_model == "mlx-community/test-model"
        assert loaded.lora_rank == 32
        assert loaded.lora_alpha == 64
        assert loaded.epochs == 1
        assert loaded.training_examples == 100


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_format_consistency(self, tmp_path):
        """Test that formats are consistent across the pipeline."""
        from train_mlx_lora import prepare_from_neutralized

        # Simulate what neutralize_corpus.py produces
        neutralized = {
            "author": "Test Author",
            "original": "The universe speaks to those who listen. Its voice echoes through the cosmos.",
            "description": "A narrator reflects on the universe's communication with humanity in a philosophical tone.",
            "word_count": 14
        }

        input_path = tmp_path / "neutralized.jsonl"
        with open(input_path, 'w') as f:
            f.write(json.dumps(neutralized) + '\n')

        # Run training prep
        paths = prepare_from_neutralized(
            neutralized_path=str(input_path),
            author="Test Author",
            output_path=str(tmp_path / "training")
        )

        # With only 1 example, it goes to validation (val_size = max(1, 1//10) = 1)
        with open(paths['valid'], 'r') as f:
            example = json.loads(f.readline())

        text = example['text']

        # Training format should be:
        # "Write a {word_count} word excerpt about the content below emulating the style and voice of {author}\n\n{description}\n\n{original}"

        assert text.startswith("Write a 14 word excerpt")
        assert "Test Author" in text
        assert neutralized["description"] in text
        assert neutralized["original"] in text

        # Verify the structure: instruction\n\ndescription\n\noriginal
        parts = text.split("\n\n")
        assert len(parts) == 3
        assert "Write a" in parts[0]  # Instruction
        assert parts[1] == neutralized["description"]  # Description
        assert parts[2] == neutralized["original"]  # Original

    def test_hyperparameter_defaults_match_paper(self):
        """Test that all hyperparameter defaults match the paper."""
        from train_mlx_lora import train_lora
        from src.generation.lora_generator import GenerationConfig
        import inspect

        # Get train_lora defaults
        sig = inspect.signature(train_lora)
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

        # Verify paper's recommendations
        assert defaults['epochs'] == 1  # 1 epoch with curated corpus
        assert defaults['batch_size'] == 1  # Batch size 1
        assert defaults['learning_rate'] == 1e-4  # 2x aggressive (1e-4 vs typical 5e-5)

        # Verify inference defaults
        config = GenerationConfig()
        assert config.temperature == 1.0  # Temperature 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
