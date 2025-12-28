"""Unit tests for statistical analysis."""

import pytest
from src.corpus.analyzer import StatisticalAnalyzer, FeatureVector
from src.corpus.preprocessor import TextPreprocessor, ProcessedParagraph


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return StatisticalAnalyzer()

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return TextPreprocessor()

    def test_analyze_text_basic(self, analyzer):
        """Test basic text analysis."""
        text = "First sentence here. Second sentence is longer. Third one."
        features = analyzer.analyze_text(text)

        assert features.sentence_count == 3
        assert features.avg_sentence_length > 0
        assert features.min_sentence_length > 0
        assert features.max_sentence_length >= features.min_sentence_length

    def test_sentence_length_calculation(self, analyzer):
        """Test sentence length calculation."""
        text = "One two three four. Five six seven eight."
        features = analyzer.analyze_text(text)

        # Each sentence has 4 words
        assert features.avg_sentence_length == 4.0
        assert features.min_sentence_length == 4
        assert features.max_sentence_length == 4

    def test_burstiness_uniform(self, analyzer):
        """Test burstiness with uniform sentences."""
        text = "One two three. Four five six. Seven eight nine."
        features = analyzer.analyze_text(text)

        # Uniform length should have low burstiness
        assert features.burstiness < 0.1

    def test_burstiness_variable(self, analyzer):
        """Test burstiness with variable sentences."""
        text = "Short. This is a much longer sentence with many more words. Medium length here."
        features = analyzer.analyze_text(text)

        # Variable length should have higher burstiness
        assert features.burstiness > 0.3

    def test_perspective_first_person(self, analyzer):
        """Test first person detection."""
        text = "I went to the store. I bought some milk. It was my favorite."
        features = analyzer.analyze_text(text)

        assert features.perspective == "first_person_singular"

    def test_perspective_third_person(self, analyzer):
        """Test third person detection."""
        text = "He walked to the park. She met him there. They talked."
        features = analyzer.analyze_text(text)

        assert features.perspective == "third_person"

    def test_punctuation_frequency(self, analyzer):
        """Test punctuation frequency calculation."""
        text = "First sentence; second clause. Question here? Yes! Another—with em dash."
        features = analyzer.analyze_text(text)

        assert "semicolon" in features.punctuation_freq
        assert "question" in features.punctuation_freq
        assert "exclamation" in features.punctuation_freq
        assert "em_dash" in features.punctuation_freq

    def test_pos_distribution(self, analyzer):
        """Test POS tag distribution."""
        text = "The quick brown fox jumps over the lazy dog."
        features = analyzer.analyze_text(text)

        assert "NOUN" in features.pos_distribution
        assert "VERB" in features.pos_distribution
        assert "ADJ" in features.pos_distribution

        # Should sum to 1.0 (normalized)
        total = sum(features.pos_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_top_words_extraction(self, analyzer):
        """Test top words extraction."""
        text = "The scientist conducted experiments in the laboratory. The results were fascinating."
        features = analyzer.analyze_text(text)

        assert len(features.top_words) > 0
        assert all(isinstance(w, str) for w in features.top_words)

    def test_analyze_sentences_list(self, analyzer):
        """Test analyzing a list of sentences."""
        sentences = ["First sentence here.", "Second sentence.", "Third one."]
        features = analyzer.analyze_sentences(sentences)

        assert features.sentence_count == 3

    def test_analyze_document(self, analyzer, preprocessor):
        """Test analyzing a processed document."""
        text = "First paragraph sentence.\n\nSecond paragraph. With two sentences."
        doc = preprocessor.process(text)

        features = analyzer.analyze_document(doc)

        assert features.sentence_count == 3

    def test_analyze_paragraph(self, analyzer):
        """Test analyzing a single paragraph."""
        para = ProcessedParagraph(
            text="Sentence one. Sentence two. Sentence three.",
            sentences=["Sentence one.", "Sentence two.", "Sentence three."],
            index=0,
            role="BODY"
        )

        features = analyzer.analyze_paragraph(para)

        assert features.sentence_count == 3

    def test_empty_text(self, analyzer):
        """Test handling of empty text."""
        features = analyzer.analyze_text("")

        assert features.sentence_count == 0
        assert features.avg_sentence_length == 0

    def test_feature_vector_to_dict(self, analyzer):
        """Test converting feature vector to dictionary."""
        features = analyzer.analyze_text("Sample text here.")
        d = features.to_dict()

        assert "avg_sentence_length" in d
        assert "burstiness" in d
        assert "perspective" in d
        assert "punctuation_freq" in d

    def test_feature_vector_from_dict(self):
        """Test creating feature vector from dictionary."""
        data = {
            "avg_sentence_length": 15.5,
            "burstiness": 0.3,
            "perspective": "first_person_singular",
            "sentence_count": 10
        }

        features = FeatureVector.from_dict(data)

        assert features.avg_sentence_length == 15.5
        assert features.burstiness == 0.3
        assert features.perspective == "first_person_singular"


class TestSimilarityCalculation:
    """Test similarity calculations."""

    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer()

    def test_identical_features_similarity(self, analyzer):
        """Test similarity of identical feature vectors."""
        # Use text with varied punctuation to ensure non-empty punctuation frequency
        text = "Sample text here; with some content. Is this a question? Yes—it is!"
        features = analyzer.analyze_text(text)

        similarity = analyzer.compute_similarity(features, features)

        # Identical features should have high similarity
        assert similarity >= 0.99

    def test_different_features_similarity(self, analyzer):
        """Test similarity of different feature vectors."""
        text1 = "Short. Very short."
        text2 = "This is a much longer sentence with many more words in it for comparison purposes."

        features1 = analyzer.analyze_text(text1)
        features2 = analyzer.analyze_text(text2)

        similarity = analyzer.compute_similarity(features1, features2)

        # Should be less than 1.0 (not identical)
        assert similarity < 1.0
        # Should be greater than 0 (some similarity)
        assert similarity >= 0

    def test_cosine_similarity_empty(self, analyzer):
        """Test cosine similarity with empty vectors."""
        similarity = analyzer._cosine_similarity({}, {})
        assert similarity == 0.0

    def test_cosine_similarity_identical(self, analyzer):
        """Test cosine similarity with identical vectors."""
        vec = {"a": 0.5, "b": 0.3, "c": 0.2}
        similarity = analyzer._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self, analyzer):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = {"a": 1.0, "b": 0.0}
        vec2 = {"a": 0.0, "b": 1.0}
        similarity = analyzer._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0)


class TestPunctuationAnalysis:
    """Test punctuation frequency analysis."""

    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer()

    def test_em_dash_detection(self, analyzer):
        """Test em dash detection."""
        text = "This is a test—with an em dash—in the middle."
        freq = analyzer._calculate_punctuation_freq(text)

        assert freq["em_dash"] > 0

    def test_semicolon_detection(self, analyzer):
        """Test semicolon detection."""
        text = "First clause; second clause; third clause."
        freq = analyzer._calculate_punctuation_freq(text)

        assert freq["semicolon"] > 0

    def test_frequency_normalization(self, analyzer):
        """Test that frequencies are per 1000 chars."""
        text = "a" * 1000 + ";"  # 1001 chars with 1 semicolon
        freq = analyzer._calculate_punctuation_freq(text)

        # Should be approximately 1.0 per 1000 chars
        assert freq["semicolon"] == pytest.approx(0.999, rel=0.01)
