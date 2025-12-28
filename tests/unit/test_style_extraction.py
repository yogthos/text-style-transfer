"""Tests for enhanced style extraction from corpus."""

import pytest
from src.corpus.style_extractor import (
    TransitionExtractor,
    VoiceAnalyzer,
    OpenerExtractor,
    PhraseExtractor,
    PunctuationAnalyzer,
    StyleFeatures,
)


class TestTransitionExtractor:
    """Tests for transition word extraction."""

    @pytest.fixture
    def extractor(self):
        return TransitionExtractor()

    @pytest.fixture
    def sample_text_with_transitions(self):
        return """
        The theory was groundbreaking. However, it faced criticism.
        Therefore, researchers had to adapt their methods.
        Moreover, new evidence emerged that supported the claims.
        The first experiment failed. Nevertheless, they persisted.
        Consequently, the field evolved rapidly.
        Furthermore, additional studies confirmed the findings.
        Thus, the paradigm shift was complete.
        Although some disagreed, the consensus grew.
        """

    def test_extracts_adversative_connectors(self, extractor, sample_text_with_transitions):
        """Should extract adversative connectors like 'however', 'nevertheless'."""
        result = extractor.extract(sample_text_with_transitions)

        assert "adversative" in result
        adversative_words = [w for w, _ in result["adversative"]]
        assert "however" in adversative_words or "However" in adversative_words
        assert "nevertheless" in adversative_words or "Nevertheless" in adversative_words

    def test_extracts_causal_connectors(self, extractor, sample_text_with_transitions):
        """Should extract causal connectors like 'therefore', 'thus'."""
        result = extractor.extract(sample_text_with_transitions)

        assert "causal" in result
        causal_words = [w.lower() for w, _ in result["causal"]]
        assert "therefore" in causal_words
        assert "thus" in causal_words
        assert "consequently" in causal_words

    def test_extracts_additive_connectors(self, extractor, sample_text_with_transitions):
        """Should extract additive connectors like 'moreover', 'furthermore'."""
        result = extractor.extract(sample_text_with_transitions)

        assert "additive" in result
        additive_words = [w.lower() for w, _ in result["additive"]]
        assert "moreover" in additive_words
        assert "furthermore" in additive_words

    def test_returns_frequencies(self, extractor):
        """Should return frequency counts for each connector."""
        text = "However, this happened. However, that happened too."
        result = extractor.extract(text)

        # Find "however" and check its frequency
        for word, freq in result.get("adversative", []):
            if word.lower() == "however":
                assert freq == 2
                break

    def test_handles_empty_text(self, extractor):
        """Should handle empty text gracefully."""
        result = extractor.extract("")
        assert isinstance(result, dict)
        assert all(isinstance(v, list) for v in result.values())

    def test_extracts_multi_word_connectors(self, extractor):
        """Should extract multi-word connectors like 'on the other hand'."""
        text = "The first option is good. On the other hand, the second has merit."
        result = extractor.extract(text)

        adversative_words = [w.lower() for w, _ in result.get("adversative", [])]
        assert "on the other hand" in adversative_words

    def test_categorizes_connectors_correctly(self, extractor):
        """Each connector should be in only one category."""
        text = """
        Therefore, we proceed. However, caution is needed.
        Moreover, we must consider alternatives.
        """
        result = extractor.extract(text)

        all_words = []
        for category, words in result.items():
            for word, _ in words:
                assert word.lower() not in [w.lower() for w in all_words], \
                    f"'{word}' appears in multiple categories"
                all_words.append(word)


class TestVoiceAnalyzer:
    """Tests for active/passive voice analysis."""

    @pytest.fixture
    def analyzer(self):
        return VoiceAnalyzer()

    def test_detects_active_voice(self, analyzer):
        """Should detect active voice sentences."""
        text = "The scientist conducted the experiment."
        result = analyzer.analyze(text)

        assert result["active_ratio"] > 0.8

    def test_detects_passive_voice(self, analyzer):
        """Should detect passive voice sentences."""
        text = "The experiment was conducted by the scientist."
        result = analyzer.analyze(text)

        assert result["passive_ratio"] > 0.8

    def test_calculates_ratio_for_mixed_text(self, analyzer):
        """Should calculate accurate ratio for mixed voice text."""
        text = """
        The researcher analyzed the data. The results were published.
        Critics questioned the methodology. The study was defended vigorously.
        """
        result = analyzer.analyze(text)

        # 2 active, 2 passive = 0.5 ratio
        assert 0.4 <= result["active_ratio"] <= 0.6

    def test_handles_empty_text(self, analyzer):
        """Should handle empty text gracefully."""
        result = analyzer.analyze("")
        assert "active_ratio" in result
        assert "passive_ratio" in result


class TestOpenerExtractor:
    """Tests for sentence opener pattern extraction."""

    @pytest.fixture
    def extractor(self):
        return OpenerExtractor()

    def test_extracts_sentence_openers(self, extractor):
        """Should extract first words/phrases of sentences."""
        text = """
        The theory emerged in the 1950s. Many scientists contributed.
        The implications were profound. Some researchers disagreed.
        """
        result = extractor.extract(text)

        assert "openers" in result
        opener_words = [o.lower() for o, _ in result["openers"]]
        assert "the" in opener_words
        assert "many" in opener_words or "some" in opener_words

    def test_extracts_openers_by_position(self, extractor):
        """Should categorize openers by sentence position."""
        paragraphs = [
            "First point here. Supporting detail follows. Conclusion of paragraph.",
            "Second main point. More details provided. Final thought here."
        ]
        text = "\n\n".join(paragraphs)
        result = extractor.extract(text)

        assert "first_sentence" in result
        assert "middle_sentence" in result
        assert "last_sentence" in result

    def test_returns_frequencies(self, extractor):
        """Should count frequency of each opener."""
        text = """
        The first point. The second point. The third point.
        A different opener. The fourth point.
        """
        result = extractor.extract(text)

        # "The" should have frequency 4
        for opener, freq in result["openers"]:
            if opener.lower() == "the":
                assert freq == 4
                break


class TestPhraseExtractor:
    """Tests for signature phrase extraction."""

    @pytest.fixture
    def extractor(self):
        return PhraseExtractor()

    def test_extracts_ngrams(self, extractor):
        """Should extract 2-grams and 3-grams."""
        text = """
        The material conditions determine consciousness.
        Material conditions shape society.
        The material conditions of production are fundamental.
        """
        result = extractor.extract(text)

        assert "bigrams" in result
        assert "trigrams" in result

        # "material conditions" should be extracted
        bigrams = [phrase.lower() for phrase, _ in result["bigrams"]]
        assert "material conditions" in bigrams

    def test_filters_stopword_only_phrases(self, extractor):
        """Should not return phrases that are only stopwords."""
        text = "The and the or the but the."
        result = extractor.extract(text)

        # Phrases like "the and" or "and the" should be filtered
        for phrase, _ in result.get("bigrams", []):
            words = phrase.lower().split()
            assert not all(w in ["the", "and", "or", "but", "a", "an"] for w in words)

    def test_returns_signature_phrases(self, extractor):
        """Should identify signature phrases (high frequency, distinctive)."""
        text = """
        The dialectical process unfolds through contradiction.
        The dialectical process reveals hidden tensions.
        The dialectical process transforms reality.
        Normal phrase here. Another normal phrase.
        """
        result = extractor.extract(text)

        assert "signature_phrases" in result
        signatures = [p.lower() for p, _ in result["signature_phrases"]]
        assert "dialectical process" in signatures


class TestPunctuationAnalyzer:
    """Tests for punctuation pattern analysis."""

    @pytest.fixture
    def analyzer(self):
        return PunctuationAnalyzer()

    def test_counts_semicolons(self, analyzer):
        """Should count semicolon usage."""
        text = "First clause; second clause. Another sentence; with semicolon."
        result = analyzer.analyze(text)

        assert "semicolon" in result
        assert result["semicolon"]["count"] == 2

    def test_counts_em_dashes(self, analyzer):
        """Should count em-dash usage."""
        text = "The theory—revolutionary as it was—changed everything."
        result = analyzer.analyze(text)

        assert "em_dash" in result
        assert result["em_dash"]["count"] >= 1

    def test_counts_parentheticals(self, analyzer):
        """Should count parenthetical usage."""
        text = "The concept (first introduced in 1950) spread quickly (especially in Europe)."
        result = analyzer.analyze(text)

        assert "parenthetical" in result
        assert result["parenthetical"]["count"] == 2

    def test_calculates_per_sentence_frequency(self, analyzer):
        """Should calculate frequency per sentence."""
        text = "First; clause. Second sentence. Third; here; now."
        result = analyzer.analyze(text)

        # 3 semicolons in 3 sentences = 1.0 per sentence
        assert "semicolon" in result
        assert result["semicolon"]["per_sentence"] == 1.0

    def test_handles_text_without_special_punctuation(self, analyzer):
        """Should handle text with no special punctuation."""
        text = "Simple sentence. Another simple one. Nothing fancy here."
        result = analyzer.analyze(text)

        assert result["semicolon"]["count"] == 0
        assert result["em_dash"]["count"] == 0


class TestStyleFeatures:
    """Tests for combined style features extraction."""

    @pytest.fixture
    def sample_corpus(self):
        return """
        The material conditions of society determine its consciousness. However,
        this relationship is dialectical; consciousness can also influence conditions.

        Therefore, we must examine both aspects. The productive forces—machinery,
        technology, and human labor—shape social relations. Moreover, these relations
        evolve through contradiction and struggle.

        The dialectical process reveals hidden tensions. Nevertheless, resolution
        emerges through synthesis. Thus, history progresses through stages.
        """

    def test_extracts_all_features(self, sample_corpus):
        """Should extract all style features from corpus."""
        features = StyleFeatures.extract_from_text(sample_corpus)

        assert features.transitions is not None
        assert features.voice_ratio is not None
        assert features.openers is not None
        assert features.phrases is not None
        assert features.punctuation is not None

    def test_features_are_serializable(self, sample_corpus):
        """Style features should be JSON-serializable."""
        import json

        features = StyleFeatures.extract_from_text(sample_corpus)
        json_str = json.dumps(features.to_dict())

        assert json_str is not None
        reloaded = json.loads(json_str)
        assert "transitions" in reloaded
        assert "voice_ratio" in reloaded

    def test_features_can_be_merged(self, sample_corpus):
        """Should be able to merge features from multiple documents."""
        features1 = StyleFeatures.extract_from_text(sample_corpus)
        features2 = StyleFeatures.extract_from_text(sample_corpus)

        merged = StyleFeatures.merge([features1, features2])

        # Merged should have combined frequencies
        assert merged is not None
