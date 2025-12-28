"""Unit tests for NLP utilities."""

import pytest
from src.utils.nlp import (
    split_into_sentences,
    split_into_paragraphs,
    extract_citations,
    remove_citations,
    extract_entities,
    extract_keywords,
    count_words,
    calculate_burstiness,
    get_pos_distribution,
    get_dependency_depth,
    detect_perspective,
    compute_semantic_similarity,
    classify_by_similarity,
)


class TestSentenceSplitting:
    """Test sentence splitting functionality."""

    def test_basic_sentences(self):
        """Test splitting basic sentences."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two."

    def test_empty_input(self):
        """Test empty input returns empty list."""
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []
        assert split_into_sentences(None) == []

    def test_single_sentence(self):
        """Test single sentence without period."""
        text = "Just one sentence"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1

    def test_preserves_citations(self):
        """Test that citations are preserved in sentences."""
        text = "This has a citation[^1]. Another sentence here."
        sentences = split_into_sentences(text)
        assert "[^1]" in sentences[0]

    def test_question_exclamation(self):
        """Test question marks and exclamation points."""
        text = "Is this a question? Yes it is! And a statement."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3


class TestParagraphSplitting:
    """Test paragraph splitting functionality."""

    def test_double_newline_split(self):
        """Test splitting on double newlines."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 3
        assert paragraphs[0] == "First paragraph."

    def test_single_newline_fallback(self):
        """Test fallback to single newlines when no double newlines."""
        text = "Line one.\nLine two.\nLine three."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 3

    def test_empty_input(self):
        """Test empty input returns empty list."""
        assert split_into_paragraphs("") == []
        assert split_into_paragraphs("   ") == []

    def test_single_paragraph(self):
        """Test single paragraph with no newlines."""
        text = "This is a single paragraph with multiple sentences. Here is another."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 1

    def test_mixed_newlines(self):
        """Test text with both double and extra newlines."""
        text = "First para.\n\n\nSecond para.\n\nThird para."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 3


class TestCitationHandling:
    """Test citation extraction and removal."""

    def test_extract_citations(self):
        """Test extracting footnote-style citations."""
        text = "Some text[^1] and more text[^2] here."
        citations = extract_citations(text)
        assert len(citations) == 2
        assert citations[0][0] == "[^1]"
        assert citations[1][0] == "[^2]"

    def test_extract_citations_positions(self):
        """Test that citation positions are correct."""
        text = "Text[^1] more[^2]"
        citations = extract_citations(text)
        assert citations[0][1] == 4  # Position of [^1]
        assert citations[1][1] == 13  # Position of [^2]

    def test_no_citations(self):
        """Test text with no citations."""
        text = "Plain text without any citations."
        citations = extract_citations(text)
        assert len(citations) == 0

    def test_remove_citations(self):
        """Test removing citations from text."""
        text = "Some text[^1] and more text[^2] here."
        clean = remove_citations(text)
        assert clean == "Some text and more text here."

    def test_remove_citations_none(self):
        """Test removing citations when none exist."""
        text = "Plain text."
        clean = remove_citations(text)
        assert clean == text


class TestEntityExtraction:
    """Test named entity extraction."""

    def test_extract_person_entities(self):
        """Test extracting person entities."""
        text = "Albert Einstein developed the theory of relativity."
        entities = extract_entities(text)
        entity_texts = [e[0] for e in entities]
        assert "Albert Einstein" in entity_texts

    def test_extract_location_entities(self):
        """Test extracting location entities."""
        text = "The conference was held in New York City."
        entities = extract_entities(text)
        entity_texts = [e[0] for e in entities]
        # spaCy should recognize New York City as a location
        assert any("New York" in t for t in entity_texts)

    def test_empty_text(self):
        """Test empty text returns empty list."""
        entities = extract_entities("")
        assert entities == []


class TestKeywordExtraction:
    """Test keyword extraction."""

    def test_extract_keywords(self):
        """Test extracting keywords from text."""
        text = "The scientist conducted experiments in the laboratory. The results were fascinating."
        keywords = extract_keywords(text, top_n=5)
        assert len(keywords) <= 5
        assert all(isinstance(k, str) for k in keywords)

    def test_keyword_lemmatization(self):
        """Test that keywords are lemmatized."""
        text = "The dogs were running through the forests."
        keywords = extract_keywords(text)
        # Should get lemmatized forms
        assert "dog" in keywords or "run" in keywords or "forest" in keywords

    def test_top_n_limit(self):
        """Test that top_n limits results."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        keywords = extract_keywords(text, top_n=3)
        assert len(keywords) <= 3


class TestWordCount:
    """Test word counting."""

    def test_basic_count(self):
        """Test basic word counting."""
        assert count_words("one two three") == 3
        assert count_words("word") == 1

    def test_empty_text(self):
        """Test empty text returns 0."""
        assert count_words("") == 0
        assert count_words(None) == 0

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        assert count_words("one  two   three") == 3


class TestBurstiness:
    """Test burstiness calculation."""

    def test_uniform_sentences(self):
        """Test burstiness with uniform sentence lengths."""
        sentences = [
            "One two three four.",
            "Five six seven eight.",
            "Nine ten eleven twelve."
        ]
        burstiness = calculate_burstiness(sentences)
        # Uniform lengths should have low burstiness
        assert burstiness < 0.1

    def test_variable_sentences(self):
        """Test burstiness with variable sentence lengths."""
        sentences = [
            "Short.",
            "This is a much longer sentence with many more words in it.",
            "Medium length here."
        ]
        burstiness = calculate_burstiness(sentences)
        # Variable lengths should have higher burstiness
        assert burstiness > 0.3

    def test_single_sentence(self):
        """Test burstiness with single sentence returns 0."""
        sentences = ["Just one sentence here."]
        burstiness = calculate_burstiness(sentences)
        assert burstiness == 0.0

    def test_empty_list(self):
        """Test burstiness with empty list returns 0."""
        burstiness = calculate_burstiness([])
        assert burstiness == 0.0


class TestPOSDistribution:
    """Test POS tag distribution."""

    def test_basic_distribution(self):
        """Test getting POS distribution."""
        text = "The quick brown fox jumps over the lazy dog."
        distribution = get_pos_distribution(text)
        # Should have various POS tags
        assert "NOUN" in distribution
        assert "VERB" in distribution
        assert "ADJ" in distribution

    def test_empty_text(self):
        """Test empty text returns empty distribution."""
        distribution = get_pos_distribution("")
        assert distribution == {}


class TestDependencyDepth:
    """Test dependency tree depth calculation."""

    def test_simple_sentence_depth(self):
        """Test depth of simple sentence."""
        text = "The cat sat."
        depth = get_dependency_depth(text)
        assert depth >= 1

    def test_complex_sentence_depth(self):
        """Test depth of complex sentence is higher."""
        simple = "The cat sat."
        complex_text = "The very large cat that I saw yesterday sat quietly on the old wooden mat."

        simple_depth = get_dependency_depth(simple)
        complex_depth = get_dependency_depth(complex_text)

        # Complex sentence should have greater depth
        assert complex_depth >= simple_depth

    def test_empty_text(self):
        """Test empty text returns 0."""
        depth = get_dependency_depth("")
        assert depth == 0.0


class TestPerspectiveDetection:
    """Test perspective detection."""

    def test_first_person_singular(self):
        """Test detecting first person singular."""
        text = "I went to the store. I bought some milk. It was my favorite brand."
        perspective = detect_perspective(text)
        assert perspective == "first_person_singular"

    def test_first_person_plural(self):
        """Test detecting first person plural."""
        text = "We went to the conference. Our team presented. We were excited about our results."
        perspective = detect_perspective(text)
        assert perspective == "first_person_plural"

    def test_third_person(self):
        """Test detecting third person."""
        text = "He walked down the street. She saw him from across the road. They waved."
        perspective = detect_perspective(text)
        assert perspective == "third_person"

    def test_neutral_text(self):
        """Test text without clear pronouns defaults to third person."""
        text = "The sun rises in the east. Mountains stand tall."
        perspective = detect_perspective(text)
        assert perspective == "third_person"


class TestSemanticSimilarity:
    """Test semantic similarity computation."""

    def test_similar_texts_high_score(self):
        """Test that similar texts have high similarity."""
        text1 = "The cat sat on the mat."
        text2 = "A cat is sitting on the rug."
        similarity = compute_semantic_similarity(text1, text2)
        # Similar sentences should have reasonable similarity
        assert similarity > 0.5

    def test_different_texts_lower_score(self):
        """Test that different texts have lower similarity."""
        text1 = "The cat sat on the mat."
        text2 = "Financial markets crashed today."
        similarity = compute_semantic_similarity(text1, text2)
        # Unrelated sentences should have lower similarity
        assert similarity < 0.7

    def test_identical_texts_high_score(self):
        """Test that identical texts have high similarity."""
        text = "The quick brown fox jumps."
        similarity = compute_semantic_similarity(text, text)
        # Identical texts should have similarity close to 1
        assert similarity > 0.99

    def test_empty_text_returns_zero(self):
        """Test that empty text returns zero similarity."""
        similarity = compute_semantic_similarity("", "Some text")
        assert similarity == 0.0


class TestClassifyBySimilarity:
    """Test classification by semantic similarity."""

    def test_classify_question(self):
        """Test classifying a question-like sentence."""
        prototypes = {
            "QUESTION": ["asking a question", "inquiring about something", "wondering why"],
            "STATEMENT": ["making a claim", "stating a fact"]
        }
        # Use a sentence with strong question semantics
        text = "I am asking why this happens and inquiring about the reason."
        category, score = classify_by_similarity(text, prototypes)
        # Should match QUESTION due to semantic similarity
        assert category == "QUESTION"
        assert score > 0.3

    def test_classify_statement(self):
        """Test classifying a factual statement."""
        prototypes = {
            "QUESTION": ["asking a question", "inquiring about something"],
            "STATEMENT": ["stating a fact", "making a factual claim"]
        }
        text = "The earth revolves around the sun."
        category, score = classify_by_similarity(text, prototypes)
        assert category == "STATEMENT"

    def test_default_to_first_category_below_threshold(self):
        """Test that below threshold, returns first category."""
        prototypes = {
            "CAT_A": ["completely unrelated prototype"],
            "CAT_B": ["another unrelated prototype"]
        }
        text = "This is about something entirely different."
        category, score = classify_by_similarity(text, prototypes, threshold=0.9)
        # Should default to first category if all scores below threshold
        assert category == "CAT_A"

    def test_empty_text_returns_first_category(self):
        """Test empty text returns first category with zero score."""
        prototypes = {
            "FIRST": ["some prototype"],
            "SECOND": ["another prototype"]
        }
        category, score = classify_by_similarity("", prototypes)
        assert category == "FIRST"
        assert score == 0.0
