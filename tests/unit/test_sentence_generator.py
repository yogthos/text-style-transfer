"""Unit tests for sentence generator."""

import pytest
from unittest.mock import MagicMock, patch

from src.generation.sentence_generator import (
    SentenceGenerator,
    MultiPassGenerator,
    GeneratedSentence,
    GeneratedParagraph,
)
from src.models.plan import (
    SentencePlan,
    SentenceNode,
    SentenceRole,
    TransitionType,
)
from src.models.graph import PropositionNode, SemanticGraph
from src.ingestion.context_analyzer import GlobalContext, ParagraphContext


class TestGeneratedSentence:
    """Test GeneratedSentence dataclass."""

    def test_create_generated_sentence(self):
        """Test creating a generated sentence."""
        sentence = GeneratedSentence(
            text="This is a test sentence.",
            node_id="s1",
            word_count=5,
            target_length=5
        )

        assert sentence.text == "This is a test sentence."
        assert sentence.word_count == 5
        assert sentence.target_length == 5

    def test_length_accuracy_exact(self):
        """Test length accuracy when exact."""
        sentence = GeneratedSentence(
            text="Test", node_id="s1",
            word_count=10, target_length=10
        )

        assert sentence.length_accuracy == 1.0

    def test_length_accuracy_over(self):
        """Test length accuracy when over target."""
        sentence = GeneratedSentence(
            text="Test", node_id="s1",
            word_count=15, target_length=10
        )

        # 5/10 = 0.5 deviation, so accuracy = 0.5
        assert sentence.length_accuracy == 0.5

    def test_length_accuracy_under(self):
        """Test length accuracy when under target."""
        sentence = GeneratedSentence(
            text="Test", node_id="s1",
            word_count=5, target_length=10
        )

        assert sentence.length_accuracy == 0.5

    def test_length_accuracy_zero_target(self):
        """Test length accuracy with zero target."""
        sentence = GeneratedSentence(
            text="Test", node_id="s1",
            word_count=5, target_length=0
        )

        assert sentence.length_accuracy == 1.0


class TestGeneratedParagraph:
    """Test GeneratedParagraph dataclass."""

    @pytest.fixture
    def sample_sentences(self):
        """Create sample generated sentences."""
        return [
            GeneratedSentence("First sentence here.", "s1", 3, 3),
            GeneratedSentence("Second sentence is longer.", "s2", 4, 4),
            GeneratedSentence("Third.", "s3", 1, 1),
        ]

    @pytest.fixture
    def sample_plan(self):
        """Create sample sentence plan."""
        return SentencePlan(
            nodes=[
                SentenceNode(id="s1", propositions=[], target_length=3),
                SentenceNode(id="s2", propositions=[], target_length=4),
                SentenceNode(id="s3", propositions=[], target_length=1),
            ],
            paragraph_role="BODY"
        )

    def test_text_property(self, sample_sentences, sample_plan):
        """Test combined text property."""
        para = GeneratedParagraph(sample_sentences, sample_plan, 0)

        expected = "First sentence here. Second sentence is longer. Third."
        assert para.text == expected

    def test_word_count(self, sample_sentences, sample_plan):
        """Test word count calculation."""
        para = GeneratedParagraph(sample_sentences, sample_plan, 0)

        assert para.word_count == 8

    def test_avg_length_accuracy(self, sample_sentences, sample_plan):
        """Test average length accuracy."""
        para = GeneratedParagraph(sample_sentences, sample_plan, 0)

        # All sentences match target exactly
        assert para.avg_length_accuracy == 1.0

    def test_avg_length_accuracy_empty(self, sample_plan):
        """Test average length accuracy with no sentences."""
        para = GeneratedParagraph([], sample_plan, 0)

        assert para.avg_length_accuracy == 0.0


class TestSentenceGenerator:
    """Test SentenceGenerator functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        llm = MagicMock()
        llm.call.return_value = "This is a generated test sentence."
        llm.generate.return_value = "This is a generated test sentence."
        return llm

    @pytest.fixture
    def global_context(self):
        """Create sample global context."""
        return GlobalContext(
            thesis="Test thesis.",
            intent="inform",
            keywords=["test"],
            perspective="third_person",
            style_dna="Direct style.",
            author_name="Test Author",
            target_burstiness=0.2,
            target_sentence_length=12.0,
            top_vocabulary=["word"],
            total_paragraphs=1,
            processed_paragraphs=0
        )

    @pytest.fixture
    def generator(self, mock_llm, global_context):
        """Create sentence generator."""
        return SentenceGenerator(mock_llm, global_context)

    @pytest.fixture
    def sentence_node(self):
        """Create sample sentence node."""
        prop = PropositionNode(id="p1", text="Test.", subject="S", verb="is")
        return SentenceNode(
            id="s1",
            propositions=[prop],
            role=SentenceRole.THESIS,
            target_length=10
        )

    @pytest.fixture
    def sentence_plan(self, sentence_node):
        """Create sample sentence plan."""
        return SentencePlan(
            nodes=[sentence_node],
            paragraph_role="BODY"
        )

    @pytest.fixture
    def paragraph_context(self):
        """Create sample paragraph context."""
        graph = SemanticGraph(nodes=[], edges=[])
        return ParagraphContext(
            paragraph_idx=0,
            role="BODY",
            semantic_graph=graph,
            previous_summary="",
            sentence_count_target=1,
            total_propositions=1
        )

    def test_generate_sentence(self, generator, sentence_node, sentence_plan):
        """Test single sentence generation."""
        result = generator.generate_sentence(sentence_node, sentence_plan)

        assert isinstance(result, GeneratedSentence)
        assert result.text.endswith('.')
        assert result.node_id == "s1"

    def test_generate_sentence_calls_llm(self, generator, sentence_node, sentence_plan, mock_llm):
        """Test that generator calls LLM."""
        generator.generate_sentence(sentence_node, sentence_plan)

        mock_llm.call.assert_called_once()
        call_args = mock_llm.call.call_args
        assert "temperature" in call_args.kwargs

    def test_generate_paragraph(self, generator, sentence_plan, paragraph_context):
        """Test paragraph generation."""
        result = generator.generate_paragraph(sentence_plan, paragraph_context)

        assert isinstance(result, GeneratedParagraph)
        assert len(result.sentences) == 1
        assert result.paragraph_idx == 0

    def test_clean_sentence_removes_prefix(self, generator):
        """Test sentence cleaning removes LLM prefixes."""
        raw = "Here's the sentence: This is the actual sentence."
        cleaned = generator._clean_sentence(raw)

        assert cleaned == "This is the actual sentence."

    def test_clean_sentence_removes_quotes(self, generator):
        """Test sentence cleaning removes surrounding quotes."""
        raw = '"This is a quoted sentence."'
        cleaned = generator._clean_sentence(raw)

        assert cleaned == "This is a quoted sentence."

    def test_clean_sentence_adds_period(self, generator):
        """Test sentence cleaning adds period if missing."""
        raw = "This sentence has no ending"
        cleaned = generator._clean_sentence(raw)

        assert cleaned == "This sentence has no ending."

    def test_clean_sentence_handles_newlines(self, generator):
        """Test sentence cleaning removes newlines."""
        raw = "This sentence\nhas newlines\nin it."
        cleaned = generator._clean_sentence(raw)

        assert "\n" not in cleaned
        assert cleaned == "This sentence has newlines in it."

    def test_revise_sentence(self, generator, sentence_node):
        """Test sentence revision."""
        original = GeneratedSentence(
            text="Original text.",
            node_id="s1",
            word_count=2,
            target_length=10
        )

        revised = generator.revise_sentence(
            original, "Make it longer.", sentence_node
        )

        assert isinstance(revised, GeneratedSentence)
        assert revised.revision_count == 1

    def test_parse_paragraph_response_newlines(self, generator):
        """Test parsing response with newline separation."""
        response = "First sentence.\nSecond sentence.\nThird sentence."
        plan = SentencePlan(nodes=[
            SentenceNode(id="s1", propositions=[], target_length=2),
            SentenceNode(id="s2", propositions=[], target_length=2),
            SentenceNode(id="s3", propositions=[], target_length=2),
        ])

        sentences = generator._parse_paragraph_response(response, plan)

        assert len(sentences) == 3
        assert sentences[0].text == "First sentence."
        assert sentences[1].text == "Second sentence."

    def test_parse_paragraph_response_periods(self, generator):
        """Test parsing response with period separation."""
        response = "First sentence. Second sentence. Third sentence."
        plan = SentencePlan(nodes=[
            SentenceNode(id="s1", propositions=[], target_length=2),
            SentenceNode(id="s2", propositions=[], target_length=2),
            SentenceNode(id="s3", propositions=[], target_length=2),
        ])

        sentences = generator._parse_paragraph_response(response, plan)

        assert len(sentences) == 3


class TestMultiPassGenerator:
    """Test MultiPassGenerator functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        llm = MagicMock()
        # Return numbered alternatives
        llm.call.return_value = """1. First alternative sentence.
2. Second alternative sentence.
3. Third alternative sentence."""
        llm.generate.return_value = """1. First alternative sentence.
2. Second alternative sentence.
3. Third alternative sentence."""
        return llm

    @pytest.fixture
    def global_context(self):
        """Create sample global context."""
        return GlobalContext(
            thesis="Test.", intent="inform", keywords=[],
            perspective="third_person", style_dna="Direct.",
            author_name="Author", target_burstiness=0.2,
            target_sentence_length=12.0, top_vocabulary=[],
            total_paragraphs=1, processed_paragraphs=0
        )

    @pytest.fixture
    def generator(self, mock_llm, global_context):
        """Create multi-pass generator."""
        return MultiPassGenerator(
            mock_llm, global_context,
            num_alternatives=3
        )

    @pytest.fixture
    def sentence_node(self):
        """Create sample sentence node."""
        prop = PropositionNode(id="p1", text="Test.", subject="S", verb="is")
        return SentenceNode(
            id="s1", propositions=[prop], target_length=4
        )

    def test_generate_alternatives(self, generator, sentence_node):
        """Test generating alternatives."""
        alternatives = generator._generate_alternatives(sentence_node, None)

        assert len(alternatives) <= 3
        assert all(isinstance(a, str) for a in alternatives)

    def test_select_best_by_length(self, generator, sentence_node):
        """Test selecting best by length accuracy."""
        alternatives = [
            "Very short.",  # 2 words
            "Four words here now.",  # 4 words (target)
            "This one is much too long for our needs.",  # 9 words
        ]

        best = generator._select_best(alternatives, sentence_node)

        # Should select the one closest to target (4 words)
        assert best == "Four words here now."

    def test_select_best_with_keywords(self, generator):
        """Test selection favors keyword inclusion."""
        node = SentenceNode(
            id="s1", propositions=[],
            target_length=4, keywords=["special"]
        )

        alternatives = [
            "Four words here now.",
            "Special word included here.",  # Contains keyword
            "Another option here too.",
        ]

        best = generator._select_best(alternatives, node)

        # Should favor the one with keyword
        assert "special" in best.lower()

    def test_generate_sentence_returns_alternatives(self, generator, sentence_node):
        """Test that generated sentence includes alternatives."""
        plan = SentencePlan(nodes=[sentence_node], paragraph_role="BODY")

        result = generator.generate_sentence(sentence_node, plan)

        assert isinstance(result, GeneratedSentence)
        # Should have alternatives stored
        assert len(result.alternatives) > 0

    def test_single_alternative_handling(self, generator, sentence_node):
        """Test handling of single alternative."""
        alternatives = ["Only one option."]

        best = generator._select_best(alternatives, sentence_node)

        assert best == "Only one option."
