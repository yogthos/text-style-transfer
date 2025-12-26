"""Unit tests for global context analysis."""

import pytest
from src.ingestion.context_analyzer import (
    GlobalContextAnalyzer,
    GlobalContext,
    ParagraphContext,
)
from src.ingestion.graph_builder import DocumentGraphBuilder
from src.corpus.preprocessor import TextPreprocessor
from src.models.style import AuthorProfile, StyleProfile
from src.models.graph import SemanticGraph, ParagraphRole, RhetoricalIntent


class TestGlobalContext:
    """Test GlobalContext data class."""

    @pytest.fixture
    def sample_context(self):
        """Create sample global context."""
        return GlobalContext(
            thesis="This is the main thesis.",
            intent="persuade",
            keywords=["keyword1", "keyword2", "keyword3"],
            perspective="first_person_singular",
            style_dna="Writes with clarity and precision.",
            author_name="Test Author",
            target_burstiness=0.3,
            target_sentence_length=15.0,
            top_vocabulary=["however", "therefore", "indeed"],
            total_paragraphs=5,
            processed_paragraphs=0,
        )

    def test_to_system_prompt(self, sample_context):
        """Test system prompt generation."""
        prompt = sample_context.to_system_prompt()

        assert "Test Author" in prompt
        assert "clarity and precision" in prompt
        assert "persuade" in prompt
        assert "first_person_singular" in prompt
        # Note: sentence length stats removed from system prompt to reduce mechanical output
        assert "AVOID" in prompt  # Anti-pattern guidance

    def test_to_dict(self, sample_context):
        """Test dictionary conversion."""
        d = sample_context.to_dict()

        assert d["thesis"] == "This is the main thesis."
        assert d["intent"] == "persuade"
        assert len(d["keywords"]) == 3
        assert d["author_name"] == "Test Author"
        assert d["total_paragraphs"] == 5


class TestParagraphContext:
    """Test ParagraphContext data class."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample semantic graph."""
        from src.models.graph import PropositionNode

        nodes = [
            PropositionNode(id="p1", text="First proposition.", subject="First", verb="is"),
            PropositionNode(id="p2", text="Second proposition.", subject="Second", verb="follows"),
        ]
        return SemanticGraph(
            nodes=nodes,
            edges=[],
            paragraph_idx=0,
            role=ParagraphRole.BODY
        )

    def test_to_prompt_section(self, sample_graph):
        """Test prompt section generation."""
        context = ParagraphContext(
            paragraph_idx=2,
            role="BODY",
            semantic_graph=sample_graph,
            previous_summary="Previous content discussed.",
            sentence_count_target=3,
            total_propositions=2
        )

        prompt = context.to_prompt_section()

        assert "PARAGRAPH 3" in prompt  # 0-indexed + 1
        assert "BODY" in prompt
        assert "Previous content" in prompt
        assert "First proposition" in prompt


class TestGlobalContextAnalyzer:
    """Test GlobalContextAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer without LLM."""
        return GlobalContextAnalyzer(llm_provider=None)

    @pytest.fixture
    def doc_builder(self):
        """Create document graph builder."""
        return DocumentGraphBuilder(llm_provider=None, use_llm_relationships=False)

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor."""
        return TextPreprocessor()

    @pytest.fixture
    def style_profile(self):
        """Create sample style profile."""
        author = AuthorProfile(
            name="Test Author",
            style_dna="Writes with eloquence and precision.",
            top_vocab=["however", "therefore", "indeed", "moreover"],
            avg_sentence_length=18.0,
            burstiness=0.35,
            punctuation_freq={"semicolon": 0.02},
            perspective="third_person"
        )
        return StyleProfile.from_author(author)

    def test_analyze_creates_context(
        self, analyzer, doc_builder, preprocessor, style_profile
    ):
        """Test that analyze creates a valid GlobalContext."""
        text = """This is the introduction with the main thesis.

        The body explains the details with supporting arguments.

        The conclusion summarizes the key points."""

        document = preprocessor.process(text)
        doc_graph = doc_builder.build_from_document(document)

        context = analyzer.analyze(doc_graph, style_profile)

        assert isinstance(context, GlobalContext)
        assert context.author_name == "Test Author"
        assert context.target_sentence_length == 18.0
        assert context.target_burstiness == 0.35
        assert context.total_paragraphs == 3

    def test_thesis_extraction(
        self, analyzer, doc_builder, preprocessor, style_profile
    ):
        """Test thesis extraction from document."""
        text = """The introduction sets up the argument.

        Supporting details follow here.

        The conclusion wraps things up."""

        document = preprocessor.process(text)
        doc_graph = doc_builder.build_from_document(document)

        context = analyzer.analyze(doc_graph, style_profile)

        assert context.thesis  # Should have extracted thesis

    def test_create_paragraph_context(
        self, analyzer, doc_builder, preprocessor, style_profile
    ):
        """Test paragraph context creation."""
        text = """Introduction paragraph.

        Body paragraph with content.

        Conclusion paragraph."""

        document = preprocessor.process(text)
        doc_graph = doc_builder.build_from_document(document)
        global_ctx = analyzer.analyze(doc_graph, style_profile)

        para_ctx = analyzer.create_paragraph_context(
            paragraph_graph=doc_graph.paragraphs[1],
            global_context=global_ctx,
            previous_paragraphs=["First generated paragraph."]
        )

        assert isinstance(para_ctx, ParagraphContext)
        assert para_ctx.paragraph_idx == 1
        assert para_ctx.previous_summary  # Should have summary

    def test_create_paragraph_context_no_previous(
        self, analyzer, doc_builder, preprocessor, style_profile
    ):
        """Test paragraph context with no previous paragraphs."""
        text = """Single paragraph document."""

        document = preprocessor.process(text)
        doc_graph = doc_builder.build_from_document(document)
        global_ctx = analyzer.analyze(doc_graph, style_profile)

        para_ctx = analyzer.create_paragraph_context(
            paragraph_graph=doc_graph.paragraphs[0],
            global_context=global_ctx,
            previous_paragraphs=[]
        )

        assert para_ctx.previous_summary == ""

    def test_update_context_after_paragraph(
        self, analyzer, doc_builder, preprocessor, style_profile
    ):
        """Test context updates after paragraph generation."""
        text = """Introduction paragraph.

        Body paragraph."""

        document = preprocessor.process(text)
        doc_graph = doc_builder.build_from_document(document)
        context = analyzer.analyze(doc_graph, style_profile)

        assert context.processed_paragraphs == 0

        analyzer.update_context_after_paragraph(
            context,
            "Generated paragraph text goes here."
        )

        assert context.processed_paragraphs == 1
        assert context.generated_summary  # Should have summary

    def test_multiple_paragraph_updates(
        self, analyzer, doc_builder, preprocessor, style_profile
    ):
        """Test multiple paragraph updates accumulate."""
        text = """Para one.

        Para two.

        Para three."""

        document = preprocessor.process(text)
        doc_graph = doc_builder.build_from_document(document)
        context = analyzer.analyze(doc_graph, style_profile)

        analyzer.update_context_after_paragraph(context, "First generated.")
        analyzer.update_context_after_paragraph(context, "Second generated.")

        assert context.processed_paragraphs == 2


class TestContextIntegration:
    """Integration tests for context flow."""

    @pytest.fixture
    def full_pipeline(self):
        """Set up full pipeline components."""
        return {
            "analyzer": GlobalContextAnalyzer(llm_provider=None),
            "doc_builder": DocumentGraphBuilder(llm_provider=None, use_llm_relationships=False),
            "preprocessor": TextPreprocessor(),
        }

    @pytest.fixture
    def style_profile(self):
        """Create style profile."""
        author = AuthorProfile(
            name="Integration Test Author",
            style_dna="Formal and academic writing style.",
            top_vocab=["thus", "hence", "therefore"],
            avg_sentence_length=20.0,
            burstiness=0.25,
        )
        return StyleProfile.from_author(author)

    def test_full_context_flow(self, full_pipeline, style_profile):
        """Test complete context creation and usage flow."""
        text = """The research examines an important topic.

        The methodology involved careful analysis.

        The results demonstrate clear findings."""

        # Process document
        document = full_pipeline["preprocessor"].process(text)
        doc_graph = full_pipeline["doc_builder"].build_from_document(document)

        # Create global context
        global_ctx = full_pipeline["analyzer"].analyze(doc_graph, style_profile)

        # Process each paragraph
        generated = []
        for i, para_graph in enumerate(doc_graph.paragraphs):
            para_ctx = full_pipeline["analyzer"].create_paragraph_context(
                paragraph_graph=para_graph,
                global_context=global_ctx,
                previous_paragraphs=generated
            )

            # Simulate generation
            generated_text = f"Generated paragraph {i+1}."
            generated.append(generated_text)

            # Update context
            full_pipeline["analyzer"].update_context_after_paragraph(
                global_ctx, generated_text
            )

        assert global_ctx.processed_paragraphs == 3
        assert len(generated) == 3
