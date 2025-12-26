"""Global context analysis for documents."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ..models.graph import DocumentGraph, SemanticGraph
from ..models.style import StyleProfile
from ..llm.provider import LLMProvider
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GlobalContext:
    """Document-level context for style transfer.

    This context is set once at the start of document processing
    and persists throughout generation.
    """
    # Document content
    thesis: str
    intent: str  # persuade, inform, narrate, explain
    keywords: List[str]
    perspective: str  # first_person_singular, first_person_plural, third_person

    # Style target
    style_dna: str
    author_name: str
    target_burstiness: float
    target_sentence_length: float
    top_vocabulary: List[str]

    # Author style features (from corpus analysis)
    author_transitions: Dict[str, List[str]] = field(default_factory=dict)
    author_openers: List[str] = field(default_factory=list)
    author_voice_ratio: float = 0.5  # 0=passive, 1=active
    author_signature_phrases: List[str] = field(default_factory=list)

    # Processing state
    total_paragraphs: int = 0
    processed_paragraphs: int = 0
    generated_summary: str = ""  # Summary of previously generated text

    def _get_author_transition_guidance(self) -> str:
        """Generate transition guidance from author's actual usage."""
        if not self.author_transitions:
            return ""

        # Get the author's preferred transitions (top 2-3 from each category)
        preferred = []
        for category, words in self.author_transitions.items():
            if words:
                preferred.extend(words[:2])

        if not preferred:
            return ""

        return f"- Uses connectors like: {', '.join(preferred[:6])}"

    def _get_author_opener_guidance(self) -> str:
        """Generate opener guidance from author's actual sentence starts."""
        if not self.author_openers:
            return ""

        # Show variety of openers the author uses
        openers = self.author_openers[:8]
        return f"- Varies sentence openings (examples: {', '.join(openers)})"

    def _get_voice_guidance(self) -> str:
        """Generate voice guidance from author's active/passive ratio."""
        if self.author_voice_ratio > 0.7:
            return "- Primarily uses active voice"
        elif self.author_voice_ratio < 0.3:
            return "- Often uses passive voice"
        else:
            return "- Mixes active and passive voice naturally"

    def to_system_prompt(self) -> str:
        """Convert context to LLM system prompt.

        Returns:
            System prompt string with document context.
        """
        # Build dynamic author style guidance from corpus analysis
        style_hints = []

        transition_hint = self._get_author_transition_guidance()
        if transition_hint:
            style_hints.append(transition_hint)

        opener_hint = self._get_author_opener_guidance()
        if opener_hint:
            style_hints.append(opener_hint)

        voice_hint = self._get_voice_guidance()
        if voice_hint:
            style_hints.append(voice_hint)

        if self.author_signature_phrases:
            phrases = self.author_signature_phrases[:4]
            style_hints.append(f"- Characteristic phrases: {', '.join(phrases)}")

        style_guidance = "\n".join(style_hints) if style_hints else "- Write naturally in the author's voice"

        return f"""You are adapting text to match {self.author_name}'s writing style.

STYLE DESCRIPTION:
{self.style_dna}

DOCUMENT:
- Main idea: {self.thesis}
- Intent: {self.intent}
- Perspective: {self.perspective}

AUTHOR'S PATTERNS (from corpus analysis):
{style_guidance}

GUIDELINES:
1. Preserve ALL semantic meaning - every fact and idea must appear in output
2. Write naturally in the target style - don't mechanically apply rules
3. Vary your sentence openings - avoid starting multiple sentences the same way
4. Let transitions flow naturally from the author's patterns above
5. Maintain {self.perspective} perspective

AVOID:
- Mechanical insertion of vocabulary words
- Formulaic sentence structures
- Over-explaining or padding"""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "thesis": self.thesis,
            "intent": self.intent,
            "keywords": self.keywords,
            "perspective": self.perspective,
            "style_dna": self.style_dna,
            "author_name": self.author_name,
            "target_burstiness": self.target_burstiness,
            "target_sentence_length": self.target_sentence_length,
            "top_vocabulary": self.top_vocabulary,
            "total_paragraphs": self.total_paragraphs,
            "processed_paragraphs": self.processed_paragraphs,
        }


@dataclass
class ParagraphContext:
    """Context for a specific paragraph being processed.

    Set once per paragraph, contains paragraph-specific info
    plus reference to global context.
    """
    paragraph_idx: int
    role: str  # INTRO, BODY, CONCLUSION
    semantic_graph: SemanticGraph
    previous_summary: str  # Summary of previously generated paragraphs
    sentence_count_target: int
    total_propositions: int

    def to_prompt_section(self) -> str:
        """Convert to prompt section for paragraph context.

        Returns:
            Prompt section string.
        """
        # Summarize the semantic graph
        props = [f"- {node.text}" for node in self.semantic_graph.nodes[:5]]
        props_text = "\n".join(props)
        if len(self.semantic_graph.nodes) > 5:
            props_text += f"\n  ... and {len(self.semantic_graph.nodes) - 5} more propositions"

        return f"""PARAGRAPH {self.paragraph_idx + 1} ({self.role}):
Previous context: {self.previous_summary or 'This is the first paragraph.'}

Semantic content to express:
{props_text}

Target: ~{self.sentence_count_target} sentences"""


class GlobalContextAnalyzer:
    """Analyzes documents to extract global context for generation.

    Creates the GlobalContext object that persists throughout
    document processing, enabling coherent multi-paragraph generation.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize analyzer.

        Args:
            llm_provider: Optional LLM for enhanced analysis.
        """
        self.llm_provider = llm_provider

    def analyze(
        self,
        document_graph: DocumentGraph,
        style_profile: StyleProfile
    ) -> GlobalContext:
        """Analyze document and style to create global context.

        Args:
            document_graph: Semantic graph of the document.
            style_profile: Target style profile.

        Returns:
            GlobalContext instance.
        """
        # Extract thesis (use document graph's or generate)
        thesis = document_graph.thesis
        if not thesis and document_graph.paragraphs:
            thesis = self._extract_thesis_from_graph(document_graph)

        # Get style profile effective values
        author = style_profile.primary_author

        # Extract author's preferred transitions (just the words, not frequencies)
        author_transitions = {}
        for category, word_freqs in author.transitions.items():
            author_transitions[category] = [word for word, _ in word_freqs[:5]]

        # Extract author's preferred sentence openers
        all_openers = author.sentence_openers.get("openers", [])
        author_openers = [opener for opener, _ in all_openers[:10]]

        # Extract signature phrases (just the phrases)
        author_phrases = [phrase for phrase, _ in author.signature_phrases[:6]]

        context = GlobalContext(
            thesis=thesis,
            intent=document_graph.intent,
            keywords=document_graph.keywords,
            perspective=document_graph.perspective,
            style_dna=style_profile.get_effective_style_dna(),
            author_name=style_profile.get_author_name(),
            target_burstiness=style_profile.get_effective_burstiness(),
            target_sentence_length=style_profile.get_effective_avg_sentence_length(),
            top_vocabulary=style_profile.get_effective_vocab(),
            # Author style features from corpus
            author_transitions=author_transitions,
            author_openers=author_openers,
            author_voice_ratio=author.voice_ratio,
            author_signature_phrases=author_phrases,
            # Processing state
            total_paragraphs=len(document_graph.paragraphs),
            processed_paragraphs=0
        )

        logger.info(
            f"Created global context: intent={context.intent}, "
            f"target_author={context.author_name}, "
            f"paragraphs={context.total_paragraphs}"
        )

        return context

    def _extract_thesis_from_graph(self, document_graph: DocumentGraph) -> str:
        """Extract thesis from semantic graph when not available.

        Args:
            document_graph: Document's semantic graph.

        Returns:
            Thesis string.
        """
        if not document_graph.paragraphs:
            return ""

        # Get first paragraph (intro)
        intro = document_graph.paragraphs[0]
        if intro.nodes:
            # Use the first proposition or two
            thesis_parts = [n.text for n in intro.nodes[:2]]
            return " ".join(thesis_parts)

        return ""

    def create_paragraph_context(
        self,
        paragraph_graph: SemanticGraph,
        global_context: GlobalContext,
        previous_paragraphs: List[str]
    ) -> ParagraphContext:
        """Create context for a specific paragraph.

        Args:
            paragraph_graph: Semantic graph for the paragraph.
            global_context: Global document context.
            previous_paragraphs: Previously generated paragraph texts.

        Returns:
            ParagraphContext instance.
        """
        # Summarize previous paragraphs
        if previous_paragraphs:
            summary = self._summarize_paragraphs(previous_paragraphs)
        else:
            summary = ""

        # Calculate target sentence count
        # Based on proposition count and target sentence length
        prop_count = len(paragraph_graph.nodes)
        avg_props_per_sentence = 1.5  # Rough estimate
        sentence_target = max(1, int(prop_count / avg_props_per_sentence))

        return ParagraphContext(
            paragraph_idx=paragraph_graph.paragraph_idx,
            role=paragraph_graph.role.value,
            semantic_graph=paragraph_graph,
            previous_summary=summary,
            sentence_count_target=sentence_target,
            total_propositions=prop_count
        )

    def _summarize_paragraphs(self, paragraphs: List[str]) -> str:
        """Create a brief summary of previous paragraphs.

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            Summary string.
        """
        if not paragraphs:
            return ""

        # Simple approach: take first sentence of each paragraph
        summaries = []
        for para in paragraphs[-3:]:  # Last 3 paragraphs
            sentences = para.split('.')
            if sentences:
                first_sent = sentences[0].strip()
                if first_sent:
                    summaries.append(first_sent)

        if len(paragraphs) > 3:
            prefix = f"[{len(paragraphs) - 3} earlier paragraphs...] "
        else:
            prefix = ""

        return prefix + ". ".join(summaries) + "."

    def update_context_after_paragraph(
        self,
        global_context: GlobalContext,
        generated_paragraph: str
    ) -> None:
        """Update global context after generating a paragraph.

        Args:
            global_context: Context to update.
            generated_paragraph: The just-generated paragraph.
        """
        global_context.processed_paragraphs += 1

        # Update summary
        if global_context.generated_summary:
            global_context.generated_summary += " " + self._get_first_sentence(generated_paragraph)
        else:
            global_context.generated_summary = self._get_first_sentence(generated_paragraph)

    def _get_first_sentence(self, text: str) -> str:
        """Get first sentence from text.

        Args:
            text: Text to extract from.

        Returns:
            First sentence.
        """
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip() + "."
        return text[:100] + "..."
