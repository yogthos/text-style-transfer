"""Sentence generation using LLM."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from typing import TYPE_CHECKING

from ..llm.provider import LLMProvider
from ..llm.session import LLMSession
from ..models.plan import SentencePlan, SentenceNode
from ..models.style import StyleProfile
from ..ingestion.context_analyzer import GlobalContext, ParagraphContext
from .prompt_builder import PromptBuilder, MultiSentencePromptBuilder
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..corpus.indexer import CorpusIndexer

logger = get_logger(__name__)


@dataclass
class GeneratedSentence:
    """A generated sentence with metadata."""
    text: str
    node_id: str
    word_count: int
    target_length: int
    alternatives: List[str] = field(default_factory=list)
    revision_count: int = 0

    @property
    def length_accuracy(self) -> float:
        """Calculate accuracy of length vs target."""
        if self.target_length == 0:
            return 1.0
        diff = abs(self.word_count - self.target_length)
        return max(0.0, 1.0 - diff / self.target_length)


@dataclass
class GeneratedParagraph:
    """A generated paragraph with its sentences."""
    sentences: List[GeneratedSentence]
    plan: SentencePlan
    paragraph_idx: int

    @property
    def text(self) -> str:
        """Get combined paragraph text."""
        return " ".join(s.text for s in self.sentences)

    @property
    def word_count(self) -> int:
        """Get total word count."""
        return sum(s.word_count for s in self.sentences)

    @property
    def avg_length_accuracy(self) -> float:
        """Average length accuracy across sentences."""
        if not self.sentences:
            return 0.0
        return sum(s.length_accuracy for s in self.sentences) / len(self.sentences)


class SentenceGenerator:
    """Generates sentences using LLM based on sentence plans.

    This is the core generation component that:
    1. Takes sentence plans
    2. Builds appropriate prompts
    3. Calls the LLM
    4. Post-processes results
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        global_context: GlobalContext,
        temperature: float = 0.7,
        style_profile: Optional[StyleProfile] = None,
        indexer: Optional["CorpusIndexer"] = None
    ):
        """Initialize sentence generator.

        Args:
            llm_provider: LLM provider instance.
            global_context: Document-level context.
            temperature: Generation temperature.
            style_profile: Target author's style profile (for author-specific transitions).
            indexer: Optional corpus indexer for RAG-based style examples.
        """
        self.llm_provider = llm_provider
        self.global_context = global_context
        self.temperature = temperature
        self.style_profile = style_profile
        self.indexer = indexer
        self.prompt_builder = PromptBuilder(global_context, style_profile, indexer)

    def generate_paragraph(
        self,
        plan: SentencePlan,
        paragraph_context: ParagraphContext
    ) -> GeneratedParagraph:
        """Generate a complete paragraph from a plan.

        Args:
            plan: Sentence plan for the paragraph.
            paragraph_context: Paragraph-specific context.

        Returns:
            GeneratedParagraph with all sentences.
        """
        logger.info(
            f"Generating paragraph {paragraph_context.paragraph_idx}: "
            f"{len(plan.nodes)} sentences planned"
        )

        # Reset entity tracking for new paragraph
        self.prompt_builder.reset_entity_tracking()

        sentences = []
        previous_sentence = None

        for node in plan.nodes:
            generated = self.generate_sentence(node, plan, previous_sentence)
            sentences.append(generated)
            previous_sentence = generated.text

            # Register generated sentence for transition tracking
            self.prompt_builder.register_generated_sentence(generated.text)

            # Register entities from this sentence's propositions for future reference
            entities = []
            for prop in node.propositions:
                entities.extend(prop.entities)
            self.prompt_builder.register_introduced_entities(entities)

        paragraph = GeneratedParagraph(
            sentences=sentences,
            plan=plan,
            paragraph_idx=paragraph_context.paragraph_idx
        )

        logger.info(
            f"Generated paragraph: {paragraph.word_count} words, "
            f"avg length accuracy: {paragraph.avg_length_accuracy:.2%}"
        )

        return paragraph

    def generate_sentence(
        self,
        node: SentenceNode,
        plan: SentencePlan,
        previous_sentence: Optional[str] = None
    ) -> GeneratedSentence:
        """Generate a single sentence.

        Args:
            node: Sentence node with specifications.
            plan: Full sentence plan for context.
            previous_sentence: The previous sentence for flow.

        Returns:
            GeneratedSentence object.
        """
        prompt = self.prompt_builder.build_sentence_prompt(
            node, plan, previous_sentence
        )

        # Generate using LLM
        response = self.llm_provider.call(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            temperature=self.temperature,
            max_tokens=150  # Single sentence
        )

        # Clean and validate
        text = self._clean_sentence(response)
        word_count = len(text.split())

        generated = GeneratedSentence(
            text=text,
            node_id=node.id,
            word_count=word_count,
            target_length=node.target_length
        )

        logger.debug(
            f"Generated sentence: {word_count}/{node.target_length} words, "
            f"accuracy: {generated.length_accuracy:.2%}"
        )

        return generated

    def generate_paragraph_batch(
        self,
        plan: SentencePlan,
        paragraph_context: ParagraphContext
    ) -> GeneratedParagraph:
        """Generate entire paragraph in one LLM call.

        More efficient but less controllable than sentence-by-sentence.

        Args:
            plan: Sentence plan.
            paragraph_context: Paragraph context.

        Returns:
            GeneratedParagraph.
        """
        prompt = self.prompt_builder.build_paragraph_prompt(
            plan, paragraph_context
        )

        # Calculate appropriate max tokens
        total_target_words = sum(n.target_length for n in plan.nodes)
        max_tokens = int(total_target_words * 1.5)  # Allow some buffer

        response = self.llm_provider.call(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            temperature=self.temperature,
            max_tokens=max_tokens
        )

        # Parse response into sentences
        sentences = self._parse_paragraph_response(response, plan)

        return GeneratedParagraph(
            sentences=sentences,
            plan=plan,
            paragraph_idx=paragraph_context.paragraph_idx
        )

    def revise_sentence(
        self,
        original: GeneratedSentence,
        feedback: str,
        node: SentenceNode
    ) -> GeneratedSentence:
        """Revise a sentence based on feedback.

        Args:
            original: Original generated sentence.
            feedback: What to improve.
            node: Original sentence specification.

        Returns:
            Revised GeneratedSentence.
        """
        prompt = self.prompt_builder.build_revision_prompt(
            original.text, feedback, node
        )

        response = self.llm_provider.call(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            temperature=self.temperature * 0.8,  # Slightly lower for revision
            max_tokens=150
        )

        text = self._clean_sentence(response)
        word_count = len(text.split())

        return GeneratedSentence(
            text=text,
            node_id=original.node_id,
            word_count=word_count,
            target_length=original.target_length,
            revision_count=original.revision_count + 1
        )

    def _clean_sentence(self, text: str) -> str:
        """Clean LLM output to get clean sentence.

        Args:
            text: Raw LLM output.

        Returns:
            Cleaned sentence text.
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the sentence:",
            "Here is the sentence:",
            "Sentence:",
            "Output:",
            "Result:",
        ]
        for prefix in prefixes_to_remove:
            if text.strip().lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Remove quotes if wrapped
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Ensure ends with sentence-ending punctuation
        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'

        # Remove newlines
        text = ' '.join(text.split())

        return text

    def _parse_paragraph_response(
        self,
        response: str,
        plan: SentencePlan
    ) -> List[GeneratedSentence]:
        """Parse paragraph response into individual sentences.

        Args:
            response: LLM response text.
            plan: Original sentence plan.

        Returns:
            List of GeneratedSentence objects.
        """
        # Try to split by newlines first (as instructed)
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]

        # If not enough lines, split by sentence-ending punctuation
        if len(lines) < len(plan.nodes):
            # Use regex to split on sentence boundaries
            pattern = r'(?<=[.!?])\s+'
            lines = re.split(pattern, response.strip())
            lines = [l.strip() for l in lines if l.strip()]

        sentences = []
        for i, node in enumerate(plan.nodes):
            if i < len(lines):
                text = self._clean_sentence(lines[i])
            else:
                # Fallback: generate missing sentence individually
                logger.warning(f"Missing sentence {i+1}, generating individually")
                text = f"[Missing sentence {i+1}]"

            word_count = len(text.split())
            sentences.append(GeneratedSentence(
                text=text,
                node_id=node.id,
                word_count=word_count,
                target_length=node.target_length
            ))

        return sentences


class MultiPassGenerator(SentenceGenerator):
    """Generator that produces multiple alternatives and selects best."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        global_context: GlobalContext,
        temperature: float = 0.8,
        num_alternatives: int = 3,
        style_profile: Optional[StyleProfile] = None,
        indexer: Optional["CorpusIndexer"] = None
    ):
        """Initialize multi-pass generator.

        Args:
            llm_provider: LLM provider.
            global_context: Global context.
            temperature: Base temperature.
            num_alternatives: Number of alternatives per sentence.
            style_profile: Target author's style profile.
            indexer: Optional corpus indexer for RAG.
        """
        super().__init__(llm_provider, global_context, temperature, style_profile, indexer)
        self.num_alternatives = num_alternatives
        self.multi_prompt_builder = MultiSentencePromptBuilder(global_context, style_profile, indexer)

    def generate_sentence(
        self,
        node: SentenceNode,
        plan: SentencePlan,
        previous_sentence: Optional[str] = None
    ) -> GeneratedSentence:
        """Generate sentence with alternatives and select best.

        Args:
            node: Sentence specification.
            plan: Full plan for context.
            previous_sentence: Previous sentence.

        Returns:
            Best GeneratedSentence.
        """
        # Generate alternatives
        alternatives = self._generate_alternatives(node, previous_sentence)

        if not alternatives:
            # Fallback to single generation
            return super().generate_sentence(node, plan, previous_sentence)

        # Score and select best
        best_text = self._select_best(alternatives, node)

        word_count = len(best_text.split())
        return GeneratedSentence(
            text=best_text,
            node_id=node.id,
            word_count=word_count,
            target_length=node.target_length,
            alternatives=alternatives
        )

    def _generate_alternatives(
        self,
        node: SentenceNode,
        previous_sentence: Optional[str]
    ) -> List[str]:
        """Generate multiple sentence alternatives.

        Args:
            node: Sentence specification.
            previous_sentence: Previous sentence.

        Returns:
            List of alternative sentences.
        """
        prompt = self.multi_prompt_builder.build_alternatives_prompt(
            node, previous_sentence, self.num_alternatives
        )

        response = self.llm_provider.call(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            temperature=self.temperature,
            max_tokens=300
        )

        # Parse numbered alternatives
        alternatives = []
        lines = response.strip().split('\n')
        for line in lines:
            # Remove numbering (1. or 1: or 1)
            cleaned = re.sub(r'^\d+[\.\):\s]+', '', line.strip())
            if cleaned:
                alternatives.append(self._clean_sentence(cleaned))

        return alternatives[:self.num_alternatives]

    def _select_best(
        self,
        alternatives: List[str],
        node: SentenceNode
    ) -> str:
        """Select best alternative using scoring.

        Args:
            alternatives: Candidate sentences.
            node: Sentence specification.

        Returns:
            Best sentence text.
        """
        if len(alternatives) == 1:
            return alternatives[0]

        # Score by length accuracy
        best_score = -1
        best_text = alternatives[0]

        for alt in alternatives:
            word_count = len(alt.split())
            diff = abs(word_count - node.target_length)
            score = 1.0 - (diff / max(node.target_length, 1))

            # Bonus for containing keywords
            keyword_bonus = sum(
                0.1 for kw in node.keywords
                if kw.lower() in alt.lower()
            )
            score += keyword_bonus

            if score > best_score:
                best_score = score
                best_text = alt

        return best_text


class SessionBasedGenerator(SentenceGenerator):
    """Generator that maintains conversation context with LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        global_context: GlobalContext,
        temperature: float = 0.7,
        style_profile: Optional[StyleProfile] = None,
        indexer: Optional["CorpusIndexer"] = None
    ):
        """Initialize session-based generator.

        Args:
            llm_provider: LLM provider.
            global_context: Global context.
            temperature: Generation temperature.
            style_profile: Target author's style profile.
            indexer: Optional corpus indexer for RAG.
        """
        super().__init__(llm_provider, global_context, temperature, style_profile, indexer)
        self._session = None

    def start_document_session(self) -> None:
        """Start a new session for document generation."""
        self._session = LLMSession(
            provider=self.llm_provider,
            system_prompt=self.global_context.to_system_prompt()
        )
        logger.info("Started document generation session")

    def generate_paragraph(
        self,
        plan: SentencePlan,
        paragraph_context: ParagraphContext
    ) -> GeneratedParagraph:
        """Generate paragraph within session context.

        Args:
            plan: Sentence plan.
            paragraph_context: Paragraph context.

        Returns:
            GeneratedParagraph.
        """
        if not self._session:
            self.start_document_session()

        # Add paragraph context to session
        context_prompt = paragraph_context.to_prompt_section()
        self._session.add_assistant_response(f"[Context: {context_prompt}]")

        # Generate sentences maintaining session history
        sentences = []
        previous_sentence = None

        for node in plan.nodes:
            prompt = self.prompt_builder.build_sentence_prompt(
                node, plan, previous_sentence
            )

            response = self._session.send(
                prompt.user_prompt,
                temperature=self.temperature
            )

            text = self._clean_sentence(response)
            word_count = len(text.split())

            sentences.append(GeneratedSentence(
                text=text,
                node_id=node.id,
                word_count=word_count,
                target_length=node.target_length
            ))
            previous_sentence = text

        return GeneratedParagraph(
            sentences=sentences,
            plan=plan,
            paragraph_idx=paragraph_context.paragraph_idx
        )

    def end_document_session(self) -> None:
        """End the document session."""
        self._session = None
        logger.info("Ended document generation session")
