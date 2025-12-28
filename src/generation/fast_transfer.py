"""Fast style transfer pipeline using LoRA.

This module provides a simplified style transfer pipeline that uses
LoRA-adapted models for fast, consistent style transfer.

Pipeline:
1. Extract propositions (what to say)
2. Generate styled text (single LoRA call)
3. Verify meaning preserved (lightweight check)
4. Optional single repair pass

Target: <30 seconds per paragraph (vs 10+ minutes with evolutionary approach).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import time

from .lora_generator import LoRAStyleGenerator, GenerationConfig
from ..utils.nlp import (
    split_into_paragraphs,
    split_into_sentences,
    filter_headings,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransferConfig:
    """Configuration for fast style transfer."""

    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Verification settings
    verify_entailment: bool = True
    entailment_threshold: float = 0.7

    # Quality critic settings
    use_quality_critic: bool = True  # Enable quality checking with explicit fix instructions
    word_cluster_threshold: int = 3  # Words used 3+ times trigger warning

    # Repair settings
    max_repair_attempts: int = 2  # Allow 2 repair attempts with critic feedback
    repair_temperature: float = 0.5  # Lower temp for repair

    # Post-processing settings
    reduce_repetition: bool = True
    repetition_threshold: int = 3  # Words used 3+ times get replaced

    # Content extraction
    skip_headings: bool = True
    min_paragraph_words: int = 10  # Skip very short paragraphs


@dataclass
class TransferStats:
    """Statistics from a transfer operation."""

    paragraphs_processed: int = 0
    paragraphs_repaired: int = 0
    quality_issues_found: int = 0
    quality_issues_fixed: int = 0
    words_replaced: int = 0
    total_time_seconds: float = 0.0
    avg_time_per_paragraph: float = 0.0
    entailment_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paragraphs_processed": self.paragraphs_processed,
            "paragraphs_repaired": self.paragraphs_repaired,
            "words_replaced": self.words_replaced,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_time_per_paragraph": round(self.avg_time_per_paragraph, 2),
            "avg_entailment_score": round(
                sum(self.entailment_scores) / len(self.entailment_scores), 3
            ) if self.entailment_scores else 0.0,
        }


class PropositionExtractor:
    """Extract semantic propositions from text.

    Uses lightweight heuristics for speed:
    - Split into sentences
    - Extract key claims
    - Identify entities and relationships
    """

    def __init__(self):
        """Initialize the extractor."""
        pass

    def extract(self, text: str) -> List[str]:
        """Extract propositions from text.

        Args:
            text: Input text.

        Returns:
            List of proposition strings.
        """
        sentences = split_into_sentences(text)

        if not sentences:
            return [text] if text.strip() else []

        # For now, use sentences as propositions
        # Future: use semantic parsing or LLM extraction
        propositions = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 3:  # Skip very short sentences
                propositions.append(sent)

        return propositions

    def extract_with_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract propositions and keywords.

        Args:
            text: Input text.

        Returns:
            Tuple of (propositions, keywords).
        """
        propositions = self.extract(text)
        keywords = extract_keywords(text, top_n=10)
        return propositions, keywords


class FastStyleTransfer:
    """Simplified style transfer using LoRA.

    This is the main entry point for fast style transfer. It replaces
    the complex evolutionary pipeline with a streamlined approach:

    1. Extract propositions (rule-based or LLM)
    2. Single LoRA generation pass
    3. Lightweight verification
    4. Optional single repair pass

    Example usage:
        transfer = FastStyleTransfer(
            adapter_path="lora_adapters/sagan",
            author_name="Carl Sagan",
        )

        result = transfer.transfer_document(input_text)
        print(result)
    """

    def __init__(
        self,
        adapter_path: Optional[str],
        author_name: str,
        config: Optional[TransferConfig] = None,
        verify_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the fast transfer pipeline.

        Args:
            adapter_path: Path to LoRA adapter directory, or None for base model.
            author_name: Author name for prompts.
            config: Transfer configuration.
            verify_fn: Optional verification function (original, output) -> score.
        """
        self.config = config or TransferConfig()
        self.author = author_name
        self.verify_fn = verify_fn

        # Initialize generator
        gen_config = GenerationConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        self.generator = LoRAStyleGenerator(
            adapter_path=adapter_path,
            config=gen_config,
        )

        # Initialize proposition extractor
        self.prop_extractor = PropositionExtractor()

        # Initialize repetition reducer for post-processing
        self.repetition_reducer = None
        if self.config.reduce_repetition:
            from ..vocabulary.repetition_reducer import RepetitionReducer
            self.repetition_reducer = RepetitionReducer(
                threshold=self.config.repetition_threshold
            )

        # Initialize quality critic for explicit fix instructions
        self.quality_critic = None
        if self.config.use_quality_critic:
            from ..validation.quality_critic import QualityCritic
            self.quality_critic = QualityCritic(
                cluster_threshold=self.config.word_cluster_threshold
            )

        # Set up entailment verifier if requested
        if self.config.verify_entailment and self.verify_fn is None:
            self.verify_fn = self._create_default_verifier()

    def _create_default_verifier(self) -> Callable[[str, str], float]:
        """Create default entailment verifier."""
        try:
            import sys
            import warnings
            import logging
            from io import StringIO
            from sentence_transformers import CrossEncoder

            # Suppress all output during model loading (position_ids mismatch report)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Suppress transformers logging
                transformers_logger = logging.getLogger("transformers")
                old_level = transformers_logger.level
                transformers_logger.setLevel(logging.ERROR)
                # Suppress stdout/stderr during loading (for LOAD REPORT)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                try:
                    model = CrossEncoder(
                        "cross-encoder/nli-deberta-v3-small",
                        max_length=512,
                    )
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    transformers_logger.setLevel(old_level)

            def verify(original: str, output: str) -> float:
                """Verify content preservation via entailment."""
                # Check if output entails original content
                scores = model.predict([(original, output)])
                # Model returns [contradiction, neutral, entailment]
                if len(scores.shape) > 1:
                    return float(scores[0][2])  # Entailment score
                return float(scores[0])

            logger.info("Using NLI-based entailment verification")
            return verify

        except ImportError:
            logger.warning(
                "sentence-transformers not available, using similarity fallback"
            )
            return self._similarity_verifier

    def _similarity_verifier(self, original: str, output: str) -> float:
        """Fallback verifier using keyword overlap."""
        from ..utils.nlp import compute_semantic_similarity

        return compute_semantic_similarity(original, output)

    def transfer_paragraph(
        self,
        paragraph: str,
        previous: Optional[str] = None,
        stats: Optional['TransferStats'] = None,
    ) -> Tuple[str, float]:
        """Transfer a single paragraph with quality checking and repair.

        Args:
            paragraph: Source paragraph.
            previous: Previous output paragraph for continuity.
            stats: Optional stats object to update.

        Returns:
            Tuple of (styled_paragraph, entailment_score).
        """
        # Skip very short paragraphs
        if len(paragraph.split()) < self.config.min_paragraph_words:
            logger.debug(f"Skipping short paragraph: {paragraph[:50]}...")
            return paragraph, 1.0

        word_count = len(paragraph.split())
        logger.debug(f"Translating paragraph: {word_count} words")

        # Generate with token limit based on input (allow 1.5x for style variation)
        max_tokens = max(100, int(word_count * 1.8))

        output = self.generator.generate(
            content=paragraph,
            author=self.author,
            context=previous,
            max_tokens=max_tokens,
        )

        # Quality critique and repair loop
        if self.quality_critic:
            for attempt in range(self.config.max_repair_attempts):
                critique = self.quality_critic.critique(paragraph, output)

                if stats:
                    stats.quality_issues_found += len(critique.issues)

                if not critique.has_critical_issues:
                    logger.debug(f"Quality check passed (attempt {attempt + 1})")
                    break

                # Log issues
                for issue in critique.issues:
                    if issue.severity == "critical":
                        logger.warning(f"Quality issue: {issue.description}")

                # Generate repair prompt with explicit fix instructions
                repair_prompt = self.quality_critic.get_repair_system_prompt(
                    self.author, critique
                )

                logger.info(f"Repair attempt {attempt + 1}: {len(critique.issues)} issues")

                # Lower temperature for repair
                old_temp = self.generator.config.temperature
                self.generator.config.temperature = self.config.repair_temperature

                try:
                    output = self.generator.generate(
                        content=paragraph,
                        author=self.author,
                        context=previous,
                        system_override=repair_prompt,
                        max_tokens=max_tokens,
                    )
                finally:
                    self.generator.config.temperature = old_temp

                if stats:
                    stats.quality_issues_fixed += 1

        # Verify if configured
        score = 1.0
        if self.verify_fn:
            score = self.verify_fn(paragraph, output)

        return output, score

    def _repair(
        self,
        original: str,
        current: str,
        propositions: List[str],
        previous: Optional[str],
    ) -> Tuple[str, float]:
        """Single repair pass for meaning preservation.

        Args:
            original: Original paragraph.
            current: Current (failed) output.
            propositions: Extracted propositions.
            previous: Previous paragraph.

        Returns:
            Tuple of (repaired_output, new_score).
        """
        logger.info("Attempting repair...")

        # Generate with lower temperature and stricter prompt
        old_temp = self.generator.config.temperature
        self.generator.config.temperature = self.config.repair_temperature

        # Use stricter system prompt for repair
        repair_system = (
            f"You are a translator who converts text into {self.author}'s writing style. "
            "CRITICAL: Translate the text faithfully. Do NOT:\n"
            "- Add new ideas, examples, or explanations\n"
            "- Expand or elaborate beyond the original\n"
            "- Change the meaning or add interpretations\n"
            "Output ONLY the translated text, nothing else."
        )

        try:
            # Estimate tokens based on input
            input_words = len(original.split())
            max_tokens = max(100, int(input_words * 2.0))

            output = self.generator.generate(
                content=original,
                author=self.author,
                context=previous,
                system_override=repair_system,
                max_tokens=max_tokens,
            )
        finally:
            self.generator.config.temperature = old_temp

        # Re-verify
        score = self.verify_fn(original, output) if self.verify_fn else 1.0
        logger.info(f"Repair result: score={score:.2f}")

        return output, score

    def transfer_document(
        self,
        text: str,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[str, TransferStats]:
        """Transfer an entire document.

        Args:
            text: Source document text.
            on_progress: Optional callback (current, total, status).

        Returns:
            Tuple of (styled_document, statistics).
        """
        start_time = time.time()
        stats = TransferStats()

        # Reset repetition reducer for new document
        if self.repetition_reducer:
            self.repetition_reducer.reset()

        # Split into paragraphs
        paragraphs = split_into_paragraphs(text)

        # Filter headings if configured
        if self.config.skip_headings:
            paragraphs = filter_headings(paragraphs)

        if not paragraphs:
            logger.warning("No content paragraphs found")
            return text, stats

        logger.info(f"Transferring {len(paragraphs)} paragraphs")

        outputs = []
        previous = None

        for i, para in enumerate(paragraphs):
            if on_progress:
                on_progress(i + 1, len(paragraphs), f"Processing paragraph {i+1}")

            para_start = time.time()

            output, score = self.transfer_paragraph(para, previous, stats)

            # Apply repetition reduction
            if self.repetition_reducer:
                output, reduction_stats = self.repetition_reducer.reduce(output)
                stats.words_replaced += reduction_stats.replacements_made

            para_time = time.time() - para_start
            logger.debug(f"Paragraph {i+1}: {para_time:.1f}s, score={score:.2f}")

            outputs.append(output)
            previous = output

            stats.paragraphs_processed += 1
            stats.entailment_scores.append(score)

            if score < self.config.entailment_threshold:
                stats.paragraphs_repaired += 1

        # Compute final stats
        stats.total_time_seconds = time.time() - start_time
        stats.avg_time_per_paragraph = (
            stats.total_time_seconds / stats.paragraphs_processed
            if stats.paragraphs_processed > 0 else 0
        )

        # Log repetition reduction summary
        if self.repetition_reducer and stats.words_replaced > 0:
            overused = self.repetition_reducer.get_overused_words(limit=5)
            if overused:
                logger.info(
                    f"Repetition reduction: {stats.words_replaced} replacements, "
                    f"top overused: {', '.join(w for w, _ in overused)}"
                )

        logger.info(
            f"Transfer complete: {stats.paragraphs_processed} paragraphs in "
            f"{stats.total_time_seconds:.1f}s "
            f"(avg {stats.avg_time_per_paragraph:.1f}s/para)"
        )

        return "\n\n".join(outputs), stats

    def switch_author(self, adapter_path: str, author_name: str) -> None:
        """Switch to a different author.

        Args:
            adapter_path: Path to new adapter.
            author_name: New author name.
        """
        self.generator.switch_adapter(adapter_path)
        self.author = author_name
        logger.info(f"Switched to author: {author_name}")


def create_fast_transfer(
    adapter_path: str,
    author_name: str,
    verify: bool = True,
    temperature: float = 0.7,
) -> FastStyleTransfer:
    """Convenience function to create a fast transfer pipeline.

    Args:
        adapter_path: Path to LoRA adapter.
        author_name: Author name.
        verify: Whether to enable entailment verification.
        temperature: Generation temperature.

    Returns:
        Configured FastStyleTransfer instance.
    """
    config = TransferConfig(
        verify_entailment=verify,
        temperature=temperature,
    )

    return FastStyleTransfer(
        adapter_path=adapter_path,
        author_name=author_name,
        config=config,
    )
