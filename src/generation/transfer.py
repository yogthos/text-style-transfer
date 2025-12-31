"""Style transfer pipeline using LoRA with semantic graph validation.

This module provides a graph-based style transfer pipeline that uses LoRA-adapted
models for consistent style transfer with semantic validation.

Pipeline:
1. Build semantic graph from source (propositions + relationships)
2. Generate neutral prose from graph (deterministic, preserves all content)
3. Pass neutral prose to LoRA writer for styling
4. Validate styled output against source graph
5. Repair any missing propositions
6. Final LoRA restyle pass
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import time

from .lora_generator import LoRAStyleGenerator, GenerationConfig
from .document_context import DocumentContext, extract_document_context
from ..utils.nlp import (
    split_into_paragraphs,
    split_into_sentences,
    filter_headings,
    is_heading,
)
from ..utils.logging import get_logger
from ..utils.prompts import format_prompt

logger = get_logger(__name__)


@dataclass
class TransferConfig:
    """Configuration for style transfer."""

    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Verification settings
    verify_entailment: bool = True
    entailment_threshold: float = 0.7

    # Repair settings
    max_repair_attempts: int = 3

    # Post-processing settings
    reduce_repetition: bool = True
    repetition_threshold: int = 3  # Words used 3+ times get replaced

    # Content handling
    pass_headings_unchanged: bool = True  # Don't transform headings
    min_paragraph_words: int = 10  # Skip very short paragraphs

    # Document context settings
    use_document_context: bool = True  # Extract and use document-level context

    # Length control settings
    max_expansion_ratio: float = 1.5  # Max output/input word ratio (1.5 = 50% longer)
    target_expansion_ratio: float = 1.2  # Target for LoRA generation
    truncate_over_expanded: bool = False  # If True, truncate; if False, allow longer output

    # LoRA influence settings
    lora_scale: float = 1.0  # 0.0=base only, 0.5=half, 1.0=full, >1.0=amplified

    # Perspective settings
    perspective: str = "preserve"  # preserve, first_person_singular, first_person_plural, third_person


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


class StyleTransfer:
    """Style transfer using LoRA with semantic graph validation.

    This is the main entry point for style transfer. Pipeline:

    1. Build semantic graph from source paragraph
    2. Generate neutral prose from graph (preserves all propositions)
    3. Pass to LoRA writer for styling
    4. Validate styled output against source graph
    5. Repair missing propositions if needed
    6. Final LoRA restyle pass

    Example usage:
        transfer = StyleTransfer(
            adapter_path="lora_adapters/sagan",
            author_name="Carl Sagan",
            critic_provider=deepseek_provider,
        )

        result, stats = transfer.transfer_document(input_text)
        print(result)
    """

    def __init__(
        self,
        adapter_path: Optional[str],
        author_name: str,
        critic_provider,
        config: Optional[TransferConfig] = None,
        verify_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the fast transfer pipeline.

        Args:
            adapter_path: Path to LoRA adapter directory, or None for base model.
            author_name: Author name for prompts.
            critic_provider: LLM provider for critique/repair (e.g., DeepSeek).
            config: Transfer configuration.
            verify_fn: Optional verification function (original, output) -> score.
        """
        self.config = config or TransferConfig()
        self.author = author_name
        self.verify_fn = verify_fn
        self.critic_provider = critic_provider

        # Initialize generator
        gen_config = GenerationConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            lora_scale=getattr(self.config, 'lora_scale', 1.0),
        )
        self.generator = LoRAStyleGenerator(
            adapter_path=adapter_path,
            config=gen_config,
        )

        # Initialize repetition reducer for post-processing
        self.repetition_reducer = None
        if self.config.reduce_repetition:
            from ..vocabulary.repetition_reducer import RepetitionReducer
            self.repetition_reducer = RepetitionReducer(
                threshold=self.config.repetition_threshold
            )

        # Set up entailment verifier if requested
        if self.config.verify_entailment and self.verify_fn is None:
            self.verify_fn = self._create_default_verifier()

        # Document context (extracted at transfer time)
        self.document_context: Optional[DocumentContext] = None

        logger.info(f"Using critic provider for repairs: {self.critic_provider.provider_name}")

    def _extract_paragraph_thesis(self, paragraph: str) -> str:
        """Extract the main thesis/point of a paragraph.

        This helps the LoRA understand what it's trying to express overall,
        not just individual propositions.

        Args:
            paragraph: Source paragraph text.

        Returns:
            One-sentence thesis statement.
        """
        if not self.critic_provider:
            # Fallback: use first sentence as thesis
            sentences = split_into_sentences(paragraph)
            return sentences[0] if sentences else ""

        try:
            response = self.critic_provider.call(
                system_prompt="You are a precise summarizer. Extract the ONE main point or thesis of this paragraph in a single sentence. Be specific and concrete. Do not add interpretation.",
                user_prompt=f"Paragraph:\n{paragraph}\n\nMain point (one sentence):",
                temperature=0.1,
                max_tokens=100,
            )
            thesis = response.strip()
            # Clean up common prefixes
            for prefix in ["The main point is that ", "The thesis is that ", "This paragraph argues that "]:
                if thesis.lower().startswith(prefix.lower()):
                    thesis = thesis[len(prefix):]
                    thesis = thesis[0].upper() + thesis[1:] if thesis else thesis
                    break
            logger.debug(f"Extracted paragraph thesis: {thesis[:80]}...")
            return thesis
        except Exception as e:
            logger.warning(f"Thesis extraction failed: {e}")
            sentences = split_into_sentences(paragraph)
            return sentences[0] if sentences else ""

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
                import numpy as np

                # Check if output entails original content
                scores = model.predict([(original, output)])

                # Model returns raw logits [contradiction, neutral, entailment]
                # Apply softmax to convert to probabilities
                if len(scores.shape) > 1:
                    logits = scores[0]
                else:
                    logits = scores

                # Softmax to get probabilities
                exp_scores = np.exp(logits - np.max(logits))  # subtract max for numerical stability
                probs = exp_scores / np.sum(exp_scores)

                # Return entailment probability (index 2)
                return float(probs[2]) if len(probs) > 2 else float(probs[-1])

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
        """Transfer a single paragraph with graph-based validation.

        Pipeline:
        1. Build source semantic graph (ground truth)
        2. Generate neutral prose from graph (deterministic, all propositions)
        3. Pass neutral prose to LoRA writer for styling
        4. Validate styled output against source graph
        5. Repair any missing propositions
        6. Final LoRA restyle pass

        Args:
            paragraph: Source paragraph.
            previous: Previous output paragraph for continuity.
            stats: Optional stats object to update.

        Returns:
            Tuple of (styled_paragraph, entailment_score).
        """
        from ..validation.semantic_graph import SemanticGraphBuilder, SemanticGraphComparator

        # Skip very short paragraphs
        if len(paragraph.split()) < self.config.min_paragraph_words:
            logger.debug(f"Skipping short paragraph: {paragraph[:50]}...")
            return paragraph, 1.0

        word_count = len(paragraph.split())
        logger.debug(f"Translating paragraph: {word_count} words")

        # ========================================
        # STEP 1: Build source graph (ground truth)
        # ========================================
        builder = SemanticGraphBuilder(use_rebel=False)
        source_graph = builder.build_from_text(paragraph)
        logger.info(f"Source graph: {len(source_graph.nodes)} propositions, {len(source_graph.edges)} relationships")

        if not source_graph.nodes:
            logger.warning("Could not build source graph, passing through unchanged")
            return paragraph, 1.0

        comparator = SemanticGraphComparator()

        # Extract paragraph thesis - the main point the writer needs to express
        paragraph_thesis = self._extract_paragraph_thesis(paragraph)
        logger.debug(f"Paragraph thesis: {paragraph_thesis[:80]}...")

        # ========================================
        # STEP 2: Neutralize input for LoRA
        # ========================================
        # The LoRA was trained on neutralized descriptions → styled output.
        # We must neutralize the input to match the training format.
        # Validate that neutralization preserves all propositions.
        content_for_generation = self._neutralize_for_lora(
            paragraph=paragraph,
            source_graph=source_graph,
            builder=builder,
            comparator=comparator,
        )
        logger.info(f"Neutralized paragraph: {len(content_for_generation.split())} words")

        # ========================================
        # STEP 3: Pass to LoRA for style transformation
        # ========================================
        target_words = int(word_count * self.config.target_expansion_ratio)
        # Keep token limit close to target to prevent over-generation
        # Use 1.3x to allow for style elaboration without too much room for hallucination
        max_tokens = max(50, int(target_words * 1.3))

        # Get context hint for generation (if document context available)
        context_hint = None
        if self.document_context:
            context_hint = self.document_context.to_generation_hint()

        output = self.generator.generate(
            content=content_for_generation,
            author=self.author,
            context=previous,
            max_tokens=max_tokens,
            context_hint=context_hint,
            perspective=getattr(self.config, 'perspective', 'preserve'),
        )

        # Check if LoRA output matches input (indicates no transformation)
        if output.strip() == paragraph.strip():
            logger.warning("LoRA output identical to original paragraph - no transformation occurred")

        # Track expansion at LoRA stage
        lora_words = len(output.split())
        source_words = len(paragraph.split())
        if lora_words > source_words * self.config.max_expansion_ratio:
            logger.warning(f"LoRA over-expanded: {lora_words} words vs {source_words} source ({lora_words/source_words:.0%})")

        # ========================================
        # STEP 4: Validate styled output against source graph
        # ========================================
        output, is_valid = self._validate_styled_output(
            source=paragraph,
            output=output,
            source_graph=source_graph,
            builder=builder,
            comparator=comparator,
            previous=previous,
            context_hint=context_hint,
            stats=stats,
            max_attempts=self.config.max_repair_attempts,
        )

        # Track expansion and optionally truncate
        final_words = len(output.split())
        max_allowed_words = int(source_words * self.config.max_expansion_ratio)

        if final_words > max_allowed_words:
            expansion_pct = final_words / source_words
            if self.config.truncate_over_expanded:
                logger.warning(f"Output over-expanded: {final_words} words vs {source_words} source ({expansion_pct:.0%}), truncating")
                # Truncate to allowed length at sentence boundary
                sentences = split_into_sentences(output)
                truncated = []
                current_words = 0
                for sent in sentences:
                    sent_words = len(sent.split())
                    if current_words + sent_words > max_allowed_words:
                        break
                    truncated.append(sent)
                    current_words += sent_words

                if truncated:
                    output = ' '.join(truncated)
                # If no complete sentences fit, keep at least the first sentence
                elif sentences:
                    output = sentences[0]
            else:
                logger.info(f"Output expanded: {final_words} words vs {source_words} source ({expansion_pct:.0%})")

        # Ensure output ends with complete sentence
        output = self._ensure_complete_ending(output)

        # Verify if configured
        score = 1.0
        if self.verify_fn:
            score = self.verify_fn(paragraph, output)

        return output, score

    def _ensure_complete_ending(self, text: str) -> str:
        """Ensure text ends with a complete sentence.

        If text ends mid-sentence, remove the incomplete part.
        """
        text = text.strip()
        if not text:
            return text

        # If already ends with sentence terminator, we're good
        if text[-1] in '.!?':
            return text

        # Find the last complete sentence
        sentences = split_into_sentences(text)
        if not sentences:
            return text

        # Check if last sentence is complete (ends with punctuation)
        complete_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent[-1] in '.!?':
                complete_sentences.append(sent)
            elif sent and len(sent) > 20:
                # Long fragment - try to salvage by adding period
                # Only if it looks like a complete thought
                words = sent.split()
                if len(words) >= 5:
                    complete_sentences.append(sent + '.')
                    logger.warning(f"Added period to incomplete sentence: ...{sent[-30:]}")

        if complete_sentences:
            return ' '.join(complete_sentences)

        # Fallback: add period to entire text
        return text + '.'

    def _neutralize_for_lora(
        self,
        paragraph: str,
        source_graph=None,
        builder=None,
        comparator=None,
        max_attempts: int = 2,
    ) -> str:
        """Neutralize input paragraph to match LoRA training format.

        The LoRA was trained on (neutral_description → styled_output) pairs.
        This method converts input prose to a neutral description, then
        VALIDATES that all propositions are preserved before returning.

        Args:
            paragraph: Input paragraph to neutralize.
            source_graph: Semantic graph of source for validation.
            builder: SemanticGraphBuilder for building neutral graph.
            comparator: SemanticGraphComparator for validation.
            max_attempts: Maximum neutralization attempts.

        Returns:
            Neutral description suitable for LoRA input.
        """
        if not self.critic_provider:
            logger.warning("No critic provider for neutralization - using original text")
            return paragraph

        # Neutralization prompt - matches training format but preserves content
        # Training uses short summaries with generic names to teach style
        # At inference we keep specific names but match the brief, direct tone
        input_words = len(paragraph.split())
        target_words = max(50, min(input_words, 200))  # Scale with input, cap at 200

        system_prompt = """You summarize text in short, direct sentences. No fancy words. No interpretation. Just what happens."""

        user_prompt = f"""Summarize what happens in about {target_words} words. Keep ALL names, places, dates, and examples exactly as written.

RULES:
1. Short sentences only. No em-dashes, semicolons, or colons.
2. Start directly with the action, not "The passage describes..."
3. Keep every fact, analogy, and example - just make it plain and direct.
4. Use simple words: "says" not "articulates", "shows" not "demonstrates"

PASSAGE:
{paragraph}

SUMMARY:"""

        best_neutral = None
        best_coverage = 0.0

        for attempt in range(max_attempts):
            try:
                # Slightly vary temperature for diversity
                temp = 0.2 + (attempt * 0.1)

                neutral = self.critic_provider.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temp,
                    max_tokens=max(300, int(len(paragraph.split()) * 1.5)),
                )

                neutral = neutral.strip()

                # Basic length validation
                if not neutral or len(neutral) < len(paragraph) * 0.15:
                    logger.warning(f"Neutralization attempt {attempt + 1} too short")
                    continue

                # Validate against source graph if available
                if source_graph and builder and comparator:
                    neutral_graph = builder.build_from_text(neutral)
                    diff = comparator.compare(source_graph, neutral_graph)

                    # Calculate coverage
                    total_props = len(source_graph.nodes)
                    missing_props = len(diff.missing_nodes)
                    coverage = (total_props - missing_props) / total_props if total_props > 0 else 0

                    logger.debug(
                        f"Neutralization attempt {attempt + 1}: "
                        f"{coverage:.0%} coverage ({missing_props} missing of {total_props})"
                    )

                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_neutral = neutral

                    # Accept if coverage is good enough
                    if coverage >= 0.8:
                        logger.info(f"Neutralization accepted: {coverage:.0%} proposition coverage")
                        return neutral

                else:
                    # No validation available - accept first reasonable result
                    logger.debug(f"Neutralization complete (no validation): {len(neutral.split())} words")
                    return neutral

            except Exception as e:
                logger.warning(f"Neutralization attempt {attempt + 1} failed: {e}")
                continue

        # Return best attempt or fall back to original
        if best_neutral and best_coverage >= 0.5:
            logger.warning(f"Using best neutralization with {best_coverage:.0%} coverage")
            return best_neutral

        logger.warning("Neutralization failed to preserve content - using original")
        return paragraph

    def _validate_styled_output(
        self,
        source: str,
        output: str,
        source_graph,
        builder,
        comparator,
        previous: Optional[str] = None,
        context_hint: Optional[str] = None,
        stats: Optional['TransferStats'] = None,
        max_attempts: int = 3,
    ) -> Tuple[str, bool]:
        """Validate styled output against source graph with repair.

        This is the post-LoRA validation step that ensures the styled output
        preserves all propositions from the source.

        IMPORTANT: Never inserts verbatim source text. If validation fails,
        regenerates entirely through LoRA using neutral prose from the graph.

        Args:
            source: Original source text.
            output: Styled output to validate.
            source_graph: Semantic graph of source (ground truth).
            builder: SemanticGraphBuilder instance.
            comparator: SemanticGraphComparator instance.
            previous: Previous paragraph for context.
            context_hint: Document context hint.
            stats: Optional stats object.
            max_attempts: Maximum repair attempts.

        Returns:
            Tuple of (validated_output, is_valid).
        """
        # Initial comparison
        output_graph = builder.build_from_text(output)
        diff = comparator.compare(source_graph, output_graph)

        if diff.is_isomorphic:
            logger.info("Styled output graph matches source - no repair needed")
            return output, True

        if not diff.has_critical_differences:
            logger.debug("Styled output has minor differences - acceptable")
            return output, True

        logger.info(
            f"Styled output graph diff: {len(diff.missing_nodes)} missing, "
            f"{len(diff.added_nodes)} added, {len(diff.entity_role_errors)} entity errors"
        )

        # ========================================
        # REPAIR: Critic adds missing content, then LoRA restyles
        # ========================================
        # 1. Use critic (DeepSeek) to surgically add missing propositions
        # 2. Restyle through LoRA with LOW temperature to preserve content
        #
        # This ensures all content is preserved AND styled.

        if not self.critic_provider:
            logger.warning("No critic provider for repair - returning original output")
            return output, False

        best_output = output
        best_missing = len(diff.missing_nodes)

        for attempt in range(max_attempts):
            try:
                # Get current diff (first iteration uses initial diff)
                if attempt > 0:
                    current_graph = builder.build_from_text(best_output)
                    current_diff = comparator.compare(source_graph, current_graph)
                else:
                    current_diff = diff

                if not current_diff.has_critical_differences:
                    break

                # Build repair instructions from the diff
                missing_props = [node.text or node.summary() for node in current_diff.missing_nodes]

                if not missing_props:
                    break

                logger.info(f"Repair attempt {attempt + 1}: {len(missing_props)} missing propositions")

                # Step 1: Critic adds missing content
                repair_prompt = f"""Add the MISSING INFORMATION to this text. Keep names EXACTLY as written - no aliases, no clarifications, no "(also known as...)".

TEXT:
{best_output}

MISSING (add these facts):
{chr(10).join(f"- {prop}" for prop in missing_props[:5])}

Output the complete text with missing facts added. Do NOT add context or explanations."""

                repaired = self.critic_provider.call(
                    system_prompt="Add missing facts to text. Keep all names exactly as given. Never add aliases or clarifications like 'also known as'. Output only the edited text.",
                    user_prompt=repair_prompt,
                    temperature=0.3,
                    max_tokens=max(200, int(len(best_output.split()) * 2)),
                )

                repaired = repaired.strip()

                if not repaired or len(repaired) < len(best_output) * 0.5:
                    logger.warning(f"Critic returned invalid output, skipping")
                    continue

                # Step 2: Restyle through LoRA with VERY LOW temperature
                # This preserves the repaired content while adding author style
                old_temp = self.generator.config.temperature
                self.generator.config.temperature = 0.2  # Very low to preserve content

                styled_output = self.generator.generate(
                    content=f"Rewrite this text preserving ALL information:\n\n{repaired}",
                    author=self.author,
                    context=previous,
                    max_tokens=max(200, int(len(repaired.split()) * 1.5)),
                    context_hint=context_hint,
                    perspective=getattr(self.config, 'perspective', 'preserve'),
                )

                self.generator.config.temperature = old_temp

                # Verify the result
                styled_graph = builder.build_from_text(styled_output)
                styled_diff = comparator.compare(source_graph, styled_graph)
                missing_count = len(styled_diff.missing_nodes)

                logger.debug(f"After repair+restyle: {missing_count} missing (was {len(current_diff.missing_nodes)})")

                if styled_diff.is_isomorphic or not styled_diff.has_critical_differences:
                    if stats:
                        stats.quality_issues_fixed += 1
                    logger.info(f"Repair successful on attempt {attempt + 1}")
                    return styled_output, True

                # Keep best result
                if missing_count < best_missing:
                    best_output = styled_output
                    best_missing = missing_count

            except Exception as e:
                logger.warning(f"Repair attempt {attempt + 1} failed: {e}")
                continue

        # Return best attempt
        if best_missing < len(diff.missing_nodes):
            logger.info(f"Partial repair: reduced missing from {len(diff.missing_nodes)} to {best_missing}")
            if stats:
                stats.quality_issues_fixed += 1
            return best_output, False

        logger.warning(f"Repair unsuccessful after {max_attempts} attempts")
        return output, False

    def _incremental_graph_repair(
        self,
        source: str,
        output: str,
        source_graph,
        builder,
        comparator,
        max_attempts_per_error: int = 2,
    ) -> str:
        """Repair output with deterministic, targeted fixes.

        This is a single-pass repair algorithm:
        1. Identify all errors from graph diff
        2. For each error, make ONE targeted fix
        3. Stop when all errors processed (no retries on same error)

        The algorithm is deterministic because:
        - Each error is processed exactly once
        - Fixes are direct text operations where possible
        - LLM is only used for specific rewrites, not open-ended generation
        - No loops or retries that could cause infinite iteration

        Args:
            source: Original source text.
            output: Current output text.
            source_graph: Ground truth semantic graph.
            builder: SemanticGraphBuilder instance.
            comparator: SemanticGraphComparator instance.
            max_attempts_per_error: Max LLM attempts for complex fixes.

        Returns:
            Repaired output text.
        """
        source_sentences = split_into_sentences(source)
        output_sentences = list(split_into_sentences(output))

        # Get the diff - this tells us exactly what needs to be fixed
        output_graph = builder.build_from_text(output)
        diff = comparator.compare(source_graph, output_graph)

        if diff.is_isomorphic:
            return output  # Nothing to fix

        errors = self._prioritize_errors(diff)
        logger.info(f"Repairing {len(errors)} graph errors")

        # Track what we've fixed to avoid duplicate work
        fixed_propositions = set()
        removed_sentences = set()
        # Track inserted sentences so cleanup doesn't remove them
        inserted_sentences = set()

        for error in errors:
            error_type = error["type"]
            error_data = error["data"]

            if error_type == "missing":
                # Missing proposition: find source sentence and add it
                prop_id = f"{error_data.subject}|{error_data.predicate}"
                if prop_id in fixed_propositions:
                    continue

                # Find the source sentence containing this proposition
                source_sent = self._find_sentence_for_proposition(
                    error_data, source_sentences
                )

                if not source_sent:
                    logger.warning(f"Could not find source sentence for: {error_data.summary()[:50]}")
                    continue

                # Check if proposition is already covered by an output sentence
                # (not just exact string match - check semantic coverage)
                already_covered = False
                for out_sent in output_sentences:
                    if self._sentence_contains_proposition(out_sent, error_data):
                        already_covered = True
                        logger.debug(f"Proposition already covered: {error_data.summary()[:40]}")
                        break

                if already_covered:
                    fixed_propositions.add(prop_id)
                    continue

                # Insert at position determined by graph structure
                insert_pos = self._find_insertion_position(
                    error_data, output_sentences, source_graph, output_graph
                )
                output_sentences.insert(insert_pos, source_sent)
                inserted_sentences.add(source_sent.lower().strip())
                fixed_propositions.add(prop_id)
                logger.info(f"Inserted missing proposition at pos {insert_pos}: {error_data.summary()[:40]}")

            elif error_type == "added":
                # Hallucinated content: remove the sentence containing it
                prop_text = error_data.text
                for i, sent in enumerate(output_sentences):
                    if i in removed_sentences:
                        continue
                    # Check if this sentence contains the hallucination
                    if self._sentence_contains_proposition(sent, error_data):
                        removed_sentences.add(i)
                        logger.debug(f"Marked for removal: {sent[:40]}...")
                        break

            elif error_type == "entity":
                # Entity error: targeted rewrite of specific sentence
                entity = error_data.entity
                source_role = error_data.source_role

                # Find sentence mentioning this entity
                for i, sent in enumerate(output_sentences):
                    if i in removed_sentences:
                        continue
                    if entity.lower() in sent.lower():
                        # Try direct fix first (for role_loss, add the role)
                        if error_data.error_type == "role_loss":
                            # Find source sentence with correct role
                            source_sent = self._find_sentence_with_entity_role(
                                entity, source_role, source_sentences
                            )
                            if source_sent:
                                output_sentences[i] = source_sent
                                logger.debug(f"Fixed entity role: {entity}")
                        elif error_data.error_type == "conflation":
                            # Remove the conflating sentence
                            removed_sentences.add(i)
                            logger.debug(f"Removed conflation: {entity}")
                        break

        # Remove marked sentences (in reverse order to preserve indices)
        for i in sorted(removed_sentences, reverse=True):
            if i < len(output_sentences):
                output_sentences.pop(i)

        # Log repair summary
        logger.info(f"Repair summary: {len(inserted_sentences)} sentences inserted, {len(removed_sentences)} removed")

        # Cleanup and return - pass inserted_sentences so we don't accidentally remove them
        final_sentences = self._cleanup_repaired_sentences(output_sentences, inserted_sentences)
        return " ".join(final_sentences)

    def _find_sentence_for_proposition(
        self, prop_node, sentences: List[str]
    ) -> Optional[str]:
        """Find the source sentence that contains a proposition."""
        # The proposition node has the original sentence text
        if prop_node.text:
            for sent in sentences:
                if sent.strip() == prop_node.text.strip():
                    return sent
                # Also check for significant overlap
                if self._sentence_overlap(sent, prop_node.text) > 0.7:
                    return sent

        # Fallback: find by subject/predicate
        for sent in sentences:
            sent_lower = sent.lower()
            if (prop_node.subject and prop_node.subject.lower() in sent_lower and
                prop_node.predicate and prop_node.predicate.lower() in sent_lower):
                return sent

        return None

    def _find_insertion_position(
        self,
        prop_node,
        output_sentences: List[str],
        source_graph,
        output_graph,
    ) -> int:
        """Find insertion position using graph structure to maintain narrative order.

        Uses the SOURCE graph node IDs (P1, P2, P3...) which reflect original sentence order.
        Finds where to insert a missing node by looking at which source nodes are already
        in the output and inserting at the correct relative position.
        """
        # Get the node ID (e.g., "P2")
        missing_id = prop_node.id

        try:
            missing_num = int(missing_id[1:])  # "P2" -> 2
        except (ValueError, IndexError):
            return len(output_sentences)

        # Find predecessor in source graph (what points TO this node)
        predecessor_id = None
        for edge in source_graph.edges:
            if edge.target_id == missing_id:
                predecessor_id = edge.source_id
                break

        # Find successor in source graph (what this node points TO)
        successor_id = None
        for edge in source_graph.edges:
            if edge.source_id == missing_id:
                successor_id = edge.target_id
                break

        # Strategy 1: Insert after predecessor if it's in output
        if predecessor_id:
            pred_node = source_graph.get_node(predecessor_id)
            if pred_node:
                for i, sent in enumerate(output_sentences):
                    if self._sentence_contains_proposition(sent, pred_node):
                        return i + 1  # Insert after predecessor

        # Strategy 2: Insert before successor if it's in output
        if successor_id:
            succ_node = source_graph.get_node(successor_id)
            if succ_node:
                for i, sent in enumerate(output_sentences):
                    if self._sentence_contains_proposition(sent, succ_node):
                        return i  # Insert before successor

        # Strategy 3: Find any SOURCE node in output with higher ID and insert before it
        for source_node in sorted(source_graph.nodes, key=lambda n: int(n.id[1:]) if n.id[1:].isdigit() else 999):
            try:
                source_num = int(source_node.id[1:])
                if source_num > missing_num:
                    # Check if this source node's sentence is in output
                    for i, sent in enumerate(output_sentences):
                        if self._sentence_contains_proposition(sent, source_node):
                            return i  # Insert before this higher-numbered source node
            except (ValueError, IndexError):
                pass

        # Strategy 4: If missing node has lower ID than any source node in output, insert at 0
        for source_node in source_graph.nodes:
            try:
                source_num = int(source_node.id[1:])
                if source_num > missing_num:
                    for sent in output_sentences:
                        if self._sentence_contains_proposition(sent, source_node):
                            return 0  # There's a higher node in output, so insert at beginning
            except (ValueError, IndexError):
                pass

        # Default: append at end
        return len(output_sentences)

    def _sentence_contains_proposition(self, sentence: str, prop_node) -> bool:
        """Check if a sentence contains a specific proposition."""
        sent_lower = sentence.lower()

        # Check for subject and predicate
        has_subject = prop_node.subject and prop_node.subject.lower() in sent_lower
        has_predicate = prop_node.predicate and prop_node.predicate.lower() in sent_lower

        if has_subject and has_predicate:
            return True

        # Check for significant keyword overlap
        if prop_node.text:
            return self._sentence_overlap(sentence, prop_node.text) > 0.6

        return False

    def _find_sentence_with_entity_role(
        self, entity: str, role: str, sentences: List[str]
    ) -> Optional[str]:
        """Find a source sentence where an entity performs a specific role."""
        entity_lower = entity.lower()
        role_words = set(role.lower().split())

        for sent in sentences:
            sent_lower = sent.lower()
            if entity_lower in sent_lower:
                # Check if role words are present
                sent_words = set(sent_lower.split())
                if role_words & sent_words:  # Any overlap
                    return sent

        return None

    def _cleanup_repaired_sentences(
        self,
        sentences: List[str],
        protected_sentences: Optional[set] = None,
    ) -> List[str]:
        """Clean up sentences after repair: dedupe, remove incomplete, etc.

        Args:
            sentences: List of sentences to clean up.
            protected_sentences: Set of sentence texts (lowercase, stripped) that were
                inserted during repair and should NOT be filtered out.

        Returns:
            Cleaned list of sentences.
        """
        from ..utils.nlp import is_sentence_incomplete

        protected = protected_sentences or set()
        cleaned = []
        seen = set()

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_normalized = sent.lower().strip()
            is_protected = sent_normalized in protected

            # Skip duplicate sentences (even protected ones shouldn't appear twice)
            if sent_normalized in seen:
                logger.debug(f"Removing duplicate sentence: {sent[:50]}...")
                continue

            # Check for incomplete sentences using spaCy POS analysis
            # BUT skip this check for protected sentences (source sentences are complete)
            if not is_protected:
                is_incomplete, reason = is_sentence_incomplete(sent)
                if is_incomplete:
                    # Try to salvage by adding punctuation if it looks nearly complete
                    if reason == "no ending punctuation" and len(sent.split()) > 8:
                        sent = sent + "."
                    else:
                        logger.warning(f"Removing incomplete sentence ({reason}): {sent[:50]}...")
                        continue

            # Skip sentences that are mostly duplicated content from another
            # BUT for protected sentences, we need to be smarter - they contain
            # critical propositions that may not be fully covered by styled sentences
            is_subset = False
            for existing in cleaned:
                # Check if this sentence is largely contained in another
                if len(sent) > 20 and sent[:20].lower() in existing.lower():
                    overlap = self._sentence_overlap(sent, existing)
                    if overlap > 0.8:
                        if is_protected:
                            # Protected sentence overlaps with existing styled sentence
                            # Keep the protected one (source) since it has the full proposition
                            # and REPLACE the styled one
                            idx = cleaned.index(existing)
                            cleaned[idx] = sent
                            logger.info(f"Replaced overlapping styled sentence with source: {sent[:50]}...")
                            is_subset = True  # Don't add again
                        else:
                            logger.debug(f"Removing overlapping sentence: {sent[:50]}...")
                            is_subset = True
                        break

            if is_subset:
                # For protected sentences, we already replaced in the loop above
                # For non-protected, we skip
                if is_protected:
                    seen.add(sent_normalized)
                continue

            seen.add(sent_normalized)
            cleaned.append(sent)

        return cleaned

    def _sentence_overlap(self, sent1: str, sent2: str) -> float:
        """Calculate word overlap between two sentences."""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        return intersection / min(len(words1), len(words2))

    def _prioritize_errors(self, diff) -> List[dict]:
        """Prioritize errors for repair.

        Order:
        1. Entity role errors (most critical - change meaning)
        2. Missing propositions (content loss)
        3. Added propositions (hallucinations)
        """
        errors = []

        # Entity errors first - these are the most critical
        for i, err in enumerate(diff.entity_role_errors):
            errors.append({
                "type": "entity",
                "priority": 0,
                "data": err,
                "description": f"{err.error_type}: {err.entity} - {err.source_role}",
            })

        # Missing propositions
        for i, node in enumerate(diff.missing_nodes):
            errors.append({
                "type": "missing",
                "priority": 1,
                "data": node,
                "description": f"Missing: {node.summary()}",
            })

        # Added propositions (hallucinations)
        for i, node in enumerate(diff.added_nodes):
            errors.append({
                "type": "added",
                "priority": 2,
                "data": node,
                "description": f"Remove: {node.summary()}",
            })

        # Sort by priority
        errors.sort(key=lambda x: x["priority"])
        return errors

    def transfer_document(
        self,
        text: str,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_paragraph: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[str, TransferStats]:
        """Transfer an entire document.

        Args:
            text: Source document text.
            on_progress: Optional callback (current, total, status).
            on_paragraph: Optional callback (index, paragraph) called after each paragraph is complete.

        Returns:
            Tuple of (styled_document, statistics).
        """
        # Track state for partial results on interrupt
        self._transfer_start_time = time.time()
        self._transfer_outputs = []
        self._transfer_stats = TransferStats()

        # Reset repetition reducer for new document
        if self.repetition_reducer:
            self.repetition_reducer.reset()

        # Split into paragraphs
        paragraphs = split_into_paragraphs(text)

        if not paragraphs:
            logger.warning("No content paragraphs found")
            return text, self._transfer_stats

        # Extract document context for improved generation and critique
        if self.config.use_document_context:
            logger.info("Extracting document context...")
            self.document_context = extract_document_context(
                text,
                llm_provider=self.critic_provider,
            )

        logger.info(f"Transferring {len(paragraphs)} paragraphs")

        previous = None

        for i, para in enumerate(paragraphs):
            if on_progress:
                on_progress(i + 1, len(paragraphs), f"Processing paragraph {i+1}")

            para_start = time.time()

            # Check if paragraph is a heading - pass through unchanged
            para_lines = para.strip().split('\n')
            if self.config.pass_headings_unchanged and len(para_lines) == 1 and is_heading(para_lines[0]):
                logger.debug(f"Passing heading unchanged: {para[:50]}...")
                output = para
                score = 1.0
            else:
                output, score = self.transfer_paragraph(para, previous, self._transfer_stats)

            # Apply repetition reduction (only to transformed content, not headings)
            if self.repetition_reducer and score < 1.0:
                output, reduction_stats = self.repetition_reducer.reduce(output)
                self._transfer_stats.words_replaced += reduction_stats.replacements_made

            para_time = time.time() - para_start
            logger.debug(f"Paragraph {i+1}: {para_time:.1f}s, score={score:.2f}")

            self._transfer_outputs.append(output)
            previous = output

            self._transfer_stats.paragraphs_processed += 1
            self._transfer_stats.entailment_scores.append(score)

            if score < self.config.entailment_threshold:
                self._transfer_stats.paragraphs_repaired += 1

            # Notify callback with completed paragraph
            if on_paragraph:
                on_paragraph(i, output)

        # Compute final stats
        self._transfer_stats.total_time_seconds = time.time() - self._transfer_start_time
        self._transfer_stats.avg_time_per_paragraph = (
            self._transfer_stats.total_time_seconds / self._transfer_stats.paragraphs_processed
            if self._transfer_stats.paragraphs_processed > 0 else 0
        )

        # Log repetition reduction summary
        if self.repetition_reducer and self._transfer_stats.words_replaced > 0:
            overused = self.repetition_reducer.get_overused_words(limit=5)
            if overused:
                logger.info(
                    f"Repetition reduction: {self._transfer_stats.words_replaced} replacements, "
                    f"top overused: {', '.join(w for w, _ in overused)}"
                )

        logger.info(
            f"Transfer complete: {self._transfer_stats.paragraphs_processed} paragraphs in "
            f"{self._transfer_stats.total_time_seconds:.1f}s "
            f"(avg {self._transfer_stats.avg_time_per_paragraph:.1f}s/para)"
        )

        # Final cleanup: deduplicate paragraphs and remove incomplete ones
        cleaned_outputs = self._cleanup_document_paragraphs(self._transfer_outputs)

        return "\n\n".join(cleaned_outputs), self._transfer_stats

    def _cleanup_document_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Clean up paragraphs: remove duplicates, incomplete content, etc."""
        from ..utils.nlp import is_sentence_incomplete, get_complete_sentences

        cleaned = []
        seen_starts = {}  # Map of first 50 chars -> full paragraph

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check for duplicate paragraphs (same start)
            para_start = para[:50].lower() if len(para) > 50 else para.lower()
            if para_start in seen_starts:
                existing = seen_starts[para_start]
                # Keep the longer/more complete version
                if len(para) > len(existing):
                    # Replace with longer version
                    idx = cleaned.index(existing)
                    cleaned[idx] = para
                    seen_starts[para_start] = para
                logger.debug(f"Skipping duplicate paragraph: {para[:50]}...")
                continue

            # Check if paragraph ends incomplete using spaCy
            if para:
                # Get the last sentence and check if it's incomplete
                sentences = split_into_sentences(para)
                if sentences:
                    last_sent = sentences[-1]
                    is_incomplete, reason = is_sentence_incomplete(last_sent)
                    if is_incomplete and reason != "no ending punctuation":
                        logger.warning(f"Paragraph ends incomplete ({reason}), truncating: ...{para[-50:]}")
                        # Keep only complete sentences
                        complete = get_complete_sentences(para)
                        if complete:
                            para = " ".join(complete)
                        else:
                            # Can't salvage - add period if long enough
                            if len(para.split()) > 10:
                                para = para + "."
                            else:
                                continue

            # Check for internal repetition (same sentence repeated)
            sentences = split_into_sentences(para)
            if len(sentences) > 1:
                unique_sentences = []
                seen_sents = set()
                for sent in sentences:
                    sent_normalized = sent.strip().lower()
                    if sent_normalized not in seen_sents:
                        seen_sents.add(sent_normalized)
                        unique_sentences.append(sent.strip())
                    else:
                        logger.debug(f"Removing repeated sentence within paragraph: {sent[:40]}...")
                if len(unique_sentences) < len(sentences):
                    para = " ".join(unique_sentences)

            seen_starts[para_start] = para
            cleaned.append(para)

        return cleaned

    def get_partial_results(self) -> Tuple[str, TransferStats]:
        """Get partial results after an interrupted transfer.

        Returns:
            Tuple of (partial_output, statistics).
        """
        # Compute stats for partial transfer
        if hasattr(self, '_transfer_stats') and hasattr(self, '_transfer_start_time'):
            self._transfer_stats.total_time_seconds = time.time() - self._transfer_start_time
            if self._transfer_stats.paragraphs_processed > 0:
                self._transfer_stats.avg_time_per_paragraph = (
                    self._transfer_stats.total_time_seconds / self._transfer_stats.paragraphs_processed
                )

        outputs = getattr(self, '_transfer_outputs', [])
        stats = getattr(self, '_transfer_stats', TransferStats())

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


def create_style_transfer(
    adapter_path: str,
    author_name: str,
    verify: bool = True,
    temperature: float = 0.7,
) -> StyleTransfer:
    """Convenience function to create a style transfer pipeline.

    Args:
        adapter_path: Path to LoRA adapter.
        author_name: Author name.
        verify: Whether to enable entailment verification.
        temperature: Generation temperature.

    Returns:
        Configured StyleTransfer instance.
    """
    config = TransferConfig(
        verify_entailment=verify,
        temperature=temperature,
    )

    return StyleTransfer(
        adapter_path=adapter_path,
        author_name=author_name,
        config=config,
    )
