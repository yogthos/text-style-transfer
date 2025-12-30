"""Style transfer pipeline using LoRA.

This module provides a style transfer pipeline that uses LoRA-adapted models
for consistent style transfer with a critic/repair loop.

Pipeline:
1. Extract document context (thesis, intent, tone)
2. For each paragraph:
   - Generate styled text (LoRA call)
   - Validate propositions preserved
   - Repair with critic if needed
3. Post-process to reduce repetition
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
    """Configuration for fast style transfer."""

    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Verification settings
    verify_entailment: bool = True
    entailment_threshold: float = 0.7

    # Proposition validation settings
    use_proposition_validation: bool = True  # Enable proposition-based validation
    proposition_threshold: float = 0.85  # Min proposition coverage (raised from 0.7 for accuracy)
    anchor_threshold: float = 0.9  # Min content anchor coverage (raised from 0.8 for accuracy)

    # Quality critic settings
    use_quality_critic: bool = True  # Enable quality checking with explicit fix instructions
    word_cluster_threshold: int = 3  # Words used 3+ times trigger warning

    # Repair settings (can be overridden by app config)
    max_repair_attempts: int = 3
    repair_temperature: float = 0.3

    # Post-processing settings
    reduce_repetition: bool = True
    repetition_threshold: int = 3  # Words used 3+ times get replaced

    # Content handling
    pass_headings_unchanged: bool = True  # Don't transform headings
    min_paragraph_words: int = 10  # Skip very short paragraphs

    # Document context settings
    use_document_context: bool = True  # Extract and use document-level context

    # Neutralization settings - convert prose to description before LoRA
    use_neutralization: bool = True  # Neutralize paragraphs before transformation
    neutralization_temperature: float = 0.3  # Low temp for consistent descriptions
    neutralization_min_tokens: int = 300  # Minimum tokens for neutralization output
    neutralization_token_multiplier: float = 1.2  # Multiplier for token calculation

    # Content anchor detection settings
    analogy_min_length: int = 10  # Minimum chars for detected analogies
    detect_phase_transitions: bool = True  # Detect "X transforms into Y" patterns

    # Hallucination detection settings
    hallucination_check_noun_phrases: bool = True  # Check for invented noun phrases
    critical_hallucination_words: str = "death,god,soul,spirit,heaven,hell,divine,eternal"

    # Length control settings
    max_expansion_ratio: float = 1.5  # Max output/input word ratio (1.5 = 50% longer)
    target_expansion_ratio: float = 1.2  # Target for LoRA generation
    truncate_over_expanded: bool = False  # If True, truncate; if False, allow longer output

    # LoRA influence settings
    lora_scale: float = 1.0  # 0.0=base only, 0.5=half, 1.0=full, >1.0=amplified

    # Perspective settings
    perspective: str = "preserve"  # preserve, first_person_singular, first_person_plural, third_person, author_voice_third_person


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


class StyleTransfer:
    """Style transfer using LoRA with critic/repair loop.

    This is the main entry point for style transfer. Pipeline:

    1. Extract propositions (rule-based or LLM)
    2. Single LoRA generation pass
    3. Lightweight verification
    4. Optional single repair pass

    Example usage:
        transfer = StyleTransfer(
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

        # Initialize proposition validator for semantic fidelity checking
        self.proposition_validator = None
        if self.config.use_proposition_validation:
            from ..validation.proposition_validator import PropositionValidator
            self.proposition_validator = PropositionValidator(
                proposition_threshold=self.config.proposition_threshold,
                anchor_threshold=self.config.anchor_threshold,
                check_noun_phrases=getattr(self.config, 'hallucination_check_noun_phrases', True),
                critical_hallucination_words=getattr(
                    self.config, 'critical_hallucination_words',
                    "death,god,soul,spirit,heaven,hell,divine,eternal"
                ),
            )

        # Set up entailment verifier if requested
        if self.config.verify_entailment and self.verify_fn is None:
            self.verify_fn = self._create_default_verifier()

        # Document context (extracted at transfer time)
        self.document_context: Optional[DocumentContext] = None

        logger.info(f"Using critic provider for repairs: {self.critic_provider.provider_name}")

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

    def _neutralize_paragraph(self, paragraph: str) -> str:
        """Convert paragraph to neutral semantic description.

        This matches the training format where the LoRA learned to transform
        descriptions into styled prose.

        Args:
            paragraph: Original paragraph text.

        Returns:
            Neutral description of the paragraph content.
        """
        if not self.critic_provider:
            logger.warning("No critic provider for neutralization, using original text")
            return paragraph

        prompt = format_prompt("neutralize_for_transfer", paragraph=paragraph)

        try:
            # Calculate tokens based on input length - need enough room for full content
            input_words = len(paragraph.split())
            min_tokens = getattr(self.config, 'neutralization_min_tokens', 300)
            token_multiplier = getattr(self.config, 'neutralization_token_multiplier', 1.2)
            neutralize_tokens = max(min_tokens, int(input_words * token_multiplier))

            description = self.critic_provider.call(
                system_prompt="You are a precise content summarizer. Describe what the text says in neutral language. Include ALL names, numbers, and specific details.",
                user_prompt=prompt,
                temperature=self.config.neutralization_temperature,
                max_tokens=neutralize_tokens,
            )

            description = description.strip()
            logger.debug(f"Raw neutralized output: {description[:200]}...")

            # Clean up any meta-language the model might add
            prefixes_to_remove = [
                "The passage ", "This text ", "The text ", "This passage ",
                "Here, ", "In this ", "The author ",
            ]
            for prefix in prefixes_to_remove:
                if description.lower().startswith(prefix.lower()):
                    description = description[len(prefix):]
                    # Capitalize first letter
                    if description:
                        description = description[0].upper() + description[1:]
                    break

            logger.debug(f"Neutralized: {len(paragraph.split())} words -> {len(description.split())} words")
            return description

        except Exception as e:
            logger.warning(f"Neutralization failed: {e}, using original text")
            return paragraph

    def _verify_and_augment_neutralization(
        self,
        neutralized: str,
        source_propositions: List,
    ) -> str:
        """Verify neutralized content contains all propositions and augment if needed.

        This is a critical step to prevent proposition loss during neutralization.
        If key content is missing from the neutralized text, we append it explicitly.

        Args:
            neutralized: Neutralized content description.
            source_propositions: Propositions extracted from source.

        Returns:
            Augmented neutralized content with any missing propositions.
        """
        if not source_propositions:
            return neutralized

        neutralized_lower = neutralized.lower()
        missing_content = []

        for prop in source_propositions:
            # Check if key entities are present
            for entity in prop.entities:
                if entity.lower() not in neutralized_lower:
                    missing_content.append(f"Mention: {entity}")

            # Check if content anchors are present
            for anchor in prop.content_anchors:
                if anchor.must_preserve and anchor.text.lower() not in neutralized_lower:
                    if anchor.anchor_type == "example":
                        missing_content.append(f"Example: {anchor.text}")
                    elif anchor.anchor_type == "statistic":
                        missing_content.append(f"Data: {anchor.text}")
                    elif anchor.anchor_type == "quote":
                        missing_content.append(f"Quote: \"{anchor.text}\"")
                    else:
                        missing_content.append(anchor.text)

            # Check if the core proposition is missing (very low keyword overlap)
            prop_keywords = set(kw.lower() for kw in prop.keywords)
            if prop_keywords:
                neutralized_words = set(neutralized_lower.split())
                overlap = len(prop_keywords & neutralized_words)
                coverage = overlap / len(prop_keywords)
                if coverage < 0.3:  # Less than 30% keyword coverage
                    # Add a summary of this proposition
                    if prop.subject and prop.verb:
                        summary = f"{prop.subject} {prop.verb}"
                        if prop.object:
                            summary += f" {prop.object}"
                        missing_content.append(f"Point: {summary}")

        # Deduplicate and filter
        seen = set()
        unique_missing = []
        for item in missing_content:
            item_lower = item.lower()
            if item_lower not in seen and item_lower not in neutralized_lower:
                seen.add(item_lower)
                unique_missing.append(item)

        if unique_missing:
            logger.warning(f"Neutralization missing {len(unique_missing)} items, augmenting")
            # Append missing content
            augmentation = "\n\nMust include:\n- " + "\n- ".join(unique_missing[:10])  # Limit to 10
            return neutralized + augmentation

        return neutralized

    def transfer_paragraph(
        self,
        paragraph: str,
        previous: Optional[str] = None,
        stats: Optional['TransferStats'] = None,
    ) -> Tuple[str, float]:
        """Transfer a single paragraph with proposition-based validation and repair.

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

        # Extract propositions from source for validation
        source_propositions = None
        if self.proposition_validator:
            source_propositions = self.proposition_validator.extract_propositions(paragraph)
            logger.debug(f"Extracted {len(source_propositions)} propositions from source")

        # Neutralize paragraph before transformation (matches LoRA training format)
        if self.config.use_neutralization:
            content_for_generation = self._neutralize_paragraph(paragraph)
            logger.info(f"Neutralized to {len(content_for_generation.split())} words")
            # Check if neutralization failed (returned original)
            if content_for_generation.strip() == paragraph.strip():
                logger.warning("Neutralization returned original text unchanged")
            # Verify neutralization preserves propositions and augment if needed
            elif source_propositions:
                content_for_generation = self._verify_and_augment_neutralization(
                    content_for_generation, source_propositions
                )
        else:
            content_for_generation = paragraph

        # Generate with token limit based on input and configured expansion ratio
        target_words = int(word_count * self.config.target_expansion_ratio)
        max_tokens = max(100, int(target_words * 1.5))  # tokens > words

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
        elif output.strip() == content_for_generation.strip():
            logger.warning("LoRA output identical to neutralized content - no style applied")

        # Track expansion at LoRA stage
        lora_words = len(output.split())
        source_words = len(paragraph.split())
        if lora_words > source_words * self.config.max_expansion_ratio:
            logger.warning(f"LoRA over-expanded: {lora_words} words vs {source_words} source ({lora_words/source_words:.0%})")

        # Proposition-based validation and repair loop
        if self.proposition_validator and source_propositions:
            for attempt in range(self.config.max_repair_attempts):
                validation = self.proposition_validator.validate(
                    paragraph, output, source_propositions
                )

                if stats:
                    stats.quality_issues_found += len(validation.missing_propositions)
                    stats.quality_issues_found += len(validation.hallucinated_content)

                if validation.is_valid:
                    logger.debug(
                        f"Proposition validation passed (attempt {attempt + 1}): "
                        f"{validation.proposition_coverage:.0%} coverage"
                    )
                    break

                # Log specific issues
                if validation.missing_entities:
                    logger.warning(f"Missing entities: {', '.join(validation.missing_entities[:3])}")
                if validation.added_entities:
                    logger.warning(f"Hallucinated entities: {', '.join(validation.added_entities[:3])}")
                if validation.missing_facts:
                    logger.warning(f"Missing facts: {len(validation.missing_facts)}")

                logger.info(
                    f"Repair attempt {attempt + 1}: "
                    f"{len(validation.missing_propositions)} missing propositions, "
                    f"{len(validation.hallucinated_content)} hallucinations"
                )

                # Use critic provider for surgical repairs
                output = self._critic_repair(
                    source=paragraph,
                    current_output=output,
                    validation=validation,
                )

                if stats:
                    stats.quality_issues_fixed += 1

        # Fallback to quality critic if proposition validator not available
        elif self.quality_critic:
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

                logger.info(f"Repair attempt {attempt + 1}: {len(critique.issues)} issues")

                # Use critic provider for repairs
                output = self._quality_critic_repair(
                    source=paragraph,
                    current_output=output,
                    critique=critique,
                )

                if stats:
                    stats.quality_issues_fixed += 1

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

    def _call_critic(
        self,
        source: str,
        current_output: str,
        instructions: List[str],
    ) -> str:
        """Call critic provider to make surgical fixes.

        Args:
            source: Original source text.
            current_output: Current styled output with issues.
            instructions: List of specific fix instructions.

        Returns:
            Repaired text.
        """
        if not instructions:
            return current_output

        instruction_text = "\n".join(f"- {inst}" for inst in instructions)

        # Use context-aware prompt if document context is available
        if self.document_context:
            system_prompt = format_prompt(
                "critic_repair_with_context",
                document_context=self.document_context.to_critic_context(),
                instructions=instruction_text
            )
        else:
            system_prompt = format_prompt(
                "critic_repair_system",
                instructions=instruction_text
            )

        # Don't pass source text to critic - only the styled output
        # This prevents the critic from copying source sentences
        user_prompt = format_prompt(
            "critic_repair_user",
            current_output=current_output
        )

        try:
            # Allow enough tokens for completion (current output + room for fixes)
            # Use 1.5x to ensure sentences can be completed
            current_words = len(current_output.split())
            max_repair_tokens = max(200, int(current_words * 1.5))

            repaired = self.critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.repair_temperature,
                max_tokens=max_repair_tokens,
            )
            logger.debug(f"Critic repair applied: {len(instructions)} fixes")
            return repaired.strip()
        except Exception as e:
            logger.warning(f"Critic repair failed: {e}, keeping original output")
            return current_output

    def _critic_repair(
        self,
        source: str,
        current_output: str,
        validation,
    ) -> str:
        """Use critic provider to surgically fix content issues.

        Args:
            source: Original source text.
            current_output: Current styled output with issues.
            validation: ValidationResult with specific issues.

        Returns:
            Repaired text with style preserved.
        """
        instructions = []

        # First priority: complete any incomplete sentences
        if validation.has_incomplete_sentences:
            instructions.append("COMPLETE the final sentence - it ends abruptly, add a proper ending")

        # Add missing propositions with FULL text (most important for preservation)
        if validation.missing_propositions:
            for match in validation.missing_propositions[:5]:
                prop = match.proposition
                # Include the full proposition text for precise repair
                if prop.text and len(prop.text) < 200:
                    instructions.append(f"MUST EXPRESS this idea: \"{prop.text}\"")
                elif prop.subject and prop.verb:
                    summary = f"{prop.subject} {prop.verb}"
                    if prop.object:
                        summary += f" {prop.object}"
                    instructions.append(f"MUST INCLUDE: {summary}")

                # Also note specific missing elements
                if match.missing_elements:
                    for elem in match.missing_elements[:2]:
                        instructions.append(f"  - Missing {elem}")

        # Add missing entities with context about what they relate to
        if validation.missing_entities:
            for entity in validation.missing_entities[:5]:
                # Skip if already covered by missing propositions
                if any(entity.lower() in str(inst).lower() for inst in instructions):
                    continue
                # Find the proposition that contains this entity for context
                context = self._find_entity_context(source, entity)
                if context:
                    instructions.append(f"ADD '{entity}' - context: {context}")
                else:
                    instructions.append(f"MENTION '{entity}' naturally in the text")

        if validation.added_entities:
            entities = ", ".join(validation.added_entities[:3])
            instructions.append(f"REMOVE these terms (not relevant): {entities}")

        # Handle hallucinated content - critical issues must be removed
        if validation.hallucinated_content:
            critical_hallucinations = [
                h for h in validation.hallucinated_content if h.severity == "critical"
            ]
            for h in critical_hallucinations[:3]:
                instructions.append(
                    f"REMOVAL REQUIRED: Delete any sentence mentioning '{h.text}' - this was not in the source"
                )

        # Add missing facts with the actual fact content
        if validation.missing_facts:
            for fact in validation.missing_facts[:3]:
                # Skip if already covered
                if any(fact.lower() in str(inst).lower() for inst in instructions):
                    continue
                instructions.append(f"INCLUDE this fact: {fact}")

        if validation.stance_violations:
            for violation in validation.stance_violations[:2]:
                instructions.append(violation)

        return self._call_critic(source, current_output, instructions)

    def _find_entity_context(self, source: str, entity: str) -> Optional[str]:
        """Find the sentence containing an entity to provide context."""
        sentences = split_into_sentences(source)
        for sent in sentences:
            if entity.lower() in sent.lower():
                # Return a shortened version of the sentence as context
                if len(sent) > 100:
                    # Find the clause containing the entity
                    words = sent.split()
                    entity_words = entity.lower().split()
                    for i, word in enumerate(words):
                        if word.lower() in entity_words:
                            start = max(0, i - 5)
                            end = min(len(words), i + len(entity_words) + 5)
                            return "..." + " ".join(words[start:end]) + "..."
                return sent[:100] + "..." if len(sent) > 100 else sent
        return None

    def _quality_critic_repair(
        self,
        source: str,
        current_output: str,
        critique,
    ) -> str:
        """Use critic provider to fix quality issues.

        Args:
            source: Original source text.
            current_output: Current styled output with issues.
            critique: QualityCritique with specific issues.

        Returns:
            Repaired text with style preserved.
        """
        instructions = []
        for issue in critique.issues:
            if issue.fix_instruction:
                instructions.append(issue.fix_instruction)

        # Always check for grammar/completeness
        instructions.append("FIX any incomplete or ungrammatical sentences")

        return self._call_critic(source, current_output, instructions)

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
        repair_system = format_prompt("repair_strict", author=self.author)

        try:
            # Estimate tokens based on input
            input_words = len(original.split())
            max_tokens = max(100, int(input_words * 2.0))

            # Get context hint for generation (if document context available)
            context_hint = None
            if self.document_context:
                context_hint = self.document_context.to_generation_hint()

            output = self.generator.generate(
                content=original,
                author=self.author,
                context=previous,
                system_override=repair_system,
                max_tokens=max_tokens,
                context_hint=context_hint,
                perspective=getattr(self.config, 'perspective', 'preserve'),
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

        return "\n\n".join(self._transfer_outputs), self._transfer_stats

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
