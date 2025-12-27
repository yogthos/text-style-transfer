"""Data-driven style transfer generator.

Uses extracted style profiles for authentic author voice matching.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
from pathlib import Path

from ..style.profile import AuthorStyleProfile
from ..style.extractor import StyleProfileExtractor
from ..style.verifier import StyleVerifier, VerificationResult
from .evolutionary_generator import EvolutionaryParagraphGenerator
from ..utils.nlp import split_into_sentences
from ..utils.logging import get_logger
from ..ingestion.proposition_extractor import PropositionExtractor
from ..validation.entailment import EntailmentVerifier, EntailmentResult

logger = get_logger(__name__)


@dataclass
class TransferResult:
    """Result of style transfer for a paragraph."""

    original: str
    transferred: str
    propositions: List[str]
    verification: VerificationResult
    sentence_count: int
    word_count: int
    entailment: Optional[EntailmentResult] = None  # NLI content preservation check


@dataclass
class DocumentTransferResult:
    """Result of style transfer for a complete document."""

    paragraphs: List[TransferResult]
    profile: AuthorStyleProfile
    overall_verification: VerificationResult

    @property
    def text(self) -> str:
        """Get full transferred text."""
        return "\n\n".join(r.transferred for r in self.paragraphs)

    @property
    def avg_score(self) -> float:
        """Average verification score."""
        if not self.paragraphs:
            return 0.0
        return sum(r.verification.overall_score for r in self.paragraphs) / len(self.paragraphs)


class DataDrivenStyleTransfer:
    """Style transfer using data-driven profile matching.

    This is the main entry point for the new generation approach:
    1. Extract style profile from corpus (Markov chains, transitions, etc.)
    2. Generate sentence-by-sentence with profile-derived constraints
    3. Verify each sentence against profile
    4. Aggregate into paragraphs with verification
    """

    def __init__(
        self,
        llm_generate: Callable[[str], str],
        corpus_paragraphs: Optional[List[str]] = None,
        profile: Optional[AuthorStyleProfile] = None,
        author_name: str = "Target",
        population_size: int = 5,
        max_generations: int = 3,
    ):
        """Initialize style transfer.

        Args:
            llm_generate: Function that takes prompt and returns LLM response.
            corpus_paragraphs: Corpus paragraphs to extract profile from.
            profile: Pre-extracted profile (overrides corpus_paragraphs).
            author_name: Author name for profile.
            population_size: Number of candidates per generation.
            max_generations: Maximum evolution iterations per sentence.
        """
        self.llm_generate = llm_generate
        self.author_name = author_name

        # Extract or use provided profile
        if profile is not None:
            self.profile = profile
        elif corpus_paragraphs:
            logger.info("Extracting style profile from corpus...")
            extractor = StyleProfileExtractor()
            self.profile = extractor.extract(corpus_paragraphs, author_name)
        else:
            raise ValueError("Must provide either corpus_paragraphs or profile")

        # Initialize components
        self.verifier = StyleVerifier(self.profile)
        self.proposition_extractor = PropositionExtractor()
        self.entailment_verifier = EntailmentVerifier(entailment_threshold=0.4)

        # Use evolutionary generator for better convergence
        self.generator = EvolutionaryParagraphGenerator(
            profile=self.profile,
            llm_generate=llm_generate,
            population_size=population_size,
            max_generations=max_generations,
        )

        logger.info(
            f"Initialized DataDrivenStyleTransfer for {author_name} "
            f"(mean length={self.profile.length_profile.mean:.1f}, "
            f"burstiness={self.profile.length_profile.burstiness:.3f}, "
            f"population={population_size}, generations={max_generations})"
        )

    def transfer_paragraph(
        self,
        paragraph: str,
        previous_context: Optional[str] = None,
    ) -> TransferResult:
        """Transfer a single paragraph to target style.

        Args:
            paragraph: Source paragraph text.
            previous_context: Optional previous paragraph for context.

        Returns:
            TransferResult with transferred text and verification.
        """
        # Extract propositions from source with RST roles
        propositions = self.proposition_extractor.extract_from_text(paragraph)

        if not propositions:
            # Fallback: use sentences as propositions
            sentences = split_into_sentences(paragraph)
            proposition_texts = sentences if sentences else [paragraph]
            rst_info = None
        else:
            proposition_texts = [p.text for p in propositions]
            # Build RST info for generator
            rst_info = [
                {
                    "role": p.rst_role,
                    "relation": p.rst_relation,
                    "parent_idx": p.parent_nucleus_idx,
                    "entities": p.entities,
                }
                for p in propositions
            ]

        logger.debug(f"Extracted {len(proposition_texts)} propositions")

        # Generate paragraph using evolutionary approach with RST awareness
        transferred = self.generator.generate_paragraph(
            proposition_texts,
            rst_info=rst_info
        )

        # Verify against profile
        verification = self.verifier.verify_paragraph(transferred)

        # Verify content preservation using NLI entailment
        entailment = self.entailment_verifier.verify(proposition_texts, transferred)

        # Log content preservation issues
        if not entailment.is_acceptable:
            logger.warning(
                f"Content preservation issue: {entailment.coverage_ratio:.0%} propositions entailed"
            )
            if entailment.lost_propositions:
                for lost in entailment.lost_propositions[:3]:
                    logger.warning(f"  Lost: {lost[:60]}...")

        # Calculate stats
        sentences = split_into_sentences(transferred)
        word_count = len(transferred.split())

        return TransferResult(
            original=paragraph,
            transferred=transferred,
            propositions=proposition_texts,
            verification=verification,
            sentence_count=len(sentences),
            word_count=word_count,
            entailment=entailment,
        )

    def transfer_document(
        self,
        paragraphs: List[str],
        on_paragraph_complete: Optional[Callable[[int, TransferResult], None]] = None,
    ) -> DocumentTransferResult:
        """Transfer a complete document to target style.

        Args:
            paragraphs: List of source paragraphs.
            on_paragraph_complete: Optional callback after each paragraph.

        Returns:
            DocumentTransferResult with all transferred paragraphs.
        """
        logger.info(f"Transferring {len(paragraphs)} paragraphs")

        results = []
        previous_context = None

        for i, para in enumerate(paragraphs):
            logger.info(f"Processing paragraph {i+1}/{len(paragraphs)}")

            result = self.transfer_paragraph(para, previous_context)
            results.append(result)

            # Update context for next paragraph
            previous_context = result.transferred

            # Call completion callback
            if on_paragraph_complete:
                on_paragraph_complete(i, result)

            # Log verification
            v = result.verification
            logger.info(
                f"  Verification: score={v.overall_score:.2f}, "
                f"acceptable={v.is_acceptable}, issues={len(v.issues)}"
            )

        # Compute overall verification
        full_text = "\n\n".join(r.transferred for r in results)
        overall_verification = self.verifier.verify_paragraph(full_text)

        return DocumentTransferResult(
            paragraphs=results,
            profile=self.profile,
            overall_verification=overall_verification,
        )

    def get_profile_summary(self) -> str:
        """Get human-readable profile summary."""
        p = self.profile
        lp = p.length_profile
        tp = p.transition_profile

        lines = [
            f"Style Profile: {p.author_name}",
            f"  Corpus: {p.corpus_word_count} words, {p.corpus_sentence_count} sentences",
            f"  Sentence Length: mean={lp.mean:.1f}, std={lp.std:.1f}",
            f"  Burstiness: {lp.burstiness:.3f}",
            f"  No-Transition Ratio: {tp.no_transition_ratio:.1%}",
        ]

        # Top transitions per category
        for cat, probs in [
            ("Causal", tp.causal),
            ("Adversative", tp.adversative),
            ("Additive", tp.additive),
        ]:
            if probs:
                top = sorted(probs.items(), key=lambda x: -x[1])[:3]
                words = ", ".join(f"{w}({p:.0%})" for w, p in top)
                lines.append(f"  {cat}: {words}")

        return "\n".join(lines)


def create_transfer_pipeline(
    corpus_paragraphs: List[str],
    llm_generate: Callable[[str], str],
    author_name: str = "Target",
) -> DataDrivenStyleTransfer:
    """Create a transfer pipeline from corpus.

    Args:
        corpus_paragraphs: Paragraphs from target author's corpus.
        llm_generate: Function to generate LLM responses.
        author_name: Name of target author.

    Returns:
        Configured DataDrivenStyleTransfer instance.
    """
    return DataDrivenStyleTransfer(
        llm_generate=llm_generate,
        corpus_paragraphs=corpus_paragraphs,
        author_name=author_name,
    )


def load_profile_and_create_transfer(
    profile_path: str,
    llm_generate: Callable[[str], str],
) -> DataDrivenStyleTransfer:
    """Create transfer pipeline from saved profile.

    Args:
        profile_path: Path to saved profile JSON.
        llm_generate: Function to generate LLM responses.

    Returns:
        Configured DataDrivenStyleTransfer instance.
    """
    profile = AuthorStyleProfile.load(profile_path)
    return DataDrivenStyleTransfer(
        llm_generate=llm_generate,
        profile=profile,
    )
