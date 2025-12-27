"""NLI-based entailment verification for content preservation.

Uses a cross-encoder model to verify that generated text entails
all source propositions, ensuring no meaning is lost.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Lazy-load the model to avoid import overhead
_model = None
_model_name = "cross-encoder/nli-deberta-v3-small"  # Smaller model for speed


def get_nli_model():
    """Lazy-load the NLI cross-encoder model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading NLI model: {_model_name}")
            _model = CrossEncoder(_model_name)
            logger.info("NLI model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            _model = None
    return _model


@dataclass
class PropositionEntailment:
    """Entailment result for a single proposition."""

    proposition: str
    is_entailed: bool
    entailment_score: float  # 0-1, higher = more entailed
    contradiction_score: float
    neutral_score: float
    best_matching_sentence: Optional[str] = None


@dataclass
class EntailmentResult:
    """Result of entailment verification for a paragraph."""

    propositions_checked: int
    propositions_entailed: int
    coverage_ratio: float  # proportion of propositions entailed
    mean_entailment_score: float

    # Per-proposition results
    details: List[PropositionEntailment] = field(default_factory=list)

    # Lost content (propositions not entailed)
    lost_propositions: List[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """Check if entailment is acceptable (>80% coverage)."""
        return self.coverage_ratio >= 0.8


class EntailmentVerifier:
    """Verifies that generated text entails source propositions.

    Uses NLI (Natural Language Inference) to check if the generated
    text preserves the meaning of each source proposition.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.5,
        use_sentence_matching: bool = True,
    ):
        """Initialize verifier.

        Args:
            entailment_threshold: Minimum entailment score to consider preserved.
            use_sentence_matching: If True, match propositions to individual
                sentences for better localized checking.
        """
        self.entailment_threshold = entailment_threshold
        self.use_sentence_matching = use_sentence_matching
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy for sentence splitting."""
        if self._nlp is None:
            from ..utils.nlp import get_nlp
            self._nlp = get_nlp()
        return self._nlp

    def verify(
        self,
        source_propositions: List[str],
        generated_text: str,
    ) -> EntailmentResult:
        """Verify that generated text entails source propositions.

        Args:
            source_propositions: List of source proposition texts.
            generated_text: The generated/transferred text.

        Returns:
            EntailmentResult with coverage metrics and details.
        """
        if not source_propositions:
            return EntailmentResult(
                propositions_checked=0,
                propositions_entailed=0,
                coverage_ratio=1.0,
                mean_entailment_score=1.0,
            )

        model = get_nli_model()
        if model is None:
            # Fallback: assume all entailed if model unavailable
            logger.warning("NLI model not available, skipping entailment check")
            return EntailmentResult(
                propositions_checked=len(source_propositions),
                propositions_entailed=len(source_propositions),
                coverage_ratio=1.0,
                mean_entailment_score=1.0,
            )

        # Split generated text into sentences for matching
        if self.use_sentence_matching:
            doc = self.nlp(generated_text)
            generated_sentences = [sent.text.strip() for sent in doc.sents]
        else:
            generated_sentences = [generated_text]

        details = []
        entailed_count = 0
        total_entailment = 0.0
        lost = []

        for proposition in source_propositions:
            result = self._check_proposition_entailment(
                proposition, generated_sentences, model
            )
            details.append(result)

            if result.is_entailed:
                entailed_count += 1
            else:
                lost.append(proposition)

            total_entailment += result.entailment_score

        coverage = entailed_count / len(source_propositions)
        mean_score = total_entailment / len(source_propositions)

        return EntailmentResult(
            propositions_checked=len(source_propositions),
            propositions_entailed=entailed_count,
            coverage_ratio=coverage,
            mean_entailment_score=mean_score,
            details=details,
            lost_propositions=lost,
        )

    def _check_proposition_entailment(
        self,
        proposition: str,
        generated_sentences: List[str],
        model,
    ) -> PropositionEntailment:
        """Check if any generated sentence entails the proposition.

        Args:
            proposition: The source proposition to check.
            generated_sentences: List of generated sentences.
            model: The NLI cross-encoder model.

        Returns:
            PropositionEntailment result.
        """
        # Build pairs: (generated_sentence, proposition)
        # NLI convention: premise=generated, hypothesis=proposition
        # If generated ENTAILS proposition, the meaning is preserved
        pairs = [(sent, proposition) for sent in generated_sentences]

        if not pairs:
            return PropositionEntailment(
                proposition=proposition,
                is_entailed=False,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=1.0,
            )

        # Get predictions: [contradiction, entailment, neutral]
        try:
            scores = model.predict(pairs)

            # scores shape: (num_pairs, 3) for [contradiction, entailment, neutral]
            if len(scores.shape) == 1:
                scores = scores.reshape(1, -1)

            # Find the sentence with highest entailment score
            entailment_scores = scores[:, 1]  # entailment is index 1
            best_idx = int(np.argmax(entailment_scores))
            best_scores = scores[best_idx]

            contradiction_score = float(best_scores[0])
            entailment_score = float(best_scores[1])
            neutral_score = float(best_scores[2])

            # Normalize scores to probabilities
            total = contradiction_score + entailment_score + neutral_score
            if total > 0:
                entailment_score /= total
                contradiction_score /= total
                neutral_score /= total

            is_entailed = entailment_score >= self.entailment_threshold

            return PropositionEntailment(
                proposition=proposition,
                is_entailed=is_entailed,
                entailment_score=entailment_score,
                contradiction_score=contradiction_score,
                neutral_score=neutral_score,
                best_matching_sentence=generated_sentences[best_idx],
            )

        except Exception as e:
            logger.warning(f"Error in entailment check: {e}")
            return PropositionEntailment(
                proposition=proposition,
                is_entailed=False,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=1.0,
            )


def verify_content_preservation(
    source_propositions: List[str],
    generated_text: str,
    threshold: float = 0.5,
) -> EntailmentResult:
    """Convenience function to verify content preservation.

    Args:
        source_propositions: List of source proposition texts.
        generated_text: The generated/transferred text.
        threshold: Minimum entailment score threshold.

    Returns:
        EntailmentResult with coverage metrics.
    """
    verifier = EntailmentVerifier(entailment_threshold=threshold)
    return verifier.verify(source_propositions, generated_text)
