"""Soft scoring module for gradient-based evolution.

This module provides a fitness function that calculates continuous raw_score
for evolution guidance, separate from the binary pass/fail decision.
"""

from typing import Dict, Tuple, Optional, List
from src.validator.semantic_critic import SemanticCritic
from src.ingestion.blueprint import SemanticBlueprint


class SoftScorer:
    """Calculates continuous fitness scores for evolution guidance.

    Unlike the semantic critic which uses hard gates and binary pass/fail,
    this scorer always returns a continuous raw_score to guide incremental
    improvement, even when candidates don't fully pass all gates.
    """

    def __init__(self, config_path: str = "config.json"):
        """Initialize the soft scorer.

        Args:
            config_path: Path to configuration file.
        """
        self.critic = SemanticCritic(config_path=config_path)
        self.config_path = config_path

    def calculate_raw_score(
        self,
        generated_text: str,
        input_blueprint: SemanticBlueprint,
        style_lexicon: Optional[List[str]] = None,
        skeleton: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate raw fitness score for evolution guidance.

        This score is used to guide evolution by allowing incremental
        improvements. A "bad" sentence (0.4) can be replaced by a "slightly
        less bad" sentence (0.5), allowing the system to climb the fitness hill.

        Formula: raw_score = (recall * 0.5) + (fluency * 0.3) + (similarity * 0.2)

        Weights prioritize:
        1. Meaning preservation (recall) - 50%
        2. Grammatical fluency - 30%
        3. Semantic similarity - 20%

        Args:
            generated_text: Generated text to evaluate.
            input_blueprint: Original input blueprint.

        Returns:
            Tuple of (raw_score, metrics_dict) where:
            - raw_score: Continuous score 0.0-1.0 for evolution guidance
            - metrics_dict: Dictionary with individual metric scores
        """
        # Get evaluation from semantic critic
        result = self.critic.evaluate(generated_text, input_blueprint, skeleton=skeleton)

        # Extract individual scores
        recall_score = result.get("recall_score", 0.0)
        adherence_score = result.get("adherence_score", 1.0)  # Default to 1.0 if no skeleton
        fluency_score = result.get("fluency_score", 0.0)  # Keep for backward compatibility

        # Calculate semantic similarity if available
        # CRITICAL: Calculate similarity BEFORE checking if result passed
        # This ensures we get a meaningful similarity score even if critic rejected early
        similarity_score = 0.0
        if self.critic.semantic_model and input_blueprint.original_text:
            try:
                similarity_score = self.critic._calculate_semantic_similarity(
                    input_blueprint.original_text,
                    generated_text
                )
            except Exception:
                # If similarity calculation fails, use precision as fallback
                precision_score = result.get("precision_score", 0.0)
                similarity_score = precision_score
        else:
            # Fallback: use precision as proxy for similarity
            precision_score = result.get("precision_score", 0.0)
            similarity_score = precision_score

        # If critic rejected early (score=0.0) but we have similarity data, use it
        # This allows evolution to work even when hard gates reject candidates
        if result.get("score", 0.0) == 0.0 and similarity_score > 0.0:
            # Candidate failed hard gate but has some similarity - give it a chance
            # Use similarity as the primary signal when other scores are zeroed
            recall_score = max(recall_score, similarity_score * 0.8)  # Partial credit
            fluency_score = max(fluency_score, 0.5)  # Assume moderate fluency if not measured

        # Weighted fitness formula
        if skeleton:
            # When skeleton is provided, use adherence-based formula
            raw_score = (recall_score * 0.5) + (adherence_score * 0.5)
        else:
            # Fallback to original formula when no skeleton
            w_recall = 0.5
            w_fluency = 0.3
            w_similarity = 0.2
            raw_score = (
                recall_score * w_recall +
                fluency_score * w_fluency +
                similarity_score * w_similarity
            )

        # PERFECTION BOOST: If the sentence is semantically complete and grammatically perfect,
        # do not let a low vector similarity score drag it down.
        # This ensures that a perfect translation gets a passing grade, even if the vector
        # embeddings drift slightly due to stylistic choices.
        if recall_score >= 0.95 and fluency_score >= 0.95:
            # Boost the score to at least 0.92 for near-perfect meaning and grammar
            raw_score = max(raw_score, 0.92)

        # Style density bonus removed - conflicts with simplified approach

        # Ensure raw_score is never zero (unless truly empty/broken)
        # Minimum score of 0.01 allows evolution to improve from very bad states
        if raw_score == 0.0 and generated_text and generated_text.strip():
            # If text exists but score is 0, give minimal score to allow improvement
            raw_score = 0.01

        metrics = {
            "recall": recall_score,
            "adherence": adherence_score,
            "fluency": fluency_score,
            "similarity": similarity_score,
            "precision": result.get("precision_score", 0.0),
            "pass": result.get("pass", False)
        }

        return raw_score, metrics

    def evaluate_with_raw_score(
        self,
        generated_text: str,
        input_blueprint: SemanticBlueprint,
        style_lexicon: Optional[List[str]] = None,
        skeleton: Optional[str] = None
    ) -> Dict[str, any]:
        """Evaluate text and return both pass status and raw_score.

        This is the main interface for evolution - it provides both:
        - pass: Boolean for final acceptance
        - raw_score: Continuous score for evolution guidance

        Args:
            generated_text: Generated text to evaluate.
            input_blueprint: Original input blueprint.

        Returns:
            Dictionary with:
            - pass: bool - Whether text passes all gates
            - raw_score: float - Continuous fitness score for evolution
            - recall_score: float
            - fluency_score: float
            - similarity_score: float
            - precision_score: float
            - feedback: str
            - score: float - Original weighted score from critic
        """
        # Get critic evaluation
        critic_result = self.critic.evaluate(generated_text, input_blueprint, allowed_style_words=style_lexicon, skeleton=skeleton)

        # Calculate raw_score
        raw_score, metrics = self.calculate_raw_score(generated_text, input_blueprint, style_lexicon=style_lexicon, skeleton=skeleton)

        # Combine results
        return {
            "pass": critic_result.get("pass", False),
            "raw_score": raw_score,
            "recall_score": metrics["recall"],
            "adherence_score": metrics.get("adherence", 1.0),
            "fluency_score": metrics["fluency"],
            "similarity_score": metrics["similarity"],
            "precision_score": metrics["precision"],
            "score": critic_result.get("score", 0.0),  # Original weighted score
            "feedback": critic_result.get("feedback", "")
        }

