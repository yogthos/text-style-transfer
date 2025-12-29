"""Validation modules for semantic preservation.

Core validation for the LoRA pipeline:
- SemanticVerifier: Semantic fidelity checking with repair hints
- QualityCritic: Explicit fix instructions for quality issues
- PropositionValidator: Proposition-level validation with repair instructions
"""

from .semantic_verifier import (
    SemanticVerifier,
    VerificationResult,
    VerificationIssue,
    verify_semantic_fidelity,
)
from .quality_critic import (
    QualityCritic,
    QualityCritique,
    QualityIssue,
)
from .proposition_validator import (
    PropositionValidator,
    PropositionMatch,
    HallucinatedContent,
    ValidationResult as PropositionValidationResult,
    create_proposition_validator,
)

__all__ = [
    "SemanticVerifier",
    "VerificationResult",
    "VerificationIssue",
    "verify_semantic_fidelity",
    "QualityCritic",
    "QualityCritique",
    "QualityIssue",
    "PropositionValidator",
    "PropositionMatch",
    "HallucinatedContent",
    "PropositionValidationResult",
    "create_proposition_validator",
]
