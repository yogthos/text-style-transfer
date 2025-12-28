"""Rhetorical structure analysis and template generation."""

from .function_classifier import (
    SentenceFunction,
    SentenceFunctionClassifier,
    ClassifiedSentence,
    classify_sentence_function,
)

from .template_generator import (
    TemplateSlot,
    RhetoricalTemplate,
    RhetoricalTemplateGenerator,
    generate_template,
)

from .proposition_mapper import (
    MappedProposition,
    MappingResult,
    PropositionMapper,
    map_propositions_to_template,
    FUNCTION_DESCRIPTIONS,
)

__all__ = [
    "SentenceFunction",
    "SentenceFunctionClassifier",
    "ClassifiedSentence",
    "classify_sentence_function",
    "TemplateSlot",
    "RhetoricalTemplate",
    "RhetoricalTemplateGenerator",
    "generate_template",
    "MappedProposition",
    "MappingResult",
    "PropositionMapper",
    "map_propositions_to_template",
    "FUNCTION_DESCRIPTIONS",
]
