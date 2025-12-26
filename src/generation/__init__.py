"""Generation module for styled text output."""

from .prompt_builder import (
    PromptBuilder,
    MultiSentencePromptBuilder,
    GenerationPrompt,
    DEFAULT_TRANSITION_WORDS,
    TRANSITION_TYPE_TO_CATEGORY,
)
from .sentence_generator import (
    SentenceGenerator,
    MultiPassGenerator,
    SessionBasedGenerator,
    GeneratedSentence,
    GeneratedParagraph,
)
from .critics import (
    Critic,
    CriticPanel,
    CriticFeedback,
    CriticType,
    ValidationResult,
    LengthCritic,
    KeywordCritic,
    FluencyCritic,
    SemanticCritic,
    StyleCritic,
    VoiceCritic,
    PunctuationCritic,
)
from .style_metrics import (
    StyleScorer,
    StyleScore,
    VocabularyScorer,
    VoiceScorer,
    SentenceLengthScorer,
    PunctuationScorer,
)

__all__ = [
    # Prompt building
    "PromptBuilder",
    "MultiSentencePromptBuilder",
    "GenerationPrompt",
    "DEFAULT_TRANSITION_WORDS",
    "TRANSITION_TYPE_TO_CATEGORY",
    # Sentence generation
    "SentenceGenerator",
    "MultiPassGenerator",
    "SessionBasedGenerator",
    "GeneratedSentence",
    "GeneratedParagraph",
    # Validation critics
    "Critic",
    "CriticPanel",
    "CriticFeedback",
    "CriticType",
    "ValidationResult",
    "LengthCritic",
    "KeywordCritic",
    "FluencyCritic",
    "SemanticCritic",
    "StyleCritic",
    "VoiceCritic",
    "PunctuationCritic",
    # Style metrics
    "StyleScorer",
    "StyleScore",
    "VocabularyScorer",
    "VoiceScorer",
    "SentenceLengthScorer",
    "PunctuationScorer",
]
