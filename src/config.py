"""Configuration management for the style transfer pipeline."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StyleBlendingWeights:
    """Fitness weights when style blending is enabled."""
    enabled_weight: float = 0.15
    content_with_blending: float = 0.35
    length_with_blending: float = 0.18
    transition_with_blending: float = 0.12
    vocabulary_with_blending: float = 0.12
    fluency_with_blending: float = 0.08


@dataclass
class FitnessWeightsConfig:
    """Configuration for fitness function weights."""
    content: float = 0.40
    length: float = 0.20
    transition: float = 0.15
    vocabulary: float = 0.15
    fluency: float = 0.10
    style_blending: StyleBlendingWeights = field(default_factory=StyleBlendingWeights)


@dataclass
class ThresholdsConfig:
    """Configuration for various strictness thresholds."""
    overuse_word_count: int = 3  # Word appearing more than this is "overused"
    severe_overuse_count: int = 5  # Severe overuse penalty threshold
    entailment_score: float = 0.5  # Min entailment for semantic preservation
    delta_score: float = 1.5  # Burrows' Delta threshold
    content_preservation_min: float = 0.5  # Min content preservation ratio
    novelty_min: float = 0.95  # Min novelty for anachronistic tests
    anachronistic_pass_rate: float = 0.9  # Min pass rate for style generalization


@dataclass
class BlendingConfig:
    """Configuration for SLERP-based author style blending."""
    enabled: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_dir: str = "centroid_cache/"
    authors: Dict[str, float] = field(default_factory=dict)  # author_name -> weight


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider."""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120


@dataclass
class LLMProviderRoles:
    """Configuration for role-based LLM provider assignment.

    Allows using different providers for different tasks:
    - writer: Fast local model for style generation (e.g., MLX with LoRA)
    - critic: Smarter API model for validation and repair (e.g., DeepSeek)
    """
    writer: str = "mlx"  # Provider for generation
    critic: str = "deepseek"  # Provider for critique/repair


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProviderRoles = field(default_factory=LLMProviderRoles)
    providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    max_retries: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0

    def get_provider_config(self, provider_name: str) -> LLMProviderConfig:
        """Get configuration for a specific provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        return self.providers[provider_name]

    def get_writer_provider(self) -> str:
        """Get the provider name for generation/writing tasks."""
        return self.provider.writer

    def get_critic_provider(self) -> str:
        """Get the provider name for critique/repair tasks."""
        return self.provider.critic


@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB."""
    persist_path: str = "atlas_cache/"
    embedding_model: str = "all-mpnet-base-v2"


@dataclass
class CorpusConfig:
    """Configuration for corpus processing."""
    min_sentences_per_paragraph: int = 2
    style_audit_threshold: int = 4
    opener_percentage: float = 0.15
    closer_percentage: float = 0.15


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_repair_attempts: int = 3  # Max critic repair attempts per paragraph
    repair_temperature: float = 0.3  # Low temperature for precise edits
    entailment_threshold: float = 0.7  # Min score for semantic preservation
    proposition_threshold: float = 0.7  # Min proposition coverage
    anchor_threshold: float = 0.8  # Min content anchor coverage
    repetition_threshold: int = 3  # Words used N+ times get replaced


@dataclass
class SemanticValidationConfig:
    """Configuration for semantic validation."""
    min_proposition_coverage: float = 0.9
    max_hallucinated_entities: int = 0
    require_citation_preservation: bool = True


@dataclass
class StatisticalValidationConfig:
    """Configuration for statistical validation."""
    length_tolerance: float = 0.2
    burstiness_tolerance: float = 0.3
    min_vocab_match: float = 0.5


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    semantic: SemanticValidationConfig = field(default_factory=SemanticValidationConfig)
    statistical: StatisticalValidationConfig = field(default_factory=StatisticalValidationConfig)


@dataclass
class ContextBudgetConfig:
    """Token budget configuration for context management."""
    system_prompt_max: int = 1500
    user_prompt_max: int = 600
    keep_last_n_sentences: int = 5
    max_conversation_tokens: int = 8000


@dataclass
class VoiceInjectionConfig:
    """Configuration for voice profile injection into generation."""
    enabled: bool = True
    assertiveness_weight: float = 0.7  # How much to weight assertiveness patterns
    rhetorical_weight: float = 0.8  # How much to weight rhetorical patterns


@dataclass
class StyleConfig:
    """Configuration for style transfer settings."""
    perspective: str = "preserve"  # preserve, first_person_singular, first_person_plural, third_person
    voice_injection: VoiceInjectionConfig = field(default_factory=VoiceInjectionConfig)
    blending: BlendingConfig = field(default_factory=BlendingConfig)

    def validate_perspective(self) -> bool:
        """Check if perspective setting is valid."""
        valid_perspectives = {
            "preserve",
            "first_person_singular",
            "first_person_plural",
            "third_person",
        }
        return self.perspective in valid_perspectives


@dataclass
class Config:
    """Main configuration container."""
    fitness_weights: FitnessWeightsConfig = field(default_factory=FitnessWeightsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    context_budget: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    log_level: str = "INFO"
    log_json: bool = False


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in string values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.environ.get(env_var, "")
        if not resolved:
            logger.warning(f"Environment variable {env_var} not set")
        return resolved
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _parse_llm_provider_config(data: Dict) -> LLMProviderConfig:
    """Parse LLM provider configuration."""
    return LLMProviderConfig(
        api_key=_resolve_env_vars(data.get("api_key", "")),
        base_url=data.get("base_url", ""),
        model=data.get("model", ""),
        max_tokens=data.get("max_tokens", 4096),
        temperature=data.get("temperature", 0.7),
        timeout=data.get("timeout", 120),
    )


def _parse_llm_config(data: Dict) -> LLMConfig:
    """Parse LLM configuration section."""
    providers = {}
    for name, provider_data in data.get("providers", {}).items():
        providers[name] = _parse_llm_provider_config(provider_data)

    retry_config = data.get("retry", {})
    provider_data = data.get("provider", {})

    provider_roles = LLMProviderRoles(
        writer=provider_data.get("writer", "mlx"),
        critic=provider_data.get("critic", "deepseek"),
    )

    return LLMConfig(
        provider=provider_roles,
        providers=providers,
        max_retries=retry_config.get("max_attempts", 5),
        base_delay=retry_config.get("base_delay", 2.0),
        max_delay=retry_config.get("max_delay", 60.0),
    )


def load_config(config_path: str = "config.json") -> Config:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please copy config.json.sample to config.json and configure it."
        )

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

    # Parse each section
    config = Config()

    # Parse fitness weights
    if "fitness_weights" in data:
        fw_data = data["fitness_weights"]
        blending_data = fw_data.get("style_blending", {})
        config.fitness_weights = FitnessWeightsConfig(
            content=fw_data.get("content", 0.40),
            length=fw_data.get("length", 0.20),
            transition=fw_data.get("transition", 0.15),
            vocabulary=fw_data.get("vocabulary", 0.15),
            fluency=fw_data.get("fluency", 0.10),
            style_blending=StyleBlendingWeights(
                enabled_weight=blending_data.get("enabled_weight", 0.15),
                content_with_blending=blending_data.get("content_with_blending", 0.35),
                length_with_blending=blending_data.get("length_with_blending", 0.18),
                transition_with_blending=blending_data.get("transition_with_blending", 0.12),
                vocabulary_with_blending=blending_data.get("vocabulary_with_blending", 0.12),
                fluency_with_blending=blending_data.get("fluency_with_blending", 0.08),
            ),
        )

    # Parse thresholds
    if "thresholds" in data:
        th_data = data["thresholds"]
        config.thresholds = ThresholdsConfig(
            overuse_word_count=th_data.get("overuse_word_count", 3),
            severe_overuse_count=th_data.get("severe_overuse_count", 5),
            entailment_score=th_data.get("entailment_score", 0.5),
            delta_score=th_data.get("delta_score", 1.5),
            content_preservation_min=th_data.get("content_preservation_min", 0.5),
            novelty_min=th_data.get("novelty_min", 0.95),
            anachronistic_pass_rate=th_data.get("anachronistic_pass_rate", 0.9),
        )

    if "llm" in data:
        config.llm = _parse_llm_config(data["llm"])

    if "chromadb" in data:
        config.chromadb = ChromaDBConfig(
            persist_path=data["chromadb"].get("persist_path", "atlas_cache/"),
            embedding_model=data["chromadb"].get("embedding_model", "all-mpnet-base-v2"),
        )

    if "corpus" in data:
        config.corpus = CorpusConfig(
            min_sentences_per_paragraph=data["corpus"].get("min_sentences_per_paragraph", 2),
            style_audit_threshold=data["corpus"].get("style_audit_threshold", 4),
            opener_percentage=data["corpus"].get("opener_percentage", 0.15),
            closer_percentage=data["corpus"].get("closer_percentage", 0.15),
        )

    if "generation" in data:
        config.generation = GenerationConfig(
            max_repair_attempts=data["generation"].get("max_repair_attempts", 3),
            repair_temperature=data["generation"].get("repair_temperature", 0.3),
            entailment_threshold=data["generation"].get("entailment_threshold", 0.7),
            proposition_threshold=data["generation"].get("proposition_threshold", 0.7),
            anchor_threshold=data["generation"].get("anchor_threshold", 0.8),
            repetition_threshold=data["generation"].get("repetition_threshold", 3),
        )

    if "style" in data:
        style_data = data["style"]
        voice_data = style_data.get("voice_injection", {})
        blending_data = style_data.get("blending", {})
        config.style = StyleConfig(
            perspective=style_data.get("perspective", "preserve"),
            voice_injection=VoiceInjectionConfig(
                enabled=voice_data.get("enabled", True),
                assertiveness_weight=voice_data.get("assertiveness_weight", 0.7),
                rhetorical_weight=voice_data.get("rhetorical_weight", 0.8),
            ),
            blending=BlendingConfig(
                enabled=blending_data.get("enabled", False),
                embedding_model=blending_data.get("embedding_model", "all-MiniLM-L6-v2"),
                cache_dir=blending_data.get("cache_dir", "centroid_cache/"),
                authors=blending_data.get("authors", {}),
            ),
        )
        if not config.style.validate_perspective():
            logger.warning(
                f"Invalid perspective '{config.style.perspective}', using 'preserve'"
            )
            config.style.perspective = "preserve"

    if "validation" in data:
        val_data = data["validation"]
        if "semantic" in val_data:
            config.validation.semantic = SemanticValidationConfig(
                min_proposition_coverage=val_data["semantic"].get("min_proposition_coverage", 0.9),
                max_hallucinated_entities=val_data["semantic"].get("max_hallucinated_entities", 0),
                require_citation_preservation=val_data["semantic"].get("require_citation_preservation", True),
            )
        if "statistical" in val_data:
            config.validation.statistical = StatisticalValidationConfig(
                length_tolerance=val_data["statistical"].get("length_tolerance", 0.2),
                burstiness_tolerance=val_data["statistical"].get("burstiness_tolerance", 0.3),
                min_vocab_match=val_data["statistical"].get("min_vocab_match", 0.5),
            )

    if "context_budget" in data:
        config.context_budget = ContextBudgetConfig(
            system_prompt_max=data["context_budget"].get("system_prompt_max", 1500),
            user_prompt_max=data["context_budget"].get("user_prompt_max", 600),
            keep_last_n_sentences=data["context_budget"].get("keep_last_n_sentences", 5),
            max_conversation_tokens=data["context_budget"].get("max_conversation_tokens", 8000),
        )

    config.log_level = data.get("log_level", "INFO")
    config.log_json = data.get("log_json", False)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def create_default_config() -> Dict:
    """Create a default configuration dictionary."""
    return {
        "llm": {
            "provider": {
                "writer": "mlx",
                "critic": "deepseek"
            },
            "providers": {
                "deepseek": {
                    "api_key": "${DEEPSEEK_API_KEY}",
                    "base_url": "https://api.deepseek.com",
                    "model": "deepseek-chat",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "timeout": 120
                },
                "mlx": {
                    "model": "mlx-community/Qwen3-8B-4bit",
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "llama3",
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            },
            "retry": {
                "max_attempts": 5,
                "base_delay": 2,
                "max_delay": 60
            }
        },
        "generation": {
            "max_repair_attempts": 3,
            "repair_temperature": 0.3,
            "entailment_threshold": 0.7,
            "proposition_threshold": 0.7,
            "anchor_threshold": 0.8,
            "repetition_threshold": 3
        },
        "log_level": "INFO"
    }
