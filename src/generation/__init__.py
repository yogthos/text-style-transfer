"""Generation module for data-driven style transfer."""

from .evolutionary_generator import (
    EvolutionarySentenceGenerator,
    EvolutionaryParagraphGenerator,
    Candidate,
    GenerationState,
)
from .data_driven_generator import (
    DataDrivenStyleTransfer,
    TransferResult,
    DocumentTransferResult,
    create_transfer_pipeline,
    load_profile_and_create_transfer,
)

# LoRA-based fast transfer (optional - requires MLX)
try:
    from .lora_generator import (
        LoRAStyleGenerator,
        GenerationConfig,
        AdapterMetadata,
    )
    from .fast_transfer import (
        FastStyleTransfer,
        TransferConfig,
        TransferStats,
        PropositionExtractor,
        create_fast_transfer,
    )
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

__all__ = [
    # Evolutionary generation
    "EvolutionarySentenceGenerator",
    "EvolutionaryParagraphGenerator",
    "Candidate",
    "GenerationState",
    # Data-driven transfer pipeline
    "DataDrivenStyleTransfer",
    "TransferResult",
    "DocumentTransferResult",
    "create_transfer_pipeline",
    "load_profile_and_create_transfer",
    # LoRA-based fast transfer
    "LORA_AVAILABLE",
]

# Add LoRA exports if available
if LORA_AVAILABLE:
    __all__.extend([
        "LoRAStyleGenerator",
        "GenerationConfig",
        "AdapterMetadata",
        "FastStyleTransfer",
        "TransferConfig",
        "TransferStats",
        "PropositionExtractor",
        "create_fast_transfer",
    ])
