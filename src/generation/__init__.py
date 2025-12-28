"""Generation module for LoRA-based style transfer.

The primary pipeline uses LoRA adapters for fast, consistent style transfer:
- LoRAStyleGenerator: Core generation using MLX LoRA adapters
- FastStyleTransfer: High-level pipeline orchestration
"""

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

__all__ = [
    # LoRA generation
    "LoRAStyleGenerator",
    "GenerationConfig",
    "AdapterMetadata",
    # Fast transfer pipeline
    "FastStyleTransfer",
    "TransferConfig",
    "TransferStats",
    "PropositionExtractor",
    "create_fast_transfer",
]
