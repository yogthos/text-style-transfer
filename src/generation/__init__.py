"""Generation module for LoRA-based style transfer.

The primary pipeline uses LoRA adapters for fast, consistent style transfer:
- LoRAStyleGenerator: Core generation using MLX LoRA adapters
- StyleTransfer: High-level pipeline with semantic graph validation
- DocumentContext: Document-level context for improved coherence
"""

from .lora_generator import (
    LoRAStyleGenerator,
    GenerationConfig,
    AdapterMetadata,
)
from .transfer import (
    StyleTransfer,
    TransferConfig,
    TransferStats,
    create_style_transfer,
)
from .document_context import (
    DocumentContext,
    DocumentContextExtractor,
    extract_document_context,
)

__all__ = [
    # LoRA generation
    "LoRAStyleGenerator",
    "GenerationConfig",
    "AdapterMetadata",
    # Style transfer pipeline
    "StyleTransfer",
    "TransferConfig",
    "TransferStats",
    "create_style_transfer",
    # Document context
    "DocumentContext",
    "DocumentContextExtractor",
    "extract_document_context",
]
