"""SFT (Supervised Fine-Tuning) dataset generation for author style transfer.

Based on research from arXiv 2510.13939 and the Book SFT Pipeline.
"""

from .dataset_generator import (
    InstructionTemplate,
    DatasetGenerator,
    generate_sft_dataset,
)
from .mlx_dataset import (
    MLXTrainingExample,
    MLXDatasetGenerator,
    generate_mlx_dataset,
)

__all__ = [
    # Original dataset generator
    "InstructionTemplate",
    "DatasetGenerator",
    "generate_sft_dataset",
    # MLX-optimized generator
    "MLXTrainingExample",
    "MLXDatasetGenerator",
    "generate_mlx_dataset",
]
