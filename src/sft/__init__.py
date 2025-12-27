"""SFT (Supervised Fine-Tuning) dataset generation for author style transfer.

Based on research from arXiv 2510.13939 and the Book SFT Pipeline.
"""

from .dataset_generator import (
    InstructionTemplate,
    DatasetGenerator,
    generate_sft_dataset,
)

__all__ = [
    "InstructionTemplate",
    "DatasetGenerator",
    "generate_sft_dataset",
]
