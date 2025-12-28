"""Vocabulary control for style transfer.

The LoRA pipeline uses RepetitionReducer for post-processing:
- Tracks word usage across the document
- Replaces overused words with synonyms
- Replaces LLM-speak with simpler alternatives
"""

from .repetition_reducer import RepetitionReducer, ReductionStats

__all__ = [
    "RepetitionReducer",
    "ReductionStats",
]
