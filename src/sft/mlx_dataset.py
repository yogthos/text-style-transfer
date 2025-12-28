"""MLX-optimized dataset generation for LoRA style transfer training.

This module generates training data optimized for teaching a model to transfer
style while preserving content. Key insight: train on (content_description,
styled_passage) pairs where the description is style-neutral, forcing the model
to learn style separately from content.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path

from ..utils.nlp import split_into_sentences, split_into_paragraphs
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Prompt to generate neutral paraphrase (same content, no distinctive style)
NEUTRALIZE_PROMPT = """Convert distinctive prose to plain neutral English. Keep the SAME word count.

EXAMPLE:
Distinctive (32 words): "The cosmos is vast beyond imagining. We are, each of us, a tiny speck in an ocean of stars. And yet - in this vastness - our minds can grasp infinity itself."
Neutral (32 words): "The universe is very large. Each person is a small part of a huge number of stars. Despite this large scale, human minds are able to understand the concept of infinity."

NOW DO THIS:
Distinctive ({word_count} words): "{chunk}"
Neutral ({word_count} words):"""

# Fallback: simple description for when LLM unavailable
CONTENT_EXTRACTION_PROMPT = """Describe the content of this passage in neutral, factual terms.

Focus ONLY on:
- Main claims or arguments made
- Key entities mentioned (people, places, concepts)
- Logical relationships (X causes Y, A contrasts with B)
- Any specific examples or evidence cited

Do NOT describe:
- Writing style, tone, or rhythm
- Sentence structure or word choice
- Rhetorical techniques used

Be concise (2-4 sentences).

Passage:
{chunk}

Neutral content description:"""


@dataclass
class MLXTrainingExample:
    """A training example optimized for MLX LoRA training.

    Format is designed to teach style transfer:
    - System: Minimal, just sets the author identity
    - User: Style-neutral content description
    - Assistant: Actual styled passage from corpus
    """

    system: str
    user: str
    assistant: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MLX-compatible format."""
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ]
        }

    def to_text_format(self) -> str:
        """Convert to simple text format for training."""
        return (
            f"<|system|>\n{self.system}\n"
            f"<|user|>\n{self.user}\n"
            f"<|assistant|>\n{self.assistant}"
        )


class MLXDatasetGenerator:
    """Generate training data optimized for style transfer with MLX.

    Key insight: Train on (content_description, styled_passage) pairs.
    This teaches the model to render ANY content in the author's style,
    rather than memorizing corpus content.
    """

    def __init__(
        self,
        llm_generate: Optional[Callable[[str], str]] = None,
        min_chunk_words: int = 50,
        max_chunk_words: int = 200,
    ):
        """Initialize the MLX dataset generator.

        Args:
            llm_generate: Function to call LLM for content description generation.
                         Signature: (prompt: str) -> str
            min_chunk_words: Minimum words per training chunk.
            max_chunk_words: Maximum words per training chunk.

        Note: Smaller chunks (50-200 words) match inference paragraph size,
        teaching the model to work at the same granularity.
        """
        self.llm_generate = llm_generate
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words

    def segment_corpus(self, paragraphs: List[str]) -> List[str]:
        """Segment corpus into optimal chunks for style capture.

        Chunks are 150-400 words to capture:
        - Paragraph-level rhythm and flow
        - Sentence transitions (where style lives)
        - Sufficient context for coherent generation

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            List of text chunks.
        """
        chunks = []
        current_chunk = []
        current_words = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_words = len(para.split())

            # Check if adding this paragraph exceeds max
            if (current_words + para_words > self.max_chunk_words and
                    current_words >= self.min_chunk_words):
                # Finalize current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk - no overlap for cleaner training
                current_chunk = []
                current_words = 0

            current_chunk.append(para)
            current_words += para_words

        # Handle final chunk
        if current_chunk and current_words >= self.min_chunk_words:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(chunk_text)

        logger.info(
            f"Segmented corpus into {len(chunks)} chunks "
            f"(target: {self.min_chunk_words}-{self.max_chunk_words} words)"
        )
        return chunks

    def generate_neutral_paraphrase(self, chunk: str) -> str:
        """Generate a neutral paraphrase of the chunk.

        Creates a plain English version with the same content but no
        distinctive stylistic features. This becomes the training INPUT,
        teaching the model to transform neutral text into styled text.

        Args:
            chunk: Text chunk from corpus (author's styled text).

        Returns:
            Neutral paraphrase with same content, no distinctive style.
        """
        word_count = len(chunk.split())

        if not self.llm_generate:
            # Fallback: return chunk as-is (less effective but works)
            logger.warning("No LLM available for paraphrase generation")
            return chunk

        prompt = NEUTRALIZE_PROMPT.format(chunk=chunk[:2000], word_count=word_count)

        try:
            response = self.llm_generate(prompt)
            paraphrase = response.strip()

            # Strip any prefix like 'Neutral (47 words): "...'
            paraphrase = re.sub(r'^Neutral\s*\(\d+\s*words?\)\s*:\s*', '', paraphrase)
            # Remove surrounding quotes if present
            if paraphrase.startswith('"') and paraphrase.endswith('"'):
                paraphrase = paraphrase[1:-1]
            paraphrase = paraphrase.strip()

            # Ensure we got a reasonable paraphrase
            para_words = len(paraphrase.split())
            if para_words < word_count * 0.5 or para_words > word_count * 1.5:
                logger.warning(f"Paraphrase length mismatch: {para_words} vs {word_count}")
                # Try to use it anyway if it's reasonable
                if para_words < 20:
                    return chunk  # Fallback to original

            return paraphrase

        except Exception as e:
            logger.warning(f"Neutral paraphrase generation failed: {e}")
            return chunk  # Fallback to original

    def _heuristic_description(self, chunk: str) -> str:
        """Generate heuristic content description without LLM.

        Extracts key nouns, verbs, and creates a simple description.

        Args:
            chunk: Text chunk.

        Returns:
            Simple content description.
        """
        from ..utils.nlp import extract_keywords

        keywords = extract_keywords(chunk, top_n=8)
        sentences = split_into_sentences(chunk)

        # Use first sentence as topic anchor
        topic = sentences[0][:100] if sentences else ""

        if keywords:
            keyword_str = ", ".join(keywords[:5])
            return f"A passage about {keyword_str}. {topic}"
        else:
            return f"Continue from: {topic}"

    def create_training_example(
        self,
        chunk: str,
        neutral_input: str,
        author: str,
    ) -> MLXTrainingExample:
        """Create a single training example.

        Training uses neutral text as input and styled text as output:
        - Input: Plain, neutral paraphrase (no distinctive style)
        - Output: Author's original text (with distinctive style)

        This teaches the model to TRANSFORM neutral text into the author's
        distinctive voice, rather than just copying.

        Args:
            chunk: Original text chunk in author's style (becomes output).
            neutral_input: Neutral paraphrase of the chunk (becomes input).
            author: Author name.

        Returns:
            MLXTrainingExample ready for training.
        """
        # System prompt emphasizes style transformation
        system = f"You write exactly like {author}. Transform the text into {author}'s distinctive voice and style."

        # Input: neutral text, Output: author's styled text
        # This teaches: plain text → distinctive author style
        return MLXTrainingExample(
            system=system,
            user=neutral_input,  # Neutral/plain version
            assistant=chunk,     # Author's original styled text
        )

    def generate_dataset(
        self,
        paragraphs: List[str],
        author: str,
        generate_paraphrases: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[MLXTrainingExample]:
        """Generate complete MLX training dataset from corpus.

        Each example pairs a neutral paraphrase (input) with the author's
        original styled text (output), teaching style transformation.

        Args:
            paragraphs: List of paragraph texts from corpus.
            author: Author name.
            generate_paraphrases: Whether to generate LLM paraphrases.
            progress_callback: Optional callback (current, total).

        Returns:
            List of MLXTrainingExample objects.
        """
        # Segment into chunks
        chunks = self.segment_corpus(paragraphs)

        if not chunks:
            logger.warning("No valid chunks generated from corpus")
            return []

        logger.info(f"Generating {len(chunks)} training examples for {author}")
        if generate_paraphrases and self.llm_generate:
            logger.info("Generating neutral paraphrases for each chunk (this may take a while)...")

        examples = []

        for i, chunk in enumerate(chunks):
            # Generate neutral paraphrase as training input
            if generate_paraphrases and self.llm_generate:
                neutral_input = self.generate_neutral_paraphrase(chunk)
            else:
                # Fallback: use chunk as-is (less effective)
                neutral_input = chunk

            # Create training example: neutral → styled
            example = self.create_training_example(
                chunk=chunk,           # Author's original (output)
                neutral_input=neutral_input,  # Neutral version (input)
                author=author,
            )
            examples.append(example)

            # Progress reporting
            if progress_callback:
                progress_callback(i + 1, len(chunks))

            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")

        logger.info(f"Generated {len(examples)} training examples")
        return examples

    def save_dataset(
        self,
        examples: List[MLXTrainingExample],
        output_path: str,
        split_validation: float = 0.1,
    ) -> Dict[str, str]:
        """Save dataset in MLX-compatible format.

        Creates train and validation splits for proper training.

        Args:
            examples: List of training examples.
            output_path: Base path for output files.
            split_validation: Fraction for validation set.

        Returns:
            Dict with paths to train and validation files.
        """
        import random

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Shuffle and split
        shuffled = examples.copy()
        random.shuffle(shuffled)

        val_size = max(1, int(len(shuffled) * split_validation))
        train_examples = shuffled[val_size:]
        val_examples = shuffled[:val_size]

        # Determine output paths
        train_path = str(path.with_suffix('.train.jsonl'))
        val_path = str(path.with_suffix('.valid.jsonl'))

        # Save training set
        with open(train_path, 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example.to_dict()) + '\n')

        # Save validation set
        with open(val_path, 'w') as f:
            for example in val_examples:
                f.write(json.dumps(example.to_dict()) + '\n')

        logger.info(
            f"Saved {len(train_examples)} training and "
            f"{len(val_examples)} validation examples"
        )

        return {
            'train': train_path,
            'valid': val_path,
        }


def generate_mlx_dataset(
    corpus_text: str,
    author: str,
    output_path: str,
    llm_generate: Optional[Callable[[str], str]] = None,
    min_chunk_words: int = 150,
    max_chunk_words: int = 400,
) -> Dict[str, str]:
    """Convenience function to generate MLX training dataset.

    Args:
        corpus_text: Full corpus text.
        author: Author name.
        output_path: Base path for output files.
        llm_generate: Optional LLM function for description generation.
        min_chunk_words: Minimum words per chunk.
        max_chunk_words: Maximum words per chunk.

    Returns:
        Dict with paths to train and validation files.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]

    if not paragraphs:
        raise ValueError("No paragraphs found in corpus text")

    # Create generator
    generator = MLXDatasetGenerator(
        llm_generate=llm_generate,
        min_chunk_words=min_chunk_words,
        max_chunk_words=max_chunk_words,
    )

    # Generate examples
    examples = generator.generate_dataset(
        paragraphs=paragraphs,
        author=author,
        generate_paraphrases=(llm_generate is not None),
    )

    # Save dataset
    return generator.save_dataset(examples, output_path)
