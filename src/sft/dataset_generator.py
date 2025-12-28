"""Generate SFT datasets from author corpora for style transfer fine-tuning.

Based on research from:
- arXiv 2510.13939: "Readers Prefer Outputs of AI Trained on Copyrighted Books"
- Book SFT Pipeline: https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering

Key insights:
- 150-400 word chunks capture style better (transitions matter)
- 15 instruction templates prevent attention collapse
- 5 system prompt variants add diversity
- Conversation format enables instruction-following
"""

import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path

from ..utils.nlp import split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Instruction templates for variety (prevents memorization)
INSTRUCTION_TEMPLATES = [
    "Write a passage in the style of {author} about: {description}",
    "Channel {author}'s distinctive voice to describe: {description}",
    "Compose a scene as {author} would write it: {description}",
    "In {author}'s literary style, write about: {description}",
    "Emulate {author}'s prose to convey: {description}",
    "Write this scene with {author}'s characteristic rhythm: {description}",
    "Capture {author}'s voice in describing: {description}",
    "Using {author}'s narrative techniques, write: {description}",
    "In the manner of {author}, compose: {description}",
    "Write as {author} would about: {description}",
    "Adopt {author}'s style for this passage: {description}",
    "Channel the prose style of {author}: {description}",
    "Write with {author}'s distinctive patterns: {description}",
    "Compose in {author}'s voice: {description}",
    "Emulate {author}'s writing to describe: {description}",
]

# System prompts for variety
SYSTEM_PROMPTS = [
    "You are an expert creative writer capable of emulating specific literary styles with authentic voice.",
    "You are a literary author who can write in various styles while maintaining authentic voice and rhythm.",
    "You are a skilled writer who captures the essence of different authorial voices.",
    "You are an author capable of writing in various literary styles with authentic voice.",
    "You are a creative writer specializing in literary style emulation.",
]


@dataclass
class InstructionTemplate:
    """Template for generating training instructions."""

    instruction: str
    system_prompt: str

    def format(self, author: str, description: str) -> str:
        """Format the instruction with author and description."""
        return self.instruction.format(author=author, description=description)


@dataclass
class TrainingExample:
    """A single training example in conversation format."""

    system: str
    user: str
    assistant: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to conversation format for training."""
        return {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ]
        }

    def to_chatml(self) -> str:
        """Convert to ChatML format string."""
        return (
            f"<|im_start|>system\n{self.system}<|im_end|>\n"
            f"<|im_start|>user\n{self.user}<|im_end|>\n"
            f"<|im_start|>assistant\n{self.assistant}<|im_end|>"
        )


class DatasetGenerator:
    """Generate SFT datasets from author corpus.

    Key features:
    - Segments corpus into optimal 150-400 word chunks
    - Generates content descriptions for each chunk
    - Creates multiple variants per chunk with different templates
    - Outputs in conversation format for training
    """

    def __init__(
        self,
        llm_generate: Optional[Callable[[str], str]] = None,
        min_chunk_words: int = 150,
        max_chunk_words: int = 400,
        variants_per_chunk: int = 2,
    ):
        """Initialize the dataset generator.

        Args:
            llm_generate: Function to call LLM for instruction generation.
            min_chunk_words: Minimum words per chunk.
            max_chunk_words: Maximum words per chunk.
            variants_per_chunk: Number of template variants per chunk.
        """
        self.llm_generate = llm_generate
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.variants_per_chunk = variants_per_chunk

    def segment_corpus(self, paragraphs: List[str]) -> List[str]:
        """Segment corpus into optimal chunks for style capture.

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            List of text chunks, each 150-400 words.
        """
        chunks = []
        current_chunk = []
        current_words = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_words = len(para.split())

            # Finalize chunk if adding this paragraph exceeds max
            if (current_words + para_words > self.max_chunk_words and
                    current_words >= self.min_chunk_words):
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap for continuity
                last_para = current_chunk[-1] if current_chunk else ""
                sentences = split_into_sentences(last_para)
                overlap = sentences[-1] if sentences else ""
                current_chunk = [overlap] if overlap else []
                current_words = len(overlap.split()) if overlap else 0

            current_chunk.append(para)
            current_words += para_words

        # Handle final chunk
        if current_chunk and current_words >= self.min_chunk_words:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(chunk_text)

        logger.info(f"Segmented corpus into {len(chunks)} chunks "
                   f"({self.min_chunk_words}-{self.max_chunk_words} words each)")
        return chunks

    def generate_instruction(self, chunk: str) -> str:
        """Generate a description of what happens in a chunk.

        Args:
            chunk: Text chunk from corpus.

        Returns:
            2-3 sentence description of the chunk content.
        """
        if not self.llm_generate:
            # Fallback: use first sentence as instruction
            sentences = split_into_sentences(chunk)
            if sentences:
                return f"Continue from: {sentences[0][:100]}..."
            return "Write a passage in the author's style."

        prompt = """Describe what happens in this passage in 2-3 sentences.
Focus on: characters present, actions, emotions, setting.
Do NOT quote the text directly. Be concise.

Passage:
{chunk}

Description:""".format(chunk=chunk[:1500])  # Limit input length

        try:
            response = self.llm_generate(prompt)
            # Clean up response
            description = response.strip()
            # Limit length
            if len(description) > 300:
                description = description[:300] + "..."
            return description
        except Exception as e:
            logger.warning(f"Instruction generation failed: {e}")
            sentences = split_into_sentences(chunk)
            if sentences:
                return f"Continue from: {sentences[0][:100]}..."
            return "Write a passage in the author's style."

    def create_training_example(
        self,
        chunk: str,
        instruction: str,
        author: str,
        template_idx: int,
        system_idx: int,
    ) -> TrainingExample:
        """Create a single training example.

        Args:
            chunk: Original text chunk (becomes assistant response).
            instruction: Generated description of chunk content.
            author: Author name.
            template_idx: Index for instruction template rotation.
            system_idx: Index for system prompt rotation.

        Returns:
            TrainingExample in conversation format.
        """
        template = INSTRUCTION_TEMPLATES[template_idx % len(INSTRUCTION_TEMPLATES)]
        system = SYSTEM_PROMPTS[system_idx % len(SYSTEM_PROMPTS)]

        user_message = template.format(author=author, description=instruction)

        return TrainingExample(
            system=system,
            user=user_message,
            assistant=chunk,
        )

    def generate_dataset(
        self,
        paragraphs: List[str],
        author: str,
        generate_instructions: bool = True,
    ) -> List[TrainingExample]:
        """Generate complete SFT dataset from corpus.

        Args:
            paragraphs: List of paragraph texts from corpus.
            author: Author name.
            generate_instructions: Whether to generate descriptions via LLM.

        Returns:
            List of TrainingExample objects.
        """
        # Segment corpus into chunks
        chunks = self.segment_corpus(paragraphs)

        if not chunks:
            logger.warning("No valid chunks generated from corpus")
            return []

        examples = []
        total_variants = len(chunks) * self.variants_per_chunk

        logger.info(f"Generating {total_variants} training examples "
                   f"({len(chunks)} chunks x {self.variants_per_chunk} variants)")

        for i, chunk in enumerate(chunks):
            # Generate instruction for this chunk
            if generate_instructions and self.llm_generate:
                instruction = self.generate_instruction(chunk)
            else:
                # Use heuristic instruction
                sentences = split_into_sentences(chunk)
                first_sentence = sentences[0] if sentences else ""
                instruction = f"a passage beginning with themes from: {first_sentence[:80]}..."

            # Create multiple variants with different templates
            for variant in range(self.variants_per_chunk):
                example = self.create_training_example(
                    chunk=chunk,
                    instruction=instruction,
                    author=author,
                    template_idx=i * self.variants_per_chunk + variant,
                    system_idx=i % len(SYSTEM_PROMPTS),
                )
                examples.append(example)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")

        logger.info(f"Generated {len(examples)} training examples")
        return examples

    def save_dataset(
        self,
        examples: List[TrainingExample],
        output_path: str,
        format: str = "jsonl",
    ) -> None:
        """Save dataset to file.

        Args:
            examples: List of training examples.
            output_path: Path to output file.
            format: Output format ('jsonl', 'json', or 'chatml').
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(path, "w") as f:
                for example in examples:
                    f.write(json.dumps(example.to_dict()) + "\n")

        elif format == "json":
            data = [example.to_dict() for example in examples]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "chatml":
            with open(path, "w") as f:
                for example in examples:
                    f.write(example.to_chatml() + "\n\n")

        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {len(examples)} examples to {path}")


def generate_sft_dataset(
    corpus_text: str,
    author: str,
    output_path: str,
    llm_generate: Optional[Callable[[str], str]] = None,
    min_chunk_words: int = 150,
    max_chunk_words: int = 400,
    variants_per_chunk: int = 2,
    format: str = "jsonl",
) -> List[TrainingExample]:
    """Convenience function to generate and save SFT dataset.

    Args:
        corpus_text: Full corpus text (will be split into paragraphs).
        author: Author name.
        output_path: Path to save dataset.
        llm_generate: Optional LLM function for instruction generation.
        min_chunk_words: Minimum words per chunk.
        max_chunk_words: Maximum words per chunk.
        variants_per_chunk: Number of variants per chunk.
        format: Output format ('jsonl', 'json', or 'chatml').

    Returns:
        List of generated TrainingExample objects.
    """
    # Split corpus into paragraphs
    paragraphs = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]

    if not paragraphs:
        raise ValueError("No paragraphs found in corpus text")

    # Create generator
    generator = DatasetGenerator(
        llm_generate=llm_generate,
        min_chunk_words=min_chunk_words,
        max_chunk_words=max_chunk_words,
        variants_per_chunk=variants_per_chunk,
    )

    # Generate examples
    examples = generator.generate_dataset(
        paragraphs=paragraphs,
        author=author,
        generate_instructions=(llm_generate is not None),
    )

    # Save to file
    generator.save_dataset(examples, output_path, format)

    return examples
