# Text Style Transfer

Transform text to match a target author's writing style while preserving semantic meaning. Uses LoRA-adapted language models for fast, consistent style transfer.

## Features

- **LoRA-Based Generation**: Fine-tuned adapters capture author style in model weights
- **Human-Like Output**: Instruction-based training produces text that passes AI detection
- **Fast Transfer**: ~5-10 seconds per paragraph
- **Semantic Preservation**: NLI-based entailment verification ensures meaning is preserved
- **Post-Processing**: Reduces word repetition and removes LLM-speak

## Requirements

- Python 3.9+
- Apple Silicon Mac (for MLX-based training/inference)
- ~18GB RAM for inference (bf16), ~50GB for training

## Installation

```bash
# Clone repository
git clone <repository-url>
cd text-style-transfer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Configure Base Model

Copy `config.json.sample` to `config.json`:

```json
{
  "llm": {
    "providers": {
      "mlx": {
        "model": "mlx-community/Qwen3-8B-Base-bf16",
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9
      }
    }
  }
}
```

**Model options:**
| Model | Size | Memory (Training) | Use Case |
|-------|------|-------------------|----------|
| `mlx-community/Qwen3-8B-Base-bf16` | ~16GB | ~50GB | Best quality |
| `mlx-community/Qwen3-8B-4bit` | ~4GB | ~20GB | Limited memory |

> **Important:** Use **base models only** (not Instruct). Instruct-tuned models have response patterns that resist style overwriting. Base models are blank canvases for LoRA adaptation.

### 2. Train a LoRA Adapter

Training is a two-step process: describe the corpus, then train.

```bash
# Step 1: Generate content descriptions from author's text
python scripts/describe_corpus.py \
    --input data/corpus/author.txt \
    --output data/described/author.jsonl \
    --author "Author Name" \
    --workers 4

# Step 2: Train LoRA adapter
python scripts/train_mlx_lora.py \
    --from-described data/described/author.jsonl \
    --author "Author Name" \
    --train \
    --output lora_adapters/author
```

### 3. Transfer Text

```bash
python cli.py transfer input.txt --author "Author Name" --output output.txt
```

## How Training Works

The key insight: **"Style lives in the transitions."**

### Instruction-Based Training

Unlike naive approaches that train on `neutral_text → styled_text` (which just teaches word substitution), this system uses **instruction-based training**:

```
Input:  "The passage discusses the vastness of space and humanity's small place within it."
Output: "The cosmos is vast beyond imagining. We are, each of us, tiny specks in an ocean of stars."
```

The model learns to **generate style from scratch** given only a content description, not to transform existing text.

### Key Techniques

1. **Overlapping chunks**: Text is segmented with 2-sentence overlap between chunks to capture transition patterns where style emerges
2. **Template rotation**: 15 different prompt templates prevent memorization and force attention on stylistic patterns
3. **Plain text format**: Base models use simple text completion, not chat templates
4. **Moderate learning rate**: 5e-5 to prevent catastrophic forgetting while still learning style

### Training Data Format

The `describe_corpus.py` script:
1. Segments the corpus into 150-400 word chunks with overlap
2. Generates 2-3 sentence descriptions of each chunk using rotating templates
3. Creates training pairs: `description → original_styled_text`

```
Training example:
{
  "text": "Write in the style of Mao.\n\n{description}\n\n{original_text}"
}
```

## Commands

### `describe_corpus.py` - Prepare Training Data

Generate instruction descriptions from author's corpus:

```bash
python scripts/describe_corpus.py \
    --input data/corpus/author.txt \
    --output data/described/author.jsonl \
    --author "Author Name" \
    --workers 4
```

Options:
| Option | Default | Description |
|--------|---------|-------------|
| `--min-words` | 150 | Minimum words per chunk |
| `--max-words` | 400 | Maximum words per chunk |
| `--overlap` | 2 | Overlap sentences between chunks |
| `--workers` | 1 | Parallel workers (2-4 for MLX, 8+ for Ollama) |
| `--llm` | mlx | LLM provider (mlx or ollama:model) |

### `train_mlx_lora.py` - Train LoRA Adapter

```bash
python scripts/train_mlx_lora.py \
    --from-described data/described/author.jsonl \
    --author "Author Name" \
    --train \
    --output lora_adapters/author
```

Options:
| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 5 | Training epochs |
| `--batch-size` | 2 | Batch size |
| `--learning-rate` | 5e-5 | Learning rate (moderate to prevent forgetting) |
| `--rank` | 32 | LoRA rank (higher = more capacity) |
| `--alpha` | 64 | LoRA alpha scaling |
| `--resume` | - | Resume from last checkpoint |

### `transfer` - Style Transfer

```bash
python cli.py transfer <input_file> --author "<Author Name>" [options]

# Examples:
python cli.py transfer essay.txt --author "Mao"
python cli.py transfer essay.txt --author "Mao" --output styled.txt
python cli.py transfer essay.txt --author "Mao" --temperature 0.5
```

Options:
| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | stdout | Output file path |
| `--adapter` | `lora_adapters/<author>` | Custom adapter path |
| `--temperature, -t` | 0.5 | Generation temperature |
| `--threshold` | 0.7 | Entailment threshold for repair |
| `--no-verify` | false | Skip entailment verification |
| `--no-repair` | false | Skip repair attempts |
| `--no-reduce-repetition` | false | Skip post-processing |

### `analyze` - Analyze Text

```bash
python cli.py analyze <file>
```

### `list` - List Adapters

```bash
python cli.py list
```

## Architecture

```
Input Text (N words)
    │
    ▼
┌─────────────────────────────┐
│ 1. PARAGRAPH SPLITTING      │  Split into paragraphs
│    - Filter headings        │  for processing
│    - Track word counts      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 2. LORA GENERATION          │  Single forward pass with
│    - Style in adapter       │  author-specific LoRA weights
│    - Plain text prompt      │  Output ≈ input length
│    - ~5-10s per paragraph   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 3. CONTENT VERIFICATION     │  NLI entailment check
│    - Score > 0.7 = pass     │  ensures meaning preserved
│    - Repair if needed       │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 4. POST-PROCESSING          │  Replace overused words
│    - Reduce repetition      │  with synonyms
│    - Remove LLM-speak       │
└─────────────────────────────┘
    │
    ▼
Output Text (≈N words)
```

## Project Structure

```
text-style-transfer/
├── cli.py                    # Command-line interface
├── config.json               # Configuration (copy from .sample)
├── requirements.txt          # Python dependencies
├── src/
│   ├── generation/           # Core generation
│   │   ├── lora_generator.py # MLX LoRA inference
│   │   └── fast_transfer.py  # Pipeline orchestration
│   ├── validation/           # Content verification
│   │   └── semantic_verifier.py
│   ├── vocabulary/           # Post-processing
│   │   └── repetition_reducer.py
│   ├── llm/                  # LLM providers
│   │   └── mlx_provider.py   # MLX local (self-contained)
│   ├── corpus/               # Corpus loading
│   └── utils/                # NLP utilities
├── scripts/
│   ├── describe_corpus.py    # Generate training descriptions
│   ├── train_mlx_lora.py     # LoRA training script
│   └── neutralize_corpus.py  # Legacy: neutral paraphrases
├── lora_adapters/            # Trained adapters
└── tests/                    # Unit tests
```

## Training Your Own Adapter

### Step 1: Prepare Corpus

Create a plain text file with the author's writing:
- **Recommended size**: 50KB+ of text (more = better style capture)
- **Format**: Clean paragraphs separated by blank lines
- **Remove**: Headers, footnotes, citations, non-prose content

### Step 2: Generate Descriptions

```bash
python scripts/describe_corpus.py \
    --input data/corpus/my_author.txt \
    --output data/described/my_author.jsonl \
    --author "My Author" \
    --workers 4
```

This generates content descriptions using rotating templates. Takes ~5-10 minutes for 50KB corpus with 4 workers.

### Step 3: Train LoRA Adapter

```bash
python scripts/train_mlx_lora.py \
    --from-described data/described/my_author.jsonl \
    --author "My Author" \
    --train \
    --output lora_adapters/my_author
```

Training takes ~15-30 minutes. Monitor the loss:
- Initial train loss: ~2-3
- Final train loss: ~0.5-1.5
- If loss stays high (>5), there may be a format issue

### Step 4: Test Transfer

```bash
python cli.py transfer test_input.txt --author "My Author"
```

### Step 5: Verify with AI Detector (Optional)

Test output with GPTZero or similar. Well-trained adapters produce text that reads as human-written.

## Performance

| Metric | Value |
|--------|-------|
| Time per paragraph | 5-10 seconds |
| LLM calls per paragraph | 1 |
| Memory (inference, bf16) | ~18GB |
| Memory (training, bf16, rank 32) | ~50GB |
| Description generation | ~5-10 min per 50KB |
| Training time (5 epochs) | ~30-45 min |

## Troubleshooting

### MLX Not Available

This project requires Apple Silicon. For other platforms, the architecture supports adding PyTorch backends.

### Out of Memory During Training

Option 1: Use 4-bit model:
```json
{
  "llm": {
    "providers": {
      "mlx": {
        "model": "mlx-community/Qwen3-8B-4bit"
      }
    }
  }
}
```

Option 2: Reduce batch size:
```bash
python scripts/train_mlx_lora.py --train ... --batch-size 1
```

### High Loss Values During Training

If train loss stays above 5-7:
- Ensure you're using `--from-described` (not `--from-neutralized`)
- Verify the base model is a **base** model (not Instruct)
- Check training data format has `"text"` field (not `"messages"`)

### Resume Interrupted Training

```bash
python scripts/train_mlx_lora.py \
    --from-described data/described/author.jsonl \
    --author "Author" \
    --train \
    --output lora_adapters/author \
    --resume
```

### Output Detected as AI-Generated

- Use `describe_corpus.py` (instruction-based), not `neutralize_corpus.py`
- Ensure overlapping chunks: `--overlap 2` or higher
- Verify template rotation is working (check `template_idx` varies in output)
- Use higher LoRA rank: `--rank 32`
- Use moderate learning rate: `--learning-rate 5e-5` (higher rates cause model corruption)

### spaCy Model Missing

```bash
python -m spacy download en_core_web_sm
```

## License

MIT License - See LICENSE file for details.
