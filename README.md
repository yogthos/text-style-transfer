# Text Style Transfer

Transform text to match a target author's writing style while preserving semantic meaning. Uses LoRA-adapted language models for fast, consistent style transfer.

## Features

- **LoRA-Based Generation**: Fine-tuned adapters capture author style in model weights
- **Faithful Translation**: Training approach ensures output matches input length (no hallucination)
- **Fast Transfer**: ~5-10 seconds per paragraph
- **Semantic Preservation**: NLI-based entailment verification ensures meaning is preserved
- **Post-Processing**: Reduces word repetition and removes LLM-speak

## Requirements

- Python 3.9+
- Apple Silicon Mac (for MLX-based training/inference)
- ~8GB RAM for inference, ~16GB for training

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

Edit `config.json` to set your MLX base model:

```json
{
  "llm": {
    "providers": {
      "mlx": {
        "model": "mlx-community/Qwen3-8B-bf16",
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9
      }
    }
  }
}
```

**Model options:**
| Model | Size | Quality | Use Case |
|-------|------|---------|----------|
| `mlx-community/Qwen3-8B-bf16` | ~16GB | Best | Production training |
| `mlx-community/Qwen3-8B-4bit` | ~4.3GB | Good | Limited memory |

> **Important:** Use base models (not Instruct). Instruct-tuned models have response patterns that resist style overwriting. Base models are blank canvases for LoRA adaptation.

### 2. Train a LoRA Adapter

Training is a two-step process: neutralize the corpus, then train.

```bash
# Step 1: Convert author's distinctive text to neutral paraphrases
python scripts/neutralize_corpus.py \
    --input data/corpus/author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name"

# Step 2: Train LoRA adapter on neutral → author pairs
python scripts/train_mlx_lora.py \
    --from-neutralized data/neutralized/author.jsonl \
    --author "Author Name" \
    --train \
    --output lora_adapters/author
```

### 3. Transfer Text

```bash
# Transform text to author's style
python cli.py transfer input.txt --author "Author Name" --output output.txt
```

## Commands

### `transfer` - Style Transfer

Transform input text to match an author's style:

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
| `--no-adapter` | false | Use base model without LoRA (for testing) |

### `train` - Train LoRA Adapter

Training uses the self-contained MLX pipeline (no external services needed).

**Step 1: Neutralize corpus** - Convert author's distinctive text to plain English:

```bash
python scripts/neutralize_corpus.py \
    --input data/corpus/author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name"
```

Neutralization options:
| Option | Default | Description |
|--------|---------|-------------|
| `--min-words` | 50 | Minimum words per chunk |
| `--max-words` | 150 | Maximum words per chunk |
| `--llm` | mlx | LLM provider (mlx or ollama:model) |

**Step 2: Train LoRA adapter** on neutral → author pairs:

```bash
python scripts/train_mlx_lora.py \
    --from-neutralized data/neutralized/author.jsonl \
    --author "Author Name" \
    --train \
    --output lora_adapters/author \
    --epochs 3
```

Training options:
| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 2 | Batch size |
| `--learning-rate` | 5e-5 | Learning rate |
| `--rank` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha scaling |

### `analyze` - Analyze Text

Get style metrics for any text file:

```bash
python cli.py analyze <file>

# Example output:
Analysis of: essay.txt

Structure:
  Paragraphs: 5
  Sentences: 23
  Words: 412

Sentence Length:
  Mean: 17.9 words
  Std: 8.2
  Range: 5 - 42 words
  Burstiness: 0.458
```

### `list` - List Adapters

Show available trained adapters:

```bash
python cli.py list

# Output:
Available LoRA adapters:
  mao: rank=16
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
│    - Word count constraint  │  Output ≈ input length
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

## How Training Works

The training approach uses **neutral → author pairs** to teach pure style transformation:

### Training Data Format

```
Input:  "The universe is very large. Each person is a small part of many stars."
Output: "The cosmos is vast beyond imagining. We are, each of us, tiny specks in an ocean of stars."
```

The model learns to transform plain English into the author's distinctive voice.

### Why This Works

1. **Base model as blank canvas**: No pre-existing instruction patterns to fight against
2. **Neutral input**: Generic text has no style - the model must add all stylistic elements
3. **Meaning preservation**: Input and output express the same ideas, just differently
4. **Length awareness**: Training pairs have similar word counts

### Training Process

```
Author Corpus (distinctive prose)
         │
         ▼
    ┌─────────────┐
    │ Neutralize  │  MLX base model converts to plain English
    └─────────────┘
         │
         ▼
  Neutral → Author Pairs
         │
         ▼
    ┌─────────────┐
    │ LoRA Train  │  Adapter learns style transformation
    └─────────────┘
         │
         ▼
   Trained Adapter
```

This differs from naive approaches that fine-tune on author text directly, which can lead to memorization rather than style learning.

## Project Structure

```
text-style-transfer/
├── cli.py                    # Command-line interface
├── requirements.txt          # Python dependencies
├── src/
│   ├── generation/           # Core generation
│   │   ├── lora_generator.py # MLX LoRA inference
│   │   └── fast_transfer.py  # Pipeline orchestration
│   ├── validation/           # Content verification
│   │   ├── entailment.py     # NLI-based checking
│   │   └── semantic_verifier.py
│   ├── vocabulary/           # Post-processing
│   │   └── repetition_reducer.py
│   ├── sft/                  # Training data generation
│   │   └── mlx_dataset.py    # Dataset for MLX LoRA
│   ├── llm/                  # LLM providers
│   │   ├── mlx_provider.py   # MLX local (self-contained)
│   │   ├── deepseek.py       # DeepSeek API
│   │   └── ollama.py         # Ollama local
│   ├── corpus/               # Corpus loading
│   └── utils/                # NLP utilities
├── scripts/
│   ├── neutralize_corpus.py  # Convert corpus to neutral pairs
│   ├── train_mlx_lora.py     # LoRA training script
│   └── fast_restyle.py       # Direct inference script
├── lora_adapters/            # Trained adapters
└── archive/                  # Deprecated modules
```

## Training Your Own Adapter

### Step 1: Prepare Corpus

Create a plain text file with the author's writing:
- **Recommended size**: 50KB+ of text (more = better style capture)
- **Format**: Clean paragraphs separated by blank lines
- **Remove**: Headers, footnotes, citations, non-prose content

Place the corpus in `data/corpus/author_name.txt`.

### Step 2: Configure Base Model

Edit `config.json` to select your base model:

```json
{
  "llm": {
    "providers": {
      "mlx": {
        "model": "mlx-community/Qwen3-8B-bf16",
        "max_tokens": 512,
        "temperature": 0.3
      }
    }
  }
}
```

**Why base models?** Instruct-tuned models have learned response patterns (helpfulness, safety guardrails) that resist style overwriting. Base models have no such patterns - they're blank canvases that fully absorb the author's voice during LoRA training.

### Step 3: Neutralize Corpus (Once Per Author)

Convert the author's distinctive prose to plain neutral English. This creates training pairs: neutral input → author-style output.

```bash
python scripts/neutralize_corpus.py \
    --input data/corpus/my_author.txt \
    --output data/neutralized/my_author.jsonl \
    --author "Author Name"
```

This runs the base model to paraphrase each chunk into generic prose. Takes ~1-2 minutes per 10KB of corpus.

> **Note:** The neutralized corpus is model-agnostic. You only need to run this once per author, then you can train adapters on different base models using the same `author.jsonl` file.

### Step 4: Train LoRA Adapter

Train the adapter on neutral → author pairs:

```bash
python scripts/train_mlx_lora.py \
    --from-neutralized data/neutralized/my_author.jsonl \
    --author "Author Name" \
    --train \
    --output lora_adapters/my_author \
    --epochs 3
```

Training takes ~10-15 minutes for a typical corpus. The adapter learns to transform neutral text into the author's distinctive voice.

### Step 5: Test Transfer

```bash
python cli.py transfer test_input.txt --author "Author Name"
```

### Tips for Best Results

- **More data is better**: 100KB+ corpus gives richer style capture
- **Use bf16 model**: Higher precision = better LoRA quality
- **Clean corpus**: Remove non-prose elements that could confuse the model
- **Adjust epochs**: 3-5 epochs typically works well; more can overfit
- **Lower temperature**: Use 0.3-0.5 for more consistent output

## Performance

| Metric | Value |
|--------|-------|
| Time per paragraph | 5-10 seconds |
| LLM calls per paragraph | 1 |
| Memory (inference, 4-bit) | ~8GB |
| Memory (inference, bf16) | ~18GB |
| Memory (training, bf16) | ~20GB |
| Neutralization time | ~1-2 min per 10KB |
| Training time (3 epochs) | ~10-15 min |

## Troubleshooting

### MLX Not Available

This project requires Apple Silicon. For other platforms, the architecture supports adding PyTorch backends.

### Out of Memory During Training

Use 4-bit model instead of bf16:
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

Or reduce batch size:
```bash
python scripts/train_mlx_lora.py --train ... --batch-size 1
```

### Neutralization Output Contains Thinking/Reasoning Text

The base model may output reasoning before the actual response. The script automatically filters this, but if issues persist:
- Check that you're using a base model (not Instruct)
- The filtering in `neutralize_corpus.py` handles common patterns

### Adapter Trained on Wrong Model

LoRA adapters are model-specific. If you switch base models (e.g., from 4-bit to bf16), you must retrain the adapter. However, you can reuse the neutralized corpus - it's model-agnostic:

```bash
# No need to re-neutralize! Just retrain the adapter on the new model
python scripts/train_mlx_lora.py \
    --from-neutralized data/neutralized/author.jsonl \
    --author "Author" \
    --train \
    --output lora_adapters/author_bf16
```

### Low Quality Output

- **Use bf16 model**: Higher precision = better style capture
- **More training data**: 50KB+ corpus recommended
- **Lower temperature**: `--temperature 0.3` for more consistent output
- **More epochs**: `--epochs 5` (but watch for overfitting)

### spaCy Model Missing

```bash
python -m spacy download en_core_web_sm
```

## License

MIT License - See LICENSE file for details.
