# Text Style Transfer

Transform text to match a target author's writing style while preserving semantic meaning. Uses LoRA-adapted language models for fast, consistent style transfer with a critic/repair loop to ensure content fidelity.

## Features

- **LoRA-Based Generation**: Fine-tuned adapters capture author style in model weights
- **Critic/Repair Loop**: DeepSeek validates content preservation and fixes issues
- **Proposition Validation**: Ensures all facts and entities are preserved
- **Fast Transfer**: ~15-30 seconds per paragraph
- **Post-Processing**: Reduces word repetition and removes LLM-speak

## Requirements

- Python 3.9+
- Apple Silicon Mac (for MLX-based training/inference)
- ~18GB RAM for inference, ~50GB for training
- DeepSeek API key (for critic/repair loop)

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

### 1. Configure

Copy `config.json.sample` to `config.json` and add your DeepSeek API key:

```json
{
  "llm": {
    "provider": {
      "writer": "mlx",
      "critic": "deepseek"
    },
    "providers": {
      "deepseek": {
        "api_key": "your-api-key",
        "model": "deepseek-chat"
      },
      "mlx": {
        "model": "mlx-community/Qwen3-8B-4bit"
      }
    }
  }
}
```

### 2. Train a LoRA Adapter

Training is a two-step process:

```bash
# Step 1: Generate training data (instruction back-translation)
python scripts/neutralize_corpus.py \
    --input data/corpus/author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name"

# Step 2: Train LoRA adapter
python scripts/train_mlx_lora.py \
    --from-neutralized data/neutralized/author.jsonl \
    --author "Author Name" \
    --train \
    --output lora_adapters/author
```

### 3. Transfer Text

```bash
python restyle.py input.txt -o output.txt \
    --adapter lora_adapters/author \
    --author "Author Name"
```

## Pipeline Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────┐
│ 1. PARAGRAPH SPLITTING      │  Split into paragraphs
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 2. LORA GENERATION (MLX)    │  Generate styled text
│    - Style in adapter       │  using LoRA weights
│    - ~5-10s per paragraph   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 3. PROPOSITION VALIDATION   │  Check all facts/entities
│    - Extract propositions   │  are preserved
│    - Identify missing info  │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 4. CRITIC REPAIR (DeepSeek) │  Surgical fixes for:
│    - Missing entities       │  - Missing facts
│    - Hallucinations        │  - Grammar issues
│    - Up to 3 attempts      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ 5. POST-PROCESSING          │  Reduce repetition
│    - Remove LLM-speak       │  Replace overused words
└─────────────────────────────┘
    │
    ▼
Output Text
```

## Commands

### `restyle.py` - Style Transfer

```bash
python restyle.py <input> -o <output> --adapter <path> --author "<name>"

# Options:
#   --adapter PATH      Path to LoRA adapter directory
#   --author NAME       Author name (or auto-detected from adapter metadata)
#   --temperature FLOAT Generation temperature (default: 0.7)
#   --no-verify         Disable entailment verification
#   -c, --config PATH   Config file path (default: config.json)
#   -v, --verbose       Verbose output
#   --list-adapters     List available adapters
```

### `neutralize_corpus.py` - Generate Training Data

Uses instruction back-translation to create training pairs:

```bash
python scripts/neutralize_corpus.py \
    --input data/corpus/author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name" \
    --workers 4  # For Ollama only; MLX is single-GPU
```

Options:
| Option | Default | Description |
|--------|---------|-------------|
| `--min-words` | 250 | Minimum words per chunk |
| `--max-words` | 650 | Maximum words per chunk |
| `--workers` | 1 | Parallel workers (Ollama only) |
| `--llm` | mlx | LLM provider (mlx or ollama:model) |

### `train_mlx_lora.py` - Train LoRA Adapter

```bash
python scripts/train_mlx_lora.py \
    --from-neutralized data/neutralized/author.jsonl \
    --author "Author Name" \
    --train \
    --output lora_adapters/author
```

Options:
| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 1 | Training epochs (1 is often enough with curated data) |
| `--batch-size` | 1 | Batch size |
| `--learning-rate` | 1e-4 | Learning rate |
| `--rank` | 16 | LoRA rank |
| `--alpha` | 32 | LoRA alpha scaling |
| `--resume` | - | Resume from last checkpoint |

### `curate_corpus.py` - Corpus Curation (Optional)

For large corpuses, curate to ~0.9M tokens for optimal training:

```bash
python scripts/curate_corpus.py \
    --input data/corpus/author_full.txt \
    --output data/corpus/author.txt \
    --target-tokens 900000
```

## Project Structure

```
text-style-transfer/
├── restyle.py                # Main CLI for style transfer
├── config.json               # Configuration (copy from .sample)
├── requirements.txt          # Python dependencies
├── prompts/                  # Prompt templates (editable)
│   ├── style_transfer_system.txt    # Main style transfer prompt
│   ├── critic_repair_system.txt     # Critic repair prompt
│   └── ...                          # Other prompts
├── src/
│   ├── config.py             # Configuration loading
│   ├── generation/           # Core generation
│   │   ├── lora_generator.py # MLX LoRA inference
│   │   └── fast_transfer.py  # Pipeline orchestration
│   ├── validation/           # Content verification
│   │   ├── semantic_verifier.py    # Semantic fidelity
│   │   ├── proposition_validator.py # Fact checking
│   │   └── quality_critic.py       # Quality issues
│   ├── vocabulary/           # Post-processing
│   │   └── repetition_reducer.py
│   ├── llm/                  # LLM providers
│   │   ├── mlx_provider.py   # MLX local inference
│   │   ├── deepseek.py       # DeepSeek API
│   │   └── ollama.py         # Ollama local
│   ├── ingestion/            # Proposition extraction
│   ├── corpus/               # Corpus loading
│   └── utils/                # NLP utilities, prompt loading
├── scripts/
│   ├── neutralize_corpus.py  # Generate training data
│   ├── train_mlx_lora.py     # LoRA training
│   ├── curate_corpus.py      # Corpus curation
│   └── clean_sample.py       # Text cleaning utility
├── lora_adapters/            # Trained adapters
└── tests/                    # Unit tests (264 tests)
```

## Customizing Prompts

All prompts used by the pipeline are stored as text files in the `prompts/` directory. You can edit these files to customize behavior without modifying code.

Key prompts:
- `style_transfer_system.txt` - Main system prompt for style generation
- `critic_repair_system.txt` - System prompt for the critic/repair loop
- `critic_repair_user.txt` - User prompt template for repairs
- `repair_strict.txt` - Strict repair prompt for content preservation

Prompts use Python's `{variable}` syntax for substitution. Available variables depend on the prompt (e.g., `{author}`, `{instructions}`).

## Training Your Own Adapter

### Step 1: Prepare Corpus

Create a plain text file with the author's writing:
- **Recommended size**: 50KB-500KB (0.9M tokens optimal)
- **Format**: Clean paragraphs separated by blank lines
- **Remove**: Headers, footnotes, non-prose content

For large corpuses, curate first:
```bash
python scripts/curate_corpus.py --input full.txt --output curated.txt
```

### Step 2: Generate Training Data

```bash
python scripts/neutralize_corpus.py \
    --input data/corpus/author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name"
```

This generates content descriptions using instruction back-translation. The model learns to generate styled text from content descriptions.

### Step 3: Train LoRA Adapter

```bash
python scripts/train_mlx_lora.py \
    --from-neutralized data/neutralized/author.jsonl \
    --author "Author Name" \
    --train \
    --output lora_adapters/author
```

### Step 4: Test Transfer

```bash
python restyle.py test.txt -o output.txt \
    --adapter lora_adapters/author \
    --author "Author Name" \
    -v
```

## Performance

| Metric | Value |
|--------|-------|
| Time per paragraph | 15-30 seconds |
| Memory (inference, 4-bit) | ~8GB |
| Memory (training, bf16) | ~50GB |
| Training data generation | ~30-60 min per corpus |
| Training time | ~15-30 min |

## Troubleshooting

### MLX Not Available

This project requires Apple Silicon. For other platforms, use Ollama as the LLM provider.

### Out of Memory During Training

Use 4-bit model in config.json:
```json
"mlx": {
  "model": "mlx-community/Qwen3-8B-4bit"
}
```

Or reduce batch size:
```bash
python scripts/train_mlx_lora.py --train ... --batch-size 1
```

### Missing DeepSeek API Key

The critic/repair loop requires a DeepSeek API key. Set it in config.json or as environment variable:
```bash
export DEEPSEEK_API_KEY="your-key"
```

### Resume Interrupted Training

```bash
python scripts/train_mlx_lora.py \
    --from-neutralized data/neutralized/author.jsonl \
    --author "Author" \
    --train \
    --output lora_adapters/author \
    --resume
```

### spaCy Model Missing

```bash
python -m spacy download en_core_web_sm
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## License

MIT License - See LICENSE file for details.
