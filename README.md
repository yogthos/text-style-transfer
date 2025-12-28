# Text Style Transfer

Transform text to match a target author's writing style while preserving semantic meaning. Uses graph-based semantic representation and LLM generation to produce stylistically accurate output.

## Features

- **Style Analysis**: Extract statistical fingerprints from author corpora (sentence length, burstiness, vocabulary patterns)
- **Semantic Preservation**: Graph-based representation ensures meaning is preserved during transformation
- **Rhythm Matching**: Reproduces the target author's sentence length variation patterns
- **Validation Critics**: Multi-critic validation ensures output quality (semantic, style, fluency, length)
- **RAG-Enhanced Generation**: ChromaDB index enables retrieval of similar style patterns

## Installation

### 1. Clone and Setup Virtual Environment

```bash
git clone <repository-url>
cd text-style-transfer

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate # for bash
source venv/bin/activate.fish # for fish
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (large model with word vectors)
python -m spacy download en_core_web_lg
```

### 2. Configure LLM Provider

Copy the sample configuration and edit with your API credentials:

```bash
cp config.json.sample config.json
```

Edit `config.json` with your settings:

```json
{
  "llm": {
    "provider": "deepseek",
    "providers": {
      "deepseek": {
        "api_key": "${DEEPSEEK_API_KEY}",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "max_tokens": 4096,
        "temperature": 0.7
      },
      "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3"
      }
    }
  },
  "chromadb": {
    "persist_path": "atlas_cache/"
  }
}
```

Set your API key as an environment variable:

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

## Usage

### Quick Start

```bash
# 1. Ingest an author's corpus
python cli.py ingest styles/sample_sagan.txt --author "Carl Sagan"

# 2. Transform text to that author's style
python cli.py transfer input.txt --author "Carl Sagan" --output output.txt
```

### Commands

#### `ingest` - Build Style Index

Load an author's corpus and build a style profile:

```bash
python cli.py ingest <corpus_path> --author "<Author Name>"

# Examples:
python cli.py ingest styles/sample_hitchens.txt --author "Christopher Hitchens"
python cli.py ingest styles/ --author "Mixed Authors"  # Directory of files
```

This creates:
- Style profile with statistical metrics (sentence length, burstiness, vocabulary)
- ChromaDB index of paragraph structures for RAG retrieval

#### `transfer` - Transform Text

Transform input text to match an author's style:

```bash
python cli.py transfer <input_file> --author "<Author Name>" [--output <output_file>]

# Examples:
python cli.py transfer essay.txt --author "Carl Sagan"
python cli.py transfer essay.txt --author "Carl Sagan" --output transformed.txt
python cli.py transfer essay.txt --author "Carl Sagan" --max-revisions 3
```

Options:
- `--output, -o`: Output file (prints to stdout if not specified)
- `--max-revisions, -r`: Maximum revision attempts per sentence (default: 2)

#### `analyze` - Analyze Text Style

Get style metrics for any text file:

```bash
python cli.py analyze <file>

# Example:
python cli.py analyze essay.txt
```

Output:
```
Analysis of: essay.txt

Structure:
  Paragraphs: 5
  Sentences: 23
  Words: 412

Style Metrics:
  Average sentence length: 17.9 words
  Sentence length range: 5 - 42 words
  Burstiness: 0.342
  Avg dependency depth: 4.52
  Perspective: third_person

POS Distribution:
  NOUN: 15.4%
  PUNCT: 12.1%
  VERB: 10.4%
  ADP: 10.0%
  PRON: 9.6%

Top Vocabulary:
  concept, system, process, analysis, method, ...
```

#### `list-authors` - Show Indexed Authors

```bash
python cli.py list-authors
```

### Configuration Options

Full `config.json` options:

```json
{
  "llm": {
    "provider": "deepseek",
    "providers": {
      "deepseek": {
        "api_key": "${DEEPSEEK_API_KEY}",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "max_tokens": 4096,
        "temperature": 0.7,
        "timeout": 120
      }
    },
    "retry": {
      "max_attempts": 5,
      "base_delay": 2,
      "max_delay": 60
    }
  },
  "chromadb": {
    "persist_path": "atlas_cache/",
    "embedding_model": "all-mpnet-base-v2"
  },
  "generation": {
    "max_repair_retries": 5,
    "rag_examples_count": 5,
    "length_tolerance": 0.2
  },
  "validation": {
    "semantic": {
      "min_proposition_coverage": 0.9,
      "max_hallucinated_entities": 0
    },
    "statistical": {
      "length_tolerance": 0.2,
      "burstiness_tolerance": 0.3,
      "min_vocab_match": 0.5
    }
  },
  "log_level": "INFO"
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input Text                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Corpus Ingestion                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Preprocessor │→ │   Analyzer   │→ │   Indexer    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Semantic Deconstruction                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Proposition │→ │ Relationship │→ │    Graph     │          │
│  │  Extractor   │  │  Detector    │  │   Builder    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Planning                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Graph Matcher│→ │    Rhythm    │→ │   Sentence   │          │
│  │   (RAG)      │  │   Planner    │  │   Planner    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: Generation & Validation                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Prompt     │→ │  Sentence    │→ │   Critic     │          │
│  │   Builder    │  │  Generator   │  │   Panel      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Styled Output                              │
└─────────────────────────────────────────────────────────────────┘
```

## Included Sample Corpora

The `styles/` directory contains sample corpora from various authors:

| Author | File | Size |
|--------|------|------|
| Richard Dawkins | `sample_dawkins.txt` | ~800KB |
| Carl Sagan | `sample_sagan.txt` | ~700KB |
| Richard Feynman | `sample_feyman.txt` | ~470KB |
| Christopher Hitchens | `sample_hitchens.txt` | ~180KB |
| Douglas Hofstadter | `sample_hofstadter.txt` | ~900KB |
| H.P. Lovecraft | `sample_lovecraft.txt` | ~2.6MB |
| Michael Parenti | `sample_parenti.txt` | ~325KB |

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/unit/ -v

# Run specific test file
python -m pytest tests/unit/test_critics.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html
```

### Project Structure

```
text-style-transfer/
├── cli.py                 # Command-line interface
├── config.json            # Configuration (create from sample)
├── config.json.sample     # Sample configuration
├── requirements.txt       # Python dependencies
├── src/
│   ├── config.py          # Configuration loading
│   ├── corpus/            # Corpus processing
│   │   ├── loader.py      # Load corpus files
│   │   ├── preprocessor.py # Text preprocessing
│   │   ├── analyzer.py    # Statistical analysis
│   │   ├── profiler.py    # Style profile creation
│   │   └── indexer.py     # ChromaDB indexing
│   ├── ingestion/         # Semantic analysis
│   │   ├── proposition_extractor.py  # SVO extraction
│   │   ├── relationship_detector.py  # Edge detection
│   │   ├── graph_builder.py          # Graph construction
│   │   └── context_analyzer.py       # Global context
│   ├── planning/          # Sentence planning
│   │   ├── graph_matcher.py   # RAG matching
│   │   ├── rhythm_planner.py  # Length patterns
│   │   └── sentence_planner.py # Plan creation
│   ├── generation/        # Text generation
│   │   ├── prompt_builder.py     # Prompt construction
│   │   ├── sentence_generator.py # LLM generation
│   │   └── critics.py            # Validation
│   ├── llm/               # LLM providers
│   │   ├── provider.py    # Base class
│   │   ├── deepseek.py    # DeepSeek API
│   │   └── ollama.py      # Ollama local
│   ├── models/            # Data models
│   │   ├── graph.py       # Semantic graph
│   │   ├── plan.py        # Sentence plan
│   │   └── style.py       # Style profile
│   └── utils/             # Utilities
│       ├── nlp.py         # NLP utilities
│       └── logging.py     # Structured logging
├── styles/                # Sample author corpora
└── tests/                 # Test suite
    └── unit/              # Unit tests
```

## Key Concepts

### Burstiness
A measure of sentence length variation (coefficient of variation). Low burstiness (~0.1) indicates uniform sentence lengths; high burstiness (~0.4+) indicates varied lengths like Hitchens' punchy style.

### Semantic Graph
Represents text as propositions (subject-verb-object triples) connected by relationships (CAUSES, CONTRASTS, ELABORATES, etc.). Ensures meaning preservation during transformation.

### Style DNA
A natural language description of an author's characteristic patterns, vocabulary preferences, and rhetorical strategies.

## Troubleshooting

### ChromaDB Issues

If you encounter ChromaDB errors:
```bash
# Clear the index and rebuild
rm -rf atlas_cache/
python cli.py ingest <corpus> --author "<Author>"
```

### spaCy Model Missing

```bash
python -m spacy download en_core_web_lg
```

### Rate Limits

The system includes exponential backoff for API rate limits. Adjust in config:
```json
"retry": {
  "max_attempts": 10,
  "base_delay": 5,
  "max_delay": 120
}
```

## License

MIT License - See LICENSE file for details.
