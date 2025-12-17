# Text Style Transfer Pipeline

A sophisticated text style transfer system that transforms input text to match the style of a sample text while preserving semantic meaning. The pipeline uses a multi-phase approach combining statistical analysis, semantic extraction, and LLM-based generation with validation and retry mechanisms.

## Overview

The system implements a "Symphony Pipeline" with five main components:

- **The Composer (Input)**: Extracts semantic meaning from input text
- **The Theorist (Analyzer)**: Analyzes sample text to extract style characteristics
- **The Conductor (Planner)**: Plans sentence structure based on context and style
- **The Musician (LLM)**: Generates style-transferred text using constraints
- **The Critic (Validator)**: Scores output and retries if quality is insufficient

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone or navigate to the project directory:
```bash
cd text-style-transfer
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data (done automatically on first run, but can be done manually):
```bash
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('vader_lexicon')"
```

## Configuration

The project uses `config.json` for configuration. Here's the structure:

```json
{
  "provider": "deepseek",
  "deepseek": {
    "api_key": "your-api-key-here",
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "editor_model": "deepseek-chat",
    "critic_model": "deepseek-chat"
  },
  "sample": {
    "file": "prompts/sample_mao.txt"
  }
}
```

### Configuration Options

- **provider**: LLM provider to use (`"deepseek"` or `"ollama"`)
- **deepseek.api_key**: Your DeepSeek API key (get one at https://platform.deepseek.com)
- **deepseek.api_url**: DeepSeek API endpoint (usually the default)
- **deepseek.editor_model**: Model for text generation (default: `"deepseek-chat"`)
- **sample.file**: Path to the sample text file that defines the target style

### Getting an API Key

1. Sign up at https://platform.deepseek.com
2. Navigate to API keys section
3. Create a new API key
4. Copy the key and paste it into `config.json`

## Usage

### Command Line Interface (Recommended)

Use the main CLI entry point:

```bash
source venv/bin/activate
python3 restyle.py input/small.md -o output/small.md
```

With additional options:

```bash
# Specify a custom sample file
python3 restyle.py input/small.md -o output/small.md --sample prompts/custom_sample.txt

# Adjust retry and threshold settings
python3 restyle.py input/small.md -o output/small.md --max-retries 5 --score-threshold 0.8

# Enable verbose output
python3 restyle.py input/small.md -o output/small.md -v
```

### CLI Options

- `input`: Input text file to transform (required)
- `-o, --output`: Output file path (required)
- `-s, --sample`: Sample text file defining target style (optional, uses config.json default)
- `-c, --config`: Configuration file path (default: `config.json`)
- `--max-retries`: Maximum retry attempts per sentence (default: 3)
- `--score-threshold`: Minimum score to accept output (default: 0.75)
- `-v, --verbose`: Enable verbose output

### Alternative: Using the Helper Script

You can also use the simpler helper script:

```bash
source venv/bin/activate
python3 run_pipeline.py input/small.md output/small.md
```

### Python API

Use the Python API directly:

```bash
source venv/bin/activate
python3 -c "from src.pipeline import run_pipeline; run_pipeline(input_file='input/small.md', output_file='output/small.md')"
```

### Programmatic Usage

```python
from src.pipeline import run_pipeline, process_text

# Using file paths
output = run_pipeline(
    input_file="input/small.md",
    output_file="output/small.md"
)

# Using text directly
input_text = "Your input text here."
sample_text = "Your style sample text here."
output = process_text(
    input_text=input_text,
    sample_text=sample_text,
    config_path="config.json"
)
```

## Project Structure

```
text-style-transfer/
├── src/
│   ├── models.py              # Data structures (StyleProfile, ContentUnit)
│   ├── analyzer/
│   │   ├── vocabulary.py       # Vocabulary mapping
│   │   └── structure.py        # Markov models for structure
│   ├── ingestion/
│   │   └── semantic.py         # Semantic meaning extraction
│   ├── planner/
│   │   └── flow.py             # Template selection
│   ├── generator/
│   │   └── llm_interface.py    # LLM generation
│   ├── validator/
│   │   └── scorer.py           # Output scoring
│   └── pipeline.py             # Main pipeline with feedback loop
├── tests/                       # Test files
├── input/                       # Input text files
├── output/                      # Generated output files
├── prompts/                    # Sample style files
├── config.json                  # Configuration file
├── requirements.txt             # Python dependencies
└── run_pipeline.py              # Convenience script
```

## How It Works

1. **Style Analysis**: Analyzes the sample text to extract:
   - Vocabulary mappings (generic → style-specific synonyms)
   - POS tag transition probabilities
   - Sentence type flow patterns

2. **Meaning Extraction**: Parses input text to extract:
   - Subject-Verb-Object triples
   - Named entities
   - Sentiment polarity

3. **Structure Planning**: Selects appropriate sentence templates based on:
   - Previous sentence structure
   - Input sentiment
   - Statistical patterns from sample text

4. **Constrained Generation**: Uses LLM to generate text that:
   - Preserves semantic meaning
   - Follows the planned structure
   - Uses style-specific vocabulary

5. **Validation & Retry**: Scores output on:
   - Meaning preservation (BERTScore-like similarity)
   - Style matching (POS distribution)
   - Structure adherence
   - Retries up to 3 times if score is below threshold

## Testing

Run all tests:

```bash
source venv/bin/activate
python3 tests/test_models.py
python3 tests/test_vocabulary.py
python3 tests/test_structure.py
python3 tests/test_semantic.py
python3 tests/test_flow.py
python3 tests/test_llm_interface.py
python3 tests/test_validator.py
```

Or run them all at once:

```bash
source venv/bin/activate
for test in tests/test_*.py; do python3 "$test"; done
```

Note: Some tests require a valid API key in `config.json` and will be skipped if unavailable.

## Dependencies

- **nltk**: Natural language processing and tokenization
- **numpy**: Numerical computations for Markov models
- **scikit-learn**: Machine learning utilities
- **sentence-transformers**: Semantic similarity calculations
- **requests**: HTTP requests for API calls

See `requirements.txt` for complete list and versions.

## Troubleshooting

### API Key Issues

If you see API errors:
1. Verify your API key in `config.json`
2. Check that you have sufficient API credits
3. Ensure the API endpoint URL is correct

### NLTK Data Missing

If you see NLTK resource errors:
```bash
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('vader_lexicon')"
```

### Import Errors

Make sure you're running from the project root directory and the virtual environment is activated.

### Low Quality Output

The pipeline includes retry mechanisms, but if output quality is consistently low:
- Try a longer or more representative sample text
- Adjust `score_threshold` in `process_text()` (default: 0.75)
- Increase `max_retries` (default: 3)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

