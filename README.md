# Text Style Transfer Pipeline

A sophisticated text style transfer system that transforms input text to match the style of a sample text while preserving semantic meaning. The pipeline uses a **Style Atlas** architecture with RAG-based dual retrieval, combining semantic embeddings, style clustering, and LLM-based generation with adversarial validation.

## Overview

The system implements a **Style Atlas Pipeline** with the following components:

- **Style Atlas Builder**: Builds a dual-vector index (semantic + style) from sample text using ChromaDB
- **Style Navigator**: Uses Markov chains to predict style cluster transitions and retrieves relevant references
- **Dual RAG Retrieval**:
  - **Situation Match**: Finds semantically similar paragraphs for vocabulary grounding
  - **Structure Match**: Finds paragraphs matching target style cluster with length constraints
- **Prompt Assembler**: Constructs constrained prompts that separate vocabulary and structure guidance
- **Adversarial Critic**: LLM-based evaluator that provides feedback for iterative refinement

## Installation

### Prerequisites

- Python 3.12 or higher (required for ChromaDB compatibility)
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
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"
```

5. Download spaCy model (optional, for better dependency parsing):
```bash
python3 -m spacy download en_core_web_sm
```

## Configuration

The project uses `config.json` for configuration. Here's the complete structure:

```json
{
  "provider": "deepseek",
  "ollama": {
    "url": "http://localhost:11434/api/generate",
    "editor_model": "qwen3:32b",
    "critic_model": "deepseek-r1:8b"
  },
  "deepseek": {
    "api_key": "your-api-key-here",
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "editor_model": "deepseek-chat",
    "critic_model": "deepseek-chat"
  },
  "sample": {
    "file": "styles/sample_sagan.txt"
  },
  "atlas": {
    "persist_path": "atlas_cache/",
    "num_clusters": 5,
    "similarity_threshold": 0.3
  },
  "critic": {
    "min_score": 0.75,
    "min_pipeline_score": 0.6,
    "max_retries": 5,
    "max_pipeline_retries": 2
  },
  "blend": {
    "authors": ["Sagan"],
    "ratio": 0.5
  }
}
```

### Configuration Options

#### Provider Settings

- **provider**: LLM provider to use (`"deepseek"` or `"ollama"`)

#### DeepSeek Configuration

- **deepseek.api_key**: Your DeepSeek API key (get one at https://platform.deepseek.com)
- **deepseek.api_url**: DeepSeek API endpoint (default: `"https://api.deepseek.com/v1/chat/completions"`)
- **deepseek.editor_model**: Model for text generation (default: `"deepseek-chat"`)
- **deepseek.critic_model**: Model for critic evaluation (default: `"deepseek-chat"`)

#### Ollama Configuration (Alternative Provider)

- **ollama.url**: Ollama API endpoint (default: `"http://localhost:11434/api/generate"`)
- **ollama.editor_model**: Model name for text generation (e.g., `"qwen3:32b"`)
- **ollama.critic_model**: Model name for critic evaluation (e.g., `"deepseek-r1:8b"`)

#### Sample Text

- **sample.file**: Path to the sample text file that defines the target style (relative to project root, typically in `styles/` folder)

#### Style Atlas Settings

- **atlas.persist_path**: Directory to cache/load Style Atlas (optional, speeds up subsequent runs)
- **atlas.num_clusters**: Number of K-means clusters for style grouping (default: `5`, recommended: 3-7)
- **atlas.similarity_threshold**: Minimum similarity score (0-1) for situation match retrieval (default: `0.3`, lower = more matches)

#### Critic Settings

- **critic.min_score**: Minimum score (0-1) required for critic to accept generated text (default: `0.75`)
- **critic.min_pipeline_score**: Minimum score (0-1) for pipeline-level acceptance (default: `0.6`, more lenient than min_score)
- **critic.max_retries**: Maximum retry attempts per sentence within critic loop (default: `5`)
- **critic.max_pipeline_retries**: Maximum retry attempts at pipeline level when score is below threshold (default: `2`)

#### Style Blending Settings

- **blend.authors**: List of author identifiers to blend (e.g., `["Hemingway", "Lovecraft"]`)
  - Single author: `["Sagan"]` → single-author mode (backward compatible)
  - Multiple authors: `["Hemingway", "Lovecraft"]` → blend mode
- **blend.ratio**: Blend ratio (0.0 to 1.0, default: `0.5`)
  - `0.0` = All Author A (first in list)
  - `1.0` = All Author B (second in list)
  - `0.5` = Balanced blend
  - Can be overridden via `--blend-ratio` CLI flag

### Getting an API Key

1. Sign up at https://platform.deepseek.com
2. Navigate to API keys section
3. Create a new API key
4. Copy the key and paste it into `config.json`

## Usage

### ChromaDB Management

Before using `restyle.py`, you need to load author styles into ChromaDB. ChromaDB operations are handled by separate scripts in the `scripts/` folder.

#### Loading Styles

Load author styles into ChromaDB using `scripts/load_style.py`:

```bash
# Load a single author style
python3 scripts/load_style.py --style-file styles/sample_sagan.txt --author "Sagan"

# Load multiple author styles in one call
python3 scripts/load_style.py \
  --style-file styles/sample_hemingway.txt --author "Hemingway" \
  --style-file styles/sample_lovecraft.txt --author "Lovecraft"

# Specify custom config and cache directory
python3 scripts/load_style.py \
  --style-file styles/sample_sagan.txt --author "Sagan" \
  --config config.json \
  --atlas-cache atlas_cache/
```

#### Listing Loaded Styles

List all author styles currently loaded in ChromaDB using `scripts/list_styles.py`:

```bash
# List all loaded authors with basic statistics
python3 scripts/list_styles.py

# Show detailed statistics including cluster IDs
python3 scripts/list_styles.py --verbose

# Specify custom collection name and cache directory
python3 scripts/list_styles.py \
  --collection-name "style_atlas" \
  --atlas-cache atlas_cache/
```

#### Clearing ChromaDB

Clear ChromaDB collections using `scripts/clear_chromadb.py`:

```bash
# Clear entire collection
python3 scripts/clear_chromadb.py --all

# Clear specific author's data
python3 scripts/clear_chromadb.py --author "Sagan"

# Specify custom collection name and cache directory
python3 scripts/clear_chromadb.py --all \
  --collection-name "style_atlas" \
  --atlas-cache atlas_cache/
```

### Text Restyling

Once styles are loaded into ChromaDB, use `restyle.py` to transform text:

```bash
source venv/bin/activate
python3 restyle.py input/small.md -o output/small.md
```

With additional options:

```bash
# Specify custom config file
python3 restyle.py input/small.md -o output/small.md --config config.json

# Specify ChromaDB cache directory
python3 restyle.py input/small.md -o output/small.md --atlas-cache atlas_cache/

# Override blend ratio (for style blending)
python3 restyle.py input/small.md -o output/small.md --blend-ratio 0.7

# Adjust retry settings
python3 restyle.py input/small.md -o output/small.md --max-retries 5

# Enable verbose output
python3 restyle.py input/small.md -o output/small.md -v
```

### CLI Options

#### restyle.py

- `input`: Input text file to transform (required)
- `-o, --output`: Output file path (required)
- `-c, --config`: Configuration file path (default: `config.json`)
- `--max-retries`: Maximum retry attempts per sentence (default: 3, overrides config.json)
- `--atlas-cache`: Path to ChromaDB persistence directory (overrides config.json)
- `--blend-ratio FLOAT`: Override blend ratio from config (0.0 to 1.0, CLI overrides config.json)
- `-v, --verbose`: Enable verbose output

#### scripts/load_style.py

- `--style-file FILE`: Path to style text file (required, can be specified multiple times)
- `--author NAME`: Author name to tag the style (required, must match number of `--style-file` arguments)
- `--config PATH`: Path to config.json (default: `config.json`)
- `--atlas-cache PATH`: Path to ChromaDB persistence directory (overrides config.json)
- `--num-clusters N`: Number of K-means clusters (overrides config.json)
- `--collection-name NAME`: Collection name (overrides config.json)
- `-v, --verbose`: Enable verbose output

#### scripts/list_styles.py

- `--config PATH`: Path to config.json (default: `config.json`)
- `--atlas-cache PATH`: Path to ChromaDB persistence directory (overrides config.json)
- `--collection-name NAME`: Collection name to list (overrides config.json)
- `-v, --verbose`: Enable verbose output with detailed statistics

#### scripts/clear_chromadb.py

- `--collection-name NAME`: Collection name to clear (default: from config.json or "style_atlas")
- `--author NAME`: Author name to clear (clears only this author's data)
- `--atlas-cache PATH`: Path to ChromaDB persistence directory (overrides config.json)
- `--config PATH`: Path to config.json (default: `config.json`)
- `--all`: Clear entire collection (default if no `--author` specified)
- `-v, --verbose`: Enable verbose output

### Python API

Use the Python API directly:

```python
from src.pipeline import run_pipeline, process_text

# Using file paths
output = run_pipeline(
    input_file="input/small.md",
    output_file="output/small.md",
    atlas_persist_path="atlas_cache/"
)

# Using text directly
input_text = "Your input text here."
sample_text = "Your style sample text here."
output = process_text(
    input_text=input_text,
    sample_text=sample_text,
    config_path="config.json",
    atlas_cache_path="atlas_cache/"
)
```

## Project Structure

```
text-style-transfer/
├── src/
│   ├── models.py              # Data structures (ContentUnit)
│   ├── analyzer/
│   │   ├── style_metrics.py   # Style vector extraction
│   │   └── structure.py        # Sentence type classification
│   ├── atlas/
│   │   ├── builder.py         # Style Atlas construction
│   │   └── navigator.py       # Cluster navigation and RAG retrieval
│   ├── ingestion/
│   │   └── semantic.py        # Semantic meaning extraction
│   ├── generator/
│   │   ├── llm_interface.py   # LLM generation
│   │   └── prompt_builder.py  # RAG-based prompt assembly
│   ├── validator/
│   │   └── critic.py          # Adversarial critic evaluation
│   └── pipeline.py            # Main pipeline orchestration
├── tests/                      # Test files
├── input/                      # Input text files
├── output/                     # Generated output files
├── styles/                     # Sample style text files
│   └── sample_*.txt           # Author style samples (e.g., sample_sagan.txt)
├── prompts/                    # LLM prompt templates
│   ├── generator_system.md    # Generator system prompt template
│   ├── generator_examples.md  # Generator examples template
│   ├── generation_prompt.md   # Generation prompt template (single-author)
│   ├── generation_blended.md  # Blended style prompt template
│   ├── critic_system.md       # Critic system prompt template
│   └── critic_user.md         # Critic user prompt template
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
└── restyle.py                 # CLI entry point
```

## How It Works

The following diagram illustrates the complete pipeline flow:

```mermaid
flowchart TD
    Start([Start: Input Text + Sample Text]) --> BuildAtlas[Build Style Atlas]

    BuildAtlas --> ChunkText[Chunk Sample Text into Paragraphs]
    ChunkText --> GenEmbeddings[Generate Dual Embeddings:<br/>Semantic + Style Vectors]
    GenEmbeddings --> StoreChromaDB[Store in ChromaDB with Metadata]
    StoreChromaDB --> Cluster[K-means Clustering on Style Vectors]
    Cluster --> AtlasComplete[Style Atlas Complete]

    Start --> ExtractMeaning[Extract Meaning from Input]
    ExtractMeaning --> ExtractSVO[Extract SVO Triples, Entities, Content Words]
    ExtractSVO --> GroupParagraphs[Group into Paragraphs]

    AtlasComplete --> BuildMarkov[Build Cluster Markov Chain]
    GroupParagraphs --> ProcessPara[Process Each Paragraph]

    BuildMarkov --> ProcessPara
    ProcessPara --> ProcessSentence[Process Each Sentence]

    ProcessSentence --> PredictCluster[Predict Current Style Cluster]
    PredictCluster --> DualRAG[Dual RAG Retrieval]

    DualRAG --> SituationMatch[Situation Match:<br/>Semantic Similarity Query]
    DualRAG --> StructureMatch[Structure Match:<br/>Cluster + Length Filter]

    SituationMatch --> BuildPrompt[Build Constrained Prompt]
    StructureMatch --> BuildPrompt

    BuildPrompt --> LLMGen[LLM Generation]
    LLMGen --> CriticEval[Critic Evaluation]

    CriticEval --> PassCheck{Pass Quality<br/>Threshold?}
    PassCheck -->|No| RetryCount{Retries<br/>&lt; Max?}
    RetryCount -->|Yes| Feedback[Generate Feedback]
    Feedback --> BuildPrompt
    RetryCount -->|No| AcceptBest[Accept Best Result]

    PassCheck -->|Yes| AcceptBest
    AcceptBest --> MoreSentences{More<br/>Sentences?}
    MoreSentences -->|Yes| ProcessSentence
    MoreSentences -->|No| MoreParagraphs{More<br/>Paragraphs?}
    MoreParagraphs -->|Yes| ProcessPara
    MoreParagraphs -->|No| End([Output Generated Text])
```

1. **Style Atlas Construction**:
   - Chunks sample text into paragraphs
   - Generates dual embeddings: semantic (sentence-transformers) and style (deterministic metrics)
   - Stores in ChromaDB with metadata (word count, sentence count, cluster ID)
   - Runs K-means clustering on style vectors to group paragraphs into style clusters

2. **Meaning Extraction**: Parses input text to extract:
   - Subject-Verb-Object triples
   - Named entities
   - Content words
   - Paragraph and sentence position metadata

3. **Style Navigation**:
   - Builds Markov chain from cluster sequence in sample text
   - Predicts next style cluster based on current generated text
   - Determines appropriate style state for each input segment

4. **Dual RAG Retrieval**:
   - **Situation Match**: Queries ChromaDB by semantic similarity to find paragraphs about similar topics (vocabulary grounding)
     - Uses configurable `similarity_threshold` (default: 0.3) to filter matches
   - **Structure Match**: Queries ChromaDB by cluster ID and filters by length ratio (0.7x-1.5x) to find rhythm/structure examples
     - Uses stochastic selection with history tracking to prevent repetition
     - Prefers candidates with better length matches
   - **Blend Mode**: When multiple authors are configured, uses `StyleBlender` to find "bridge texts" via vector interpolation
     - Calculates author centroids (average style vectors)
     - Interpolates between author styles: `target_vec = (vec_a * (1 - ratio)) + (vec_b * ratio)`
     - Finds paragraphs that naturally blend both styles

5. **Constrained Generation**: Uses LLM with RAG-based prompts that:
   - Explicitly separate vocabulary guidance (from situation match) and structure guidance (from structure match)
   - Include detailed structure analysis (punctuation, clauses, voice, complexity)
   - Include length constraints to prevent expansion (1:1 sentence mapping)
   - Preserve citations `[^number]` and direct quotations exactly
   - Use low temperature (0.1) for precise structure matching
   - Load prompts from markdown templates for easy customization

6. **Adversarial Validation**:
   - Critic LLM evaluates generated text against both structure and situation matches
   - Provides specific, actionable feedback formatted as numbered action items
   - Accumulates feedback across retry attempts for cumulative hints
   - Two-level retry system:
     - **Critic-level retries**: Up to `critic.max_retries` attempts per sentence
     - **Pipeline-level retries**: Up to `critic.max_pipeline_retries` attempts when score is below threshold
   - Adaptive threshold adjustment when structure matches are poor quality
   - Enforces minimum score thresholds with `ConvergenceError` if not met

## Testing

Run all tests:

```bash
source venv/bin/activate
python3 tests/test_models.py
python3 tests/test_atlas.py
python3 tests/test_semantic.py
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
- **numpy**: Numerical computations
- **scikit-learn**: K-means clustering and machine learning utilities
- **sentence-transformers**: Semantic embeddings
- **chromadb**: Vector database for RAG retrieval
- **spacy**: Dependency parsing (optional, with fallback)
- **requests**: HTTP requests for API calls
- **pydantic-settings**: Configuration management

See `requirements.txt` for complete list and versions.

## Prompt Templates

The system uses markdown template files in the `prompts/` directory for all LLM prompts. This makes it easy to customize prompts without modifying code.

### Generator Prompts

- **`prompts/generator_system.md`**: System message defining the generator's role and directives
  - Template variable: `{author_name}` - Target author name
- **`prompts/generator_examples.md`**: Examples section added to system message
  - Shows good vs bad examples of style transfer
  - Demonstrates structure matching requirements
- **`prompts/generation_prompt.md`**: Main generation prompt template (single-author mode)
  - Template variables:
    - `{structure_match}` - Structural reference text
    - `{structure_instructions}` - Formatted structure analysis
    - `{situation_match_label}` - Label for situational reference
    - `{situation_match_content}` - Situational reference content
    - `{vocab_block}` - Global vocabulary injection block
    - `{input_text}` - Original input text
    - `{input_word_count}` - Input word count
    - `{target_word_count}` - Target output word count
    - `{avg_sentence_len}` - Average sentence length from structure match
- **`prompts/generation_blended.md`**: Blended style generation prompt template
  - Template variables:
    - `{bridge_template}` - Bridge text that connects two author styles
    - `{hybrid_vocab}` - Comma-separated list of words from both authors
    - `{author_a}` - First author name
    - `{author_b}` - Second author name
    - `{blend_desc}` - Description of the blend (e.g., "primarily Hemingway with subtle Lovecraft influences")
    - `{input_text}` - Original input text

### Critic Prompts

- **`prompts/critic_system.md`**: System message for the critic evaluator
  - Defines evaluation criteria and output format
- **`prompts/critic_user.md`**: User prompt template for critic evaluation
  - Template variables:
    - `{structure_section}` - Structural reference section
    - `{situation_section}` - Situational reference section
    - `{original_section}` - Original text section (if provided)
    - `{generated_text}` - Generated text to evaluate
    - `{preservation_checks}` - Conditional preservation requirements
    - `{preservation_instruction}` - Conditional preservation instructions

### Customizing Prompts

To customize prompts, edit the corresponding markdown file in `prompts/`. The system will automatically load the updated templates on the next run. Template variables use Python's `.format()` syntax with curly braces `{variable_name}`.

## Troubleshooting

### API Key Issues

If you see API errors:
1. Verify your API key in `config.json`
2. Check that you have sufficient API credits
3. Ensure the API endpoint URL is correct

### NLTK Data Missing

If you see NLTK resource errors:
```bash
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"
```

### ChromaDB Issues

If you see ChromaDB errors:
- Ensure Python 3.12+ is being used (required for ChromaDB compatibility)
- Try rebuilding the virtual environment: `python3.12 -m venv venv`
- Check that `onnxruntime` is installed (required dependency)
- If ChromaDB collection is corrupted, clear it: `python3 scripts/clear_chromadb.py --all`
- If collection is empty, load styles first: `python3 scripts/load_style.py --style-file <file> --author <name>`
- When loading multiple authors, use the same `--atlas-cache` path to ensure they're added to the same collection
- Author styles are stored with prefixed IDs (e.g., `Hemingway_para_0`) to avoid conflicts
- If `restyle.py` says collection is empty, you need to load styles first using `scripts/load_style.py`

### Import Errors

Make sure you're running from the project root directory and the virtual environment is activated.

### Low Quality Output

The pipeline includes retry mechanisms, but if output quality is consistently low:
- Try a longer or more representative sample text
- Adjust critic thresholds in `config.json`:
  - Lower `critic.min_score` (default: 0.75) to be more lenient
  - Increase `critic.max_retries` (default: 5) for more attempts
- Lower `atlas.similarity_threshold` (default: 0.3) to get more situation matches
- Ensure the sample text has sufficient variety in style
- Check that the Style Atlas is being built correctly (check logs with `-v`)
- Review prompt templates in `prompts/` - they may need adjustment for your use case
- For style blending: Ensure both authors have sufficient text loaded (at least 10-20 paragraphs each)
- If bridge texts aren't found, try adjusting `blend.ratio` or ensure authors have overlapping style characteristics

### Length Expansion Issues

If output is much longer than input:
- The length-constrained retrieval should prevent this
- Check that the atlas was built with length metadata (rebuild if needed)
- Verify that structure matches are being filtered by length ratio

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
