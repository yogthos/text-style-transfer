# Text Style Transfer

Transform text to match a target author's style while preserving semantic meaning. Uses a multi-layered architecture with:

- **Style Atlas**: ChromaDB-based vector store for paragraph-level style retrieval
- **Paragraph Atlas**: Statistical archetype system with Markov chain transitions for paragraph generation
- **Style RAG**: Dynamic retrieval of semantically relevant style fragments (3-sentence windows) for concrete phrasing examples
- **Semantic Translation**: Neutral summary extraction to preserve meaning while removing source style
- **Statistical Generation**: Paragraph-level generation using archetype statistics, style palettes, and iterative refinement
- **Semantic Validation**: Multi-metric validation ensuring meaning preservation and style alignment

## Dependencies

This project requires the following Python packages (see `requirements.txt` for versions):
- **spacy** (>=3.7.0) - Natural language processing and grammatical validation
- **nltk** (>=3.8.0) - Text tokenization and linguistic analysis
- **numpy** (>=1.24.0) - Numerical computations
- **scikit-learn** (>=1.3.0) - Machine learning (K-means clustering for style atlas)
- **pandas** (>=2.0.0) - Data manipulation and analysis
- **sentence-transformers** (>=2.2.0) - Semantic embeddings and similarity calculations
- **torch** (>=2.0.0) - PyTorch (required by sentence-transformers for neural network operations)
- **requests** (>=2.31.0) - HTTP requests for LLM API calls
- **chromadb** (>=0.4.0) - Vector database for style atlas storage
- **jsonrepair** (>=0.19.0) - JSON repair utilities
- **tiktoken** (>=0.5.0) - Token counting for LLM APIs
- **pytest** (>=7.0.0) - Testing framework

Additional setup required:
- **spaCy English model**: `en_core_web_sm` (automatically downloaded on first use, or manually with `python3 -m spacy download en_core_web_sm`)
- **NLTK data**: punkt, punkt_tab, averaged_perceptron_tagger_eng, vader_lexicon (downloaded automatically on first run)
- **Sentence Transformer models**: `all-mpnet-base-v2` (automatically downloaded on first use, ~420MB)

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first run, or manually):
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('vader_lexicon', quiet=True)"

# Download spaCy model (required for grammatical validation):
python3 -m spacy download en_core_web_sm

# grab the model for the RAG
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

### Configuration

1. Copy `config.json` and set your API key:
   ```json
   {
     "provider": "deepseek",
     "deepseek": {
       "api_key": "your-api-key-here"
     },
     "blend": {
       "authors": ["Mao"]
     }
   }
   ```

2. Load author styles into ChromaDB:
   ```bash
   python3 scripts/load_style.py --style-file styles/sample_mao.txt --author "Mao"
   ```

3. Build paragraph atlas (for statistical generation):
   ```bash
   python3 scripts/build_paragraph_atlas.py styles/sample_mao.txt --author "Mao"
   ```

4. Build Style RAG index (for dynamic style palette retrieval):
   ```bash
   python3 tools/build_rag_index.py --author "Mao"
   ```

5. Transform text:
   ```bash
   python3 restyle.py input/small.md -o output/small.md
   ```

## Usage

### Loading Author Styles

**1. Load Style Atlas** (for paragraph-level style retrieval):
```bash
# Single author
python3 scripts/load_style.py --style-file styles/sample_mao.txt --author "Mao"

# Multiple authors
python3 scripts/load_style.py \
  --style-file styles/sample_hemingway.txt --author "Hemingway" \
  --style-file styles/sample_lovecraft.txt --author "Lovecraft"
```

**2. Build Paragraph Atlas** (for statistical archetype generation):
```bash
python3 scripts/build_paragraph_atlas.py styles/sample_mao.txt --author "Mao"
```

This creates:
- `atlas_cache/paragraph_atlas/{author}/archetypes.json` - Paragraph archetype statistics
- `atlas_cache/paragraph_atlas/{author}/transition_matrix.json` - Markov chain transitions
- `atlas_cache/paragraph_atlas/{author}/chroma/` - ChromaDB collection with paragraph examples

**3. Build Style RAG Index** (for dynamic style palette retrieval):
```bash
# Uses default corpus file: styles/sample_{author}.txt
python3 tools/build_rag_index.py --author "Mao"

# Or specify custom corpus file
python3 tools/build_rag_index.py --author "Mao" --corpus-file path/to/corpus.txt
```

This creates:
- `atlas_cache/paragraph_atlas/{author}/style_fragments_chroma/` - ChromaDB collection with 3-sentence style fragments
- Uses high-fidelity `all-mpnet-base-v2` embeddings for semantic retrieval

**4. Generate Style DNA** (optional, for style profiling):
```bash
python3 scripts/generate_style_dna.py --author "Mao"
```

**5. List Loaded Styles**:
```bash
python3 scripts/list_styles.py
```

### Transforming Text

```bash
# Basic usage
python3 restyle.py input/small.md -o output/small.md

# With options
python3 restyle.py input/small.md -o output/small.md \
  --max-retries 5 \
  --verbose
```

**CLI Options:**
- `input`: Input text file (required)
- `-o, --output`: Output file path (required)
- `-c, --config`: Config file path (default: `config.json`)
- `--max-retries`: Max retry attempts per sentence (default: 3)
- `--atlas-cache`: ChromaDB persistence directory (overrides config)
- `--blend-ratio`: Style blending ratio (0.0-1.0, default: 0.6)
- `-v, --verbose`: Enable verbose output

### Python API

```python
from src.pipeline import run_pipeline

output = run_pipeline(
    input_file="input/small.md",
    output_file="output/small.md",
    config_path="config.json",
    verbose=True
)
```

## Configuration

### Provider Settings

```json
{
  "provider": "deepseek",
  "deepseek": {
    "api_key": "your-api-key-here",
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "editor_model": "deepseek-chat",
    "critic_model": "deepseek-chat"
  }
}
```

Supported providers: `deepseek`, `ollama`, `glm`, `gemini`

### Author Configuration

```json
{
  "blend": {
    "authors": ["Mao"]
  }
}
```

The first author in the list is used. Style DNA is loaded from the Style Registry (`atlas_cache/author_profiles.json`).

### Key Configuration Sections

**Paragraph Atlas** (statistical archetype generation):
```json
{
  "paragraph_atlas": {
    "path": "atlas_cache/paragraph_atlas",
    "default_archetype": 0
  }
}
```

Controls the statistical archetype system used for paragraph generation. The paragraph atlas contains:
- **Archetypes**: Statistical patterns (sentence length, sentence count, burstiness, style type)
- **Transition Matrix**: Markov chain probabilities for archetype transitions
- **ChromaDB Collection**: Full example paragraphs for each archetype

**Generation** (statistical generation parameters):
```json
{
  "generation": {
    "temperature": 0.8,
    "max_tokens": 1500,
    "num_candidates": 4,
    "max_retries": 2,
    "compliance_threshold": 0.85
  }
}
```

Controls the statistical paragraph generation process:
- `temperature`: Generation temperature (default: 0.8)
- `max_tokens`: Maximum tokens per paragraph (default: 1500)
- `num_candidates`: Number of candidates to generate per round (default: 4)
- `max_retries`: Maximum refinement rounds if compliance is low (default: 2)
- `compliance_threshold`: Minimum statistical compliance score to accept (default: 0.85)

**Semantic Critic** (validation thresholds):
```json
{
  "semantic_critic": {
    "recall_threshold": 0.85,
    "precision_threshold": 0.60,
    "similarity_threshold": 0.7,
    "fluency_threshold": 0.8,
    "weights": {
      "accuracy": 0.5,
      "fluency": 0.1,
      "style": 0.1,
      "thesis_alignment": 0.15,
      "intent_compliance": 0.1,
      "keyword_coverage": 0.05
    }
  }
}
```

The `weights` section controls how different metrics contribute to the composite score:
- `accuracy`: Semantic accuracy (proposition recall/precision)
- `fluency`: Grammatical fluency
- `style`: Style alignment with target author
- `thesis_alignment`: Alignment with document thesis (when global context is available)
- `intent_compliance`: Compliance with document intent (when global context is available)
- `keyword_coverage`: Coverage of document keywords (when global context is available)

**Atlas** (Style Atlas settings):
```json
{
  "atlas": {
    "persist_path": "atlas_cache/",
    "num_clusters": 5,
    "min_structure_words": 4,
    "max_length_ratio": 1.8,
    "min_length_ratio": 0.6
  }
}
```

**Translator** (generation parameters):
```json
{
  "translator": {
    "temperature": 0.5,
    "max_tokens": 300
  }
}
```

**Critic** (evaluation settings):
```json
{
  "critic": {
    "min_score": 0.6,
    "max_retries": 5,
    "good_enough_threshold": 0.8
  }
}
```

**Global Context** (document-level context for style transfer):
```json
{
  "global_context": {
    "enabled": true,
    "max_summary_tokens": 500
  }
}
```

When enabled, the system uses document-level context (thesis, intent, keywords) to improve style transfer quality. This is particularly useful for maintaining consistency across long documents.

**Style RAG** (Dynamic Style Palette retrieval):
```json
{
  "style_rag": {
    "num_fragments": 8,
    "window_size": 3,
    "overlap": 1,
    "embedding_model": "all-mpnet-base-v2"
  }
}
```

Controls the Style RAG system that retrieves semantically relevant style fragments from the author's corpus:
- `num_fragments`: Number of style fragments to retrieve per paragraph (default: 8)
- `window_size`: Number of sentences per fragment (default: 3)
- `overlap`: Number of sentences to overlap between fragments (default: 1)
- `embedding_model`: Sentence transformer model for semantic embeddings (default: "all-mpnet-base-v2")

The Style RAG system retrieves actual phrases and sentence structures from the author's corpus that are semantically similar to the content being generated, providing concrete examples for the LLM to mimic.

**Evolutionary** (evolutionary generation parameters):
```json
{
  "evolutionary": {
    "batch_size": 40,
    "variants_per_skeleton": {
      "strict_adherence": 10,
      "high_style": 10,
      "experimental": 10,
      "simplified": 10
    },
    "max_generations": 3,
    "convergence_threshold": 0.95,
    "top_k_parents": 10,
    "breeding_children": 10,
    "min_keyword_presence": 0.5
  }
}
```

Controls the evolutionary generation process for sentence-level style transfer.

## Project Structure

```
text-style-transfer/
├── src/
│   ├── pipeline.py              # Main pipeline orchestration
│   ├── atlas/
│   │   ├── builder.py          # Style Atlas construction
│   │   ├── navigator.py        # RAG retrieval
│   │   ├── paragraph_atlas.py  # Paragraph archetype loader
│   │   ├── style_rag.py        # Style fragment retrieval (RAG)
│   │   └── style_registry.py   # Style DNA storage
│   ├── generator/
│   │   ├── translator.py       # Text generation
│   │   ├── semantic_translator.py # Neutral summary extraction
│   │   └── mutation_operators.py # Prompt templates
│   ├── validator/
│   │   ├── semantic_critic.py  # Semantic validation
│   │   └── statistical_critic.py # Statistical validation
│   ├── analyzer/
│   │   ├── style_extractor.py  # Style DNA extraction
│   │   ├── structuralizer.py   # Rhythm extraction
│   │   └── structure_extractor.py # Structural template extraction
│   └── analysis/
│       └── semantic_analyzer.py # Proposition extraction
├── prompts/                     # LLM prompt templates (markdown)
├── scripts/
│   ├── load_style.py           # Load author styles into Style Atlas
│   ├── build_paragraph_atlas.py # Build paragraph archetype atlas
│   ├── generate_style_dna.py   # Generate Style DNA profiles
│   ├── list_styles.py          # List loaded authors
│   └── clear_chromadb.py       # Clear ChromaDB collections
├── tools/
│   └── build_rag_index.py      # Build Style RAG fragment index
├── styles/                      # Author corpus files
├── input/                       # Input text files
├── output/                      # Generated output files
├── atlas_cache/                 # ChromaDB persistence directory
├── config.json                  # Configuration
├── restyle.py                   # CLI entry point
└── run_pipeline.py             # Alternative CLI entry point
```

## How It Works

### Pipeline Flow

```mermaid
flowchart TD
    Start([Input Text]) --> LoadAtlas[Load Style Atlas & Paragraph Atlas]
    LoadAtlas --> GetDNA[Get Style DNA from Registry]
    GetDNA --> ProcessPara[Process Each Paragraph]

    ProcessPara --> ExtractNeutral[Extract Neutral Summary]
    ExtractNeutral --> RetrievePalette[Retrieve Style Palette via RAG]
    RetrievePalette --> SelectArchetype[Select Statistical Archetype]
    SelectArchetype --> GetRhythm[Get Rhythm Reference Example]
    GetRhythm --> GenerateCandidates[Generate Multiple Candidates]
    GenerateCandidates --> EvaluateStats[Evaluate with Statistical Critic]
    EvaluateStats --> PassCheck{Compliance<br/>Score OK?}
    PassCheck -->|Yes| AcceptPara[Accept Paragraph]
    PassCheck -->|No| Refine[Refinement Round]
    Refine --> GenerateCandidates

    AcceptPara --> UpdateMarkov[Update Markov Chain State]
    UpdateMarkov --> MorePara{More<br/>Paragraphs?}
    MorePara -->|Yes| ProcessPara
    MorePara -->|No| End([Output Text])
```

### Key Components

1. **Style Atlas**: ChromaDB-based vector store with dual embeddings (semantic + style) and K-means clustering for paragraph-level style retrieval
2. **Paragraph Atlas**: Statistical archetype system with Markov chain transitions for generating paragraphs matching author's structural patterns
3. **Style RAG**: Dynamic retrieval of semantically relevant style fragments (3-sentence windows) to provide concrete phrasing examples during generation
4. **Semantic Translator**: Extracts neutral logical summaries from input text, removing style while preserving meaning
5. **Statistical Critic**: Validates generated paragraphs against statistical archetype parameters (sentence length, sentence count, burstiness)
6. **Semantic Critic**: Validates generated text using proposition recall and style alignment metrics
7. **Style Registry**: Sidecar JSON storage for author Style DNA profiles

### Statistical Paragraph Generation Process

The current implementation uses **Statistical Archetype Generation**:

1. **Neutral Summary Extraction**: Convert input paragraph to a neutral logical summary, removing style while preserving semantic content
2. **Style Palette Retrieval**: Use Style RAG to retrieve 5-10 semantically relevant style fragments from the author's corpus
3. **Archetype Selection**: Use Markov chain to select the next paragraph archetype based on the previous paragraph's archetype
4. **Archetype Description**: Load statistical parameters (avg sentence length, avg sentences per paragraph, burstiness, style type)
5. **Rhythm Reference**: Retrieve a full example paragraph matching the selected archetype from ChromaDB
6. **Generation**: Generate multiple candidate paragraphs using:
   - Neutral summary as content source
   - Style palette fragments as phrasing examples
   - Archetype statistics as structural constraints
   - Rhythm reference as flow model
7. **Evaluation**: Score candidates using Statistical Critic (compliance with archetype parameters)
8. **Refinement**: If compliance is below threshold, generate improved candidates with feedback
9. **Markov Update**: Update Markov chain state for next paragraph continuity

## Testing

Run tests:
```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run individual test files
python3 tests/test_paragraph_rhythm_extraction.py
python3 tests/test_translate_paragraph_contract.py
python3 tests/test_pipeline_fallback_contract.py
python3 tests/test_quality_improvements.py

# Or run all tests with pytest
pytest tests/
```

## Troubleshooting

**Atlas not found**: Load styles first using `scripts/load_style.py`

**Paragraph Atlas not found**: Build paragraph atlas using `scripts/build_paragraph_atlas.py`

**Style RAG collection not found**: Build RAG index using `tools/build_rag_index.py --author <name>`

**Author not found**: Check `blend.authors` in config.json matches loaded author names

**Low quality output**:
- Adjust `generation.compliance_threshold` or `semantic_critic.recall_threshold`
- Ensure Style RAG index is built for better style palette retrieval (`tools/build_rag_index.py`)
- Check that paragraph atlas is built for statistical generation (`scripts/build_paragraph_atlas.py`)
- Verify that Style DNA is generated for the author (`scripts/generate_style_dna.py`)
- Increase `generation.num_candidates` for more diverse generation
- Adjust `generation.temperature` (higher = more creative, lower = more conservative)

**Import errors**: Ensure virtual environment is activated and dependencies are installed

**Missing spaCy model**: The model is automatically downloaded on first use. If issues occur, run `python3 -m spacy download en_core_web_sm` manually

**Missing NLTK data**: The code will attempt to download required NLTK data automatically on first run

**Missing embedding model**: The `all-mpnet-base-v2` model (~420MB) is automatically downloaded by sentence-transformers on first use

**ChromaDB errors**: Ensure ChromaDB is properly installed and the atlas_cache directory is writable
