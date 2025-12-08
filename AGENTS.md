# AI Text Depattern - Agents Guide

This document provides essential information for agents working with the AI Text Depattern codebase.

## Project Overview

AI Text Depattern is a Python tool that rewrites AI-generated text to sound more natural and human-like using Few-Shot Style Transfer. It matches a human writing sample while preserving all original meaning.

## Core Architecture

### Main Components

- **humanizer.py**: Main entry point and core `AgenticHumanizer` class
- **markov.py**: Style transfer agent using Markov chains for stylistic analysis
- **glm.py**: GLM (Z.AI) API provider implementation
- **deepseek.py**: DeepSeek API provider implementation
- **prompts/**: Contains system prompts and templates for different AI models

### Key Dependencies

- PyTorch & Transformers: For GPT-2 Large model used in perplexity scoring
- spaCy: Natural language processing for style analysis
- requests: HTTP client for API communications
- sqlite3: Database for Markov chain style patterns

## Essential Commands

### Setup and Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# For local Ollama models
ollama pull qwen3:32b
ollama pull deepseek-r1:8b
ollama serve
```

### Running the Tool

```bash
# Basic usage
python humanizer.py input/generated.md

# With custom output path
python humanizer.py input/generated.md output/humanized.md

# Default output goes to output/<input_filename>
```

## Configuration

### Primary Config File: config.json

- **provider**: Choose between "ollama", "glm", or "deepseek"
- **ollama**: Local model settings (url, editor_model, critic_model)
- **glm**: Z.AI API settings (api_key, api_url, models)
- **deepseek**: DeepSeek API settings (api_key, api_url, models)

### API Keys

Can be set in config.json or environment variables:
- `export GLM_API_KEY=your-api-key-here`
- `export DEEPSEEK_API_KEY=your-api-key-here`

### Style Configuration

- **prompts/sample.txt**: Target writing style sample
- **prompts/editor.md**: Style transfer instructions
- **prompts/system.md**: AI detection patterns
- **prompts/structural_*.md**: Structural analysis and editing prompts

## Code Organization and Patterns

### Class Structure

1. **AgenticHumanizer** (humanizer.py): Main orchestrator class
   - Initializes providers and models
   - Manages the humanization pipeline
   - Handles device selection (MPS/CUDA/CPU)

2. **Provider Classes**: Abstract API access
   - `GLMProvider`: Z.AI API wrapper
   - `DeepSeekProvider`: DeepSeek API wrapper
   - Both implement similar `call()` method signature

3. **StyleTransferAgent** (markov.py): Markov chain analysis
   - Learns style patterns from sample text
   - Stores patterns in SQLite database (style_brain.db)

### Key Methods

- `humanize(text, max_retries=3)`: Main text processing pipeline
- `_call_provider()`: Abstract method for API calls
- `_calculate_perplexity()`: GPT-2 based scoring
- `_analyze_ai_patterns()`: Detect AI-like text patterns

### Pipeline Flow

1. Extract structural patterns from style sample
2. Apply structural editing to break AI patterns
3. Perform style transfer using Few-Shot Learning
4. Validate results with critic model
5. Score with perplexity evaluation
6. Repeat if validation fails (up to max_retries)

## Important Gotchas

### Model Loading

- First run downloads GPT-2 Large (~3GB) for perplexity scoring
- Device selection prioritizes MPS (macOS) > CUDA > CPU
- Chunk size limited to 1024 tokens to avoid context window issues

### File Paths

- All paths use `Path` from pathlib for cross-platform compatibility
- Database file (style_brain.db) created automatically if missing
- Default output directory: `output/`

### Prompt System

- System prompts loaded from `prompts/` directory at initialization
- Each prompt has specific role (editor, critic, structural analysis)
- Prompts contain detailed instructions and constraints for AI models

### Error Handling

- API key validation on provider initialization
- Graceful fallback for missing features
- Comprehensive error messages for configuration issues

## Testing and Validation

### No Formal Test Suite

- No test files exist in the codebase
- Validation happens through the critic model during processing
- Manual testing by processing sample files

### Quality Checks

The system includes multiple validation layers:
- AI pattern detection (system.md prompts)
- Meaning preservation checks
- Perplexity scoring using GPT-2
- Burstiness and sentence variation analysis

## Development Patterns

### Code Style

- Standard Python conventions
- Type hints in some newer code
- Comprehensive docstrings for classes and methods
- Console output for progress tracking

### Adding New Providers

To add a new AI provider:
1. Create provider class similar to `GLMProvider`
2. Implement `call()` method with consistent signature
3. Add provider configuration to `config.json` schema
4. Update `AgenticHumanizer.__init__()` to initialize new provider
5. Add provider to validation logic

### Modifying Prompts

- Edit files in `prompts/` directory
- Restart application to reload prompts
- Test changes with various input types
- Maintain prompt structure and formatting

## Database Schema

The Markov chain uses SQLite with tables for:
- Word transitions and probabilities
- Sentence patterns
- Style signatures

Database is automatically created and trained from the sample text on first run.

## Performance Considerations

- GPU acceleration recommended (MPS on macOS, CUDA elsewhere)
- Large documents are chunked to fit model context windows
- API rate limits may affect processing speed
- Local models via Ollama avoid API costs but require hardware resources