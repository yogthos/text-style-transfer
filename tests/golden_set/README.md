# Golden Paragraph Regression Suite

## Overview

The Golden Paragraph Regression Suite ensures that linguistic quality doesn't degrade over time. It stores manually verified "perfect" outputs and compares new generations against them using semantic similarity.

## Directory Structure

- `examples/` - JSON files containing golden examples
- `test_golden_regression.py` - Regression test runner
- `curate_golden_set.py` - Helper script for creating new golden examples

## Golden Example Format

Each golden example is stored as a JSON file in `examples/`:

```json
{
  "id": "golden_001",
  "input_text": "Original paragraph text...",
  "author_name": "AuthorName",
  "archetype_id": 3,
  "perspective": "first_person_singular",
  "expected_output": "Perfect output paragraph...",
  "expected_similarity": 0.95,
  "metadata": {
    "created_date": "2024-01-01",
    "verified_by": "manual",
    "notes": "Excellent example of complex sentence structure"
  }
}
```

## Creating Golden Examples

### Method 1: Using the Curation Script

```bash
python tests/golden_set/curate_golden_set.py \
  --input "Your input text here" \
  --author "AuthorName" \
  --output-id "golden_001"
```

### Method 2: Manual Creation

1. Generate a paragraph using the system
2. Manually verify it's 10/10 quality
3. Create a JSON file in `examples/` following the format above
4. Set `expected_similarity` to the semantic similarity score (0.0-1.0)

## Running Regression Tests

```bash
# Run all golden regression tests
pytest tests/golden_set/test_golden_regression.py

# Run with verbose output
pytest tests/golden_set/test_golden_regression.py -v

# Run specific golden example
pytest tests/golden_set/test_golden_regression.py::test_golden_001
```

## Test Behavior

- **CI/CD Mode**: Uses mocked LLM responses, compares against cached golden outputs
- **Local Mode**: Can optionally use real LLM (set `USE_REAL_LLM=true`)
- **Comparison**: Uses semantic similarity (embedding-based), no LLM call needed
- **Threshold**: Fails if similarity drops by >15% relative to expected

## Adding New Golden Examples

1. Generate a high-quality output manually
2. Verify it meets all quality criteria:
   - No action echo
   - Proper grounding (concrete ending)
   - Correct perspective
   - No zipper merge issues
   - Natural flow
3. Use `curate_golden_set.py` or create JSON manually
4. Commit to version control

## Maintenance

- Review golden examples periodically (quarterly)
- Update if author style evolves significantly
- Remove examples that no longer represent target quality
- Keep 10-15 examples for good coverage

