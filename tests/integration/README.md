# Integration Tests for Linguistic Quality

This directory contains integration tests that validate linguistic quality alongside code quality.

## Test Files

- `test_linguistic_quality.py` - Tests for zipper merge, action echo, grounding, and perspective lock
- `test_structural_integrity.py` - Tests for impossible constraints, empty slots, and sledgehammer convergence
- `test_narrative_flow.py` - Parametrized tests for perspective locking
- `conftest.py` - Shared pytest fixtures

## Running the Tests

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run All Integration Tests

```bash
pytest tests/integration/ -v
```

### Run Specific Test Files

```bash
# Linguistic quality tests
pytest tests/integration/test_linguistic_quality.py -v

# Structural integrity tests
pytest tests/integration/test_structural_integrity.py -v

# Narrative flow tests
pytest tests/integration/test_narrative_flow.py -v
```

### Run Specific Test Classes

```bash
# Test zipper merge detection
pytest tests/integration/test_linguistic_quality.py::TestAntiStutterZipperMerge -v

# Test action echo detection
pytest tests/integration/test_linguistic_quality.py::TestActionEchoDetection -v

# Test grounding validation
pytest tests/integration/test_linguistic_quality.py::TestGroundingValidation -v

# Test perspective lock
pytest tests/integration/test_linguistic_quality.py::TestPerspectiveLock -v
```

## Test Coverage

### Linguistic Quality Tests

1. **Anti-Stutter (Zipper Merge)**
   - Full echo detection
   - Head echo detection (same 3+ words at start)
   - Tail echo detection (end of prev matches start of new)
   - Non-echoing sentences pass

2. **Action Echo Detection**
   - Detects repeated action verbs using spaCy lemmatization
   - Ignores auxiliary verbs (was, had, did)
   - Tests with "weave/wove" and "run/ran/runs" examples

3. **Grounding Validation**
   - Detects abstract/moralizing endings
   - Verifies concrete sensory details pass
   - Tests moralizing patterns

4. **Perspective Lock**
   - Verifies first person singular perspective
   - Verifies third person perspective
   - Verifies first person plural perspective
   - Tests that wrong perspective pronouns fail

### Structural Integrity Tests

1. **Impossible Constraint Test**
   - Verifies max_retries configuration exists
   - Tests sledgehammer programmatic split mechanism

2. **Empty Slot Test**
   - Tests ContentPlanner marks slots as EMPTY
   - Verifies final output length when slots are EMPTY

3. **Sledgehammer Convergence Test**
   - Tests programmatic split after max_retries
   - Verifies ||| separator is added

### Narrative Flow Tests

- Parametrized tests for perspective detection
- Tests perspective verification with various text samples
- Edge case testing (empty text, no pronouns, mixed perspectives)

## Mocking Strategy

All tests use mocked LLM responses to avoid API calls:
- Mock LLM provider in `tests/mocks/mock_llm_provider.py`
- Pre-recorded responses in `tests/mocks/llm_responses.json`
- No API keys required for testing

## CI/CD Integration

Tests run automatically in GitHub Actions:
- Stage 1: Deterministic unit tests (no LLM)
- Stage 2: Mocked integration tests (mocked LLM)

See `.github/workflows/linguistic_quality.yml` for details.

