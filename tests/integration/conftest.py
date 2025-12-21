"""Shared fixtures for integration tests."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.generator.translator import StyleTranslator
from tests.test_helpers import ensure_config_exists


@pytest.fixture(scope="session")
def config_path():
    """Ensure config exists and return path."""
    ensure_config_exists()
    return "config.json"


@pytest.fixture(scope="function")
def translator(config_path):
    """Create translator instance for tests."""
    return StyleTranslator(config_path=config_path)

