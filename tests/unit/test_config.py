"""Unit tests for configuration loading."""

import json
import os
import pytest
import tempfile
from pathlib import Path

from src.config import (
    load_config,
    create_default_config,
    Config,
    LLMConfig,
    ChromaDBConfig,
)


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_minimal_config(self):
        """Test loading a minimal valid config."""
        minimal_config = {
            "llm": {
                "provider": "deepseek",
                "providers": {
                    "deepseek": {
                        "api_key": "test-key",
                        "model": "deepseek-chat"
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.provider == "deepseek"
                assert config.llm.providers["deepseek"].api_key == "test-key"
            finally:
                os.unlink(f.name)

    def test_load_full_config(self):
        """Test loading a complete config with all sections."""
        full_config = create_default_config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(full_config, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.provider == "deepseek"
                assert config.chromadb.persist_path == "atlas_cache/"
                assert config.generation.max_repair_retries == 5
                assert config.validation.semantic.min_proposition_coverage == 0.9
            finally:
                os.unlink(f.name)

    def test_missing_config_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config("nonexistent_config.json")
        assert "Configuration file not found" in str(exc_info.value)

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            f.flush()

            try:
                with pytest.raises(ValueError) as exc_info:
                    load_config(f.name)
                assert "Invalid JSON" in str(exc_info.value)
            finally:
                os.unlink(f.name)


class TestEnvironmentVariables:
    """Test environment variable resolution."""

    def test_resolves_env_var(self):
        """Test that ${VAR} syntax resolves environment variables."""
        os.environ["TEST_API_KEY"] = "secret-from-env"

        config_data = {
            "llm": {
                "provider": "deepseek",
                "providers": {
                    "deepseek": {
                        "api_key": "${TEST_API_KEY}",
                        "model": "test"
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.providers["deepseek"].api_key == "secret-from-env"
            finally:
                os.unlink(f.name)
                del os.environ["TEST_API_KEY"]

    def test_missing_env_var_returns_empty(self):
        """Test that missing env var returns empty string with warning."""
        # Ensure var doesn't exist
        os.environ.pop("NONEXISTENT_VAR", None)

        config_data = {
            "llm": {
                "provider": "deepseek",
                "providers": {
                    "deepseek": {
                        "api_key": "${NONEXISTENT_VAR}",
                        "model": "test"
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.providers["deepseek"].api_key == ""
            finally:
                os.unlink(f.name)


class TestDefaultConfig:
    """Test default configuration creation."""

    def test_create_default_config(self):
        """Test creating default config dictionary."""
        config = create_default_config()

        assert "llm" in config
        assert "chromadb" in config
        assert "validation" in config

        assert config["llm"]["provider"] == "deepseek"
        assert "deepseek" in config["llm"]["providers"]
        assert "ollama" in config["llm"]["providers"]

    def test_default_values(self):
        """Test that Config has sensible defaults."""
        config = Config()

        assert config.llm.max_retries == 5
        assert config.chromadb.persist_path == "atlas_cache/"
        assert config.generation.max_repair_retries == 5
        assert config.validation.semantic.min_proposition_coverage == 0.9
        assert config.validation.statistical.length_tolerance == 0.2


class TestLLMConfig:
    """Test LLM configuration."""

    def test_get_provider_config(self):
        """Test getting provider-specific config."""
        config_data = {
            "llm": {
                "provider": "deepseek",
                "providers": {
                    "deepseek": {
                        "api_key": "key1",
                        "model": "model1"
                    },
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model": "model2"
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)

                deepseek_config = config.llm.get_provider_config("deepseek")
                assert deepseek_config.api_key == "key1"

                ollama_config = config.llm.get_provider_config("ollama")
                assert ollama_config.base_url == "http://localhost:11434"

                # Default provider
                default_config = config.llm.get_provider_config()
                assert default_config.api_key == "key1"
            finally:
                os.unlink(f.name)

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        llm_config = LLMConfig(provider="test", providers={})

        with pytest.raises(ValueError) as exc_info:
            llm_config.get_provider_config("unknown")
        assert "Unknown LLM provider" in str(exc_info.value)
