"""Tests for the prompts module."""

import pytest
from pathlib import Path
import tempfile
import os

from src.utils.prompts import (
    load_prompt,
    format_prompt,
    get_prompt_with_fallback,
    list_prompts,
    clear_prompt_cache,
    PROMPTS_DIR,
)


class TestLoadPrompt:
    """Tests for load_prompt function."""

    def test_load_existing_prompt(self):
        """Test loading an existing prompt file."""
        # Should load the style_transfer_system.txt prompt
        prompt = load_prompt("style_transfer_system")
        assert "You are {author}" in prompt
        assert "CRITICAL RULES" in prompt

    def test_load_nonexistent_prompt_raises_error(self):
        """Test that loading a nonexistent prompt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_xyz")

    def test_prompt_caching(self):
        """Test that prompts are cached."""
        clear_prompt_cache()
        # Load same prompt twice
        prompt1 = load_prompt("style_transfer_system")
        prompt2 = load_prompt("style_transfer_system")
        # Should be the same object due to caching
        assert prompt1 is prompt2


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_format_with_variables(self):
        """Test formatting a prompt with variables."""
        prompt = format_prompt("style_transfer_system", author="Carl Sagan")
        assert "You are Carl Sagan" in prompt
        assert "{author}" not in prompt

    def test_format_base_model_prompt(self):
        """Test formatting the base model prompt with multiple variables."""
        prompt = format_prompt(
            "style_transfer_base_model",
            target_words=250,
            author="Isaac Asimov"
        )
        assert "250 word" in prompt
        assert "Isaac Asimov" in prompt

    def test_format_critic_repair(self):
        """Test formatting the critic repair prompt."""
        instructions = "- Fix grammar\n- Add missing entity"
        prompt = format_prompt(
            "critic_repair_system",
            instructions=instructions
        )
        assert "Fix grammar" in prompt
        assert "Add missing entity" in prompt


class TestGetPromptWithFallback:
    """Tests for get_prompt_with_fallback function."""

    def test_returns_file_content_when_exists(self):
        """Test that it returns file content when file exists."""
        fallback = "This is a fallback"
        prompt = get_prompt_with_fallback(
            "style_transfer_system",
            fallback,
            author="Test Author"
        )
        assert "You are Test Author" in prompt
        assert fallback not in prompt

    def test_returns_fallback_when_not_exists(self):
        """Test that it returns fallback when file doesn't exist."""
        fallback = "This is a fallback for {author}"
        prompt = get_prompt_with_fallback(
            "nonexistent_prompt_xyz",
            fallback,
            author="Test Author"
        )
        assert "This is a fallback for Test Author" in prompt


class TestListPrompts:
    """Tests for list_prompts function."""

    def test_list_default_prompts(self):
        """Test listing prompts from default directory."""
        prompts = list_prompts()
        assert len(prompts) > 0
        assert "style_transfer_system" in prompts
        assert "critic_repair_system" in prompts

    def test_list_prompts_returns_paths(self):
        """Test that list_prompts returns Path objects."""
        prompts = list_prompts()
        for name, path in prompts.items():
            assert isinstance(path, Path)
            assert path.exists()

    def test_list_prompts_empty_directory(self, tmp_path):
        """Test listing prompts from empty directory."""
        prompts = list_prompts(tmp_path)
        assert prompts == {}


class TestClearPromptCache:
    """Tests for clear_prompt_cache function."""

    def test_clear_cache(self):
        """Test that cache can be cleared."""
        # Load a prompt to populate cache
        load_prompt("style_transfer_system")
        # Clear should not raise
        clear_prompt_cache()
        # Should still be able to load after clear
        prompt = load_prompt("style_transfer_system")
        assert prompt is not None


class TestPromptFiles:
    """Tests for the actual prompt files."""

    def test_all_prompts_have_valid_placeholders(self):
        """Test that all prompts have valid Python format placeholders."""
        prompts = list_prompts()
        for name, path in prompts.items():
            content = path.read_text()
            # Check that placeholders are valid (no unmatched braces)
            # This is a basic check - actual formatting will validate fully
            open_count = content.count('{')
            close_count = content.count('}')
            assert open_count == close_count, f"Unmatched braces in {name}"

    def test_style_transfer_prompts_exist(self):
        """Test that core style transfer prompts exist."""
        required_prompts = [
            "style_transfer_system",
            "style_transfer_base_model",
            "critic_repair_system",
            "critic_repair_user",
            "repair_strict",
        ]
        prompts = list_prompts()
        for name in required_prompts:
            assert name in prompts, f"Missing required prompt: {name}"

    def test_prompts_not_empty(self):
        """Test that no prompt files are empty."""
        prompts = list_prompts()
        for name, path in prompts.items():
            content = path.read_text()
            assert len(content.strip()) > 0, f"Empty prompt file: {name}"
