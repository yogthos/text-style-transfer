"""Helper functions for tests to avoid skipping."""
from pathlib import Path
import json
from unittest.mock import Mock, patch


def ensure_config_exists():
    """Ensure config.json exists with minimal configuration."""
    config_path = Path("config.json")
    if not config_path.exists():
        minimal_config = {
            "provider": "deepseek",
            "deepseek": {"api_key": "test-key", "model": "deepseek-chat"},
            "critic": {"fallback_pass_threshold": 0.75}
        }
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f)
    return config_path


def mock_llm_provider_for_critic():
    """Create a mock LLM provider for critic_evaluate tests."""
    mock_llm = Mock()
    mock_llm.call.return_value = json.dumps({
        "pass": True,
        "score": 0.85,
        "feedback": "Text matches structure and situation well."
    })
    mock_llm_class = Mock(return_value=mock_llm)
    return patch('src.validator.critic.LLMProvider', mock_llm_class)

