"""Mock LLM Provider for testing.

Provides mocked LLM responses to avoid API calls in tests.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from unittest.mock import Mock


class MockLLMProvider:
    """Mock LLM provider that returns pre-recorded responses."""

    def __init__(self, responses_file: Optional[Path] = None):
        """Initialize mock LLM provider.

        Args:
            responses_file: Path to JSON file with pre-recorded responses.
                           If None, uses default responses.
        """
        self.call_count = 0
        self.call_history = []

        # Load responses from file if provided
        if responses_file and responses_file.exists():
            with open(responses_file, 'r') as f:
                self.responses = json.load(f)
        else:
            # Default responses for common scenarios
            self.responses = self._get_default_responses()

    def call(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        model_type: str = "editor",
        require_json: bool = False,
        temperature: float = 0.8,
        max_tokens: int = 1500,
        timeout: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Mock LLM call that returns pre-recorded responses.

        Args:
            system_prompt: System prompt (ignored in mock)
            user_prompt: User prompt (used to select response)
            model_type: Model type (ignored in mock)
            require_json: Whether JSON is required (affects response format)
            temperature: Temperature (ignored in mock)
            max_tokens: Max tokens (ignored in mock)
            timeout: Timeout (ignored in mock)
            top_p: Top-p (ignored in mock)

        Returns:
            Mocked response string
        """
        self.call_count += 1
        self.call_history.append({
            "system_prompt": system_prompt[:100] if len(system_prompt) > 100 else system_prompt,
            "user_prompt": user_prompt[:200] if len(user_prompt) > 200 else user_prompt,
            "model_type": model_type,
            "require_json": require_json
        })

        # Try to find matching response based on prompt content
        response = self._select_response(user_prompt, require_json)

        return response

    def _select_response(self, user_prompt: str, require_json: bool) -> str:
        """Select appropriate response based on prompt content."""
        prompt_lower = user_prompt.lower()

        # Check for specific patterns in responses dict
        for key, response in self.responses.items():
            if key.lower() in prompt_lower:
                return self._format_response(response, require_json)

        # Default responses based on prompt content
        if "variant" in prompt_lower or "generate" in prompt_lower:
            # Sentence variant generation
            return self._format_response(
                "VAR: The first variant sentence.\nVAR: The second variant sentence.\nVAR: The third variant sentence.",
                require_json
            )
        elif "content distribution" in prompt_lower or "slot" in prompt_lower:
            # Content planning
            return self._format_response(
                "Content for slot 1.\nContent for slot 2.\nContent for slot 3.",
                require_json
            )
        elif require_json:
            # JSON response
            return json.dumps({
                "result": "Mocked JSON response",
                "status": "success"
            })
        else:
            # Default text response
            return "This is a mocked LLM response for testing purposes."

    def _format_response(self, response: str, require_json: bool) -> str:
        """Format response based on requirements."""
        if require_json:
            # Try to parse as JSON, if fails wrap in JSON
            try:
                json.loads(response)
                return response
            except json.JSONDecodeError:
                return json.dumps({"text": response})
        return response

    def _get_default_responses(self) -> Dict[str, Any]:
        """Get default mock responses."""
        return {
            "variant": "VAR: First variant.\nVAR: Second variant.\nVAR: Third variant.",
            "content": "Content for distribution.",
            "sentence": "A generated sentence for testing.",
            "paragraph": "A generated paragraph for testing purposes."
        }


def get_mock_llm_provider(responses_file: Optional[Path] = None) -> MockLLMProvider:
    """Factory function to create a mock LLM provider.

    Args:
        responses_file: Optional path to responses JSON file

    Returns:
        MockLLMProvider instance
    """
    if responses_file is None:
        # Try to load from default location
        default_file = Path(__file__).parent / "llm_responses.json"
        if default_file.exists():
            responses_file = default_file

    return MockLLMProvider(responses_file=responses_file)

