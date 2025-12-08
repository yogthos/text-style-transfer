import requests
import json
from typing import Optional

class GLMProvider:
    """GLM API provider implementation for Z.AI GLM models."""

    def __init__(self, api_key: str, api_url: str = "https://api.z.ai/api/paas/v4/chat/completions"):
        """
        Initialize GLM provider.

        Args:
            api_key: Z.AI API key
            api_url: API endpoint URL (default: Z.AI chat completions endpoint)
        """
        self.api_key = api_key
        self.api_url = api_url
        if not self.api_key:
            raise ValueError("GLM API key is required. Set it in config.json or GLM_API_KEY environment variable.")

    def call(self, model: str, prompt: str, system_prompt: str = "", temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 4096) -> str:
        """
        Call GLM API with a prompt.

        Args:
            model: Model name (e.g., "glm-4.6")
            prompt: User prompt text
            system_prompt: System prompt text (optional)
            temperature: Sampling temperature (default: 0.7)
            top_p: Top-p sampling parameter (default: 0.9)
            max_tokens: Maximum tokens to generate (default: 4096)

        Returns:
            Response text from the model
        """
        # Build messages array
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add user message
        messages.append({
            "role": "user",
            "content": prompt
        })

        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }

        # Make API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            # Add timeout to prevent hanging (60 seconds for generation)
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()

            result = response.json()

            # Debug: print response structure if empty
            if "choices" not in result or len(result.get("choices", [])) == 0:
                print(f"WARNING: GLM API response has no choices. Full response: {json.dumps(result, indent=2)}")
                return ""  # Return empty string, let caller handle it

            # Extract response content
            choice = result["choices"][0]

            # Check finish_reason to understand why response might be empty
            finish_reason = choice.get("finish_reason", "unknown")
            if finish_reason != "stop":
                print(f"WARNING: GLM API finish_reason: {finish_reason}. This may indicate why response is empty.")

            if "message" not in choice:
                print(f"WARNING: GLM API choice has no message. Choice: {json.dumps(choice, indent=2)}")
                return ""

            content = choice["message"].get("content", "")

            if not content or not content.strip():
                print(f"WARNING: GLM API returned empty content (finish_reason: {finish_reason})")
                print(f"Full response structure: {json.dumps(result, indent=2)[:500]}...")  # Limit output
                return ""  # Return empty string instead of raising error

            return content

        except requests.exceptions.RequestException as e:
            error_msg = f"GLM API request failed: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nResponse: {json.dumps(error_detail, indent=2)}"
                except:
                    error_msg += f"\nResponse text: {e.response.text}"
            raise RuntimeError(error_msg)

