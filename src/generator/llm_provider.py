"""Unified LLM provider interface for all supported providers.

This module provides a consistent API for interacting with different LLM providers
(DeepSeek, Ollama, GLM, Gemini) through a single class interface.
"""

import json
import requests
from typing import Dict, Optional


class LLMProvider:
    """Unified interface for LLM providers (DeepSeek, Ollama, GLM, Gemini)."""

    def __init__(self, config_path: str = "config.json", provider: Optional[str] = None):
        """Initialize LLM provider from config.

        Args:
            config_path: Path to configuration file.
            provider: Optional provider name to override config. If None, uses config value.
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.provider = provider or self.config.get("provider", "deepseek")
        # Get default timeout and context window from config
        llm_provider_config = self.config.get("llm_provider", {})
        self.default_timeout = llm_provider_config.get("timeout", 120)  # Default 120 seconds
        self.context_window = llm_provider_config.get("context_window", 128000)  # Default 128k for modern models
        self.max_output_tokens = llm_provider_config.get("max_output_tokens", 4000)  # Default 4k for output
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize provider-specific configuration."""
        if self.provider == "deepseek":
            deepseek_config = self.config.get("deepseek", {})
            self.api_key = deepseek_config.get("api_key", "").strip() if deepseek_config.get("api_key") else ""
            self.api_url = deepseek_config.get("api_url", "").strip() if deepseek_config.get("api_url") else ""
            self.editor_model = deepseek_config.get("editor_model", "deepseek-chat")
            self.critic_model = deepseek_config.get("critic_model", self.editor_model)

            if not self.api_key:
                raise ValueError(
                    "DeepSeek API key not found in config. Please set 'deepseek.api_key' in config.json. "
                    "Get your API key at https://platform.deepseek.com"
                )
            if not self.api_url:
                raise ValueError(
                    "DeepSeek API URL not found in config. Please set 'deepseek.api_url' in config.json"
                )

        elif self.provider == "ollama":
            ollama_config = self.config.get("ollama", {})
            self.api_url = ollama_config.get("url", "http://localhost:11434/api/chat")
            self.editor_model = ollama_config.get("editor_model", "mistral-nemo")
            self.critic_model = ollama_config.get("critic_model", "qwen3:8b")
            self.keep_alive = ollama_config.get("keep_alive", "10m")
            self.api_key = None  # Ollama doesn't use API keys

        elif self.provider == "glm":
            glm_config = self.config.get("glm", {})
            self.api_key = glm_config.get("api_key", "").strip() if glm_config.get("api_key") else ""
            self.api_url = glm_config.get("api_url", "").strip() if glm_config.get("api_url") else ""
            self.editor_model = glm_config.get("editor_model", "glm-4.6")
            self.critic_model = glm_config.get("critic_model", self.editor_model)

            if not self.api_key:
                raise ValueError(
                    "GLM API key not found in config. Please set 'glm.api_key' in config.json. "
                    "Get your API key at https://open.bigmodel.cn"
                )
            if not self.api_url:
                raise ValueError(
                    "GLM API URL not found in config. Please set 'glm.api_url' in config.json"
                )

        elif self.provider == "gemini":
            gemini_config = self.config.get("gemini", {})
            self.api_key = gemini_config.get("api_key", "").strip() if gemini_config.get("api_key") else ""
            self.api_url = gemini_config.get("api_url", "").strip() if gemini_config.get("api_url") else ""
            self.editor_model = gemini_config.get("editor_model", "gemini-3-flash-preview")
            self.critic_model = gemini_config.get("critic_model", self.editor_model)
            self.thinking_level = gemini_config.get("thinkingLevel", "MEDIUM")
            self.include_thoughts = gemini_config.get("includeThoughts", False)

            if not self.api_key:
                raise ValueError(
                    "Gemini API key not found in config. Please set 'gemini.api_key' in config.json. "
                    "Get your API key at https://aistudio.google.com/app/apikey"
                )
            if not self.api_url:
                raise ValueError(
                    "Gemini API URL not found in config. Please set 'gemini.api_url' in config.json"
                )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken.

        Uses model-specific tokenizer for accurate counting.
        Falls back to rough estimate if tiktoken unavailable.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        try:
            import tiktoken
            # Use cl100k_base (GPT-4/DeepSeek compatible)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to rough estimate if tiktoken not available
            return len(text) // 4
        except Exception:
            # Fallback on any error
            return len(text) // 4

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_type: str = "editor",
        require_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        verbose_context: bool = False
    ) -> str:
        """Call LLM API with unified interface.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt with the request.
            model_type: Which model to use ("editor" or "critic"). Default: "editor".
            require_json: If True, request JSON format (for critic). If False, plain text.
            temperature: Optional temperature override. Default: provider-specific.
            max_tokens: Optional max_tokens override. Default: max_output_tokens from config.
            top_p: Optional top_p override. Default: provider-specific.
            timeout: Optional timeout in seconds. Default: provider-specific (30s for DeepSeek, 60s for Ollama/GLM).
            verbose_context: If True, log context window utilization. Default: False.

        Returns:
            LLM response text.
        """
        # Use max_output_tokens as default if max_tokens not provided
        output_tokens = max_tokens if max_tokens is not None else self.max_output_tokens

        # Estimate input tokens and check context window
        input_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt)
        utilization = (input_tokens / self.context_window) * 100 if self.context_window > 0 else 0

        # Log context usage if verbose or approaching limits
        if verbose_context or utilization > 80:
            print(f"  📊 Context Usage: {input_tokens:,} input tokens / {self.context_window:,} context window ({utilization:.1f}% utilized)")
            if utilization > 80:
                print(f"  ⚠ Warning: Context window utilization is {utilization:.1f}% - approaching limit")
            if input_tokens > self.context_window:
                print(f"  ❌ Error: Input tokens ({input_tokens:,}) exceed context window ({self.context_window:,})")
                # Don't crash, but log the issue - let the API handle it

        model = self.critic_model if model_type == "critic" else self.editor_model

        if self.provider == "deepseek":
            return self._call_deepseek_api(
                system_prompt, user_prompt, model, require_json, temperature, output_tokens, top_p, timeout
            )
        elif self.provider == "ollama":
            return self._call_ollama_api(
                system_prompt, user_prompt, model, require_json, temperature, output_tokens, top_p, timeout
            )
        elif self.provider == "glm":
            return self._call_glm_api(
                system_prompt, user_prompt, model, require_json, temperature, output_tokens, top_p, timeout
            )
        elif self.provider == "gemini":
            return self._call_gemini_api(
                system_prompt, user_prompt, model, require_json, temperature, output_tokens, top_p, timeout
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_deepseek_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        require_json: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> str:
        """Call DeepSeek API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature if temperature is not None else 0.3,
        }

        if top_p is not None:
            payload["top_p"] = top_p

        # Only add JSON format requirement for critic evaluation
        if require_json:
            payload["response_format"] = {"type": "json_object"}
        else:
            # For text generation/editing, add max_tokens instead
            payload["max_tokens"] = max_tokens if max_tokens is not None else 200

        # Validate model name for DeepSeek API
        valid_deepseek_models = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner", "deepseek-chat-v3"]
        if model not in valid_deepseek_models:
            print(f"    ⚠ Warning: Model '{model}' may not be valid for DeepSeek API. Valid models: {valid_deepseek_models}")

        # Use provided timeout or default from config (or 30 seconds as fallback)
        api_timeout = timeout if timeout is not None else (getattr(self, 'default_timeout', None) or 30)

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=api_timeout)
            response.raise_for_status()

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_detail = e.response.text
                raise RuntimeError(f"DeepSeek API 400 Bad Request: {error_detail}. Check model name '{model}' and request format.")
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DeepSeek API request failed: {e}")

    def _call_ollama_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        require_json: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> str:
        """Call Ollama API."""
        # Ollama uses /api/chat endpoint
        if "/api/generate" in self.api_url:
            api_url = self.api_url.replace("/api/generate", "/api/chat")
        else:
            api_url = self.api_url

        # For JSON format requests (critic), add format instruction to system prompt
        if require_json:
            enhanced_system_prompt = system_prompt + "\n\nIMPORTANT: You must respond with valid JSON only. No additional text or explanation."
        else:
            enhanced_system_prompt = system_prompt

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,  # Always use non-streaming for consistency
            "keep_alive": self.keep_alive
        }

        # Add options for temperature, max tokens, and top_p
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if top_p is not None:
            options["top_p"] = top_p
        if require_json:
            data["format"] = "json"  # Request JSON format for Ollama
        if options:
            data["options"] = options

        # Use provided timeout or default from config (or 60 seconds as fallback)
        api_timeout = timeout if timeout is not None else (getattr(self, 'default_timeout', None) or 60)

        try:
            response = requests.post(api_url, json=data, timeout=api_timeout)
            response.raise_for_status()

            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Try to get available models for better error message
                try:
                    models_url = api_url.replace("/api/chat", "/api/tags")
                    models_response = requests.get(models_url, timeout=5)
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        available_models = [m.get("name", "") for m in models_data.get("models", [])]
                        raise RuntimeError(
                            f"Ollama model '{model}' not found. Available models: {', '.join(available_models[:5])}"
                        )
                except:
                    pass
                raise RuntimeError(f"Ollama API 404: Model '{model}' not found or endpoint incorrect. Check model name and Ollama service.")
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")

    def _call_glm_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        require_json: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> str:
        """Call GLM (Zhipu AI) API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature if temperature is not None else 0.3,
        }

        if top_p is not None:
            payload["top_p"] = top_p

        # Only add JSON format requirement for critic evaluation
        if require_json:
            payload["response_format"] = {"type": "json_object"}
        else:
            # For text generation/editing, add max_tokens instead
            payload["max_tokens"] = max_tokens if max_tokens is not None else 200

        # Use provided timeout or default from config (or 60 seconds as fallback)
        api_timeout = timeout if timeout is not None else (getattr(self, 'default_timeout', None) or 60)

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=api_timeout)
            response.raise_for_status()

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_detail = e.response.text
                raise RuntimeError(f"GLM API 400 Bad Request: {error_detail}. Check model name '{model}' and request format.")
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"GLM API request failed: {e}")

    def _call_gemini_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        require_json: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> str:
        """Call Google Gemini API."""
        # Construct full URL with API key
        # API URL format: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}
        # Note: Model is embedded in the URL path, so we use the URL from config as-is
        # If model parameter differs from URL, we could substitute it, but for now we use URL as-is
        if "?key=" in self.api_url:
            # URL already has ?key=, just append the key
            api_url = self.api_url + self.api_key
        elif "key=" in self.api_url:
            # URL already has key parameter (maybe with value), replace it
            import re
            api_url = re.sub(r'key=[^&]*', f'key={self.api_key}', self.api_url)
        else:
            # URL doesn't have key parameter, add it
            separator = "&" if "?" in self.api_url else "?"
            api_url = f"{self.api_url}{separator}key={self.api_key}"

        # Combine system and user prompts for Gemini (Gemini doesn't have separate system/user roles in the same way)
        # Gemini uses a single "contents" array, but we can put system prompt as first message
        combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

        payload = {
            "contents": [{
                "parts": [{"text": combined_prompt}]
            }],
            "generationConfig": {
                "temperature": temperature if temperature is not None else 0.3,
            }
        }

        if top_p is not None:
            payload["generationConfig"]["top_p"] = top_p

        # Add thinking config if available
        if hasattr(self, 'thinking_level') and hasattr(self, 'include_thoughts'):
            payload["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": self.include_thoughts,
                "thinkingLevel": self.thinking_level
            }

        # Add max_tokens if provided (Gemini uses maxOutputTokens)
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        # For JSON format requests (critic), add format instruction
        if require_json:
            # Gemini doesn't have a direct JSON mode, so we add instruction to prompt
            payload["contents"][0]["parts"][0]["text"] = combined_prompt + "\n\nIMPORTANT: You must respond with valid JSON only. No additional text or explanation."

        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Use provided timeout or default from config (or 60 seconds as fallback)
            api_timeout = timeout if timeout is not None else (getattr(self, 'default_timeout', None) or 60)
            response = requests.post(api_url, headers=headers, json=payload, timeout=api_timeout)
            response.raise_for_status()

            result = response.json()

            # Extract text from Gemini response structure
            # Response format: candidates[0].content.parts[0].text
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]

                # Check for finish reason (might indicate error)
                finish_reason = candidate.get("finishReason", "")
                if finish_reason and finish_reason != "STOP":
                    # Handle non-STOP finish reasons (e.g., MAX_TOKENS, SAFETY)
                    if finish_reason == "MAX_TOKENS":
                        print(f"    ⚠ Warning: Gemini response truncated (MAX_TOKENS). Consider increasing max_tokens.")
                    elif finish_reason == "SAFETY":
                        raise RuntimeError(f"Gemini API blocked content due to safety filters. Finish reason: {finish_reason}")
                    elif finish_reason == "RECITATION":
                        raise RuntimeError(f"Gemini API blocked content due to recitation detection. Finish reason: {finish_reason}")
                    else:
                        print(f"    ⚠ Warning: Gemini finish reason: {finish_reason}")

                if "content" in candidate and "parts" in candidate["content"]:
                    if len(candidate["content"]["parts"]) > 0:
                        text = candidate["content"]["parts"][0].get("text", "")
                        if text:
                            return text.strip()
                        else:
                            raise RuntimeError(f"Gemini API returned empty text. Finish reason: {finish_reason}")

            # Fallback: try to find text anywhere in response
            raise RuntimeError(f"Gemini API response format unexpected. Response: {result}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_detail = e.response.text
                raise RuntimeError(f"Gemini API 400 Bad Request: {error_detail}. Check model name '{model}' and request format.")
            elif e.response.status_code == 401:
                raise RuntimeError(f"Gemini API 401 Unauthorized: Invalid API key. Check 'gemini.api_key' in config.json.")
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini API request failed: {e}")

