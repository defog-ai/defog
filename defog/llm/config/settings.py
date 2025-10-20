import os
from typing import Optional, Dict
from .constants import (
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    DEFAULT_TEMPERATURE,
    DEEPSEEK_BASE_URL,
    OPENAI_BASE_URL,
    ALIBABA_BASE_URL,
)
from defog import config


class LLMConfig:
    """Configuration management for LLM providers."""

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        default_temperature: float = DEFAULT_TEMPERATURE,
        api_keys: Optional[Dict[str, str]] = None,
        base_urls: Optional[Dict[str, str]] = None,
        enable_parallel_tool_calls: bool = True,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_temperature = default_temperature
        self.enable_parallel_tool_calls = enable_parallel_tool_calls

        # API keys with environment fallbacks
        self.api_keys = api_keys or {}
        self._setup_api_keys()

        # Base URLs with defaults
        self.base_urls = base_urls or {}
        self._setup_base_urls()

    def _setup_api_keys(self):
        """Setup API keys with environment variable and config file fallbacks."""
        key_mappings = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "together": "TOGETHER_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "alibaba": "ALIBABA_API_KEY",
            "grok": "XAI_API_KEY",
        }

        for provider, env_var in key_mappings.items():
            if provider not in self.api_keys:
                # Try environment variable first, then fall back to config file
                self.api_keys[provider] = os.getenv(env_var) or config.get(env_var)

        # Support legacy GROK_API_KEY environment variable name
        if not self.api_keys.get("grok"):
            self.api_keys["grok"] = os.getenv("GROK_API_KEY") or config.get(
                "GROK_API_KEY"
            )

    def _setup_base_urls(self):
        """Setup base URLs with defaults."""
        default_urls = {
            "openai": OPENAI_BASE_URL,
            "deepseek": DEEPSEEK_BASE_URL,
            "alibaba": ALIBABA_BASE_URL,
            "grok": "https://api.x.ai",
        }

        for provider, url in default_urls.items():
            if provider not in self.base_urls:
                self.base_urls[provider] = url

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self.api_keys.get(provider)

    def get_base_url(self, provider: str) -> Optional[str]:
        """Get base URL for a provider."""
        return self.base_urls.get(provider)

    def validate_provider_config(self, provider: str) -> bool:
        """Validate that a provider has the required configuration."""
        api_key = self.get_api_key(provider)
        return api_key is not None and api_key != ""

    def update_config(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
