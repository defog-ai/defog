from typing import Optional

from defog import config as defog_config

from .anthropic_provider import AnthropicProvider
from ..config import LLMConfig


class GrokProvider(AnthropicProvider):
    """Grok provider implementation using Anthropic-compatible client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config=None,
    ):
        # xAI recommends using the Anthropic SDK with a different base URL.
        super().__init__(
            api_key=api_key
            or defog_config.get("XAI_API_KEY")
            or defog_config.get("GROK_API_KEY"),
            base_url=base_url or "https://api.x.ai",
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create Grok provider from config."""
        api_key = config.get_api_key("grok") or defog_config.get("XAI_API_KEY")
        base_url = config.get_base_url("grok") or "https://api.x.ai"
        return cls(api_key=api_key, base_url=base_url, config=config)

    def get_provider_name(self) -> str:
        return "grok"
