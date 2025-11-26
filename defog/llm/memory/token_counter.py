"""Token counting utilities for different LLM providers."""

from typing import List, Dict, Any, Optional, Union
import tiktoken
from functools import lru_cache
import json


class TokenCounter:
    """
    Accurate token counting for different LLM providers.

    Uses:
    - tiktoken for OpenAI models (and as fallback for all other providers)
    - API endpoints for Anthropic/Gemini when client is provided
    """

    def __init__(self):
        self._encoding_cache = {}

    @lru_cache(maxsize=10)
    def _get_openai_encoding(self, model: str):
        """Get and cache tiktoken encoding for OpenAI models."""
        try:
            # Try to get encoding for specific model
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Default encodings for different model families
            if "gpt-4o" in model:
                return tiktoken.get_encoding("o200k_base")
            else:
                # Default for gpt-4, gpt-3.5-turbo, and others
                return tiktoken.get_encoding("cl100k_base")

    def count_openai_tokens(
        self, messages: Union[str, List[Dict[str, Any]]], model: str = "gpt-4"
    ) -> int:
        """
        Count tokens for OpenAI models using tiktoken.

        Args:
            messages: Either a string or list of message dicts
            model: OpenAI model name

        Returns:
            Token count
        """
        encoding = self._get_openai_encoding(model)

        if isinstance(messages, str):
            return len(encoding.encode(messages))

        # Handle message list format
        # Based on OpenAI's token counting guide
        tokens_per_message = (
            3  # Every message follows <|im_start|>{role}\n{content}<|im_end|>\n
        )
        tokens_per_name = 1  # If there's a name, the role is omitted

        total_tokens = 0
        for message in messages:
            total_tokens += tokens_per_message

            for key, value in message.items():
                if key == "role":
                    total_tokens += len(encoding.encode(value))
                elif key == "content":
                    if isinstance(value, str):
                        total_tokens += len(encoding.encode(value))
                    else:
                        # Handle tool calls or other structured content
                        total_tokens += len(encoding.encode(json.dumps(value)))
                elif key == "name":
                    total_tokens += tokens_per_name
                    total_tokens += len(encoding.encode(value))

        total_tokens += 3  # Every reply is primed with <|im_start|>assistant<|im_sep|>
        return total_tokens

    async def count_anthropic_tokens(
        self, messages: List[Dict[str, Any]], model: str, client: Optional[Any] = None
    ) -> int:
        """
        Count tokens for Anthropic models using their API.

        Args:
            messages: List of message dicts
            model: Anthropic model name
            client: Optional Anthropic client instance

        Returns:
            Token count
        """
        if client is None:
            # Use OpenAI tokenizer as approximation
            return self.count_openai_tokens(messages, "gpt-4")

        try:
            # Use Anthropic's token counting endpoint
            response = await client.messages.count_tokens(
                model=model, messages=messages
            )
            return response.input_tokens
        except Exception:
            # Fallback to OpenAI tokenizer
            return self.count_openai_tokens(messages, "gpt-4")

    def count_gemini_tokens(
        self,
        content: Union[str, List[Dict[str, Any]]],
        model: str,
        client: Optional[Any] = None,
    ) -> int:
        """
        Count tokens for Gemini models.

        Args:
            content: Text or message list
            model: Gemini model name
            client: Optional Gemini client instance

        Returns:
            Token count
        """
        if client is None:
            # Use OpenAI tokenizer as approximation
            return self.count_openai_tokens(content, "gpt-4")

        try:
            # Extract text content
            text = self._extract_text(content)

            # Use Gemini's count_tokens method
            response = client.count_tokens(text)
            return response.total_tokens
        except Exception:
            # Fallback to OpenAI tokenizer
            return self.count_openai_tokens(content, "gpt-4")

    def count_together_tokens(
        self, messages: Union[str, List[Dict[str, Any]]], model: str
    ) -> int:
        """
        Count tokens for Together models using OpenAI tokenizer as approximation.

        Args:
            messages: Text or message list
            model: Together model name

        Returns:
            Estimated token count
        """
        # Use OpenAI tokenizer as approximation
        return self.count_openai_tokens(messages, "gpt-4")

    def count_tokens(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        model: str,
        provider: str,
        client: Optional[Any] = None,
    ) -> int:
        """
        Universal token counting method.

        Args:
            messages: Text or message list
            model: Model name
            provider: Provider name (openai, anthropic, gemini, together)
            client: Optional provider client for API-based counting

        Returns:
            Token count
        """
        provider_lower = provider.lower()

        if provider_lower == "openai":
            return self.count_openai_tokens(messages, model)
        elif provider_lower == "anthropic":
            # Anthropic count_tokens is async, so for sync context use OpenAI approximation
            if (
                client
                and hasattr(client, "messages")
                and hasattr(client.messages, "count_tokens")
            ):
                # This would need to be called in an async context
                return self.count_openai_tokens(messages, "gpt-4")
            return self.count_openai_tokens(messages, "gpt-4")
        elif provider_lower == "gemini":
            return self.count_gemini_tokens(messages, model, client)
        elif provider_lower == "together":
            return self.count_together_tokens(messages, model)
        else:
            # Default to OpenAI tokenizer
            return self.count_openai_tokens(messages, "gpt-4")

    def _extract_text(self, content: Union[str, List[Dict[str, Any]]]) -> str:
        """Extract text from various content formats."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if "content" in item:
                        texts.append(str(item["content"]))
                    elif "text" in item:
                        texts.append(str(item["text"]))
                else:
                    texts.append(str(item))
            return " ".join(texts)

        return str(content)

    def estimate_remaining_tokens(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        model: str,
        provider: str,
        max_context_tokens: int = 128000,
        response_buffer: int = 4000,
        client: Optional[Any] = None,
    ) -> int:
        """
        Estimate remaining tokens in context window.

        Args:
            messages: Current messages
            model: Model name
            provider: Provider name
            max_context_tokens: Maximum context window size
            response_buffer: Tokens to reserve for response
            client: Optional provider client

        Returns:
            Estimated remaining tokens
        """
        used_tokens = self.count_tokens(messages, model, provider, client)
        return max(0, max_context_tokens - used_tokens - response_buffer)

    def count_tool_output_tokens(self, tool_output: Any, model: str = "gpt-4") -> int:
        """
        Count tokens for a single tool output.

        Args:
            tool_output: The tool output (can be string, dict, list, etc.)
            model: Model name for tokenization (defaults to gpt-4)

        Returns:
            Token count for the tool output
        """
        # Convert tool output to string representation
        if isinstance(tool_output, str):
            output_str = tool_output
        else:
            # For non-string outputs, use JSON serialization
            try:
                output_str = json.dumps(tool_output)
            except (TypeError, ValueError):
                # Fallback to string representation
                output_str = str(tool_output)

        # Use OpenAI tokenizer to count tokens
        return self.count_openai_tokens(output_str, model)

    def truncate_tool_output(
        self, tool_output: Any, max_tokens: int, model: str = "gpt-4"
    ) -> str:
        """
        Truncate a tool output to a target token budget.

        Args:
            tool_output: The tool output (string or JSON-serializable object)
            max_tokens: Maximum tokens to keep
            model: Model name for tokenization (defaults to gpt-4)

        Returns:
            String representation trimmed to the requested token budget with a truncation marker when applicable.
        """
        # Convert to string so we can consistently trim
        if isinstance(tool_output, str):
            output_str = tool_output
        else:
            try:
                output_str = json.dumps(tool_output)
            except (TypeError, ValueError):
                output_str = str(tool_output)

        if max_tokens <= 0:
            return ""

        encoding = self._get_openai_encoding(model)
        tokens = encoding.encode(output_str)
        if len(tokens) <= max_tokens:
            return output_str

        truncated = encoding.decode(tokens[:max_tokens])
        truncated_tokens = len(tokens) - max_tokens
        return (
            f"{truncated}\n\n...[truncated {truncated_tokens} tokens from tool output]"
        )

    def validate_tool_output_size(
        self, tool_output: Any, max_tokens: int = 10000, model: str = "gpt-4"
    ) -> tuple[bool, int]:
        """
        Validate if a tool output is within the token limit.

        Args:
            tool_output: The tool output to validate
            max_tokens: Maximum allowed tokens (default: 10000)
            model: Model name for tokenization

        Returns:
            Tuple of (is_valid, token_count)
        """
        token_count = self.count_tool_output_tokens(tool_output, model)
        return token_count <= max_tokens, token_count
