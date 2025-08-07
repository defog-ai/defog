import asyncio
import traceback
import json
from typing import Dict, List, Optional, Any, Union, Callable

from .providers import (
    BaseLLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    TogetherProvider,
    DeepSeekProvider,
    MistralProvider,
)
from .providers.base import LLMResponse
from .exceptions import LLMError, ConfigurationError
from .config import LLMConfig
from .llm_providers import LLMProvider
from .citations import citations_tool
from copy import deepcopy
from .utils_mcp import get_mcp_tools

# Keep the original LLMResponse for backwards compatibility
# (it's now defined in providers.base but we re-export it here)
__all__ = [
    "LLMResponse",
    "chat_async",
    "map_model_to_provider",
    "get_provider_instance",
]


def get_provider_instance(
    provider: Union[LLMProvider, str], config: Optional[LLMConfig] = None
) -> BaseLLMProvider:
    """
    Get a provider instance based on the provider enum or string.

    Args:
        provider: LLMProvider enum or string name
        config: Optional configuration object

    Returns:
        BaseLLMProvider instance

    Raises:
        ConfigurationError: If provider is not supported or misconfigured
    """
    if config is None:
        config = LLMConfig()

    # Handle both enum and string values
    if isinstance(provider, LLMProvider):
        provider_name = provider.value
    else:
        provider_name = provider.lower()

    # Validate provider config
    if not config.validate_provider_config(provider_name):
        raise ConfigurationError(f"No API key found for provider '{provider_name}'")

    # Create provider instances
    provider_classes = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "together": TogetherProvider,
        "deepseek": DeepSeekProvider,
        "mistral": MistralProvider,
        "alibaba": OpenAIProvider,  # Alibaba uses OpenAI-compatible API
    }

    if provider_name not in provider_classes:
        raise ConfigurationError(f"Unsupported provider: {provider_name}")

    provider_class = provider_classes[provider_name]

    # Handle special cases for providers that need custom configuration
    if provider_name == "alibaba":
        return provider_class(
            api_key=config.get_api_key("alibaba"),
            base_url=config.get_base_url("alibaba"),
            config=config,
        )
    else:
        # Use the provider's from_config method for consistent initialization
        return provider_class.from_config(config)


async def chat_async(
    provider: Union[LLMProvider, str],
    model: str,
    messages: List[Dict[str, Any]],
    max_completion_tokens: Optional[int] = None,
    temperature: float = 0.0,
    response_format=None,
    verbosity: Optional[str] = None,
    store: bool = True,
    metadata: Optional[Dict[str, str]] = None,
    timeout: int = 600,
    backup_model: Optional[str] = None,
    backup_provider: Optional[Union[LLMProvider, str]] = None,
    prediction: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None,
    tools: Optional[List[Callable]] = None,
    tool_choice: Optional[str] = None,
    max_retries: Optional[int] = None,
    post_tool_function: Optional[Callable] = None,
    post_response_hook: Optional[Callable] = None,
    config: Optional[LLMConfig] = None,
    mcp_servers: Optional[List[str]] = None,
    image_result_keys: Optional[List[str]] = None,
    tool_budget: Optional[Dict[str, int]] = None,
    insert_tool_citations: bool = False,
    parallel_tool_calls: bool = False,
    tool_output_max_tokens: int = 10000,
) -> LLMResponse:
    """
    Execute a chat completion with explicit provider parameter.

    Args:
        provider: LLMProvider enum or string specifying which provider to use
        model: Model name to use
        messages: List of message dictionaries with text or multimodal content
        max_completion_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        response_format: Structured output format (Pydantic model)
        store: Whether to store the conversation
        metadata: Additional metadata
        timeout: Request timeout in seconds
        backup_model: Fallback model to use on retry
        backup_provider: Fallback provider to use on retry
        prediction: Predicted output configuration (OpenAI only)
        reasoning_effort: Reasoning effort level (o1/o3 models only)
        tools: List of tools the model can call
        tool_choice: Tool calling behavior ("auto", "required", function name)
        max_retries: Maximum number of retry attempts
        post_tool_function: Function to call after each tool execution. Must have parameters: function_name, input_args, tool_result
        post_response_hook: Function to call after each response is received from the model. Must have parameters: response, messages
        config: LLM configuration object
        mcp_servers: List of MCP server urls for streamable http servers (e.g., ["http://localhost:8000/mcp", "http://localhost:8001/mcp"])
        image_result_keys: List of keys to check in tool results for image data (e.g., ['image_base64', 'screenshot_data'])
        tool_budget: Dictionary mapping tool names to maximum allowed calls. Tools not in the dictionary have unlimited calls.
        insert_tool_citations: If True, adds citations to the response using tool outputs as source documents (OpenAI and Anthropic only)
        parallel_tool_calls: Enable parallel tool calls when set to True (default: False)
        tool_output_max_tokens: Maximum tokens allowed in tool outputs (default: 10000). Set to -1 to disable the check.

    Returns:
        LLMResponse object containing the result

    Raises:
        ConfigurationError: If provider configuration is invalid
        ProviderError: If the provider API call fails
        LLMError: For other LLM-related errors
    """
    # create a deep copy of the messages to avoid modifying the original messages
    messages = deepcopy(messages)

    if config is None:
        config = LLMConfig()

    if max_retries is None:
        max_retries = config.max_retries

    base_delay = 1  # Initial delay in seconds
    error_trace = None

    # validate that mcp servers are simple strings and nothing else
    if mcp_servers:
        for mcp_server in mcp_servers:
            if not isinstance(mcp_server, str):
                raise ValueError("mcp_servers must be a list of strings")

    if mcp_servers:
        mcp_tools = []
        for mcp_server in mcp_servers:
            mcp_tools.extend(await get_mcp_tools(mcp_server))
        if tools:
            tools = tools + mcp_tools
        else:
            tools = mcp_tools

    for attempt in range(max_retries):
        try:
            # Use backup provider/model on subsequent attempts
            current_provider = provider
            current_model = model

            if attempt > 0:
                if backup_provider is not None:
                    current_provider = backup_provider
                if backup_model is not None:
                    current_model = backup_model

            # Validate provider support for citations before execution
            if insert_tool_citations:
                if isinstance(current_provider, str):
                    provider_enum = LLMProvider(current_provider.lower())
                else:
                    provider_enum = current_provider

                if provider_enum not in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
                    raise ValueError(
                        "insert_tool_citations is only supported for OpenAI and Anthropic providers"
                    )

            # Get provider instance
            provider_instance = get_provider_instance(current_provider, config)

            if post_response_hook:
                provider_instance.validate_post_response_hook(post_response_hook)

            # Execute the chat completion
            response = await provider_instance.execute_chat(
                messages=messages,
                model=current_model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                response_format=response_format,
                verbosity=verbosity,
                tools=tools,
                tool_choice=tool_choice,
                store=store,
                metadata=metadata,
                timeout=timeout,
                prediction=prediction,
                reasoning_effort=reasoning_effort,
                post_tool_function=post_tool_function,
                post_response_hook=post_response_hook,
                image_result_keys=image_result_keys,
                tool_budget=tool_budget,
                parallel_tool_calls=parallel_tool_calls,
                tool_output_max_tokens=tool_output_max_tokens,
            )

            # Process citations if requested and provider supports it
            if insert_tool_citations and response.tool_outputs:
                # Get the provider enum (already validated above)
                if isinstance(current_provider, str):
                    provider_enum = LLMProvider(current_provider.lower())
                else:
                    provider_enum = current_provider

                # Get the original user question
                original_question = get_original_user_question(messages)

                # Convert tool outputs to documents
                documents = convert_tool_outputs_to_documents(response.tool_outputs)

                # Call citations tool to add citations to the response
                citation_blocks = await citations_tool(
                    question=original_question,
                    instructions="Add citations to the following response using the tool outputs as source documents: "
                    + response.content,
                    documents=documents,
                    model=current_model,
                    provider=provider_enum,
                    max_tokens=max_completion_tokens or 16000,
                    verbose=False,  # Don't show progress for internal citation processing
                )

                # Update response with citation-enhanced content and citations
                if citation_blocks and len(citation_blocks) > 0:
                    # Extract text content from citation blocks
                    cited_content = "".join(
                        block.get("text", "") for block in citation_blocks
                    )
                    # Only update content if citation processing produced meaningful content
                    if cited_content.strip():
                        response.content = cited_content
                        response.citations = citation_blocks
                    # If citation processing failed or returned empty content, keep original content

            return response

        except Exception as e:
            error_trace = traceback.format_exc()
            delay = base_delay * (2**attempt)  # Exponential backoff

            if attempt < max_retries - 1:  # Don't log on final attempt
                print(
                    f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...",
                    flush=True,
                )
                print(f"Error: {e}", flush=True)
                await asyncio.sleep(delay)
            else:
                # Final attempt failed, re-raise the exception
                if isinstance(e, LLMError):
                    raise e
                else:
                    traceback.print_exc()
                    raise LLMError(f"All attempts failed. Latest error: {e}") from e

    # This should never be reached, but just in case
    raise LLMError(
        f"All {max_retries} attempts at calling the chat_async function failed. "
        f"The latest error traceback was: {error_trace}"
    )


# Legacy compatibility functions - these map the old model-based routing to the new provider-based system
def map_model_to_provider(model: str) -> LLMProvider:
    """
    Map a model name to its provider for backwards compatibility.

    Args:
        model: Model name

    Returns:
        LLMProvider enum value

    Raises:
        ConfigurationError: If model cannot be mapped to a provider
    """
    if model.startswith("claude"):
        return LLMProvider.ANTHROPIC
    elif model.startswith("gemini"):
        return LLMProvider.GEMINI
    elif (
        model.startswith("gpt")
        or model.startswith("o1")
        or model.startswith("chatgpt")
        or model.startswith("o3")
        or model.startswith("o4")
    ):
        return LLMProvider.OPENAI
    elif model.startswith("deepseek"):
        return LLMProvider.DEEPSEEK
    elif model.startswith("mistral"):
        return LLMProvider.MISTRAL
    elif (
        model.startswith("meta-llama")
        or model.startswith("mistralai")
        or model.startswith("Qwen")
    ):
        return LLMProvider.TOGETHER
    elif model.startswith("qwen"):  # lowercase qwen for Alibaba Cloud
        return LLMProvider.ALIBABA
    else:
        raise ConfigurationError(f"Unknown model: {model}")


def convert_tool_outputs_to_documents(
    tool_outputs: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Convert tool outputs to document format for citation processing.

    Args:
        tool_outputs: List of tool output dictionaries

    Returns:
        List of documents with document_name and document_content
    """
    documents = []
    for output in tool_outputs:
        doc_name = f"{output['name']}_{output['tool_call_id']}"
        doc_content = f"Function: {output['name']}\nArguments: {json.dumps(output['args'])}\nResult: {json.dumps(output['result'])}"
        documents.append({"document_name": doc_name, "document_content": doc_content})
    return documents


def get_original_user_question(messages: List[Dict[str, Any]]) -> str:
    """
    Extract the original user question from the messages list.

    Args:
        messages: List of message dictionaries

    Returns:
        The content of the first user message, or empty string if not found
    """
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            # Handle both string content and list content (for multimodal)
            if isinstance(content, list):
                # Extract text from multimodal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
            return content
    return ""
