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
    GrokProvider,
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
        "grok": GrokProvider,
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
    citations_instructions: Optional[str] = None,
    parallel_tool_calls: bool = False,
    tool_output_max_tokens: int = 10000,
    tool_result_preview_max_tokens: Optional[int] = None,
    tool_sample_functions: Optional[Dict[str, Callable]] = None,
    previous_response_id: Optional[str] = None,
    citations_model: Optional[str] = None,
    citations_provider: Optional[Union[LLMProvider, str]] = None,
    citations_excluded_tools: Optional[List[str]] = None,
    citations_reasoning_effort: Optional[str] = None,
    tool_phase_complete_message: str = "exploration done, generating answer",
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
        post_tool_function: Function to call after each tool execution. Must have parameters: function_name, input_args, tool_result, tool_id
        post_response_hook: Function to call after each response is received from the model. Must have parameters: response, messages
        config: LLM configuration object
        mcp_servers: List of MCP server urls for streamable http servers (e.g., ["http://localhost:8000/mcp", "http://localhost:8001/mcp"])
        image_result_keys: List of keys to check in tool results for image data (e.g., ['image_base64', 'screenshot_data'])
        tool_budget: Dictionary mapping tool names to maximum allowed calls. Tools not in the dictionary have unlimited calls.
        tool_result_preview_max_tokens: Optional token budget for the tool output preview that is sent back to the LLM. The full tool result is still stored in tool_outputs.
        tool_sample_functions: Optional mapping of tool name to sampler function (or a single callable) used to downsample tool outputs before returning them to the LLM.
        insert_tool_citations: If True, adds citations to the response using tool outputs as source documents (OpenAI and Anthropic only)
        citations_model: Optional model to use specifically for generating citations. If provided, the provider is inferred from the model name unless citations_provider is set. If not provided, the main model is used.
        citations_provider: Optional provider to use for generating citations. Overrides provider inference from citations_model. Accepts LLMProvider enum or string name. If not provided, the main provider is used.
        citations_excluded_tools: Optional list of tool names to exclude from the documents sent for citation generation.
        parallel_tool_calls: Enable parallel tool calls when set to True (default: False)
        tool_output_max_tokens: Maximum tokens allowed in tool outputs (default: 10000). Set to -1 to disable the check.
        previous_response_id: Optional id of a previous response when continuing conversations (supported for OpenAI, Anthropic/Grok, Gemini).
        citations_reasoning_effort: Reasoning effort level for citations
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
            # Determine which provider/model will be used for citations if enabled
            if insert_tool_citations:
                # Determine provider for the main call
                if isinstance(current_provider, str):
                    provider_enum_for_main = LLMProvider(current_provider.lower())
                else:
                    provider_enum_for_main = current_provider

                # Determine citations provider and model
                # Provider precedence: explicit citations_provider > inferred from citations_model > main provider
                if citations_provider is not None:
                    if isinstance(citations_provider, str):
                        citations_provider_enum = LLMProvider(
                            citations_provider.lower()
                        )
                    else:
                        citations_provider_enum = citations_provider
                elif citations_model:
                    try:
                        citations_provider_enum = map_model_to_provider(citations_model)
                    except Exception as e:
                        raise ConfigurationError(
                            f"Unable to map citations_model '{citations_model}' to a provider: {e}"
                        )
                else:
                    citations_provider_enum = provider_enum_for_main

                citations_model_to_use = (
                    citations_model if citations_model else current_model
                )

                # If both provider and model were provided, check for an obvious mismatch and warn via error
                if citations_provider is not None and citations_model:
                    try:
                        inferred = map_model_to_provider(citations_model)
                        if inferred != citations_provider_enum:
                            raise ConfigurationError(
                                f"citations_model '{citations_model}' appears to map to provider '{inferred.value}', "
                                f"but citations_provider was set to '{citations_provider_enum.value}'. Please align them."
                            )
                    except Exception:
                        # If mapping fails, we already raised earlier; keep going otherwise
                        pass

                # Validate provider support for citations
                if citations_provider_enum not in [
                    LLMProvider.OPENAI,
                    LLMProvider.ANTHROPIC,
                ]:
                    raise ValueError(
                        "insert_tool_citations is only supported for OpenAI and Anthropic providers"
                    )

                # Validate that we have credentials for the citations provider
                provider_name = (
                    citations_provider_enum.value
                    if hasattr(citations_provider_enum, "value")
                    else str(citations_provider_enum)
                )
                if not config.validate_provider_config(provider_name):
                    raise ConfigurationError(
                        f"No API key found for citations provider '{provider_name}'"
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
                tool_result_preview_max_tokens=tool_result_preview_max_tokens,
                tool_sample_functions=tool_sample_functions,
                previous_response_id=previous_response_id,
                return_tool_outputs_only=insert_tool_citations,
                tool_phase_complete_message=tool_phase_complete_message,
            )

            # Process citations if requested and we have tool outputs
            if insert_tool_citations and response.tool_outputs:
                # Determine citations provider/model (recalculate to use the actual values)
                if isinstance(current_provider, str):
                    provider_enum_for_main = LLMProvider(current_provider.lower())
                else:
                    provider_enum_for_main = current_provider

                # Provider precedence: explicit citations_provider > inferred from citations_model > main provider
                if citations_provider is not None:
                    if isinstance(citations_provider, str):
                        citations_provider_enum = LLMProvider(
                            citations_provider.lower()
                        )
                    else:
                        citations_provider_enum = citations_provider
                elif citations_model:
                    citations_provider_enum = map_model_to_provider(citations_model)
                else:
                    citations_provider_enum = provider_enum_for_main

                citations_model_to_use = (
                    citations_model if citations_model else current_model
                )

                # Get the original user question
                original_question = get_original_user_question(messages)

                # Optionally filter out tool outputs from citation documents
                tool_outputs_for_citation = response.tool_outputs
                if citations_excluded_tools:
                    excluded_set = set(citations_excluded_tools)
                    tool_outputs_for_citation = [
                        o
                        for o in response.tool_outputs
                        if o.get("name") not in excluded_set
                    ]

                # filter out all tool outputs that errored out
                tool_outputs_for_citation = [
                    o
                    for o in tool_outputs_for_citation
                    if not o.get("error") and not o.get("exception")
                ]

                # Convert tool outputs to documents
                documents = convert_tool_outputs_to_documents(tool_outputs_for_citation)

                # If no documents remain after filtering, skip citation enhancement
                if not documents:
                    return response

                # Call citations tool to add citations to the response
                if citations_instructions:
                    citation_instructions = citations_instructions
                elif response.content:
                    citation_instructions = (
                        "Add citations to the following response using the tool outputs "
                        f"as source documents: {response.content}"
                    )
                else:
                    citation_instructions = (
                        "Use the tool outputs as source documents to answer the user's question. "
                        "Add citations that reference the relevant tool call for each fact."
                    )
                citation_blocks = await citations_tool(
                    question=original_question,
                    instructions=citation_instructions,
                    documents=documents,
                    model=citations_model_to_use,
                    provider=citations_provider_enum,
                    max_tokens=max_completion_tokens or 16000,
                    verbose=False,  # Don't show progress for internal citation processing
                    reasoning_effort=citations_reasoning_effort,
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

                # Add citation generation cost to the overall cost if available
                citation_cost = getattr(citation_blocks, "cost_in_cents", None)
                if citation_cost is not None:
                    response.cost_in_cents = (
                        response.cost_in_cents or 0
                    ) + citation_cost

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
    elif model.startswith("grok"):
        return LLMProvider.GROK
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
        tool_name = output["name"]
        tool_args = output["args"]
        if "result_for_llm" in output:
            tool_result = output["result_for_llm"]
        else:
            tool_result = output["result"]

        doc_content = f"Function: {tool_name}\nArguments: {json.dumps(tool_args)}\nResult: {json.dumps(tool_result)}"
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
