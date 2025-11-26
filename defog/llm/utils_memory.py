"""Chat utilities with memory management support."""

from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass

from .utils import chat_async
from .llm_providers import LLMProvider
from .providers.base import LLMResponse
from .config import LLMConfig
from .memory import MemoryManager, compactify_messages, TokenCounter


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    enabled: bool = True
    token_threshold: int = 50000  # ~50k tokens before compactifying
    preserve_last_n_messages: int = 10
    summary_max_tokens: int = 4000
    max_context_tokens: int = 128000  # 128k context window


async def chat_async_with_memory(
    provider: Union[LLMProvider, str],
    model: str,
    messages: List[Dict[str, str]],
    memory_manager: Optional[MemoryManager] = None,
    memory_config: Optional[MemoryConfig] = None,
    auto_compactify: bool = True,
    max_completion_tokens: Optional[int] = None,
    temperature: float = 0.0,
    response_format=None,
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
    tool_result_preview_max_tokens: Optional[int] = None,
    tool_sample_functions: Optional[Dict[str, Callable]] = None,
    config: Optional[LLMConfig] = None,
) -> LLMResponse:
    """
    Execute a chat completion with memory management support.

    This function extends chat_async with automatic conversation memory management
    and compactification when approaching token limits.

    Args:
        provider: LLM provider to use
        model: Model name
        messages: List of message dictionaries
        memory_manager: Optional MemoryManager instance (created if not provided)
        memory_config: Memory configuration settings
        auto_compactify: Whether to automatically compactify when needed
        ... (all other chat_async parameters)

    Returns:
        LLMResponse object with the result
    """
    # Initialize memory config if not provided
    if memory_config is None:
        memory_config = MemoryConfig()

    # Initialize memory manager if not provided and memory is enabled
    if memory_manager is None and memory_config.enabled:
        memory_manager = MemoryManager(
            token_threshold=memory_config.token_threshold,
            preserve_last_n_messages=memory_config.preserve_last_n_messages,
            summary_max_tokens=memory_config.summary_max_tokens,
            enabled=memory_config.enabled,
        )

    # If memory is disabled, just pass through to regular chat_async
    if not memory_config.enabled or memory_manager is None:
        return await chat_async(
            provider=provider,
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            store=store,
            metadata=metadata,
            timeout=timeout,
            backup_model=backup_model,
            backup_provider=backup_provider,
            prediction=prediction,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            max_retries=max_retries,
            post_tool_function=post_tool_function,
            tool_result_preview_max_tokens=tool_result_preview_max_tokens,
            tool_sample_functions=tool_sample_functions,
            config=config,
        )

    # Get current messages from memory manager
    memory_manager.get_current_messages()

    # Add new messages to memory
    token_counter = TokenCounter()
    new_tokens = token_counter.count_tokens(messages, model, str(provider))
    memory_manager.add_messages(messages, new_tokens)

    # Check if we should compactify
    if auto_compactify and memory_manager.should_compactify():
        # Get messages to summarize and preserve
        system_messages, messages_to_summarize, preserved_messages = (
            memory_manager.get_messages_for_compactification()
        )

        # Compactify messages
        compactified_messages, new_token_count = await compactify_messages(
            system_messages=system_messages,
            messages_to_summarize=messages_to_summarize,
            preserved_messages=preserved_messages,
            provider=str(provider),
            model=model,
            max_summary_tokens=memory_config.summary_max_tokens,
            config=config,  # Pass config for API credentials
        )

        # Update memory manager with compactified messages
        if compactified_messages and len(compactified_messages) > len(system_messages):
            # Find the summary message (first non-system message in the result)
            summary_idx = len(system_messages)
            summary_message = compactified_messages[summary_idx]
            memory_manager.update_after_compactification(
                system_messages=system_messages,
                summary_message=summary_message,
                preserved_messages=preserved_messages,
                new_token_count=new_token_count,
            )

        # Use compactified messages for the API call
        messages_for_api = compactified_messages
    else:
        # Use all messages from memory
        messages_for_api = memory_manager.get_current_messages()

    # Make the API call with the potentially compactified messages
    response = await chat_async(
        provider=provider,
        model=model,
        messages=messages_for_api,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        response_format=response_format,
        store=store,
        metadata=metadata,
        timeout=timeout,
        backup_model=backup_model,
        backup_provider=backup_provider,
        prediction=prediction,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice=tool_choice,
        max_retries=max_retries,
        post_tool_function=post_tool_function,
        tool_result_preview_max_tokens=tool_result_preview_max_tokens,
        tool_sample_functions=tool_sample_functions,
        config=config,
    )

    # Add the assistant's response to memory
    assistant_message = {"role": "assistant", "content": response.content}
    response_tokens = response.output_tokens or 0
    memory_manager.add_messages([assistant_message], response_tokens)

    # Add memory stats to response metadata
    if hasattr(response, "_memory_stats"):
        response._memory_stats = memory_manager.get_stats()

    return response


# Convenience function for creating a memory manager
def create_memory_manager(
    token_threshold: int = 100000,
    preserve_last_n_messages: int = 20,
    summary_max_tokens: int = 10000,
    enabled: bool = True,
) -> MemoryManager:
    """
    Create a new MemoryManager instance.

    Args:
        token_threshold: Token count threshold for triggering compactification
        preserve_last_n_messages: Number of recent messages to always preserve
        summary_max_tokens: Maximum tokens for the summary
        enabled: Whether memory management is enabled

    Returns:
        Configured MemoryManager instance
    """
    return MemoryManager(
        token_threshold=token_threshold,
        preserve_last_n_messages=preserve_last_n_messages,
        summary_max_tokens=summary_max_tokens,
        enabled=enabled,
    )
