"""Message compactification utilities for conversation memory management."""

from typing import List, Dict, Any, Tuple
from .token_counter import TokenCounter


async def compactify_messages(
    system_messages: List[Dict[str, Any]],
    messages_to_summarize: List[Dict[str, Any]],
    preserved_messages: List[Dict[str, Any]],
    provider: str,
    model: str,
    max_summary_tokens: int = 2000,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Compactify a conversation by summarizing older messages while preserving system messages.

    Args:
        system_messages: System messages to preserve at the beginning
        messages_to_summarize: Messages to be summarized
        preserved_messages: Recent messages to keep as-is
        provider: LLM provider to use for summarization
        model: Model to use for summarization
        max_summary_tokens: Maximum tokens for the summary
        **kwargs: Additional arguments for the LLM call

    Returns:
        Tuple of (new_messages, total_token_count)
    """
    if not messages_to_summarize:
        # Nothing to summarize, return system + preserved messages
        all_messages = system_messages + preserved_messages
        token_counter = TokenCounter()
        total_tokens = token_counter.count_tokens(all_messages, model, provider)
        return all_messages, total_tokens

    # Create a summary prompt
    summary_prompt = _create_summary_prompt(messages_to_summarize, max_summary_tokens)

    # Generate summary using the same provider/model
    from ..utils import chat_async

    summary_response = await chat_async(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,  # Lower temperature for more consistent summaries
        **kwargs,
    )

    # Create summary message
    summary_message = {
        "role": "user",  # Summary as user context
        "content": f"[Previous conversation summary]\n{summary_response.content}",
    }

    # Combine: system messages + summary + preserved messages
    new_messages = system_messages + [summary_message] + preserved_messages

    # Calculate new token count
    token_counter = TokenCounter()
    total_tokens = token_counter.count_tokens(new_messages, model, provider)

    return new_messages, total_tokens


def _create_summary_prompt(messages: List[Dict[str, Any]], max_tokens: int) -> str:
    """Create a prompt for summarizing conversation history."""

    # Format messages for summary
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle different content types
        if isinstance(content, dict):
            # Tool calls or structured content
            content = f"[Structured content: {type(content).__name__}]"
        elif isinstance(content, list):
            # Multiple content items
            content = f"[Multiple content items: {len(content)}]"

        formatted_messages.append(f"{role.upper()}: {content}")

    conversation_text = "\n\n".join(formatted_messages)

    prompt = f"""Please provide a concise summary of the following conversation. 
Focus on:
1. Key topics discussed
2. Important decisions or conclusions reached
3. Any unresolved questions or ongoing tasks
4. Critical context that should be preserved

Keep the summary under {max_tokens // 4} words (approximately {max_tokens} tokens).

Conversation:
{conversation_text}

Summary:"""

    return prompt


async def smart_compactify(
    memory_manager, provider: str, model: str, **kwargs
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Intelligently compactify messages using the memory manager.

    This is a convenience function that works with a MemoryManager instance.

    Args:
        memory_manager: MemoryManager instance
        provider: LLM provider
        model: Model name
        **kwargs: Additional LLM arguments

    Returns:
        Tuple of (new_messages, total_token_count)
    """
    if not memory_manager.should_compactify():
        return (
            memory_manager.get_current_messages(),
            memory_manager.history.total_tokens,
        )

    # Get messages to summarize and preserve
    system_messages, messages_to_summarize, preserved_messages = (
        memory_manager.get_messages_for_compactification()
    )

    # Perform compactification
    new_messages, new_token_count = await compactify_messages(
        system_messages=system_messages,
        messages_to_summarize=messages_to_summarize,
        preserved_messages=preserved_messages,
        provider=provider,
        model=model,
        max_summary_tokens=memory_manager.summary_max_tokens,
        **kwargs,
    )

    # Update memory manager
    if new_messages and len(new_messages) > len(system_messages):
        # Find the summary message (first non-system message in the result)
        summary_idx = len(system_messages)
        summary_message = new_messages[summary_idx]
        memory_manager.update_after_compactification(
            system_messages=system_messages,
            summary_message=summary_message,
            preserved_messages=preserved_messages,
            new_token_count=new_token_count,
        )

    return new_messages, new_token_count
