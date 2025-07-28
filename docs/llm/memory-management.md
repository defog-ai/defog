## Memory Management

Automatically manage long conversations by intelligently summarizing older messages.

### Basic Memory Usage

```python
from defog.llm import chat_async_with_memory, create_memory_manager, MemoryConfig

# Create a memory manager
memory_manager = create_memory_manager(
    token_threshold=50000,      # Compactify when reaching 50k tokens
    preserve_last_n_messages=10, # Keep last 10 messages intact
    summary_max_tokens=2000,    # Max tokens for summary
    enabled=True
)

# System messages are automatically preserved
response1 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful Python tutor."},
        {"role": "user", "content": "Tell me about Python"}
    ],
    memory_manager=memory_manager
)

# Continue conversation - memory is automatically managed
response2 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "What about its use in data science?"}],
    memory_manager=memory_manager
)

# Check memory stats
stats = memory_manager.get_stats()
print(f"Total messages: {len(memory_manager.get_current_messages())}")
print(f"Compactifications: {stats['compactification_count']}")
```

### Memory Configuration Options

```python
# Use memory configuration without explicit manager
response = await chat_async_with_memory(
    provider="anthropic",
    model="claude-3-5-sonnet",
    messages=messages,
    memory_config=MemoryConfig(
        enabled=True,
        token_threshold=100000,       # 100k tokens before compactification
        preserve_last_n_messages=10,
        summary_max_tokens=4000,
        preserve_system_messages=True, # Always preserve system messages
        preserve_tool_calls=True,      # Keep function calls in memory
        compression_ratio=0.3          # Target 30% compression
    )
)
```

### Advanced Memory Management

```python
from defog.llm.memory import EnhancedMemoryManager

# Enhanced memory with cross-agent support
memory_manager = EnhancedMemoryManager(
    token_threshold=50000,
    preserve_last_n_messages=10,
    summary_max_tokens=2000,
    
    # Advanced options
    enable_cross_agent_memory=True,    # Share memory across agents
    memory_tags=["research", "analysis"], # Tag memories
    preserve_images=False,             # Exclude images from memory
    
    # Custom summarization
    summary_model="gpt-4o-mini",       # Use cheaper model for summaries
    summary_provider="openai",
    summary_instructions="Focus on key decisions and findings"
)

# Agent 1 with memory
response1 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Research topic X"}],
    memory_manager=memory_manager,
    agent_id="researcher"
)

# Agent 2 can access Agent 1's memory
response2 = await chat_async_with_memory(
    provider="anthropic",
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Summarize the research findings"}],
    memory_manager=memory_manager,
    agent_id="summarizer",
    include_cross_agent_memory=True
)
```