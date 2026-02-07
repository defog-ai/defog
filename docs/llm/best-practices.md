## Best Practices

### 1. Provider Selection

- **OpenAI**: Best for function calling, structured output, general purpose
- **Anthropic**: Best for long context, complex reasoning, document analysis
- **Gemini**: Best for cost-effectiveness, multimodal tasks, large documents
- **Together**: Best for open-source models, specific use cases

### 2. Error Handling

```python
from defog.llm.utils import chat_async
from defog.llm.exceptions import LLMError, RateLimitError, ContextLengthError

try:
    response = await chat_async(...)
except RateLimitError:
    # Wait and retry
    await asyncio.sleep(60)
    response = await chat_async(...)
except ContextLengthError:
    # Chunk content or reduce message history
    response = await chat_async(...)
except LLMError as e:
    print(f"LLM error: {e}")
```

### 3. Performance Optimization

- Use appropriate models for tasks (don't use GPT-4 for simple tasks)
- Enable caching for repeated operations (especially PDFs)
- Use structured output for consistent parsing
- Batch operations when possible
- Manage conversation history to stay within context limits

### 4. Cost Optimization

- Use cheaper models for summarization (e.g., gpt-4o-mini)
- Enable input caching for repeated content
- Set max_tokens appropriately
- Use token counting before sending requests
- Monitor costs with tracking utilities

### 5. Security Considerations

- Validate all tool inputs with Pydantic models
- Restrict code interpreter capabilities appropriately
- Use domain filtering for web searches
- Implement rate limiting for production use
- Sanitize any user-provided content