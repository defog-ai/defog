# Multimodal Support

The library supports image inputs across all major providers with automatic format conversion.

## Image Input Examples

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider
import base64

# Using base64-encoded images
with open("image.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }
]

# Using image URLs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
    }
]

# Works with all providers - automatic format conversion
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,  # or OPENAI, GEMINI
    model="claude-sonnet-4-20250514",
    messages=messages
)
```

## Multiple Images

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two charts"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{chart1_base64}"}
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{chart2_base64}"}
            }
        ]
    }
]
```