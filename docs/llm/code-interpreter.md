## Code Interpreter

Execute Python code in sandboxed environments with AI assistance.

### Basic Code Execution

```python
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.llm_providers import LLMProvider

result = await code_interpreter_tool(
    question="Analyze this CSV data and create a visualization",
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    csv_string="name,age,score\nAlice,25,95\nBob,30,87\nCarol,28,92",
    instructions="Create a bar chart showing scores by name"
)

print(result["code"])    # Generated Python code
print(result["output"])  # Execution results
# Images are returned as base64 if generated
```

### Advanced Code Interpreter Options

```python
result = await code_interpreter_tool(
    question="Perform statistical analysis",
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    csv_string=data,
    
    # Advanced options
    allowed_libraries=["pandas", "matplotlib", "seaborn", "numpy", "scipy"],
    memory_limit_mb=512,              # Memory limit
    timeout_seconds=30,               # Execution timeout
    
    # Custom environment setup
    pre_code="""
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    """,
    
    # Output preferences
    return_images=True,               # Return generated images
    image_format="png",               # png, svg, jpg
    figure_dpi=150,                   # Image quality
    
    # Security
    allow_file_access=False,          # Restrict file system access
    allow_network_access=False        # Restrict network access
)

# Access results
if result.get("images"):
    for idx, img_base64 in enumerate(result["images"]):
        with open(f"output_{idx}.png", "wb") as f:
            f.write(base64.b64decode(img_base64))
```