DEFAULT_TIMEOUT = 600  # seconds
MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.0

# Provider-specific constants
OPENAI_BASE_URL = "https://api.openai.com/v1/"

# Model families that require special handling
O_MODELS = ["o1-mini", "o1-preview", "o1", "o3-mini", "o3", "o4-mini"]

MODELS_WITHOUT_TEMPERATURE = [model for model in O_MODELS]

MODELS_WITHOUT_RESPONSE_FORMAT = [
    "o1-mini",
    "o1-preview",
]

MODELS_WITHOUT_TOOLS = ["o1-mini", "o1-preview"]

MODELS_WITH_PARALLEL_TOOL_CALLS = ["gpt-4o", "gpt-4o-mini"]

MODELS_WITH_PREDICTION = ["gpt-4o", "gpt-4o-mini"]
