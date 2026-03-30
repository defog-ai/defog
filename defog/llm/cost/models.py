MODEL_COSTS = {
    "chatgpt-4o": {"input_cost_per1k": 0.0025, "output_cost_per1k": 0.01},
    "gpt-4o": {
        "input_cost_per1k": 0.0025,
        "cached_input_cost_per1k": 0.00125,
        "output_cost_per1k": 0.01,
    },
    "gpt-4o-mini": {
        "input_cost_per1k": 0.00015,
        "cached_input_cost_per1k": 0.000075,
        "output_cost_per1k": 0.0006,
    },
    "gpt-4.1": {
        "input_cost_per1k": 0.002,
        "cached_input_cost_per1k": 0.0005,
        "output_cost_per1k": 0.008,
    },
    "gpt-4.1-mini": {
        "input_cost_per1k": 0.0004,
        "cached_input_cost_per1k": 0.0001,
        "output_cost_per1k": 0.0016,
    },
    "gpt-4.1-nano": {
        "input_cost_per1k": 0.0001,
        "cached_input_cost_per1k": 0.000025,
        "output_cost_per1k": 0.0004,
    },
    "gpt-5": {
        "input_cost_per1k": 0.00125,
        "cached_input_cost_per1k": 0.000125,
        "output_cost_per1k": 0.01,
    },
    "gpt-5-mini": {
        "input_cost_per1k": 0.00025,
        "cached_input_cost_per1k": 0.000025,
        "output_cost_per1k": 0.002,
    },
    "gpt-5-nano": {
        "input_cost_per1k": 0.00005,
        "cached_input_cost_per1k": 0.000005,
        "output_cost_per1k": 0.0004,
    },
    "o3-mini": {
        "input_cost_per1k": 0.0011,
        "cached_input_cost_per1k": 0.00055,
        "output_cost_per1k": 0.0044,
    },
    "o3": {
        "input_cost_per1k": 0.01,
        "cached_input_cost_per1k": 0.0025,
        "output_cost_per1k": 0.04,
    },
    "o4-mini": {
        "input_cost_per1k": 0.0011,
        "cached_input_cost_per1k": 0.000275,
        "output_cost_per1k": 0.0044,
    },
    "claude-3-5-sonnet": {
        "input_cost_per1k": 0.003,
        "output_cost_per1k": 0.015,
        "cached_input_cost_per1k": 0.0003,
        "cache_creation_input_cost_per1k": 0.00375,
    },
    "claude-sonnet-4": {
        "input_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.00075,
        "cache_creation_input_cost_per1k": 0.00375,
        "output_cost_per1k": 0.015,
    },
    "claude-sonnet-4-5": {
        "input_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.00075,
        "cache_creation_input_cost_per1k": 0.00375,
        "output_cost_per1k": 0.015,
    },
    "claude-sonnet-4-6": {
        "input_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.00075,
        "cache_creation_input_cost_per1k": 0.00375,
        "output_cost_per1k": 0.015,
    },
    "claude-opus-4-1": {
        "input_cost_per1k": 0.015,
        "cached_input_cost_per1k": 0.00375,
        "cache_creation_input_cost_per1k": 0.01875,
        "output_cost_per1k": 0.075,
    },
    "claude-opus-4-5": {
        "input_cost_per1k": 0.005,
        "cached_input_cost_per1k": 0.00125,
        "cache_creation_input_cost_per1k": 0.00625,
        "output_cost_per1k": 0.025,
    },
    "claude-opus-4-6": {
        "input_cost_per1k": 0.005,
        "cached_input_cost_per1k": 0.00125,
        "cache_creation_input_cost_per1k": 0.00625,
        "output_cost_per1k": 0.025,
    },
    "claude-haiku-4-5": {
        "input_cost_per1k": 0.001,
        "cached_input_cost_per1k": 0.0001,
        "cache_creation_input_cost_per1k": 0.00125,
        "output_cost_per1k": 0.005,
    },
    "claude-3-5-haiku": {
        "input_cost_per1k": 0.00025,
        "output_cost_per1k": 0.00125,
        "cached_input_cost_per1k": 0.000025,
        "cache_creation_input_cost_per1k": 0.0003125,
    },
    "claude-3-opus": {
        "input_cost_per1k": 0.015,
        "output_cost_per1k": 0.075,
        "cached_input_cost_per1k": 0.0015,
        "cache_creation_input_cost_per1k": 0.01875,
    },
    "claude-3-sonnet": {
        "input_cost_per1k": 0.003,
        "output_cost_per1k": 0.015,
        "cached_input_cost_per1k": 0.0003,
        "cache_creation_input_cost_per1k": 0.00375,
    },
    "claude-3-haiku": {
        "input_cost_per1k": 0.00025,
        "output_cost_per1k": 0.00125,
        "cached_input_cost_per1k": 0.000025,
        "cache_creation_input_cost_per1k": 0.0003125,
    },
    "gemini-2.0-flash": {
        "input_cost_per1k": 0.00010,
        "output_cost_per1k": 0.0004,
    },
    "gemini-2.5-flash": {
        "input_cost_per1k": 0.0003,
        "output_cost_per1k": 0.0025,
    },
    "gemini-2.5-pro": {
        "input_cost_per1k": 0.00125,
        "output_cost_per1k": 0.01,
    },
    "gemini-3-flash": {
        "input_cost_per1k": 0.0005,
        "output_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.00005,
    },
    "gemini-3-pro": {
        "input_cost_per1k": 0.002,
        "output_cost_per1k": 0.012,
        "cached_input_cost_per1k": 0.0002,
    },
    "gemini-3.1-flash-lite": {
        "input_cost_per1k": 0.00025,
        "output_cost_per1k": 0.0015,
        "cached_input_cost_per1k": 0.000025,
    },
    "gemini-3.1-pro": {
        "input_cost_per1k": 0.002,
        "output_cost_per1k": 0.012,
        "cached_input_cost_per1k": 0.0002,
    },
}
