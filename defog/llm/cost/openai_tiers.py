"""OpenAI Flex and Batch prices in USD per 1K tokens.

Source: https://developers.openai.com/api/docs/pricing (2026-09-15).
Each entry contains (input, cached input, output) rates for short context
and, where published, for requests above 272,000 total input tokens.
None means no separate cached-input discount or no long-context rate.
Standard prices remain in models.py.
"""

OPENAI_TIER_COSTS = {
    "batch": {
        "gpt-5.6-sol": ((0.002, 0.0002, 0.01), (0.004, 0.0004, 0.015)),
        "gpt-5.6-terra": ((0.001, 0.0001, 0.006), (0.002, 0.0002, 0.009)),
        "gpt-5.6-luna": ((0.0001, 0.00001, 0.0006), (0.0002, 0.00002, 0.0009)),
        "gpt-5.5": ((0.0025, 0.00025, 0.015), (0.005, 0.0005, 0.0225)),
        "gpt-5.5-pro": ((0.015, None, 0.09), None),
        "gpt-5.4": ((0.00125, 0.00013, 0.0075), (0.0025, 0.00025, 0.01125)),
        "gpt-5.4-mini": ((0.000375, 0.0000375, 0.00225), None),
        "gpt-5.4-nano": ((0.0001, 0.00001, 0.000625), None),
        "gpt-5.4-pro": ((0.015, None, 0.09), (0.03, None, 0.135)),
        "gpt-5": ((0.000625, 0.0000625, 0.005), None),
        "gpt-5-mini": ((0.000125, 0.0000125, 0.001), None),
        "gpt-5-nano": ((0.000025, 0.0000025, 0.0002), None),
        "gpt-4.1": ((0.001, None, 0.004), None),
        "gpt-4.1-mini": ((0.0002, None, 0.0008), None),
        "gpt-4.1-nano": ((0.00005, None, 0.0002), None),
        "gpt-4o": ((0.00125, None, 0.005), None),
        "gpt-4o-2024-05-13": ((0.0025, None, 0.0075), None),
        "gpt-4o-mini": ((0.000075, None, 0.0003), None),
        "o3": ((0.001, None, 0.004), None),
        "o4-mini": ((0.00055, None, 0.0022), None),
        "o3-mini": ((0.00055, None, 0.0022), None),
    },
    "flex": {
        "gpt-5.6-sol": ((0.002, 0.0002, 0.01), (0.004, 0.0004, 0.015)),
        "gpt-5.6-terra": ((0.001, 0.0001, 0.006), (0.002, 0.0002, 0.009)),
        "gpt-5.6-luna": ((0.0001, 0.00001, 0.0006), (0.0002, 0.00002, 0.0009)),
        "gpt-5.5": ((0.0025, 0.00025, 0.015), (0.005, 0.0005, 0.0225)),
        "gpt-5.5-pro": ((0.015, None, 0.09), None),
        "gpt-5.4": ((0.00125, 0.00013, 0.0075), (0.0025, 0.00025, 0.01125)),
        "gpt-5.4-mini": ((0.000375, 0.0000375, 0.00225), None),
        "gpt-5.4-nano": ((0.0001, 0.00001, 0.000625), None),
        "gpt-5.4-pro": ((0.015, None, 0.09), (0.03, None, 0.135)),
        "gpt-5": ((0.000625, 0.0000625, 0.005), None),
        "gpt-5-mini": ((0.000125, 0.0000125, 0.001), None),
        "gpt-5-nano": ((0.000025, 0.0000025, 0.0002), None),
        "o3": ((0.001, 0.00025, 0.004), None),
        "o4-mini": ((0.00055, 0.000138, 0.0022), None),
    },
}

# Public alias used by the provider.
for _rates in OPENAI_TIER_COSTS.values():
    _rates["gpt-5.6"] = _rates["gpt-5.6-sol"]
