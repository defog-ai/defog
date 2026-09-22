# DeepSeek-V4.1-Flash rates in USD per 1K tokens, including legacy Flash
# names now served by the same model. Peak hours are 01:00-04:00 and
# 06:00-10:00 UTC on weekdays; all other times use the off-peak rates.
# Source (2026-09-10): https://api-docs.deepseek.com/quick_start/pricing/
_DEEPSEEK_FLASH_COSTS = {
    "input_cost_per1k": 0.0003,
    "cached_input_cost_per1k": 0.000006,
    "output_cost_per1k": 0.0012,
    "off_peak_input_cost_per1k": 0.00015,
    "off_peak_cached_input_cost_per1k": 0.000003,
    "off_peak_output_cost_per1k": 0.0006,
}


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
    # Source (2026-09-23): https://developers.openai.com/api/docs/pricing
    "gpt-6-astra": {
        "input_cost_per1k": 0.01,
        "cached_input_cost_per1k": 0.001,
        "output_cost_per1k": 0.05,
    },
    "gpt-6-sol": {
        "input_cost_per1k": 0.002,
        "cached_input_cost_per1k": 0.0002,
        "output_cost_per1k": 0.01,
    },
    "gpt-6-luna": {
        "input_cost_per1k": 0.0001,
        "cached_input_cost_per1k": 0.00001,
        "output_cost_per1k": 0.0005,
    },
    # gpt-5.6 is an alias for gpt-5.6-sol.
    # Source (2026-09-15): https://developers.openai.com/api/docs/pricing
    "gpt-5.6": {
        "input_cost_per1k": 0.004,
        "cached_input_cost_per1k": 0.0004,
        "output_cost_per1k": 0.02,
    },
    "gpt-5.6-sol": {
        "input_cost_per1k": 0.004,
        "cached_input_cost_per1k": 0.0004,
        "output_cost_per1k": 0.02,
    },
    "gpt-5.6-terra": {
        "input_cost_per1k": 0.002,
        "cached_input_cost_per1k": 0.0002,
        "output_cost_per1k": 0.012,
    },
    "gpt-5.6-luna": {
        "input_cost_per1k": 0.0002,
        "cached_input_cost_per1k": 0.00002,
        "output_cost_per1k": 0.0012,
    },
    "gpt-5.5": {
        "input_cost_per1k": 0.005,
        "cached_input_cost_per1k": 0.0005,
        "output_cost_per1k": 0.03,
    },
    "gpt-5.5-pro": {
        "input_cost_per1k": 0.03,
        "output_cost_per1k": 0.18,
    },
    "gpt-5.4": {
        "input_cost_per1k": 0.0025,
        "cached_input_cost_per1k": 0.00025,
        "output_cost_per1k": 0.015,
    },
    "gpt-5.4-mini": {
        "input_cost_per1k": 0.00075,
        "cached_input_cost_per1k": 0.000075,
        "output_cost_per1k": 0.0045,
    },
    "gpt-5.4-nano": {
        "input_cost_per1k": 0.0002,
        "cached_input_cost_per1k": 0.00002,
        "output_cost_per1k": 0.00125,
    },
    "gpt-5.4-pro": {
        "input_cost_per1k": 0.03,
        "output_cost_per1k": 0.18,
    },
    "gpt-5.3-chat-latest": {
        "input_cost_per1k": 0.00175,
        "cached_input_cost_per1k": 0.000175,
        "output_cost_per1k": 0.014,
    },
    "gpt-5.3-codex": {
        "input_cost_per1k": 0.00175,
        "cached_input_cost_per1k": 0.000175,
        "output_cost_per1k": 0.014,
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
    # Speech-to-text models (Transcription API). Input is billed as audio
    # tokens; OpenAI's rule-of-thumb estimates are ~$0.006/min for
    # gpt-4o-transcribe(-diarize) and ~$0.003/min for the mini tier.
    "gpt-4o-transcribe": {
        "input_cost_per1k": 0.0025,
        "output_cost_per1k": 0.01,
    },
    "gpt-4o-mini-transcribe": {
        "input_cost_per1k": 0.00125,
        "output_cost_per1k": 0.005,
    },
    "gpt-4o-transcribe-diarize": {
        "input_cost_per1k": 0.0025,
        "output_cost_per1k": 0.01,
    },
    # Cache reads on Fable 5.1 are 2.5% of the input price, not 10%.
    # Source (2026-09-23): https://platform.claude.com/docs/en/about-claude/pricing
    "claude-fable-5-1": {
        "input_cost_per1k": 0.010,
        "cached_input_cost_per1k": 0.00025,
        "cache_creation_input_cost_per1k": 0.0125,
        "output_cost_per1k": 0.050,
    },
    "claude-fable-5": {
        "input_cost_per1k": 0.010,
        "cached_input_cost_per1k": 0.001,
        "cache_creation_input_cost_per1k": 0.0125,
        "output_cost_per1k": 0.050,
    },
    "claude-3-5-sonnet": {
        "input_cost_per1k": 0.003,
        "output_cost_per1k": 0.015,
        "cached_input_cost_per1k": 0.0003,
        "cache_creation_input_cost_per1k": 0.00375,
    },
    "claude-sonnet-4": {
        "input_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.0003,
        "cache_creation_input_cost_per1k": 0.00375,
        "output_cost_per1k": 0.015,
    },
    "claude-sonnet-4-5": {
        "input_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.0003,
        "cache_creation_input_cost_per1k": 0.00375,
        "output_cost_per1k": 0.015,
    },
    "claude-sonnet-4-6": {
        "input_cost_per1k": 0.003,
        "cached_input_cost_per1k": 0.0003,
        "cache_creation_input_cost_per1k": 0.00375,
        "output_cost_per1k": 0.015,
    },
    # Introductory pricing through 2026-08-31; becomes $3/$15 per MTok
    # (cached $0.30/MTok, 5m cache write $3.75/MTok) on 2026-09-01.
    "claude-sonnet-5": {
        "input_cost_per1k": 0.002,
        "cached_input_cost_per1k": 0.0002,
        "cache_creation_input_cost_per1k": 0.0025,
        "output_cost_per1k": 0.01,
    },
    "claude-opus-4-1": {
        "input_cost_per1k": 0.015,
        "cached_input_cost_per1k": 0.0015,
        "cache_creation_input_cost_per1k": 0.01875,
        "output_cost_per1k": 0.075,
    },
    "claude-opus-4-5": {
        "input_cost_per1k": 0.005,
        "cached_input_cost_per1k": 0.0005,
        "cache_creation_input_cost_per1k": 0.00625,
        "output_cost_per1k": 0.025,
    },
    "claude-opus-4-6": {
        "input_cost_per1k": 0.005,
        "cached_input_cost_per1k": 0.0005,
        "cache_creation_input_cost_per1k": 0.00625,
        "output_cost_per1k": 0.025,
    },
    "claude-opus-4-7": {
        "input_cost_per1k": 0.005,
        "cached_input_cost_per1k": 0.0005,
        "cache_creation_input_cost_per1k": 0.00625,
        "output_cost_per1k": 0.025,
    },
    "claude-opus-4-8": {
        "input_cost_per1k": 0.005,
        "cached_input_cost_per1k": 0.0005,
        "cache_creation_input_cost_per1k": 0.00625,
        "output_cost_per1k": 0.025,
    },
    # Launched 2026-07-24 at unchanged Opus-generation pricing.
    "claude-opus-5": {
        "input_cost_per1k": 0.005,
        "cached_input_cost_per1k": 0.0005,
        "cache_creation_input_cost_per1k": 0.00625,
        "output_cost_per1k": 0.025,
    },
    # Released 2026-09-22. Cache reads are 5% of the input price.
    # Source: https://platform.claude.com/docs/en/models/opus-5-5/overview
    "claude-opus-5-5": {
        "input_cost_per1k": 0.004,
        "cached_input_cost_per1k": 0.0002,
        "cache_creation_input_cost_per1k": 0.005,
        "output_cost_per1k": 0.02,
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
    "gemini-3.5-flash": {
        "input_cost_per1k": 0.0015,
        "output_cost_per1k": 0.009,
        "cached_input_cost_per1k": 0.00015,
    },
    "deepseek-v4-pro": {
        # Peak: 01:00-04:00 and 06:00-10:00 UTC. All other hours use the
        # off-peak rates below. Effective 2026-08-16; used here immediately
        # for forward-looking cost estimates.
        "input_cost_per1k": 0.00132,
        "cached_input_cost_per1k": 0.000044,
        "output_cost_per1k": 0.00396,
        "off_peak_input_cost_per1k": 0.00066,
        "off_peak_cached_input_cost_per1k": 0.000022,
        "off_peak_output_cost_per1k": 0.00198,
    },
    "deepseek-flash": _DEEPSEEK_FLASH_COSTS.copy(),
    "deepseek-v4.1-flash": _DEEPSEEK_FLASH_COSTS.copy(),
    "deepseek-v4-flash": _DEEPSEEK_FLASH_COSTS.copy(),
    "deepseek-v4-flash-vision-exp": _DEEPSEEK_FLASH_COSTS.copy(),
    "glm-5.2": {
        "input_cost_per1k": 0.0014,
        "cached_input_cost_per1k": 0.00026,
        "output_cost_per1k": 0.0044,
    },
    "glm-5.1": {
        "input_cost_per1k": 0.0014,
        "cached_input_cost_per1k": 0.00026,
        "output_cost_per1k": 0.0044,
    },
}
