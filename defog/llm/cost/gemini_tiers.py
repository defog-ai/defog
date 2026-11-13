"""Gemini Batch, Flex and Priority prices in USD per 1K tokens.

Source (2026-11-13): https://ai.google.dev/gemini-api/docs/pricing
Text, image and video input rates; audio input rates are not held here.
Standard prices remain in models.py.

Each entry has the same shape as a MODEL_COSTS entry and is keyed by the
MODEL_COSTS name of the model. It may hold a "long_prompt" price (prompts
of over 200,000 tokens) and a "price_change" with a start date. A model
that is absent has no published price for that tier: gemini-3.5-flash,
gemini-3-pro and gemini-2.0-flash are not on the pricing page.
"""

import re
from typing import Optional


def _rates(input_rate: float, cached_rate: float, output_rate: float) -> dict:
    return {
        "input_cost_per1k": input_rate,
        "cached_input_cost_per1k": cached_rate,
        "output_cost_per1k": output_rate,
    }


def _entry(
    rates: tuple,
    *,
    long_prompt: Optional[tuple] = None,
    change_from: Optional[str] = None,
    change_rates: Optional[tuple] = None,
) -> dict:
    entry = _rates(*rates)
    if long_prompt is not None:
        entry["long_prompt"] = {"above_prompt_tokens": 200_000, **_rates(*long_prompt)}
    if change_from is not None:
        entry["price_change"] = {"from": change_from, **_rates(*change_rates)}
    return entry


# Gemini 4 Carbon and Argon double on 2026-12-01, Gemini 3.6 and 3.8 Flash
# on 2027-01-01, on every tier.
_GEMINI_4_BATCH = dict(
    rates=(0.001, 0.00005, 0.005),
    change_from="2026-12-01",
    change_rates=(0.002, 0.0001, 0.01),
)
_GEMINI_4_PRIORITY = dict(
    rates=(0.0036, 0.00018, 0.018),
    change_from="2026-12-01",
    change_rates=(0.0072, 0.00036, 0.036),
)
_FLASH_3_6_BATCH = dict(
    rates=(0.000375, 0.0000375, 0.001875),
    change_from="2027-01-01",
    change_rates=(0.00075, 0.000075, 0.00375),
)
_FLASH_3_6_PRIORITY = dict(
    rates=(0.00135, 0.000135, 0.00675),
    change_from="2027-01-01",
    change_rates=(0.0027, 0.00027, 0.0135),
)


def _batch_costs() -> dict:
    # For gemini-3.1-pro, gemini-3-flash, gemini-2.5-pro and
    # gemini-2.5-flash the page lists the cached input price of Batch and
    # Flex as the same as Standard.
    return {
        "gemini-4-carbon": _entry(**_GEMINI_4_BATCH),
        "gemini-4-argon": _entry(**_GEMINI_4_BATCH),
        "gemini-4-hydrogen": _entry((0.00075, 0.0000375, 0.00375)),
        "gemini-3.8-flash": _entry(**_FLASH_3_6_BATCH),
        "gemini-3.6-flash": _entry(**_FLASH_3_6_BATCH),
        "gemini-3.5-flash-lite": _entry((0.00015, 0.000015, 0.00125)),
        "gemini-3.1-flash-lite": _entry((0.000125, 0.0000125, 0.00075)),
        "gemini-3.1-pro": _entry(
            (0.001, 0.0002, 0.006), long_prompt=(0.002, 0.0004, 0.009)
        ),
        "gemini-3-flash": _entry((0.00025, 0.00005, 0.0015)),
        "gemini-2.5-pro": _entry(
            (0.000625, 0.000125, 0.005), long_prompt=(0.00125, 0.00025, 0.0075)
        ),
        "gemini-2.5-flash": _entry((0.00015, 0.00003, 0.00125)),
    }


GEMINI_TIER_COSTS = {
    "batch": _batch_costs(),
    # The Flex table equals the Batch table, except for the cached input
    # price of gemini-3.5-flash-lite ($0.02 per 1M on Flex, $0.015 on Batch).
    "flex": {
        **_batch_costs(),
        "gemini-3.5-flash-lite": _entry((0.00015, 0.00002, 0.00125)),
    },
    "priority": {
        "gemini-4-carbon": _entry(**_GEMINI_4_PRIORITY),
        "gemini-4-argon": _entry(**_GEMINI_4_PRIORITY),
        "gemini-4-hydrogen": _entry((0.0027, 0.000135, 0.0135)),
        "gemini-3.8-flash": _entry(**_FLASH_3_6_PRIORITY),
        "gemini-3.6-flash": _entry(**_FLASH_3_6_PRIORITY),
        # The page lists $0.05 per 1M cached input, not 10% of $0.54.
        "gemini-3.5-flash-lite": _entry((0.00054, 0.00005, 0.0045)),
        "gemini-3.1-flash-lite": _entry((0.00045, 0.000045, 0.0027)),
        "gemini-3.1-pro": _entry(
            (0.0036, 0.00036, 0.0216), long_prompt=(0.0072, 0.00072, 0.0324)
        ),
        "gemini-3-flash": _entry((0.0009, 0.00009, 0.0054)),
        "gemini-2.5-pro": _entry(
            (0.00225, 0.000225, 0.018), long_prompt=(0.0045, 0.00045, 0.027)
        ),
        "gemini-2.5-flash": _entry((0.00054, 0.000054, 0.0045)),
    },
}

# A model id may add a preview, "latest" or numeric version suffix to the
# priced name ("gemini-3.1-pro-preview"). Any other suffix names another
# model ("gemini-2.5-flash-lite", "gemini-3-pro-image") and gets no tier
# price.
_VERSION_SUFFIX = re.compile(r"(-preview.*|-latest|-\d[\d-]*)?$")


def gemini_tier_entry(model: str, model_name: str, tier: str) -> Optional[dict]:
    """Return the ``tier`` price entry of ``model``, or None if unpublished.

    ``model_name`` is the MODEL_COSTS key that prices ``model``.
    """
    if not model.startswith(model_name):
        return None
    if not _VERSION_SUFFIX.fullmatch(model[len(model_name) :]):
        return None
    return GEMINI_TIER_COSTS[tier].get(model_name)
