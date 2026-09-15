import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import MODEL_COSTS
from .openai_tiers import (
    LONG_CONTEXT_TOKENS,
    OPENAI_LONG_CONTEXT_COSTS,
    OPENAI_TIER_COSTS,
)

# Size-tier suffixes that should route to same-tier pricing entries.
# A model ending in "-mini" must not fall back to a base-tier price,
# since mini/nano tiers are dramatically cheaper (5-25x).
_SIZE_SUFFIXES = ("mini", "nano", "flash", "lite", "pro")

_DEEPSEEK_PEAK_WINDOWS_UTC = ((1, 4), (6, 10))
_BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def _is_deepseek_peak_time(calculation_time: Optional[datetime] = None) -> bool:
    """Return whether a timestamp falls in a DeepSeek peak billing window.

    DeepSeek's peak windows are 01:00-04:00 and 06:00-10:00 UTC, Monday
    through Friday in Beijing time. The end of each window is exclusive.
    Naive datetimes are interpreted as UTC.
    """
    if calculation_time is None:
        calculation_time = datetime.now(timezone.utc)
    elif calculation_time.tzinfo is None:
        calculation_time = calculation_time.replace(tzinfo=timezone.utc)
    else:
        calculation_time = calculation_time.astimezone(timezone.utc)

    if calculation_time.astimezone(_BEIJING_TIMEZONE).isoweekday() > 5:
        return False

    return any(
        start_hour <= calculation_time.hour < end_hour
        for start_hour, end_hour in _DEEPSEEK_PEAK_WINDOWS_UTC
    )


def _split_size_suffix(name: str) -> tuple[str, str]:
    """Split size-tier model ids while allowing optional snapshot suffixes.

    Examples:
      - 'gpt-5.4-mini' -> ('gpt-5.4', 'mini')
      - 'gpt-5.5-pro-2026-04-23' -> ('gpt-5.5', 'pro')
      - 'gpt-5' -> ('gpt-5', '')
    """
    parts = name.split("-")
    for index in range(len(parts) - 1, 0, -1):
        if parts[index] in _SIZE_SUFFIXES:
            return "-".join(parts[:index]), parts[index]
    return name, ""


def _find_match(model: str) -> Optional[str]:
    """Resolve a model name to a MODEL_COSTS key.

    Preference order:
      1. Exact match.
      2. A candidate with the SAME size suffix (mini/nano/flash/lite/pro)
         whose family portion is a prefix of the model's family, separated
         by a '.' or '-'. Picks the longest such family prefix.
      3. Fall back to the loose "key-is-a-substring-of-model" match as a
         last resort (preserves prior behavior for unusual names).
    """
    if model in MODEL_COSTS:
        return model

    model_family, model_size = _split_size_suffix(model)

    best_key = None
    best_family_len = 0
    for candidate in MODEL_COSTS:
        cand_family, cand_size = _split_size_suffix(candidate)
        if cand_size != model_size:
            continue
        if cand_family == model_family:
            return candidate
        # Candidate family must be a prefix of the model family with a clean
        # version delimiter immediately after.
        if (
            model_family.startswith(cand_family)
            and len(model_family) > len(cand_family)
            and model_family[len(cand_family)] in (".", "-")
        ):
            if len(cand_family) > best_family_len:
                best_key = candidate
                best_family_len = len(cand_family)

    if best_key is not None:
        return best_key

    # Fall back to the original loose substring match for names that don't
    # follow the family-size pattern (dated suffixes, aliases, etc.).
    substring_matches = [k for k in MODEL_COSTS if k in model]
    if substring_matches:
        return max(substring_matches, key=len)
    return None


def _openai_price_name(model: str, table: dict) -> Optional[str]:
    """Return the key of ``table`` that prices ``model``, or None.

    Only exact names and their dated snapshots (``-YYYY-MM-DD``) qualify.
    """
    if model in table:
        return model
    snapshot_free = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    return snapshot_free if snapshot_free in table else None


def _cost_from_rates(
    rates: tuple,
    input_tokens: int,
    cached_input_tokens: Optional[int],
    output_tokens: int,
) -> float:
    """Price one request in cents from (input, cached input, output) USD per 1K."""
    input_rate, cached_rate, output_rate = rates
    # Models without a cache discount still bill cached tokens as input;
    # they must not disappear from the total.
    if cached_rate is None:
        cached_rate = input_rate
    return (
        input_tokens * input_rate
        + (cached_input_tokens or 0) * cached_rate
        + output_tokens * output_rate
    ) / 10


class CostCalculator:
    """Handles cost calculation for LLM usage."""

    @staticmethod
    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: Optional[int] = None,
        cache_creation_input_tokens: Optional[int] = None,
        calculation_time: Optional[datetime] = None,
        *,
        service_tier: str | None = None,
        batch: bool = False,
    ) -> Optional[float]:
        """
        Calculate cost in cents for the given token usage.

        ``calculation_time`` selects the UTC peak or off-peak rate for models
        with time-based pricing. It defaults to the current time. Naive
        datetimes are treated as UTC.

        ``input_tokens`` excludes ``cached_input_tokens``. For OpenAI,
        ``service_tier="flex"`` selects Flex prices; ``batch=True`` selects
        Batch API prices and takes precedence without stacking discounts.
        Other tiers and providers keep their existing prices. Unlisted
        OpenAI Flex/Batch prices return None instead of guessing a discount.

        Token counts describe one provider request. OpenAI requests whose
        input plus cached input exceeds 272,000 tokens use the published
        long-context rates on every tier; price separate requests
        separately rather than pricing their summed token counts.

        Returns:
            Cost in cents, or None if model pricing is not available
        """
        model_name = _find_match(model)
        if model_name is None:
            return None

        costs = MODEL_COSTS[model_name]

        is_openai = model_name.startswith(("gpt-", "chatgpt-", "o3", "o4"))
        total_input = input_tokens + (cached_input_tokens or 0)

        if (batch or service_tier == "flex") and is_openai:
            tier_costs = OPENAI_TIER_COSTS["batch" if batch else "flex"]
            # Only known names and their dated snapshots qualify. Do not
            # give a different model a discount through a loose name match.
            tier_model = _openai_price_name(model, tier_costs)
            if tier_model is None:
                return None
            rates, long_rates = tier_costs[tier_model]
            if long_rates and total_input > LONG_CONTEXT_TOKENS:
                rates = long_rates
            return _cost_from_rates(
                rates, input_tokens, cached_input_tokens, output_tokens
            )

        if is_openai and total_input > LONG_CONTEXT_TOKENS:
            # Standard requests above the long-context threshold use the
            # published long-context rates. Models with no such price keep
            # their standard price.
            long_model = _openai_price_name(model, OPENAI_LONG_CONTEXT_COSTS)
            if long_model is not None:
                return _cost_from_rates(
                    OPENAI_LONG_CONTEXT_COSTS[long_model],
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                )

        rate_prefix = ""
        if "off_peak_input_cost_per1k" in costs and not _is_deepseek_peak_time(
            calculation_time
        ):
            rate_prefix = "off_peak_"

        # Calculate base cost
        cost_in_cents = (
            input_tokens / 1000 * costs[f"{rate_prefix}input_cost_per1k"]
            + output_tokens / 1000 * costs[f"{rate_prefix}output_cost_per1k"]
        ) * 100

        # Add cached input cost if available. An OpenAI model with no cached
        # input discount (gpt-5.5-pro, gpt-5.4-pro) bills cached input at
        # the regular input rate.
        cached_rate_key = f"{rate_prefix}cached_input_cost_per1k"
        if cached_input_tokens and cached_rate_key in costs:
            cost_in_cents += (cached_input_tokens / 1000 * costs[cached_rate_key]) * 100
        elif cached_input_tokens and is_openai:
            cost_in_cents += (
                cached_input_tokens / 1000 * costs[f"{rate_prefix}input_cost_per1k"]
            ) * 100

        # Add cache creation input cost if available
        if cache_creation_input_tokens and "cache_creation_input_cost_per1k" in costs:
            cost_in_cents += (
                cache_creation_input_tokens
                / 1000
                * costs["cache_creation_input_cost_per1k"]
            ) * 100

        return cost_in_cents

    @staticmethod
    def is_model_supported(model: str) -> bool:
        """Check if cost calculation is supported for the given model."""
        return _find_match(model) is not None
