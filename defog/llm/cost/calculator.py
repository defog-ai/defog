from typing import Optional
from .models import MODEL_COSTS


# Size-tier suffixes that should route to same-tier pricing entries.
# A model ending in "-mini" must not fall back to a base-tier price,
# since mini/nano tiers are dramatically cheaper (5-25x).
_SIZE_SUFFIXES = ("mini", "nano", "flash", "lite", "pro")


def _split_size_suffix(name: str) -> tuple[str, str]:
    """Split 'gpt-5.4-mini' -> ('gpt-5.4', 'mini'); 'gpt-5' -> ('gpt-5', '')."""
    for s in _SIZE_SUFFIXES:
        token = f"-{s}"
        if name.endswith(token):
            return name[: -len(token)], s
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


class CostCalculator:
    """Handles cost calculation for LLM usage."""

    @staticmethod
    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: Optional[int] = None,
        cache_creation_input_tokens: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate cost in cents for the given token usage.

        Returns:
            Cost in cents, or None if model pricing is not available
        """
        model_name = _find_match(model)
        if model_name is None:
            return None

        costs = MODEL_COSTS[model_name]

        # Calculate base cost
        cost_in_cents = (
            input_tokens / 1000 * costs["input_cost_per1k"]
            + output_tokens / 1000 * costs["output_cost_per1k"]
        ) * 100

        # Add cached input cost if available
        if cached_input_tokens and "cached_input_cost_per1k" in costs:
            cost_in_cents += (
                cached_input_tokens / 1000 * costs["cached_input_cost_per1k"]
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
