from typing import Optional
from .models import MODEL_COSTS


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
        # Find exact match first
        if model in MODEL_COSTS:
            model_name = model
        else:
            # Attempt partial matches if no exact match
            potential_model_names = []
            for mname in MODEL_COSTS.keys():
                if mname in model:
                    potential_model_names.append(mname)

            if not potential_model_names:
                return None

            # Use the longest match
            model_name = max(potential_model_names, key=len)

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
        if model in MODEL_COSTS:
            return True

        # Check for partial matches
        return any(mname in model for mname in MODEL_COSTS.keys())
