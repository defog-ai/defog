import unittest
from defog.llm.utils import LLMResponse
from defog.llm.cost.calculator import CostCalculator


class TestLLMResponse(unittest.TestCase):
    def test_cost_calculator(self):
        # Test CostCalculator directly
        self.assertAlmostEqual(
            CostCalculator.calculate_cost("gpt-4o", 1000, 1000, 500),
            (0.0025 * 1 + 0.00125 * 0.5 + 0.01 * 1) * 100,
            places=10,
        )

        self.assertAlmostEqual(
            CostCalculator.calculate_cost("claude-haiku-4-5", 1000, 1000),
            (0.001 * 1 + 0.005 * 1) * 100,
            places=10,
        )

        # Test unsupported model
        self.assertIsNone(CostCalculator.calculate_cost("unknown-model", 1000, 1000))

    def test_token_costs(self):
        test_cases = [
            {
                "model_name": "gpt-4o",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "output_tokens": 1000,
            },
            {
                "model_name": "gpt-4o-mini",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "output_tokens": 1000,
            },
            {
                "model_name": "o1",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "output_tokens": 1000,
            },
            {
                "model_name": "o1-mini",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "output_tokens": 1000,
            },
            {
                "model_name": "o3-mini",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "output_tokens": 1000,
            },
            {
                "model_name": "deepseek-chat",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "output_tokens": 1000,
            },
            {
                "model_name": "deepseek-reasoner",
                "input_tokens": 1000,
                "cached_input_tokens": 500,
                "output_tokens": 1000,
            },
            {
                "model_name": "claude-haiku-4-5",
                "input_tokens": 1000,
                "cached_input_tokens": 0,
                "output_tokens": 1000,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case):
                # Calculate expected cost using CostCalculator
                expected_cost_in_cents = CostCalculator.calculate_cost(
                    model=case["model_name"],
                    input_tokens=case["input_tokens"],
                    output_tokens=case["output_tokens"],
                    cached_input_tokens=case["cached_input_tokens"],
                )

                # Create LLMResponse and check if cost is calculated correctly
                response = LLMResponse(
                    content="",
                    model=case["model_name"],
                    time=0.0,
                    input_tokens=case["input_tokens"],
                    output_tokens=case["output_tokens"],
                    cached_input_tokens=case["cached_input_tokens"],
                    cost_in_cents=expected_cost_in_cents,
                )

                self.assertAlmostEqual(
                    response.cost_in_cents, expected_cost_in_cents, places=10
                )


if __name__ == "__main__":
    unittest.main()
