import warnings
import unittest
import pytest
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.llm_providers import LLMProvider
from tests.conftest import skip_if_no_api_key

warnings.filterwarnings("ignore")


class TestCodeInterp(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test fixtures with sample CSV data for all tests."""
        self.sample_csv = """name,age,salary,department
John Doe,30,50000,Engineering
Jane Smith,25,45000,Marketing  
Bob Johnson,35,60000,Engineering
Alice Brown,28,55000,Sales
Charlie Wilson,32,48000,Marketing"""

        self.complex_question = "What is the average salary by department? Show the results in a table format."

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_openai_complex_analysis(self):
        """Test OpenAI provider with complex aggregation question."""

        response = await code_interpreter_tool(
            question=self.complex_question,
            model="gpt-4.1",
            provider=LLMProvider.OPENAI,
            csv_string=self.sample_csv,
        )

        self.assertIsInstance(response, dict)
        self.assertIn("code", response)
        self.assertIn("output", response)
        self.assertGreater(len(response["output"]), 0)
        # Should contain department names from our data in output
        output_text = response["output"]
        self.assertIn("Engineering", output_text)
        self.assertIn("Marketing", output_text)
        self.assertIn("Sales", output_text)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_anthropic_complex_analysis(self):
        """Test Anthropic provider with complex aggregation question."""

        response = await code_interpreter_tool(
            question=self.complex_question,
            model="claude-3-7-sonnet-latest",
            provider=LLMProvider.ANTHROPIC,
            csv_string=self.sample_csv,
        )

        self.assertIsInstance(response, dict)
        self.assertIn("code", response)
        self.assertIn("output", response)
        self.assertGreater(len(response["output"]), 0)
        # Should contain department names from our data in output
        output_text = response["output"]
        self.assertIn("Engineering", output_text)
        self.assertIn("Marketing", output_text)
        self.assertIn("Sales", output_text)

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_gemini_complex_analysis(self):
        """Test Gemini provider with complex aggregation question."""

        response = await code_interpreter_tool(
            question=self.complex_question,
            model="gemini-2.0-flash",
            provider=LLMProvider.GEMINI,
            csv_string=self.sample_csv,
        )

        self.assertIsInstance(response, dict)
        self.assertIn("code", response)
        self.assertIn("output", response)
        # Should have generated some code and output
        self.assertGreater(len(response["code"] + response["output"]), 0)

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_large_dataset_analysis(self):
        """Test analysis with larger dataset."""

        # Generate larger CSV with 100 rows
        large_csv_header = "id,name,age,salary,department,years_experience\n"
        large_csv_rows = []
        departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]

        for i in range(100):
            dept = departments[i % len(departments)]
            age = 22 + (i % 40)
            salary = 40000 + (i * 500)
            experience = max(0, age - 22)
            large_csv_rows.append(
                f"{i + 1},Employee{i + 1},{age},{salary},{dept},{experience}"
            )

        large_csv = large_csv_header + "\n".join(large_csv_rows)

        response = await code_interpreter_tool(
            question="What are the key insights about salary distribution across departments? Include summary statistics.",
            model="gpt-4.1",
            provider=LLMProvider.OPENAI,
            csv_string=large_csv,
        )

        self.assertIsInstance(response, dict)
        self.assertIn("code", response)
        self.assertIn("output", response)
        self.assertGreater(len(response["output"]), 0)
        # Should contain department names and statistical terms in output
        output_lower = response["output"].lower()
        self.assertTrue(any(dept.lower() in output_lower for dept in departments))
        self.assertTrue(
            any(
                word in output_lower
                for word in ["mean", "median", "average", "distribution"]
            )
        )

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_mathematical_calculations(self):
        """Test mathematical calculations and formulas."""

        math_csv = """product,price,quantity,discount_rate
Laptop,1000,5,0.1
Mouse,25,100,0.05
Keyboard,75,50,0.15
Monitor,300,20,0.08"""

        response = await code_interpreter_tool(
            question="Calculate the total revenue after applying discounts for each product, and find which product generates the most revenue.",
            model="claude-3-7-sonnet-latest",
            provider=LLMProvider.ANTHROPIC,
            csv_string=math_csv,
        )

        self.assertIsInstance(response, dict)
        self.assertIn("code", response)
        self.assertIn("output", response)
        self.assertGreater(len(response["output"]), 0)
        # Should contain product names and revenue calculations in output
        output_text = response["output"]
        self.assertIn("Laptop", output_text)
        self.assertTrue(any(char.isdigit() for char in output_text))

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_time_series_data(self):
        """Test analysis with time-based data."""

        time_series_csv = """date,sales,customers
2024-01-01,1000,50
2024-01-02,1200,60
2024-01-03,800,40
2024-01-04,1500,75
2024-01-05,1100,55
2024-01-06,1300,65
2024-01-07,900,45"""

        response = await code_interpreter_tool(
            question="Analyze the sales trend over time and calculate the daily growth rate.",
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            csv_string=time_series_csv,
        )

        self.assertIsInstance(response, dict)
        self.assertIn("code", response)
        self.assertIn("output", response)
        self.assertGreater(len(response["output"]), 0)
        # Should contain time-related analysis terms in output
        output_lower = response["output"].lower()
        self.assertTrue(
            any(word in output_lower for word in ["trend", "growth", "time", "daily"])
        )


if __name__ == "__main__":
    unittest.main()
