"""
Tests for TextDataExtractor
"""

import pytest
from unittest.mock import MagicMock, patch
from defog.llm.text_data_extractor import (
    TextDataExtractor,
    TextAnalysisResponse,
    DataPointIdentification,
    SchemaField,
    extract_text_data,
)


@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return """
    TRANSCRIPT: Federal Reserve Press Conference
    
    CHAIR POWELL: Good afternoon. The economy remains strong with unemployment at 4.2%.
    Inflation has moderated to 2.3% but remains above our 2% target.
    
    REPORTER: What is the Fed's stance on interest rates?
    CHAIR POWELL: We've decided to keep rates unchanged at 4.25-4.5%.
    
    REPORTER: When might we see rate cuts?
    CHAIR POWELL: We're monitoring the data closely. It depends on inflation trends.
    
    Key Economic Indicators:
    - GDP Growth: 2.5%
    - Unemployment Rate: 4.2%
    - Core PCE Inflation: 2.6%
    - Federal Funds Rate: 4.25-4.5%
    """


@pytest.fixture
def mock_analysis_response():
    """Mock response from text analysis."""
    return TextAnalysisResponse(
        document_type="transcript",
        content_description="Federal Reserve press conference transcript with Q&A",
        identified_datapoints=[
            DataPointIdentification(
                name="qa_exchanges",
                description="Question and answer exchanges between reporters and Chair",
                data_type="qa_pairs",
                schema_fields=[
                    SchemaField(
                        name="questioner", type="string", description="Person asking"
                    ),
                    SchemaField(
                        name="question", type="string", description="Question text"
                    ),
                    SchemaField(
                        name="answer", type="string", description="Answer text"
                    ),
                ],
            ),
            DataPointIdentification(
                name="economic_indicators",
                description="Key economic metrics mentioned",
                data_type="key_value_pairs",
                schema_fields=[
                    SchemaField(
                        name="gdp_growth", type="float", description="GDP growth rate"
                    ),
                    SchemaField(
                        name="unemployment_rate",
                        type="float",
                        description="Unemployment rate",
                    ),
                    SchemaField(
                        name="core_pce_inflation",
                        type="float",
                        description="Core PCE inflation",
                    ),
                    SchemaField(
                        name="federal_funds_rate",
                        type="string",
                        description="Fed funds rate",
                    ),
                ],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_text_data_extractor_initialization():
    """Test TextDataExtractor initialization."""
    extractor = TextDataExtractor(
        analysis_provider="anthropic",
        analysis_model="claude-haiku-4-5",
        max_parallel_extractions=3,
        temperature=0.5,
    )

    assert extractor.analysis_provider == "anthropic"
    assert extractor.analysis_model == "claude-haiku-4-5"
    assert extractor.max_parallel_extractions == 3
    assert extractor.temperature == 0.5
    assert extractor.enable_caching is True


@pytest.mark.asyncio
async def test_analyze_text_structure(sample_text, mock_analysis_response):
    """Test text structure analysis."""
    extractor = TextDataExtractor()

    # Mock the chat_async call
    mock_result = MagicMock()
    mock_result.content = mock_analysis_response
    mock_result.cost_in_cents = 1.5
    mock_result.input_tokens = 1000
    mock_result.output_tokens = 500
    mock_result.cached_input_tokens = 0

    with patch("defog.llm.text_data_extractor.chat_async", return_value=mock_result):
        analysis, cost_metadata = await extractor.analyze_text_structure(
            sample_text, focus_areas=["Q&A exchanges"]
        )

    assert analysis.document_type == "transcript"
    assert len(analysis.identified_datapoints) == 2
    assert analysis.identified_datapoints[0].name == "qa_exchanges"
    assert cost_metadata["cost_cents"] == 1.5
    assert cost_metadata["input_tokens"] == 1000


@pytest.mark.asyncio
async def test_extract_single_datapoint(sample_text, mock_analysis_response):
    """Test extracting a single datapoint."""
    extractor = TextDataExtractor()
    datapoint = mock_analysis_response.identified_datapoints[0]  # qa_exchanges
    schema = extractor._generate_pydantic_schema(datapoint)

    # Mock the extraction result
    mock_qa_data = {
        "exchanges": [
            {
                "questioner": "REPORTER",
                "question": "What is the Fed's stance on interest rates?",
                "answer": "We've decided to keep rates unchanged at 4.25-4.5%.",
            },
            {
                "questioner": "REPORTER",
                "question": "When might we see rate cuts?",
                "answer": "We're monitoring the data closely. It depends on inflation trends.",
            },
        ],
        "total_exchanges": 2,
    }

    mock_result = MagicMock()
    mock_result.content = mock_qa_data
    mock_result.cost_in_cents = 0.8
    mock_result.input_tokens = 500
    mock_result.output_tokens = 200
    mock_result.cached_input_tokens = 100

    with patch("defog.llm.text_data_extractor.chat_async", return_value=mock_result):
        result = await extractor.extract_single_datapoint(
            sample_text, datapoint, schema
        )

    assert result.success is True
    assert result.datapoint_name == "qa_exchanges"
    assert result.extracted_data == mock_qa_data
    assert result.cost_cents == 0.8


@pytest.mark.asyncio
async def test_extract_all_data(sample_text, mock_analysis_response):
    """Test extracting all datapoints."""
    extractor = TextDataExtractor()

    # Mock the analysis response
    mock_analysis_result = MagicMock()
    mock_analysis_result.content = mock_analysis_response
    mock_analysis_result.cost_in_cents = 1.5
    mock_analysis_result.input_tokens = 1000
    mock_analysis_result.output_tokens = 500
    mock_analysis_result.cached_input_tokens = 0

    # Mock extraction results
    mock_qa_data = {
        "exchanges": [
            {
                "questioner": "REPORTER",
                "question": "What is the Fed's stance on interest rates?",
                "answer": "We've decided to keep rates unchanged at 4.25-4.5%.",
            },
        ],
        "total_exchanges": 1,
    }

    mock_indicators_data = {
        "gdp_growth": 2.5,
        "unemployment_rate": 4.2,
        "core_pce_inflation": 2.6,
        "federal_funds_rate": "4.25-4.5%",
    }

    mock_extraction_results = [
        MagicMock(
            content=mock_qa_data,
            cost_in_cents=0.8,
            input_tokens=500,
            output_tokens=200,
            cached_input_tokens=100,
        ),
        MagicMock(
            content=mock_indicators_data,
            cost_in_cents=0.6,
            input_tokens=400,
            output_tokens=150,
            cached_input_tokens=50,
        ),
    ]

    with patch("defog.llm.text_data_extractor.chat_async") as mock_chat:
        # First call for analysis, subsequent calls for extraction
        mock_chat.side_effect = [mock_analysis_result] + mock_extraction_results

        result = await extractor.extract_all_data(sample_text)

    assert result.document_type == "transcript"
    assert result.total_datapoints_identified == 2
    assert result.successful_extractions == 2
    assert result.failed_extractions == 0
    assert result.total_cost_cents == pytest.approx(2.9)  # 1.5 + 0.8 + 0.6
    assert len(result.extraction_results) == 2


@pytest.mark.asyncio
async def test_extract_as_dict(sample_text):
    """Test extracting data as dictionary."""
    extractor = TextDataExtractor()

    # Create a minimal mock that returns the expected structure
    mock_result = MagicMock()
    mock_result.text_content_hash = "abc123"
    mock_result.document_type = "transcript"
    mock_result.total_datapoints_identified = 2
    mock_result.successful_extractions = 2
    mock_result.failed_extractions = 0
    mock_result.total_time_ms = 1500
    mock_result.total_cost_cents = 2.9
    mock_result.metadata = {"content_description": "Test transcript"}

    # Create mock extraction results
    mock_result.extraction_results = [
        MagicMock(
            success=True,
            datapoint_name="qa_exchanges",
            extracted_data={"exchanges": [], "total_exchanges": 0},
        ),
        MagicMock(
            success=True,
            datapoint_name="economic_indicators",
            extracted_data={"gdp_growth": 2.5},
        ),
    ]

    with patch.object(extractor, "extract_all_data", return_value=mock_result):
        result_dict = await extractor.extract_as_dict(sample_text)

    assert "metadata" in result_dict
    assert "data" in result_dict
    assert result_dict["metadata"]["document_type"] == "transcript"
    assert "qa_exchanges" in result_dict["data"]
    assert "economic_indicators" in result_dict["data"]


@pytest.mark.asyncio
async def test_convenience_function(sample_text):
    """Test the convenience function."""
    # Mock the entire extraction process
    mock_dict_result = {
        "metadata": {
            "document_type": "transcript",
            "extraction_summary": {
                "total_identified": 2,
                "successful": 2,
                "failed": 0,
            },
        },
        "data": {
            "qa_exchanges": {"exchanges": [], "total_exchanges": 0},
            "economic_indicators": {"gdp_growth": 2.5},
        },
    }

    with patch(
        "defog.llm.text_data_extractor.TextDataExtractor.extract_as_dict",
        return_value=mock_dict_result,
    ):
        result = await extract_text_data(sample_text, focus_areas=["Q&A"])

    assert result == mock_dict_result


@pytest.mark.asyncio
async def test_preprocess_text():
    """Test text preprocessing."""
    extractor = TextDataExtractor()

    raw_text = """
    
    Line 1
    
    
    Line 2
      Line 3 with spaces  
    
    """

    processed = extractor._preprocess_text(raw_text)
    expected = "Line 1\n\nLine 2\nLine 3 with spaces"

    assert processed == expected


@pytest.mark.asyncio
async def test_caching():
    """Test caching functionality."""
    extractor = TextDataExtractor(enable_caching=True)

    # Create mock result
    mock_result = MagicMock()
    mock_result.text_content_hash = "abc123"
    mock_result.document_type = "transcript"
    mock_result.total_datapoints_identified = 1
    mock_result.successful_extractions = 1
    mock_result.failed_extractions = 0
    mock_result.extraction_results = []
    mock_result.total_time_ms = 1000
    mock_result.total_cost_cents = 1.0
    mock_result.metadata = {}

    # Mock the analysis to return consistent results
    mock_analysis = TextAnalysisResponse(
        document_type="transcript", content_description="Test", identified_datapoints=[]
    )

    mock_analysis_result = MagicMock()
    mock_analysis_result.content = mock_analysis
    mock_analysis_result.cost_in_cents = 1.0
    mock_analysis_result.input_tokens = 100
    mock_analysis_result.output_tokens = 50
    mock_analysis_result.cached_input_tokens = 0

    with patch(
        "defog.llm.text_data_extractor.chat_async", return_value=mock_analysis_result
    ):
        # First call should analyze
        result1 = await extractor.extract_all_data("test content")

        # Second call should use cache
        result2 = await extractor.extract_all_data("test content")

        # Results should be the same object (cached)
        assert result1 is result2


@pytest.mark.asyncio
async def test_text_size_validation():
    """Test text size validation."""
    extractor = TextDataExtractor(max_text_size_mb=1)

    # Create text larger than 1MB
    large_text = "a" * (2 * 1024 * 1024)  # 2MB

    with pytest.raises(ValueError, match="Text content exceeds maximum size"):
        await extractor.extract_all_data(large_text)
