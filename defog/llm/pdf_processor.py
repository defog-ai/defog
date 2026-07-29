"""
PDF processors for analyzing PDFs with Claude and OpenAI APIs.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel, Field
import openai
from defog import config as defog_config
from .cost.calculator import CostCalculator

from .pdf_utils import download_and_process_pdf
from .utils import chat_async
from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


class PDFAnalysisInput(BaseModel):
    """Input model for PDF analysis requests."""

    url: str = Field(description="URL of the PDF to analyze")
    task: str = Field(description="Analysis task to perform on the PDF")
    response_format: Optional[Type[BaseModel]] = Field(
        default=None, description="Pydantic model for structured response"
    )


class PDFAnalysisResult(BaseModel):
    """Result model for PDF analysis."""

    success: bool = Field(description="Whether the analysis was successful")
    result: Any = Field(description="Analysis result content")
    metadata: Dict[str, Any] = Field(description="PDF metadata and processing info")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    chunks_processed: int = Field(description="Number of PDF chunks processed")


class BasePDFProcessor(ABC):
    """Abstract base class for PDF processors."""

    CONCURRENT_CHUNKS = 3  # Max concurrent chunk processing

    def __init__(
        self,
        provider: Union[str, LLMProvider],
        model: str,
        temperature: float = 0.0,
    ):
        """
        Initialize base PDF processor.

        Args:
            provider: LLM provider
            model: Model to use
            temperature: Sampling temperature
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature

    @abstractmethod
    async def _create_api_input(
        self,
        pdf_chunks: List[str],
        task: str,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> Any:
        """
        Create provider-specific API input.

        Args:
            pdf_chunks: List of base64 encoded PDF chunks
            task: Analysis task
            chunk_index: Current chunk index (0-based)
            total_chunks: Total number of chunks

        Returns:
            Provider-specific API input
        """
        pass

    @abstractmethod
    async def _call_api(
        self, api_input: Any, response_format: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        Make provider-specific API call.

        Args:
            api_input: Provider-specific API input
            response_format: Optional Pydantic model for structured response

        Returns:
            Dictionary with content, tokens, and cost information
        """
        pass

    def _get_system_prompt(self, chunk_index: int, total_chunks: int) -> str:
        """Get common system prompt for PDF analysis."""
        return f"""You are an expert PDF analyzer. Analyze the provided PDF content and complete the requested task.

PDF Context:
- This is chunk {chunk_index + 1} of {total_chunks} total chunks
- Focus on the content in this specific chunk
- If this is part of a multi-chunk document, note that context may be limited

Provide a thorough analysis based on the PDF content."""

    async def _process_single_chunk(
        self,
        pdf_chunks: List[str],
        task: str,
        chunk_index: int,
        total_chunks: int,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single PDF chunk.

        Args:
            pdf_chunks: List of base64 encoded PDF chunks
            task: Analysis task
            chunk_index: Current chunk index
            total_chunks: Total number of chunks
            response_format: Optional Pydantic model for structured response

        Returns:
            Processing result for this chunk
        """
        try:
            api_input = await self._create_api_input(
                pdf_chunks, task, chunk_index, total_chunks
            )

            result = await self._call_api(api_input, response_format)

            return {
                "success": True,
                "content": result["content"],
                "chunk_index": chunk_index,
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "cost_in_cents": result.get("cost_in_cents", 0),
                **result.get("extra", {}),  # Provider-specific fields
            }

        except Exception as e:
            logger.error(f"Error processing PDF chunk {chunk_index}: {e}")
            return {"success": False, "error": str(e), "chunk_index": chunk_index}

    async def _process_multiple_chunks(
        self,
        pdf_chunks: List[str],
        task: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDF chunks concurrently.

        Args:
            pdf_chunks: List of base64 encoded PDF chunks
            task: Analysis task
            response_format: Optional Pydantic model for structured response

        Returns:
            List of processing results for all chunks
        """
        total_chunks = len(pdf_chunks)

        # Process chunks concurrently (limit concurrency to avoid rate limits)
        semaphore = asyncio.Semaphore(self.CONCURRENT_CHUNKS)

        async def process_with_semaphore(chunk_index: int):
            async with semaphore:
                return await self._process_single_chunk(
                    pdf_chunks, task, chunk_index, total_chunks, response_format
                )

        # Create tasks for all chunks
        tasks = [process_with_semaphore(i) for i in range(total_chunks)]

        # Wait for all chunks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {"success": False, "error": str(result), "chunk_index": i}
                )
            else:
                processed_results.append(result)

        return processed_results

    def _combine_chunk_results(
        self, chunk_results: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> str:
        """
        Combine results from multiple chunks into a coherent response.

        Args:
            chunk_results: Results from processing each chunk
            metadata: PDF metadata

        Returns:
            Combined analysis result
        """
        successful_results = [r for r in chunk_results if r.get("success")]
        failed_results = [r for r in chunk_results if not r.get("success")]

        if not successful_results:
            return "Failed to process any PDF chunks successfully."

        combined_content = []

        if len(successful_results) > 1:
            combined_content.append(
                f"PDF Analysis Summary ({len(successful_results)} chunks processed):"
            )
            combined_content.append("=" * 60)

            for result in successful_results:
                chunk_idx = result["chunk_index"]
                combined_content.append(f"\n--- Chunk {chunk_idx + 1} Analysis ---")
                combined_content.append(str(result["content"]))
        else:
            # Single chunk result
            combined_content.append(str(successful_results[0]["content"]))

        if failed_results:
            combined_content.append(
                f"\n\nNote: {len(failed_results)} chunks failed to process."
            )

        return "\n".join(combined_content)

    async def analyze_pdf(
        self, url: str, task: str, response_format: Optional[Type[BaseModel]] = None
    ) -> PDFAnalysisResult:
        """
        Analyze a PDF from URL.

        Args:
            url: PDF URL to analyze
            task: Analysis task to perform
            response_format: Optional Pydantic model for structured response

        Returns:
            PDFAnalysisResult with analysis results
        """
        try:
            # Download and process PDF
            logger.info(f"Processing PDF from URL: {url}")
            pdf_chunks, metadata = await download_and_process_pdf(url)

            logger.info(f"PDF processed: {metadata['chunk_count']} chunks")

            # Process chunks
            if len(pdf_chunks) == 1:
                # Single chunk processing
                result = await self._process_single_chunk(
                    pdf_chunks, task, 0, 1, response_format
                )
                chunk_results = [result]
            else:
                # Multiple chunk processing
                chunk_results = await self._process_multiple_chunks(
                    pdf_chunks, task, response_format
                )

            # Combine results
            if (
                response_format
                and len(pdf_chunks) == 1
                and chunk_results[0].get("success")
            ):
                # Return structured response directly for single chunk
                final_result = chunk_results[0]["content"]
            else:
                # Combine text results for multiple chunks or unstructured response
                final_result = self._combine_chunk_results(chunk_results, metadata)

            # Calculate total tokens and cost
            total_input_tokens = sum(
                r.get("input_tokens", 0) for r in chunk_results if r.get("success")
            )
            total_output_tokens = sum(
                r.get("output_tokens", 0) for r in chunk_results if r.get("success")
            )
            total_cost = sum(
                r.get("cost_in_cents", 0) for r in chunk_results if r.get("success")
            )

            # Add processing metadata
            processing_metadata = {
                **metadata,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost_in_cents": total_cost,
                "successful_chunks": len(
                    [r for r in chunk_results if r.get("success")]
                ),
                "failed_chunks": len(
                    [r for r in chunk_results if not r.get("success")]
                ),
            }

            # Add provider-specific metadata
            for key in ["cached_tokens"]:
                values = [
                    r.get(key, 0)
                    for r in chunk_results
                    if r.get("success") and key in r
                ]
                if values:
                    processing_metadata[key] = sum(values)

            return PDFAnalysisResult(
                success=True,
                result=final_result,
                metadata=processing_metadata,
                chunks_processed=len(pdf_chunks),
            )

        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return PDFAnalysisResult(
                success=False,
                result=None,
                metadata={"error_type": type(e).__name__},
                error=str(e),
                chunks_processed=0,
            )


class ClaudePDFProcessor(BasePDFProcessor):
    """PDF processor that integrates with Claude's API and input caching."""

    async def _create_api_input(
        self,
        pdf_chunks: List[str],
        task: str,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> List[Dict[str, Any]]:
        """Create message content for Claude API with PDF and cache control."""
        system_content = self._get_system_prompt(chunk_index, total_chunks)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_chunks[chunk_index],
                        },
                        "cache_control": {"type": "ephemeral"},  # Cache PDF content
                    },
                    {
                        "type": "text",
                        "text": f"{system_content}\n\nThe task to be completed is: `{task}`",
                    },
                ],
            },
        ]

        return messages

    async def _call_api(
        self, api_input: Any, response_format: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """Make Claude API call."""
        # Prepare chat parameters
        chat_params = {
            "provider": self.provider,
            "model": self.model,
            "messages": api_input,
            "temperature": self.temperature,
            "reasoning_effort": "low",  # 2048 token thinking budget
        }

        # Add response format if specified
        if response_format:
            chat_params["response_format"] = response_format

        # Call Claude API
        response = await chat_async(**chat_params)

        return {
            "content": response.content,
            "input_tokens": response.input_tokens or 0,
            "output_tokens": response.output_tokens or 0,
            "cost_in_cents": response.cost_in_cents or 0,
            "extra": {
                "cached_tokens": getattr(response, "cached_input_tokens", None) or 0
            },
        }


class OpenAIPDFProcessor(BasePDFProcessor):
    """PDF processor that integrates with OpenAI's API using base64 encoding."""

    def __init__(
        self,
        provider: Union[str, LLMProvider] = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize OpenAI PDF processor with API client."""
        super().__init__(provider, model, temperature)

        self._client_kwargs = {
            "api_key": api_key or defog_config.get("OPENAI_API_KEY"),
            "base_url": base_url or "https://api.openai.com/v1/",
        }
        self.client = None

    async def analyze_pdf(
        self, url: str, task: str, response_format: Optional[Type[BaseModel]] = None
    ) -> PDFAnalysisResult:
        """Analyze a PDF while deterministically managing the OpenAI client."""
        async with openai.AsyncOpenAI(**self._client_kwargs) as client:
            self.client = client
            try:
                return await super().analyze_pdf(url, task, response_format)
            finally:
                self.client = None

    async def _create_api_input(
        self,
        pdf_chunks: List[str],
        task: str,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> List[Dict[str, Any]]:
        """Create input content for OpenAI API with PDF."""
        prompt = f"{self._get_system_prompt(chunk_index, total_chunks)}\n\nTask: {task}"

        input_content = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": f"document_chunk_{chunk_index + 1}.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf_chunks[chunk_index]}",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            },
        ]

        return input_content

    async def _call_api(
        self, api_input: Any, response_format: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """Make OpenAI API call."""
        # Prepare API parameters
        api_params = {
            "model": self.model,
            "input": api_input,
        }

        # Only add temperature if supported (reasoning models don't support it)
        if not self.model.startswith("o"):
            api_params["temperature"] = self.temperature

        # Call OpenAI API using appropriate method
        if response_format:
            # Use parse method for structured output
            api_params["text_format"] = response_format
            response = await self.client.responses.parse(**api_params)
            content = response.output_parsed
        else:
            # Use create method for unstructured output
            response = await self.client.responses.create(**api_params)
            content = response.output_text

        # Get token usage from response
        usage = getattr(response, "usage", None)
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        # Calculate cost using the actual model pricing
        cost_in_cents = CostCalculator.calculate_cost(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # If cost calculation fails, return 0
        if cost_in_cents is None:
            cost_in_cents = 0

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_in_cents": cost_in_cents,
            "extra": {},
        }
