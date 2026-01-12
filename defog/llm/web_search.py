from typing import Optional, Type, Union

from pydantic import BaseModel

from defog.llm.llm_providers import LLMProvider
from defog.llm.utils_logging import (
    ToolProgressTracker,
    SubTaskLogger,
    NoOpToolProgressTracker,
    NoOpSubTaskLogger,
)
from defog import config


async def web_search_tool(
    question: str,
    model: str,
    provider: LLMProvider,
    max_tokens: int = 8192,
    verbose: bool = True,
    response_format: Optional[Type[BaseModel]] = None,
):
    """
    Search the web for the answer to the question.

    Args:
        question: The search query/question to answer.
        model: The model to use (e.g., "gpt-4.1", "gemini-3-flash-preview").
        provider: The LLM provider (OPENAI, ANTHROPIC, or GEMINI).
        max_tokens: Maximum tokens for the response.
        verbose: Whether to log progress.
        response_format: Optional Pydantic model class for structured output.
            When provided, search_results will contain a parsed instance of this model.

    Returns:
        dict with keys:
            - usage: Token usage statistics
            - search_results: str (or Pydantic model instance if response_format provided)
            - websites_cited: List of cited sources with url and title/source
    """
    tracker_class = ToolProgressTracker if verbose else NoOpToolProgressTracker
    logger_class = SubTaskLogger if verbose else NoOpSubTaskLogger

    async with tracker_class(
        "Web Search",
        f"Searching for: {question[:50]}{'...' if len(question) > 50 else ''}",
    ) as tracker:
        subtask_logger = logger_class()
        subtask_logger.log_provider_info(
            provider.value if hasattr(provider, "value") else str(provider), model
        )

        if provider in [LLMProvider.OPENAI, LLMProvider.OPENAI.value]:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=config.get("OPENAI_API_KEY"))

            tracker.update(20, "Initiating web search")
            subtask_logger.log_search_status(question)

            # Build request parameters
            request_params = {
                "model": model,
                "tools": [{"type": "web_search"}],
                "tool_choice": "required",
                "input": question,
                "max_output_tokens": max_tokens,
            }

            # Add structured output format if provided
            if response_format:
                schema = response_format.model_json_schema()
                request_params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": schema.get("title", response_format.__name__),
                        "schema": schema | {"additionalProperties": False},
                    }
                }

            response = await client.responses.create(**request_params)
            tracker.update(80, "Processing search results")
            subtask_logger.log_subtask("Extracting citations and content", "processing")

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            output_text = response.output_text
            websites_cited = []
            for output in response.output:
                if hasattr(output, "content") and output.content:
                    for content in output.content:
                        if content.annotations:
                            for annotation in content.annotations:
                                websites_cited.append(
                                    {
                                        "url": annotation.url,
                                        "title": annotation.title,
                                    }
                                )

            subtask_logger.log_result_summary(
                "Web Search",
                {
                    "websites_found": len(websites_cited),
                    "tokens_used": usage["input_tokens"] + usage["output_tokens"],
                },
            )

            # Parse structured output if response_format provided
            search_results = output_text
            if response_format and output_text:
                search_results = response_format.model_validate_json(output_text)

            return {
                "usage": usage,
                "search_results": search_results,
                "websites_cited": websites_cited,
            }

        elif provider in [LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC.value]:
            import json

            from anthropic import AsyncAnthropic
            from anthropic.types import TextBlock

            client = AsyncAnthropic(api_key=config.get("ANTHROPIC_API_KEY"))

            tracker.update(20, "Initiating web search")
            subtask_logger.log_search_status(question, max_results=5)

            # Build request parameters
            request_params = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": question}],
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5,
                    }
                ],
                "tool_choice": {"type": "any"},
            }

            # Add system message for structured output if provided
            if response_format:
                schema_json = json.dumps(response_format.model_json_schema(), indent=2)
                request_params["system"] = (
                    f"After searching the web and gathering information, respond with a JSON object "
                    f"that matches this schema:\n\n```json\n{schema_json}\n```\n\n"
                    f"Output ONLY the JSON object, no other text."
                )

            response = await client.messages.create(**request_params)

            tracker.update(80, "Processing search results")
            subtask_logger.log_subtask("Extracting citations and content", "processing")

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            content_blocks = response.content
            # we want to use only the TextBlock class in the search results
            text_blocks = [
                block for block in content_blocks if isinstance(block, TextBlock)
            ]

            # convert the search_results into simple text with citations
            # (where citations = text + hyperlinks)
            output_text = [
                (
                    f'<a href="{block.citations[0].url}">' + block.text + "</a>"
                    if block.citations
                    else block.text
                )
                for block in text_blocks
            ]
            websites_cited = [
                {"url": block.citations[0].url, "title": block.citations[0].title}
                for block in text_blocks
                if block.citations
            ]

            subtask_logger.log_result_summary(
                "Web Search",
                {
                    "text_blocks": len(text_blocks),
                    "websites_cited": len(websites_cited),
                    "tokens_used": usage["input_tokens"] + usage["output_tokens"],
                },
            )

            # Parse structured output if response_format provided
            search_results: Union[list, BaseModel] = output_text
            if response_format and text_blocks:
                # Get the raw text from text blocks and parse as JSON
                raw_text = "".join(block.text for block in text_blocks)
                # Try to extract JSON from the response (may have surrounding text)
                import re

                json_match = re.search(r"\{[\s\S]*\}", raw_text)
                if json_match:
                    search_results = response_format.model_validate_json(
                        json_match.group()
                    )
                else:
                    # Fall back to trying the whole text
                    search_results = response_format.model_validate_json(raw_text)

            return {
                "usage": usage,
                "search_results": search_results,
                "websites_cited": websites_cited,
            }
        elif provider in [LLMProvider.GEMINI, LLMProvider.GEMINI.value]:
            from google import genai

            client = genai.Client(api_key=config.get("GEMINI_API_KEY"))

            tracker.update(20, "Initiating Google search")
            subtask_logger.log_search_status(question)

            # Build request params for Interactions API
            request_params = {
                "model": model,
                "input": question,
                "tools": [{"type": "google_search"}],
            }

            # Add structured output config if response_format provided
            if response_format:
                request_params["response_mime_type"] = "application/json"
                request_params["response_format"] = response_format.model_json_schema()

            # Use Interactions API for proper token counts
            response = await client.aio.interactions.create(**request_params)

            tracker.update(80, "Processing search results")
            subtask_logger.log_subtask("Extracting grounding metadata", "processing")

            # Extract token usage (same pattern as gemini_provider.py)
            input_tokens = 0
            output_tokens = 0
            thinking_tokens = 0

            if hasattr(response, "usage") and response.usage:
                usage_obj = response.usage
                input_tokens = getattr(usage_obj, "total_input_tokens", 0)
                output_tokens = getattr(usage_obj, "total_output_tokens", 0)
                thinking_tokens = getattr(usage_obj, "total_reasoning_tokens", 0)
                cached_tokens = getattr(usage_obj, "total_cached_tokens", 0)
                tool_use_tokens = getattr(usage_obj, "total_tool_use_tokens", 0)

            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens + thinking_tokens + tool_use_tokens,
                "cached_input_tokens": cached_tokens,
            }

            # Extract websites from grounding metadata
            websites_cited = []
            if hasattr(response, "outputs") and response.outputs:
                for output in response.outputs:
                    if (
                        hasattr(output, "grounding_metadata")
                        and output.grounding_metadata
                    ):
                        grounding = output.grounding_metadata
                        if (
                            hasattr(grounding, "grounding_chunks")
                            and grounding.grounding_chunks
                        ):
                            for chunk in grounding.grounding_chunks:
                                if hasattr(chunk, "web") and chunk.web:
                                    websites_cited.append(
                                        {
                                            "source": chunk.web.title,
                                            "url": chunk.web.uri,
                                        }
                                    )

            # Extract text from outputs
            output_text = ""
            if hasattr(response, "outputs") and response.outputs:
                for output in response.outputs:
                    if getattr(output, "type", None) == "text":
                        output_text += output.text or ""

            subtask_logger.log_result_summary(
                "Web Search",
                {
                    "websites_found": len(websites_cited),
                    "total_tokens": usage["input_tokens"]
                    + usage["output_tokens"]
                    + usage["cached_input_tokens"],
                },
            )

            # Parse structured output if response_format provided
            search_results: Union[str, BaseModel] = output_text
            if response_format and output_text:
                search_results = response_format.model_validate_json(output_text)

            return {
                "usage": usage,
                "search_results": search_results,
                "websites_cited": websites_cited,
            }

        else:
            raise ValueError(f"Provider {provider} not supported")
