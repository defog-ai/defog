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
):
    """
    Search the web for the answer to the question.
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

            response = await client.responses.create(
                model=model,
                tools=[{"type": "web_search"}],
                tool_choice="required",
                input=question,
                # in the responses API, this means both the reasoning and the output tokens
                max_output_tokens=max_tokens,
            )
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

            return {
                "usage": usage,
                "search_results": output_text,
                "websites_cited": websites_cited,
            }

        elif provider in [LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC.value]:
            from anthropic import AsyncAnthropic
            from anthropic.types import TextBlock

            client = AsyncAnthropic(api_key=config.get("ANTHROPIC_API_KEY"))

            tracker.update(20, "Initiating web search")
            subtask_logger.log_search_status(question, max_results=5)

            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": question}],
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 5,
                        # can also use allowed_domains to limit the search to specific domains
                        # can also use blocked_domains to exclude specific domains
                    }
                ],
                tool_choice={"type": "any"},
            )

            tracker.update(80, "Processing search results")
            subtask_logger.log_subtask("Extracting citations and content", "processing")

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            search_results = response.content
            # we want to use only the TextBlock class in the search results
            search_results = [
                block for block in search_results if isinstance(block, TextBlock)
            ]

            # convert the search_results into simple text with citations
            # (where citations = text + hyperlinks
            output_text = [
                (
                    f'<a href="{block.citations[0].url}">' + block.text + "</a>"
                    if block.citations
                    else block.text
                )
                for block in search_results
            ]
            websites_cited = [
                {"url": block.citations[0].url, "title": block.citations[0].title}
                for block in search_results
                if block.citations
            ]

            subtask_logger.log_result_summary(
                "Web Search",
                {
                    "text_blocks": len(search_results),
                    "websites_cited": len(websites_cited),
                    "tokens_used": usage["input_tokens"] + usage["output_tokens"],
                },
            )

            return {
                "usage": usage,
                "search_results": output_text,
                "websites_cited": websites_cited,
            }
        elif provider in [LLMProvider.GEMINI, LLMProvider.GEMINI.value]:
            from google import genai
            from google.genai.types import (
                Tool,
                GenerateContentConfig,
                GoogleSearch,
                ToolConfig,
                FunctionCallingConfig,
            )

            client = genai.Client(api_key=config.get("GEMINI_API_KEY"))
            google_search_tool = Tool(google_search=GoogleSearch())

            tracker.update(20, "Initiating Google search")
            subtask_logger.log_search_status(question)

            response = await client.aio.models.generate_content(
                model=model,
                contents=question,
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    tool_config=ToolConfig(
                        function_calling_config=FunctionCallingConfig(
                            mode="ANY",
                        ),
                    ),
                ),
            )
            tracker.update(80, "Processing search results")
            subtask_logger.log_subtask("Extracting grounding metadata", "processing")

            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "thinking_tokens": response.usage_metadata.thoughts_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }

            websites_cited = []
            if response.candidates:
                for candidate in response.candidates:
                    if (
                        candidate.grounding_metadata
                        and candidate.grounding_metadata.grounding_chunks
                    ):
                        for chunk in candidate.grounding_metadata.grounding_chunks:
                            websites_cited.append(
                                {"source": chunk.web.title, "url": chunk.web.uri}
                            )

            output_text = response.text

            subtask_logger.log_result_summary(
                "Web Search",
                {
                    "websites_found": len(websites_cited),
                    "total_tokens": usage["input_tokens"]
                    + usage["thinking_tokens"]
                    + usage["output_tokens"],
                },
            )

            return {
                "usage": usage,
                "search_results": output_text,
                "websites_cited": websites_cited,
            }

        else:
            raise ValueError(f"Provider {provider} not supported")
