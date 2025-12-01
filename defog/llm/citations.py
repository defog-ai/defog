from defog.llm.llm_providers import LLMProvider
from defog.llm.utils_logging import (
    ToolProgressTracker,
    SubTaskLogger,
    NoOpToolProgressTracker,
    NoOpSubTaskLogger,
)
from defog import config
from defog.llm.cost.calculator import CostCalculator
import asyncio


class CitationsResult(list):
    """List-like container that carries citation blocks plus cost metadata."""

    def __init__(self, blocks, cost_in_cents=None, usage=None):
        super().__init__(blocks or [])
        self.cost_in_cents = cost_in_cents
        self.usage = usage


async def upload_document_to_openai_vector_store(document, store_id):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.get("OPENAI_API_KEY"))

    file_name = document["document_name"]
    if not file_name.endswith(".txt"):
        file_name = file_name + ".txt"
    file_content = document["document_content"]
    if isinstance(file_content, str):
        # convert to bytes
        file_content = file_content.encode("utf-8")

    # first, upload the file to the vector store
    file = await client.files.create(
        file=(file_name, file_content), purpose="assistants"
    )

    # then add it to the vector store
    await client.vector_stores.files.create(
        vector_store_id=store_id,
        file_id=file.id,
    )


async def citations_tool(
    question: str,
    instructions: str,
    documents: list[dict],
    model: str,
    provider: LLMProvider,
    max_tokens: int = 32000,
    verbose: bool = True,
    reasoning_effort: str | None = None,
):
    """
    Use this tool to get an answer to a well-cited answer to a question,
    given a list of documents.
    Returns a CitationsResult (list subclass) that carries citation blocks plus an
    optional `cost_in_cents` attribute for the citation generation call.
    """
    tracker_class = ToolProgressTracker if verbose else NoOpToolProgressTracker
    logger_class = SubTaskLogger if verbose else NoOpSubTaskLogger

    async with tracker_class(
        "Citations Tool", f"Generating citations for {len(documents)} documents"
    ) as tracker:
        subtask_logger = logger_class()
        subtask_logger.log_provider_info(
            provider.value if hasattr(provider, "value") else str(provider), model
        )

        if provider in [LLMProvider.OPENAI, LLMProvider.OPENAI.value]:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=config.get("OPENAI_API_KEY"))

            # create an ephemeral vector store
            store = await client.vector_stores.create()
            store_id = store.id

            # Upload all documents in parallel
            tracker.update(10, "Uploading documents to vector store")
            subtask_logger.log_subtask("Starting document uploads", "processing")

            upload_tasks = []
            for idx, document in enumerate(documents, 1):
                subtask_logger.log_document_upload(
                    document["document_name"], idx, len(documents)
                )
                upload_tasks.append(
                    upload_document_to_openai_vector_store(document, store_id)
                )

            await asyncio.gather(*upload_tasks)
            tracker.update(40, "Documents uploaded")

            # keep polling until the vector store is ready
            is_ready = False
            while not is_ready:
                store = await client.vector_stores.files.list(vector_store_id=store_id)
                total_completed = sum(
                    1 for file in store.data if file.status == "completed"
                )
                is_ready = total_completed == len(documents)

                # Update progress based on indexing status
                progress = 40 + (total_completed / len(documents) * 40)  # 40-80% range
                tracker.update(
                    progress, f"Indexing {total_completed}/{len(documents)} files"
                )
                subtask_logger.log_vector_store_status(total_completed, len(documents))

                if not is_ready:
                    await asyncio.sleep(1)

            # get the answer
            tracker.update(80, "Generating citations")
            subtask_logger.log_subtask("Querying with file search", "processing")
            payload = {
                "model": model,
                "input": question,
                "tools": [
                    {
                        "type": "file_search",
                        "vector_store_ids": [store_id],
                    }
                ],
                "tool_choice": "required",
                "instructions": instructions,
                "max_output_tokens": max_tokens,
            }
            if reasoning_effort:
                payload["reasoning"] = {
                    "effort": reasoning_effort,
                    "summary": "auto",
                }

            response = await client.responses.create(**payload)

            usage = getattr(response, "usage", None)
            cost_in_cents = None
            if usage:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                cached_tokens = (
                    getattr(
                        getattr(usage, "input_tokens_details", None),
                        "cached_tokens",
                        0,
                    )
                    or 0
                )
                cost_in_cents = CostCalculator.calculate_cost(
                    model, input_tokens, output_tokens, cached_tokens
                )

            # convert the response to a list of blocks
            # similar to a subset of the Anthropic citations API
            blocks = []
            for part in response.output:
                if part.type == "message":
                    contents = part.content
                    for item in contents:
                        if item.type == "output_text":
                            blocks.append(
                                {
                                    "text": item.text,
                                    "type": "text",
                                    "citations": [
                                        {"document_title": i.filename}
                                        for i in item.annotations
                                    ],
                                }
                            )
            tracker.update(95, "Processing results")
            subtask_logger.log_result_summary(
                "Citations",
                {
                    "blocks_generated": len(blocks),
                    "documents_processed": len(documents),
                },
            )

            return CitationsResult(blocks, cost_in_cents=cost_in_cents, usage=usage)

        elif provider in [LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC.value]:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=config.get("ANTHROPIC_API_KEY"))

            document_contents = []
            for document in documents:
                document_contents.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": document["document_content"],
                        },
                        "title": document["document_name"],
                        "citations": {"enabled": True},
                    }
                )

            # Create content messages with citations enabled for individual tool calls
            tracker.update(50, "Preparing document contents")
            subtask_logger.log_subtask(
                f"Processing {len(documents)} documents for Anthropic", "processing"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        # Add all individual document contents
                        *document_contents,
                    ],
                }
            ]

            tracker.update(70, "Generating citations")
            subtask_logger.log_subtask(
                "Calling Anthropic API with citations", "processing"
            )

            payload = {
                "model": model,
                "messages": messages,
                "system": instructions,
                "max_tokens": max_tokens,
            }
            if reasoning_effort:
                if reasoning_effort == "low":
                    budget_tokens = 4096
                elif reasoning_effort == "medium":
                    budget_tokens = 8192
                elif reasoning_effort == "high":
                    budget_tokens = 16384
                else:
                    raise ValueError(f"Invalid reasoning effort: {reasoning_effort}")

                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }

            response = await client.messages.create(**payload)

            usage = getattr(response, "usage", None)
            cost_in_cents = None
            if usage:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                cost_in_cents = CostCalculator.calculate_cost(
                    model, input_tokens, output_tokens, None
                )

            tracker.update(90, "Processing results")
            response_with_citations = [item.to_dict() for item in response.content]

            subtask_logger.log_result_summary(
                "Citations",
                {
                    "content_blocks": len(response_with_citations),
                    "documents_processed": len(documents),
                },
            )

            return CitationsResult(
                response_with_citations, cost_in_cents=cost_in_cents, usage=usage
            )

        else:
            raise ValueError(f"Provider {provider} not supported for citations tool")
