from defog.llm.llm_providers import LLMProvider
from defog.llm.utils_logging import (
    ToolProgressTracker,
    SubTaskLogger,
    NoOpToolProgressTracker,
    NoOpSubTaskLogger,
)
from defog import config
from io import BytesIO


async def code_interpreter_tool(
    question: str,
    model: str,
    provider: LLMProvider,
    csv_string: str = "",
    instructions: str = "You are a Python programmer. You are given a question and a CSV string of data. You need to answer the question using the data. You are also given a sandboxed server environment where you can run the code.",
    verbose: bool = True,
):
    """
    Creates a python script to answer the question, where the python script is executed in a sandboxed server environment.
    """
    tracker_class = ToolProgressTracker if verbose else NoOpToolProgressTracker
    logger_class = SubTaskLogger if verbose else NoOpSubTaskLogger

    async with tracker_class(
        "Code Interpreter",
        f"Executing code to answer: {question[:50]}{'...' if len(question) > 50 else ''}",
    ) as tracker:
        subtask_logger = logger_class()
        subtask_logger.log_provider_info(
            provider.value if hasattr(provider, "value") else str(provider), model
        )

        # create a csv file from the csv_string
        if csv_string:
            tracker.update(10, "Preparing data file")
            subtask_logger.log_subtask("Creating CSV file from string", "processing")
            csv_file = BytesIO(csv_string.encode("utf-8"))
            csv_file.name = "data.csv"

        if provider in [LLMProvider.OPENAI, LLMProvider.OPENAI.value]:
            from openai import AsyncOpenAI
            from openai.types.responses import (
                ResponseCodeInterpreterToolCall,
                ResponseOutputMessage,
            )

            async with AsyncOpenAI(api_key=config.get("OPENAI_API_KEY")) as client:
                if csv_string:
                    tracker.update(20, "Uploading data file")
                    subtask_logger.log_subtask("Uploading CSV to OpenAI", "processing")
                    file = await client.files.create(
                        file=csv_file,
                        purpose="user_data",
                    )

                tracker.update(40, "Running code interpreter")
                subtask_logger.log_code_execution("python")
                response = await client.responses.create(
                    model=model,
                    tools=[
                        {
                            "type": "code_interpreter",
                            "container": {
                                "type": "auto",
                                "file_ids": [file.id] if csv_string else [],
                            },
                        }
                    ],
                    tool_choice="required",
                    input=[
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": question}],
                        }
                    ],
                    instructions=instructions,
                )
                tracker.update(80, "Processing results")
                subtask_logger.log_subtask("Extracting code and output", "processing")

                code = ""
                output_text = ""

                for chunk in response.output:
                    if isinstance(chunk, ResponseCodeInterpreterToolCall):
                        code += chunk.code
                    elif isinstance(chunk, ResponseOutputMessage):
                        for content in chunk.content:
                            output_text += content.text

                subtask_logger.log_result_summary(
                    "Code Execution",
                    {"code_length": len(code), "output_length": len(output_text)},
                )

                return {"code": code, "output": output_text}
        elif provider in [LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC.value]:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(
                api_key=config.get("ANTHROPIC_API_KEY"),
                default_headers={"anthropic-beta": "code-execution-2025-05-22"},
            )

            if csv_string:
                tracker.update(20, "Uploading data file")
                subtask_logger.log_subtask("Uploading CSV to Anthropic", "processing")
                file_object = await client.beta.files.upload(
                    file=csv_file,
                )

            tracker.update(40, "Running code execution")
            subtask_logger.log_code_execution("python")
            response = await client.messages.create(
                model=model,
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": instructions
                                + "\n\nThe question you must answer is: "
                                + question,
                            },
                            {
                                "type": "container_upload",
                                "file_id": file_object.id if csv_string else None,
                            },
                        ],
                    }
                ],
                tools=[{"type": "code_execution_20250522", "name": "code_execution"}],
                tool_choice={"type": "any"},
            )
            tracker.update(80, "Processing results")
            subtask_logger.log_subtask("Extracting code and output", "processing")

            code = ""
            output_text = ""
            for chunk in response.content:
                if chunk.type == "server_tool_use":
                    code += chunk.input["code"]
                elif chunk.type == "text":
                    output_text += chunk.text

            subtask_logger.log_result_summary(
                "Code Execution",
                {"code_length": len(code), "output_length": len(output_text)},
            )

            return {"code": code, "output": output_text}
        elif provider in [LLMProvider.GEMINI, LLMProvider.GEMINI.value]:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=config.get("GEMINI_API_KEY"))

            tracker.update(20, "Uploading data file")
            subtask_logger.log_subtask("Uploading CSV to Gemini", "processing")
            file_csv = await client.aio.files.upload(
                file=csv_file,
                config=types.UploadFileConfig(
                    mime_type="text/csv",
                    display_name="data.csv",
                ),
            )

            tracker.update(40, "Running code execution")
            subtask_logger.log_code_execution("python")
            response = await client.aio.models.generate_content(
                model=model,
                contents=[file_csv, question],
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                ),
            )

            tracker.update(80, "Processing results")
            subtask_logger.log_subtask("Extracting code and output", "processing")

            parts = response.candidates[0].content.parts

            code = ""
            output_text = ""

            for part in parts:
                if (
                    hasattr(part, "executable_code")
                    and part.executable_code is not None
                ):
                    code += part.executable_code.code
                if hasattr(part, "text") and part.text is not None:
                    output_text += part.text

            subtask_logger.log_result_summary(
                "Code Execution",
                {"code_length": len(code), "output_length": len(output_text)},
            )

            return {"code": code, "output": output_text}
