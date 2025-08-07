from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
import json
import re
from ..config.settings import LLMConfig
from ..exceptions import ProviderError
from ..tools import ToolHandler
import inspect


@dataclass
class LLMResponse:
    content: Any
    model: str
    time: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: Optional[int] = None
    output_tokens_details: Optional[Dict[str, int]] = None
    cost_in_cents: Optional[float] = None
    tool_outputs: Optional[List[Dict[str, Any]]] = None
    citations: Optional[List[Dict[str, Any]]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.config = config or LLMConfig()
        self.tool_handler = ToolHandler()

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass

    @abstractmethod
    def build_params(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format=None,
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        store: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        parallel_tool_calls: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Build parameters for the provider's API call."""
        pass

    @abstractmethod
    async def process_response(
        self,
        client,
        response,
        request_params: Dict[str, Any],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format=None,
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """Process the response from the provider."""
        pass

    @abstractmethod
    async def execute_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format=None,
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        store: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        tool_budget: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with the provider."""
        pass

    @abstractmethod
    def supports_tools(self, model: str) -> bool:
        """Check if the model supports tool calling."""
        pass

    @abstractmethod
    def supports_response_format(self, model: str) -> bool:
        """Check if the model supports structured response formats."""
        pass

    @abstractmethod
    def create_image_message(
        self,
        image_base64: Union[str, List[str]],
        description: str = "Tool generated image",
        image_detail: str = "low",
    ) -> Any:
        """Create an image message in the provider's format.

        Args:
            image_base64: Base64 encoded image string or list of strings
            description: Description text for the image(s)
            image_detail: Level of detail for image analysis (provider-specific, default: "low")

        Returns:
            Message in the provider's format (dict, object, etc.)
        """
        pass

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create provider instance from config. Override in subclasses for custom initialization."""
        provider_name = cls.__name__.lower().replace("provider", "")
        return cls(
            api_key=config.get_api_key(provider_name),
            base_url=config.get_base_url(provider_name),
            config=config,
        )

    def preprocess_messages(
        self, messages: List[Dict[str, Any]], model: str
    ) -> List[Dict[str, Any]]:
        """Preprocess messages for provider-specific requirements. Override in subclasses as needed."""
        return messages

    def parse_structured_response(self, content: str, response_format: Any) -> Any:
        """Parse and validate structured outputs."""
        if not response_format or not content:
            return content

        # Remove markdown formatting if present
        original_content = content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            # Parse JSON
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the content
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    # If all parsing fails, return original content
                    return original_content
            else:
                return original_content

        # Validate with Pydantic if applicable
        if hasattr(response_format, "__pydantic_model__") or hasattr(
            response_format, "model_validate"
        ):
            return response_format.model_validate(parsed)

        return parsed

    def calculate_token_usage(
        self, response
    ) -> Tuple[int, int, Optional[int], Optional[Dict[str, int]]]:
        """Calculate token usage including cached tokens."""
        input_tokens = 0
        cached_tokens = 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        output_tokens_details = None

        if (
            hasattr(response.usage, "prompt_tokens_details")
            and response.usage.prompt_tokens_details
        ):
            cached_tokens = (
                getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
            )
            input_tokens = (
                getattr(response.usage, "prompt_tokens", 0) or 0
            ) - cached_tokens
        else:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0

        if hasattr(response.usage, "completion_tokens_details"):
            output_tokens_details = response.usage.completion_tokens_details

        return input_tokens, output_tokens, cached_tokens, output_tokens_details

    async def execute_tool_calls_with_retry(
        self,
        tool_calls: List[Any],
        tool_dict: Dict[str, Callable],
        messages: List[Dict[str, Any]],
        post_tool_function: Optional[Callable] = None,
        consecutive_exceptions: int = 0,
        tool_handler: Optional[ToolHandler] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Tuple[List[Any], int]:
        """Common tool handling logic shared by all providers."""
        # Use provided tool_handler or fall back to self.tool_handler
        if tool_handler is None:
            tool_handler = self.tool_handler

        tool_outputs = []

        try:
            # Use provided parallel_tool_calls or fall back to config
            tool_outputs = await tool_handler.execute_tool_calls_batch(
                tool_calls,
                tool_dict,
                parallel_tool_calls=parallel_tool_calls,
                post_tool_function=post_tool_function,
            )
            consecutive_exceptions = 0
        except Exception as e:
            consecutive_exceptions += 1
            if consecutive_exceptions >= tool_handler.max_consecutive_errors:
                raise ProviderError(
                    self.get_provider_name(),
                    f"Tool execution failed after {consecutive_exceptions} consecutive errors: {str(e)}",
                )

            # Add error to messages for retry
            error_msg = f"{e}. Retries left: {tool_handler.max_consecutive_errors - consecutive_exceptions}"
            print(error_msg)
            messages.append({"role": "assistant", "content": str(e)})

        return tool_outputs, consecutive_exceptions

    def create_tool_handler_with_budget(
        self,
        tool_budget: Optional[Dict[str, int]] = None,
        image_result_keys: Optional[List[str]] = None,
        tool_output_max_tokens: Optional[int] = None,
    ) -> ToolHandler:
        """Create a ToolHandler instance with optional tool budget, image result keys, and output token limit."""
        if tool_budget or image_result_keys or tool_output_max_tokens is not None:
            return ToolHandler(
                tool_budget=tool_budget,
                image_result_keys=image_result_keys,
                tool_output_max_tokens=tool_output_max_tokens,
            )
        return self.tool_handler

    def filter_tools_by_budget(
        self, tools: Optional[List[Callable]], tool_handler: ToolHandler
    ) -> Optional[List[Callable]]:
        """Filter tools based on available budget."""
        if tools and tool_handler.tool_budget:
            return tool_handler.get_available_tools(tools)
        return tools

    def update_tools_with_budget(
        self,
        tools: Optional[List[Callable]],
        tool_handler: ToolHandler,
        request_params: Dict[str, Any],
        model: str,
    ) -> Tuple[Optional[List[Callable]], Dict[str, Callable]]:
        """Update available tools based on budget and rebuild parameters."""
        from ..utils_function_calling import get_function_specs

        if not tools or not tool_handler.tool_budget:
            return tools, tool_handler.build_tool_dict(tools) if tools else {}

        # Get available tools based on budget
        available_tools = tool_handler.get_available_tools(tools)

        if available_tools != tools:
            # Find which tools were removed
            removed_tools = []
            for tool in tools:
                if tool not in available_tools:
                    removed_tools.append(tool.__name__)

            # Tools have changed, update parameters
            if available_tools:
                # Rebuild tool specs with only available tools
                function_specs = get_function_specs(available_tools, model)
                request_params["tools"] = function_specs
                tool_dict = tool_handler.build_tool_dict(available_tools)
            else:
                # No tools available, remove tools from params
                request_params.pop("tools", None)
                request_params.pop("tool_choice", None)
                request_params.pop("parallel_tool_calls", None)
                tool_dict = {}

            # Add a note about removed tools to the last user message/input
            if removed_tools:
                budget_msg = (
                    "\n\nIMPORTANT: The following tools have exceeded their usage budget and are no longer available: "
                    + ", ".join(removed_tools)
                    + ". Do not attempt to call these tools."
                )

                if request_params.get("messages"):
                    # Chat Completions style: messages list
                    for i in range(len(request_params["messages"]) - 1, -1, -1):
                        if request_params["messages"][i].get("role") == "user":
                            if isinstance(
                                request_params["messages"][i]["content"], str
                            ):
                                request_params["messages"][i]["content"] += budget_msg
                            elif isinstance(
                                request_params["messages"][i]["content"], list
                            ):
                                request_params["messages"][i]["content"].append(
                                    {"type": "text", "text": budget_msg}
                                )
                            break
                elif request_params.get("input"):
                    # Responses style: input list with content parts
                    for i in range(len(request_params["input"]) - 1, -1, -1):
                        if request_params["input"][i].get("role") == "user":
                            content = request_params["input"][i].get("content")
                            if isinstance(content, list):
                                content.append(
                                    {"type": "input_text", "text": budget_msg}
                                )
                            elif isinstance(content, str):
                                request_params["input"][i]["content"] = [
                                    {"type": "input_text", "text": content + budget_msg}
                                ]
                            else:
                                request_params["input"][i]["content"] = [
                                    {"type": "input_text", "text": budget_msg}
                                ]
                            break

            return available_tools, tool_dict

        return tools, tool_handler.build_tool_dict(tools)

    def validate_post_response_hook(
        self, post_response_hook: Optional[Callable]
    ) -> None:
        """
        Verify that the post_response_hook function has the required parameters.
        """
        sig = inspect.signature(post_response_hook)
        required_params = ["response", "messages"]

        for param in required_params:
            if sig.parameters.get(param) is None:
                raise ValueError(
                    "post_response_hook must have parameter named `response` and `messages`."
                )

        return post_response_hook

    async def call_post_response_hook(
        self, post_response_hook: Callable, response, messages
    ):
        """Common function to call the post response hook if available. Shared by all providers."""
        try:
            if post_response_hook is None:
                return
            if inspect.iscoroutinefunction(post_response_hook):
                await post_response_hook(
                    response=response,
                    messages=messages,
                )
            else:
                post_response_hook(
                    response=response,
                    messages=messages,
                )
        except Exception as e:
            raise Exception(f"Error executing post_response_hook: {e}", e)
