class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    pass


class ProviderError(LLMError):
    """Exception raised when there's an error with a specific LLM provider."""

    def __init__(self, provider: str, message: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"Provider '{provider}': {message}")


class ToolError(LLMError):
    """Exception raised when there's an error with tool calling."""

    def __init__(self, tool_name: str, message: str, original_error: Exception = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}': {message}")


class MaxTokensError(LLMError):
    """Exception raised when maximum tokens are reached."""

    pass


class ConfigurationError(LLMError):
    """Exception raised when there's a configuration error."""

    pass


class AuthenticationError(LLMError):
    """Exception raised when there's an authentication error with a provider."""

    pass


class APIError(LLMError):
    """Exception raised when there's a general API error."""

    pass


class PauseToolExecution(BaseException):
    """Raised by a tool to suspend the agent loop for human-in-the-loop input.

    A tool raises this to pause ``chat_async`` mid-run (e.g. to ask the end
    user a clarifying question). ``chat_async`` stops the loop and returns an
    :class:`~defog.llm.providers.base.LLMResponse` with ``status="paused"``,
    the in-flight ``messages`` (with thinking blocks intact), the
    ``pending_tool_use`` that paused, and the ``pause_payload`` the tool
    passed. The caller persists those, collects the answer (possibly minutes
    later, on a different worker, or after a restart), and resumes via
    ``chat_async(..., messages=paused_messages, resume_tool_results={tool_use_id: answer})``.

    This subclasses ``BaseException`` (like ``asyncio.CancelledError``) on
    purpose: the tool-execution path is littered with ``except Exception``
    handlers that turn tool failures into retries or error strings, and a
    pause must not be swallowed by any of them. defog's tool-execution layer
    fills in ``tool_use_id`` / ``tool_name`` / ``tool_input`` (the tool itself
    does not know its own call id), and the provider fills in ``messages`` /
    ``pending_tool_use`` / ``response_id`` when it captures the conversation.

    Args:
        payload: Arbitrary, JSON-serializable data the tool wants to hand back
            to the application (e.g. the questions to ask the user). Surfaced
            as ``LLMResponse.pause_payload``.
    """

    def __init__(self, payload=None):
        self.payload = payload
        # Filled in by the tool-execution layer, which knows the call id/name.
        self.tool_use_id = None
        self.tool_name = None
        self.tool_input = None
        # Filled in by the provider when it captures the in-flight conversation.
        self.messages = None
        self.pending_tool_use = None
        self.response_id = None
        # Results of sibling tools that finished in the same batch, keyed by
        # call id. Used to preserve already-done work across the pause.
        self.completed_results = {}
        # Best-effort partial token usage at the pause point.
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.cache_creation_input_tokens = 0
        super().__init__("Tool execution paused for human-in-the-loop input")

    def attach_identity(self, tool_use_id, tool_name, tool_input):
        """Record which tool paused, if not already known."""
        if self.tool_use_id is None:
            self.tool_use_id = tool_use_id
            self.tool_name = tool_name
            self.tool_input = tool_input
        return self
