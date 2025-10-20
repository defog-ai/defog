import os
import pytest
import logging


# Set up colored logging for test skips
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


# Configure logger for test skips with respect to pytest verbosity
skip_logger = logging.getLogger("test_skip")
skip_logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))
skip_logger.addHandler(handler)


def _should_log_skips() -> bool:
    """Check if we should log test skips based on pytest verbosity."""
    import sys

    # Check for pytest verbosity flags
    verbose_flags = ["-v", "--verbose", "-s", "--capture=no"]
    return any(flag in sys.argv for flag in verbose_flags)


# Check which API keys are available
AVAILABLE_PROVIDERS = {}
AVAILABLE_MODELS = {}

# Check API keys and populate available providers
if os.getenv("ANTHROPIC_API_KEY"):
    AVAILABLE_PROVIDERS["anthropic"] = True
    AVAILABLE_MODELS["anthropic"] = [
        "claude-3-7-sonnet-latest",
        "claude-haiku-4-5",
    ]
else:
    AVAILABLE_PROVIDERS["anthropic"] = False
    AVAILABLE_MODELS["anthropic"] = []

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_PROVIDERS["openai"] = True
    AVAILABLE_MODELS["openai"] = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    ]
else:
    AVAILABLE_PROVIDERS["openai"] = False
    AVAILABLE_MODELS["openai"] = []

if os.getenv("GEMINI_API_KEY"):
    AVAILABLE_PROVIDERS["gemini"] = True
    AVAILABLE_MODELS["gemini"] = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
else:
    AVAILABLE_PROVIDERS["gemini"] = False
    AVAILABLE_MODELS["gemini"] = []

if os.getenv("DEEPSEEK_API_KEY"):
    AVAILABLE_PROVIDERS["deepseek"] = True
    AVAILABLE_MODELS["deepseek"] = ["deepseek-chat", "deepseek-reasoner"]
else:
    AVAILABLE_PROVIDERS["deepseek"] = False
    AVAILABLE_MODELS["deepseek"] = []

if os.getenv("MISTRAL_API_KEY"):
    AVAILABLE_PROVIDERS["mistral"] = True
    AVAILABLE_MODELS["mistral"] = [
        "mistral-small-latest",
        "mistral-medium-latest",
        "mistral-large-latest",
    ]
else:
    AVAILABLE_PROVIDERS["mistral"] = False
    AVAILABLE_MODELS["mistral"] = []

if os.getenv("TOGETHER_API_KEY"):
    AVAILABLE_PROVIDERS["together"] = True
    AVAILABLE_MODELS["together"] = []
else:
    AVAILABLE_PROVIDERS["together"] = False
    AVAILABLE_MODELS["together"] = []

if os.getenv("ALIBABA_API_KEY"):
    AVAILABLE_PROVIDERS["alibaba"] = True
    AVAILABLE_MODELS["alibaba"] = ["qwen-max", "qwen-plus", "qwen-turbo"]
else:
    AVAILABLE_PROVIDERS["alibaba"] = False
    AVAILABLE_MODELS["alibaba"] = []

if os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY"):
    AVAILABLE_PROVIDERS["grok"] = True
    AVAILABLE_MODELS["grok"] = [
        "grok-4",
        "grok-4-fast-non-reasoning-latest",
        "grok-4-fast-reasoning-latest",
    ]
else:
    AVAILABLE_PROVIDERS["grok"] = False
    AVAILABLE_MODELS["grok"] = []


def get_available_models():
    """Get all available models across all providers with API keys."""
    models = []
    for provider_models in AVAILABLE_MODELS.values():
        models.extend(provider_models)
    return models


def skip_if_no_api_key(provider: str):
    """Decorator to skip test if API key for provider is not available."""

    def decorator(test_func):
        if not AVAILABLE_PROVIDERS.get(provider, False) and _should_log_skips():
            skip_logger.warning(
                f"🚫 Skipping {test_func.__name__} - No API key for {provider} provider"
            )
        return pytest.mark.skipif(
            not AVAILABLE_PROVIDERS.get(provider, False),
            reason=f"No API key for {provider} provider",
        )(test_func)

    return decorator


def skip_if_no_models():
    """Decorator to skip test if no models are available (no API keys)."""

    def decorator(test_func):
        if not any(AVAILABLE_PROVIDERS.values()) and _should_log_skips():
            skip_logger.warning(
                f"🚫 Skipping {test_func.__name__} - No API keys available for any provider"
            )
        return pytest.mark.skipif(
            not any(AVAILABLE_PROVIDERS.values()),
            reason="No API keys available for any provider",
        )(test_func)

    return decorator


# Fixtures for commonly used test data
@pytest.fixture
def available_providers():
    """Return dictionary of available providers."""
    return {k: v for k, v in AVAILABLE_PROVIDERS.items() if v}


@pytest.fixture
def available_models():
    """Return list of all available models."""
    return get_available_models()


@pytest.fixture
def available_models_by_provider():
    """Return dictionary of available models by provider."""
    return {k: v for k, v in AVAILABLE_MODELS.items() if v}
