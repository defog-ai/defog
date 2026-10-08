"""Anthropic requests never carry `temperature`.

The Anthropic SDK removed `temperature` from messages.create() in 1.9.0, so
sending it raises "unexpected keyword argument 'temperature'". These tests
build request params without a live call and check the key is absent.
"""

import pytest

from defog.llm.providers.anthropic_provider import AnthropicProvider

MODELS = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-5-5",
    "claude-fable-5-1",
]


def _params(model, **kwargs):
    params, _ = AnthropicProvider(api_key="sk-test").build_params(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )
    return params


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
def test_no_temperature_without_thinking(model, temperature):
    params = _params(model, temperature=temperature, reasoning_effort=None)
    assert "temperature" not in params


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_no_temperature_with_thinking(model, effort):
    params = _params(model, temperature=0.0, reasoning_effort=effort)
    assert params["thinking"]["type"] != "disabled"
    assert "temperature" not in params


def test_opus_4_8_default_call_has_no_temperature():
    params = _params("claude-opus-4-8", temperature=0.0, reasoning_effort=None)
    assert "temperature" not in params
    assert params["model"] == "claude-opus-4-8"


def test_opus_4_8_low_effort_has_no_temperature():
    params = _params("claude-opus-4-8", temperature=0.0, reasoning_effort="low")
    assert "temperature" not in params
