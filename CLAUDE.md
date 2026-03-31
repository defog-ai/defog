# General Behaviour
- Do not be obsequious. Do not say "You are absolutely right" or "You are right". You do not have to please the user – you just need to make the code work.
- Do NOT use emotional words or exclaimations like "Perfect!", or "Great!". Be more stoic

# Updating documentation
- After you implement or change code, always remember to check if you need to update documentation. Documentation is in `README.md` and in the `docs/` folder

# Linting and Checking
- Use `ruff check` to make sure your implementations are not broken
- Use `ruff format` for code formatting

# Reviewing PRs
- Be honest, technically focused, and blunt. Pay special attention to security issues.

# Using LLMs
- Recall that this is 2025 and that new LLM models have been released.
  - Claude 4 Sonnet and Claude 4 Opus are now valid models
  - OpenAI's gpt-4.1, o3, and o4-mini are now valid models
  - Gemini's 2.5 flash and 2.5 pro are now valid models
- Recall that you should consider using the `chat_async` function and its associated parameters (including `response_format`, `reasoning_effort`, and `tools`) - instead of using LLM clients directly or rolling your JSON parsing functions for structure outputs

# Testing
- Remember to run all tests with `python -m pytest ...`. This is to ensure you are using the correct version of python and pytest
- Add `PYTHONPATH=.` when running tests, so that the tests/examples use the version of defog in this repo - instead of the machine installed version
- Use the `--envfile .env` argument to load environment variables (API keys etc.) when running tests
- Try to run scoped tests that directly affect changed code instead of running all tests at once