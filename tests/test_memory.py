"""Tests for memory management and compactification with real API calls."""

import pytest
import os
from defog.llm.memory import (
    MemoryManager,
    ConversationHistory,
    TokenCounter,
    compactify_messages,
)
from defog.llm.utils_memory import (
    MemoryConfig,
    chat_async_with_memory,
    create_memory_manager,
)


class TestConversationHistory:
    """Test ConversationHistory class."""

    def test_add_message(self):
        history = ConversationHistory()
        message = {"role": "user", "content": "Hello"}

        history.add_message(message, tokens=10)

        assert len(history.messages) == 1
        assert history.messages[0] == message
        assert history.total_tokens == 10

    def test_get_messages(self):
        history = ConversationHistory()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        for msg in messages:
            history.add_message(msg, tokens=10)

        retrieved = history.get_messages()
        assert retrieved == messages
        # Ensure it's a copy, not the original
        retrieved[0]["content"] = "Modified"
        assert history.messages[0]["content"] == "Hello"

    def test_clear(self):
        history = ConversationHistory()
        history.add_message({"role": "user", "content": "Hello"}, tokens=10)

        history.clear()

        assert len(history.messages) == 0
        assert history.total_tokens == 0

    def test_replace_messages(self):
        history = ConversationHistory()
        history.add_message({"role": "user", "content": "Old"}, tokens=10)

        new_messages = [{"role": "system", "content": "Summary"}]
        history.replace_messages(new_messages, 50)

        assert len(history.messages) == 1
        assert history.messages[0]["content"] == "Summary"
        assert history.total_tokens == 50
        assert history.compactification_count == 1
        assert history.last_compactified_at is not None


class TestMemoryManager:
    """Test MemoryManager class."""

    def test_initialization(self):
        manager = MemoryManager(
            token_threshold=1000,
            preserve_last_n_messages=5,
            summary_max_tokens=100,
            enabled=True,
        )

        assert manager.token_threshold == 1000
        assert manager.preserve_last_n_messages == 5
        assert manager.summary_max_tokens == 100
        assert manager.enabled is True

    def test_update_after_compactification(self):
        manager = MemoryManager()

        # Add initial messages
        for i in range(5):
            manager.history.add_message(
                {"role": "user", "content": f"Message {i}"}, tokens=10
            )

        system_messages = [{"role": "system", "content": "You are helpful"}]
        summary = {
            "role": "user",
            "content": "[Previous conversation summary]\nSummary of conversation",
        }
        preserved = [{"role": "user", "content": "Recent message"}]

        manager.update_after_compactification(system_messages, summary, preserved, 100)

        assert len(manager.history.messages) == 3  # system + summary + preserved
        assert manager.history.messages[0] == system_messages[0]
        assert manager.history.messages[1] == summary
        assert manager.history.messages[2] == preserved[0]
        assert manager.history.total_tokens == 100
        assert manager.history.compactification_count == 1

    def test_should_compactify(self):
        manager = MemoryManager(token_threshold=100)

        # Should not compactify initially
        assert manager.should_compactify() is False

        # Add messages to exceed threshold
        manager.add_messages([{"role": "user", "content": "Hello"}], tokens=150)

        assert manager.should_compactify() is True

    def test_should_compactify_disabled(self):
        manager = MemoryManager(token_threshold=100, enabled=False)
        manager.add_messages([{"role": "user", "content": "Hello"}], tokens=150)

        assert manager.should_compactify() is False

    def test_get_messages_for_compactification(self):
        manager = MemoryManager(preserve_last_n_messages=2)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
        ]

        for msg in messages:
            manager.history.add_message(msg, tokens=10)

        system_msgs, to_summarize, to_preserve = (
            manager.get_messages_for_compactification()
        )

        assert len(system_msgs) == 1  # System message
        assert system_msgs[0]["content"] == "You are helpful"
        assert len(to_summarize) == 3  # First 3 non-system messages
        assert len(to_preserve) == 2  # Last 2 messages
        assert to_preserve[0]["content"] == "Response 2"
        assert to_preserve[1]["content"] == "Message 3"


class TestTokenCounter:
    """Test TokenCounter class."""

    def test_count_openai_tokens_string(self):
        counter = TokenCounter()
        text = "Hello, world! This is a test message."

        # Test with string input
        tokens = counter.count_openai_tokens(text, "gpt-4")
        assert tokens > 0
        assert isinstance(tokens, int)
        print(f"Tokens: {tokens}")
        # Rough check - this text should be around 8-10 tokens
        assert 5 <= tokens <= 15

    def test_count_openai_tokens_messages(self):
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        tokens = counter.count_openai_tokens(messages, "gpt-4")
        print(f"Tokens: {tokens}")
        assert tokens > 0

        # Should be more than just the content tokens due to message formatting
        text_only = "You are helpful Hello Hi there!"
        text_tokens = counter.count_openai_tokens(text_only, "gpt-4")
        print(f"Text Tokens: {text_tokens}")
        assert tokens > text_tokens

    def test_count_tokens_different_providers(self):
        counter = TokenCounter()
        messages = [{"role": "user", "content": "Hello, how are you doing today?"}]

        # All providers should use OpenAI tokenizer as approximation
        openai_tokens = counter.count_tokens(messages, "gpt-4", "openai")
        anthropic_tokens = counter.count_tokens(
            messages, "claude-haiku-4-5", "anthropic"
        )
        gemini_tokens = counter.count_tokens(messages, "gemini-pro", "gemini")
        together_tokens = counter.count_tokens(messages, "llama-2", "together")

        # All should return the same count since they use OpenAI tokenizer
        assert openai_tokens == anthropic_tokens
        assert openai_tokens == gemini_tokens
        assert openai_tokens == together_tokens
        assert openai_tokens > 0

    def test_estimate_remaining_tokens(self):
        counter = TokenCounter()
        messages = [{"role": "user", "content": "Hello " * 100}]

        remaining = counter.estimate_remaining_tokens(
            messages, "gpt-4", "openai", max_context_tokens=1000, response_buffer=100
        )

        assert remaining < 900  # Should have used some tokens
        assert remaining >= 0  # Should not be negative


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
class TestCompactifyMessagesReal:
    """Test message compactification with real API calls."""

    @pytest.mark.asyncio
    async def test_compactify_messages_empty(self):
        """Test compactifying with no messages to summarize."""
        system_messages = [{"role": "system", "content": "You are a helpful assistant"}]
        preserved = [{"role": "user", "content": "What's the capital of France?"}]

        result, tokens = await compactify_messages(
            system_messages=system_messages,
            messages_to_summarize=[],
            preserved_messages=preserved,
            provider="anthropic",
            model="claude-haiku-4-5",
            max_summary_tokens=500,
        )

        assert result == system_messages + preserved
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_compactify_messages_with_summary(self):
        """Test compactifying with messages to summarize using real API."""
        system_messages = []  # No system messages in this test
        to_summarize = [
            {"role": "user", "content": "What's the weather like in Paris?"},
            {
                "role": "assistant",
                "content": "I don't have access to real-time weather data, but Paris typically has mild weather.",
            },
            {"role": "user", "content": "What about in summer?"},
            {
                "role": "assistant",
                "content": "Paris summers are usually warm, with temperatures around 20-25°C (68-77°F).",
            },
        ]
        preserved = [{"role": "user", "content": "Should I pack a jacket for July?"}]

        result, tokens = await compactify_messages(
            system_messages=system_messages,
            messages_to_summarize=to_summarize,
            preserved_messages=preserved,
            provider="anthropic",
            model="claude-haiku-4-5",
            max_summary_tokens=200,
        )

        # Should have summary + preserved messages
        assert len(result) == 2
        assert result[0]["role"] == "user"  # Summary is now a user message
        assert "[Previous conversation summary]" in result[0]["content"]
        assert result[1] == preserved[0]

        # Summary should mention weather/Paris
        summary_content = result[0]["content"].lower()
        assert "weather" in summary_content or "paris" in summary_content
        assert tokens > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestChatAsyncWithMemoryReal:
    """Test chat_async_with_memory with real API calls."""

    @pytest.mark.asyncio
    async def test_simple_conversation_no_compactification(self):
        """Test a simple conversation that doesn't need compactification."""
        manager = MemoryManager(token_threshold=10000)  # High threshold

        # First message
        response1 = await chat_async_with_memory(
            provider="openai",
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            memory_manager=manager,
        )

        assert response1.content is not None
        assert "4" in response1.content or "four" in response1.content.lower()

        # Check memory state
        assert len(manager.history.messages) == 2  # user + assistant
        assert manager.history.messages[0]["content"] == "What is 2+2?"
        assert manager.history.messages[1]["content"] == response1.content

        # Second message
        response2 = await chat_async_with_memory(
            provider="openai",
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "What about 3+3?"}],
            memory_manager=manager,
        )

        assert response2.content is not None
        assert "6" in response2.content or "six" in response2.content.lower()

        # Check memory has all messages
        assert len(manager.history.messages) == 4  # 2 user + 2 assistant
        assert manager.history.compactification_count == 0

    @pytest.mark.asyncio
    async def test_conversation_with_compactification(self):
        """Test a conversation that triggers compactification."""
        # Use very low threshold to force compactification
        manager = MemoryManager(
            token_threshold=100,  # Very low threshold
            preserve_last_n_messages=2,
            summary_max_tokens=200,
        )

        # Add some conversation history to trigger compactification
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about Python programming."},
            {
                "role": "assistant",
                "content": "Python is a high-level programming language known for its simplicity.",
            },
            {"role": "user", "content": "What are its main features?"},
            {
                "role": "assistant",
                "content": "Python features include dynamic typing, garbage collection, and extensive libraries.",
            },
            {"role": "user", "content": "How do I install it?"},
            {
                "role": "assistant",
                "content": "You can install Python from python.org or using package managers.",
            },
        ]

        # Add messages to history manually to simulate prior conversation
        for msg in conversation:
            manager.history.add_message(msg, tokens=50)

        # This should trigger compactification
        response = await chat_async_with_memory(
            provider="openai",
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "What's the latest version?"}],
            memory_manager=manager,
        )

        assert response.content is not None

        # Check that compactification occurred
        assert manager.history.compactification_count == 1

        # Check messages structure after compactification
        messages = manager.history.get_messages()

        # Should have system + summary + preserved + new exchange
        # Check that system message is preserved
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."

        # Find the summary message
        summary_found = False
        for msg in messages:
            if (
                msg["role"] == "user"
                and "[Previous conversation summary]" in msg["content"]
            ):
                summary_found = True
                # Summary should mention Python
                assert "python" in msg["content"].lower()
                break

        assert summary_found, "Summary message not found after compactification"


class TestMemoryConfig:
    """Test MemoryConfig dataclass."""

    def test_default_values(self):
        config = MemoryConfig()

        assert config.enabled is True
        assert config.token_threshold == 50000
        assert config.preserve_last_n_messages == 10
        assert config.summary_max_tokens == 4000
        assert config.max_context_tokens == 128000

    def test_custom_values(self):
        config = MemoryConfig(
            enabled=False,
            token_threshold=25000,
            preserve_last_n_messages=5,
            summary_max_tokens=1000,
            max_context_tokens=64000,
        )

        assert config.enabled is False
        assert config.token_threshold == 25000
        assert config.preserve_last_n_messages == 5
        assert config.summary_max_tokens == 1000
        assert config.max_context_tokens == 64000


class TestCreateMemoryManager:
    """Test create_memory_manager helper function."""

    def test_create_with_defaults(self):
        manager = create_memory_manager()

        assert isinstance(manager, MemoryManager)
        assert manager.token_threshold == 100000
        assert manager.preserve_last_n_messages == 10
        assert manager.summary_max_tokens == 2000
        assert manager.enabled is True

    def test_create_with_custom_values(self):
        manager = create_memory_manager(
            token_threshold=25000,
            preserve_last_n_messages=5,
            summary_max_tokens=1000,
            enabled=False,
        )

        assert manager.token_threshold == 25000
        assert manager.preserve_last_n_messages == 5
        assert manager.summary_max_tokens == 1000
        assert manager.enabled is False


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestLongConversationWithMemory:
    """Test a long conversation that triggers multiple compactifications."""

    @pytest.mark.asyncio
    async def test_long_conversation_multiple_compactifications(self):
        """Test a realistic long conversation with multiple compactifications."""
        # Create memory manager with low threshold to trigger multiple compactifications
        manager = create_memory_manager(
            token_threshold=3000,  # Low threshold for testing
            preserve_last_n_messages=4,
            summary_max_tokens=500,
        )

        # Add a system message
        system_message = {
            "role": "system",
            "content": "You are a helpful programming assistant.",
        }
        manager.history.add_message(system_message, tokens=10)

        # Simulate a long conversation about building a web application
        conversation_topics = [
            # Phase 1: Project setup (will trigger first compactification)
            "I want to build a todo list web application. What technologies should I use?",
            "Can you explain more about React and why it's good for this project?",
            "What about the backend? Should I use Node.js or Python?",
            "How do I set up a React project with TypeScript?",
            # Phase 2: Development (will trigger second compactification)
            "Can you show me how to create a Todo component in React?",
            "How do I connect to a PostgreSQL database?",
            "How do I implement CRUD operations for todos?",
            # Phase 3: Final steps
            "How do I implement user authentication?",
            "What's the best way to deploy this application?",
        ]

        compactification_count = 0

        for i, question in enumerate(conversation_topics):
            print(f"Question {i}: {question}")
            # Make API call
            response = await chat_async_with_memory(
                provider="openai",
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": question}],
                memory_manager=manager,
                temperature=0.3,
            )

            assert response.content is not None
            assert len(response.content) > 0

            # Check memory state
            current_messages = manager.history.get_messages()
            current_compactifications = manager.history.compactification_count

            # Verify system message is always preserved first
            assert current_messages[0]["role"] == "system"
            assert current_messages[0]["content"] == system_message["content"]

            print(f"Current messages: {current_messages}")

            # Check if compactification occurred
            if current_compactifications > compactification_count:
                # Find and verify summary message
                summary_found = False
                for msg in current_messages:
                    if (
                        msg["role"] == "user"
                        and "[Previous conversation summary]" in msg["content"]
                    ):
                        summary_found = True
                        summary_content = msg["content"].lower()

                        # Verify summary contains relevant keywords from conversation
                        # Just check that it mentions some key concepts
                        relevant_words = [
                            "todo",
                            "react",
                            "backend",
                            "api",
                            "database",
                            "component",
                            "frontend",
                            "express",
                            "deploy",
                            "authentication",
                        ]
                        assert any(
                            word in summary_content for word in relevant_words
                        ), (
                            f"Summary doesn't contain relevant keywords: {summary_content[:200]}"
                        )
                        break

                assert summary_found, (
                    f"Summary not found after compactification {current_compactifications}"
                )
                compactification_count = current_compactifications

        # Final assertions
        assert (
            manager.history.compactification_count >= 2
        )  # Should have at least 2 compactifications

        # Verify final state
        final_messages = manager.history.get_messages()
        assert final_messages[0]["role"] == "system"  # System message preserved

        # Should have summary + preserved recent messages
        summary_count = sum(
            1
            for msg in final_messages
            if "[Previous conversation summary]" in msg.get("content", "")
        )
        assert summary_count >= 1  # At least one summary

        # Verify we don't have too many messages (compactification worked)
        assert len(final_messages) <= 10  # System + summary + preserved messages

    @pytest.mark.asyncio
    async def test_memory_disabled_no_compactification(self):
        """Test that memory management can be disabled."""
        # Create manager with memory disabled
        manager = create_memory_manager(
            token_threshold=100,
            enabled=False,  # Very low threshold  # Disabled
        )

        # Add many messages
        for i in range(10):
            response = await chat_async_with_memory(
                provider="openai",
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"Question {i}: What is {i} + {i}?"}
                ],
                memory_manager=manager,
            )

            assert response.content is not None

        # No compactification should have occurred
        assert manager.history.compactification_count == 0
        assert len(manager.history.messages) == 20  # 10 user + 10 assistant messages
