"""
Tests for image support in tool results.

Tests the enhanced chat_async functionality that detects image data
in tool results and injects them as separate messages.
"""

import pytest
import base64
import logging
from io import BytesIO
from PIL import Image, ImageDraw
from typing import Optional
from pydantic import BaseModel, Field
import base64

from defog.llm.providers.gemini_provider import GeminiProvider
from defog.llm.providers.anthropic_provider import AnthropicProvider
from defog.llm.providers.openai_provider import OpenAIProvider
from defog.llm.providers.deepseek_provider import DeepSeekProvider
from defog.llm.utils_image_support import (
    detect_image_in_result,
    process_tool_results_with_images,
    validate_base64_image,
    detect_image_format,
    validate_and_process_image_data,
    safe_extract_media_type_and_data,
    MAX_IMAGE_SIZE_BYTES,
)


def create_test_image(text: str = "TEST") -> str:
    """Create a simple test image and return as base64."""
    img = Image.new("RGB", (100, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(0, 0, 0))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_multi_part_images() -> list[str]:
    """Create multiple test images to simulate multi-part screenshots."""
    return [create_test_image("PART 1"), create_test_image("PART 2")]


class MockToolResult(BaseModel):
    """Mock tool result for testing."""

    result: str = Field(description="Text result")
    image_base64: Optional[str | list[str]] = Field(
        None, description="Base64 image data - can be single string or list of strings"
    )
    other_field: Optional[str] = Field(None, description="Other data")


class MockToolBlock:
    """Mock tool call block for testing."""

    def __init__(self, tool_id: str, name: str):
        self.id = tool_id
        self.name = name


class TestImageDetection:
    """Test image detection in tool results."""

    def test_correct_image_key(self):
        """Test that image is detected with correct key."""
        image_data = create_test_image()
        result = MockToolResult(result="test", image_base64=image_data)

        text, image = detect_image_in_result(result, ["image_base64"])
        assert image == image_data
        assert text == "test"  # Should return the cleaned result field

    def test_multiple_keys(self):
        """Test detection with multiple possible keys."""
        image_data = create_test_image()
        result = MockToolResult(result="test", image_base64=image_data)

        text, image = detect_image_in_result(
            result, ["wrong_key", "image_base64", "other_key"]
        )
        assert image == image_data

    def test_multi_part_image_detection(self):
        """Test detection of multi-part images (list of base64 strings)."""
        image_list = create_multi_part_images()
        result = MockToolResult(result="multi-part screenshot", image_base64=image_list)

        text, image = detect_image_in_result(result, ["image_base64"])
        assert image == image_list
        assert text == "multi-part screenshot"


class TestToolResultProcessing:
    """Test processing of tool results with images."""

    def test_single_image(self):
        """Test processing tool results with one image."""
        image_data = create_test_image()
        tool_blocks = [MockToolBlock("1", "screenshot_tool")]
        results = [MockToolResult(result="took screenshot", image_base64=image_data)]

        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )

        assert len(tool_data_list) == 1
        tool_data = tool_data_list[0]

        # Check tool data
        assert tool_data.tool_id == "1"
        assert tool_data.tool_name == "screenshot_tool"
        assert (
            tool_data.tool_result_text == "took screenshot"
        )  # Should return cleaned text
        assert tool_data.image_data == image_data

    def test_mixed_results(self):
        """Test processing mix of results with and without images."""
        image_data = create_test_image()
        tool_blocks = [
            MockToolBlock("1", "text_tool"),
            MockToolBlock("2", "image_tool"),
        ]
        results = [
            MockToolResult(result="text only"),
            MockToolResult(result="with image", image_base64=image_data),
        ]

        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )

        assert len(tool_data_list) == 2

        # First tool should have no image
        assert tool_data_list[0].tool_name == "text_tool"
        assert tool_data_list[0].image_data is None

        # Second tool should have image
        assert tool_data_list[1].tool_name == "image_tool"
        assert tool_data_list[1].image_data == image_data

    def test_multi_part_screenshot(self):
        """Test processing tool results with multi-part images."""
        image_list = create_multi_part_images()
        tool_blocks = [MockToolBlock("1", "screenshot_tool")]
        results = [
            MockToolResult(
                result="Page was 15000px tall - split into 2 screenshot segments",
                image_base64=image_list,
            )
        ]

        tool_data_list = process_tool_results_with_images(
            tool_blocks, results, ["image_base64"]
        )

        assert len(tool_data_list) == 1
        tool_data = tool_data_list[0]

        # Check tool data
        assert tool_data.tool_id == "1"
        assert tool_data.tool_name == "screenshot_tool"
        assert "split into 2 screenshot segments" in tool_data.tool_result_text

        # Check that image data is the list of images
        assert tool_data.image_data == image_list
        assert isinstance(tool_data.image_data, list)
        assert len(tool_data.image_data) == 2


class TestProviderMessageCreation:
    """Test provider-specific image message creation."""

    def test_anthropic_image_message_creation(self):
        """Test that Anthropic provider creates correct image messages."""

        provider = AnthropicProvider(api_key="test")
        image_data = create_test_image()

        # Test single image
        msg = provider.create_image_message(image_data, "Test image")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Test image"
        assert msg["content"][1]["type"] == "image"
        assert msg["content"][1]["source"]["type"] == "base64"
        assert msg["content"][1]["source"]["data"] == image_data

        # Test multiple images
        image_list = create_multi_part_images()
        msg = provider.create_image_message(image_list, "Multi-part screenshot")
        assert len(msg["content"]) == 3  # text + 2 images
        assert msg["content"][1]["source"]["data"] == image_list[0]
        assert msg["content"][2]["source"]["data"] == image_list[1]

    def test_openai_image_message_creation(self):
        """Test that OpenAI provider creates correct image messages."""

        provider = OpenAIProvider(api_key="test")
        image_data = create_test_image()

        # Test single image
        msg = provider.create_image_message(image_data, "Test image")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Test image"
        assert msg["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in msg["content"][1]["image_url"]["url"]
        assert image_data in msg["content"][1]["image_url"]["url"]

        # Test multiple images
        image_list = create_multi_part_images()
        msg = provider.create_image_message(image_list, "Multi-part screenshot")
        assert len(msg["content"]) == 3  # text + 2 images
        for i, img_data in enumerate(image_list):
            assert img_data in msg["content"][i + 1]["image_url"]["url"]

    def test_gemini_image_message_creation(self):
        """Test that Gemini provider creates correct image messages."""

        provider = GeminiProvider(api_key="test")
        image_data = create_test_image()

        # Test single image
        msg = provider.create_image_message(image_data, "Test image")
        # Gemini returns a Content object, not a dict
        assert msg.role == "user"
        assert len(msg.parts) == 2
        assert msg.parts[0].text == "Test image"
        assert hasattr(msg.parts[1], "inline_data")  # Image part
        assert msg.parts[1].inline_data.mime_type == "image/png"

        expected_bytes = base64.b64decode(image_data)
        assert msg.parts[1].inline_data.data == expected_bytes


class TestImageValidation:
    """Test image validation functionality."""

    def test_validate_base64_image_valid_png(self):
        """Test validation of valid PNG image."""
        image_data = create_test_image()
        is_valid, error = validate_base64_image(image_data)
        assert is_valid is True
        assert error is None

    def test_validate_base64_image_invalid_data(self):
        """Test validation with invalid base64 data."""
        is_valid, error = validate_base64_image("invalid_base64!")
        assert is_valid is False
        assert "Invalid base64 encoding" in error

    def test_validate_base64_image_empty_string(self):
        """Test validation with empty string."""
        is_valid, error = validate_base64_image("")
        assert is_valid is False
        assert "non-empty string" in error

    def test_validate_base64_image_non_string(self):
        """Test validation with non-string input."""
        is_valid, error = validate_base64_image(123)
        assert is_valid is False
        assert "must be a non-empty string" in error

    def test_validate_base64_image_too_large(self):
        """Test validation with oversized image."""
        # Create a valid but oversized image by repeating a valid base64 string
        small_image = create_test_image()
        # Repeat the image data to make it larger than 20MB
        large_data = small_image * 100000  # This should exceed the size limit
        is_valid, error = validate_base64_image(large_data)
        assert is_valid is False
        # It might fail on size limit OR format validation, both are acceptable
        assert (
            "exceeds" in error and "MB limit" in error
        ) or "Unrecognized image format" in error

    def test_validate_base64_image_with_data_uri(self):
        """Test validation with data URI prefix."""
        image_data = create_test_image()
        data_uri = f"data:image/png;base64,{image_data}"
        is_valid, error = validate_base64_image(data_uri)
        assert is_valid is True
        assert error is None

    def test_validate_base64_image_malformed_data_uri(self):
        """Test validation with malformed data URI."""
        is_valid, error = validate_base64_image("data:malformed")
        assert is_valid is False
        assert "Malformed data URI prefix" in error

    def test_detect_image_format_png(self):
        """Test PNG format detection."""
        image_data = create_test_image()
        decoded = base64.b64decode(image_data)
        format_type = detect_image_format(decoded)
        assert format_type == "image/png"

    def test_detect_image_format_unknown(self):
        """Test format detection with unknown format."""
        unknown_data = b"unknown binary data"
        format_type = detect_image_format(unknown_data)
        assert format_type is None

    def test_detect_image_format_too_short(self):
        """Test format detection with insufficient data."""
        short_data = b"abc"
        format_type = detect_image_format(short_data)
        assert format_type is None

    def test_validate_and_process_image_data_single_valid(self):
        """Test processing single valid image."""
        image_data = create_test_image()
        valid_images, errors = validate_and_process_image_data(image_data)
        assert len(valid_images) == 1
        assert len(errors) == 0
        assert valid_images[0] == image_data

    def test_validate_and_process_image_data_list_mixed(self):
        """Test processing list with valid and invalid images."""
        valid_image = create_test_image()
        invalid_image = "invalid_base64!"
        image_list = [valid_image, invalid_image]

        valid_images, errors = validate_and_process_image_data(image_list)
        assert len(valid_images) == 1
        assert len(errors) == 1
        assert valid_images[0] == valid_image

    def test_validate_and_process_image_data_all_invalid(self):
        """Test processing list with all invalid images."""
        invalid_images = ["invalid1", "invalid2"]
        valid_images, errors = validate_and_process_image_data(invalid_images)
        assert len(valid_images) == 0
        assert len(errors) == 2

    def test_safe_extract_media_type_and_data_with_uri(self):
        """Test extracting media type from data URI."""
        image_data = create_test_image()
        data_uri = f"data:image/png;base64,{image_data}"
        media_type, clean_data = safe_extract_media_type_and_data(data_uri)
        assert media_type == "image/png"
        assert clean_data == image_data

    def test_safe_extract_media_type_and_data_without_uri(self):
        """Test extracting media type from raw base64."""
        image_data = create_test_image()
        media_type, clean_data = safe_extract_media_type_and_data(image_data)
        assert media_type == "image/png"  # Should detect PNG
        assert clean_data == image_data

    def test_safe_extract_media_type_and_data_invalid(self):
        """Test extracting media type from invalid data."""
        media_type, clean_data = safe_extract_media_type_and_data("invalid")
        assert media_type == "image/jpeg"  # Fallback
        assert clean_data == "invalid"


class TestProviderImageMessageValidation:
    """Test that providers properly validate images."""

    def test_anthropic_provider_invalid_image(self):
        """Test Anthropic provider with invalid image data."""

        provider = AnthropicProvider(api_key="test")

        with pytest.raises(ValueError) as exc_info:
            provider.create_image_message("invalid_base64!", "Test")

        assert "Cannot create image message" in str(exc_info.value)

    def test_openai_provider_invalid_image(self):
        """Test OpenAI provider with invalid image data."""

        provider = OpenAIProvider(api_key="test")

        with pytest.raises(ValueError) as exc_info:
            provider.create_image_message("invalid_base64!", "Test")

        assert "Cannot create image message" in str(exc_info.value)

    def test_gemini_provider_invalid_image(self):
        """Test Gemini provider with invalid image data."""

        provider = GeminiProvider(api_key="test")

        with pytest.raises(ValueError) as exc_info:
            provider.create_image_message("invalid_base64!", "Test")

        assert "Cannot create image message" in str(exc_info.value)

    def test_deepseek_provider_handles_invalid_image(self):
        """Test DeepSeek provider gracefully handles invalid images."""

        provider = DeepSeekProvider(api_key="test")

        # DeepSeek should handle invalid images gracefully (just log warnings)
        msg = provider.create_image_message("invalid_base64!", "Test")
        assert msg["role"] == "user"
        assert "Test" in msg["content"]

    def test_provider_partial_validation_success(self):
        """Test provider behavior with mixed valid/invalid images."""

        provider = AnthropicProvider(api_key="test")
        valid_image = create_test_image()
        mixed_images = [valid_image, "invalid_base64!"]

        # Should succeed with valid image, log warning for invalid
        msg = provider.create_image_message(mixed_images, "Mixed test")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2  # text + 1 valid image
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image"

    def test_openai_provider_invalid_image_detail(self):
        """Test OpenAI provider with invalid image_detail parameter."""

        provider = OpenAIProvider(api_key="test")
        valid_image = create_test_image()

        # Test invalid image_detail value
        with pytest.raises(ValueError) as exc_info:
            provider.create_image_message(valid_image, "Test", image_detail="invalid")

        assert "Invalid image_detail value" in str(exc_info.value)
        assert "Must be 'low' or 'high'" in str(exc_info.value)


class TestImageValidationEdgeCases:
    """Comprehensive edge case tests for image validation."""

    def test_empty_base64_data(self):
        """Test validation with empty base64 data."""
        is_valid, error = validate_base64_image("")
        assert is_valid is False
        assert "non-empty string" in error

    def test_none_input(self):
        """Test validation with None input."""
        is_valid, error = validate_base64_image(None)
        assert is_valid is False
        assert "non-empty string" in error

    def test_whitespace_only_base64(self):
        """Test validation with whitespace-only string."""
        is_valid, error = validate_base64_image("   \n\t   ")
        assert is_valid is False
        assert "Invalid base64 encoding" in error

    def test_malformed_data_uri_missing_comma(self):
        """Test malformed data URI without comma separator."""
        is_valid, error = validate_base64_image("data:image/png;base64")
        assert is_valid is False
        assert "Malformed data URI prefix" in error

    def test_malformed_data_uri_missing_base64(self):
        """Test malformed data URI without base64 marker."""
        is_valid, error = validate_base64_image("data:image/png,somedata")
        assert is_valid is False
        # Could fail on either base64 decoding or format recognition
        assert (
            "Invalid base64 encoding" in error or "Unrecognized image format" in error
        )

    def test_corrupted_base64_padding(self):
        """Test base64 with incorrect padding."""
        is_valid, error = validate_base64_image("YWJjZGVmZ2hp=")  # Missing padding
        assert is_valid is False
        assert "Invalid" in error  # Will fail format check

    def test_valid_base64_but_not_image(self):
        """Test valid base64 that doesn't contain image data."""
        text_base64 = base64.b64encode(b"This is plain text").decode()
        is_valid, error = validate_base64_image(text_base64)
        assert is_valid is False
        assert "Unrecognized image format" in error

    def test_truncated_image_data(self):
        """Test image data that's too short to be valid."""
        truncated = base64.b64encode(b"\xff\xd8").decode()  # Just JPEG start
        is_valid, error = validate_base64_image(truncated)
        assert is_valid is False
        assert "Unrecognized image format" in error

    def test_media_type_mismatch_warning(self, caplog):
        """Test warning when data URI media type doesn't match detected format."""
        # Create a PNG but label it as JPEG
        png_data = create_test_image()
        mislabeled = f"data:image/jpeg;base64,{png_data}"

        with caplog.at_level(logging.WARNING):
            is_valid, error = validate_base64_image(mislabeled)

        assert is_valid is True
        assert error is None
        assert "Media type mismatch" in caplog.text

    def test_extremely_large_base64_string(self):
        """Test handling of extremely large base64 strings."""
        # Create a string larger than MAX_IMAGE_SIZE_BYTES
        large_data = "A" * (MAX_IMAGE_SIZE_BYTES * 2)  # Base64 expands by ~4/3
        is_valid, error = validate_base64_image(large_data)
        assert is_valid is False
        # Should fail on size or format validation
        assert (
            "exceeds" in error and "MB limit" in error
        ) or "Unrecognized image format" in error

    def test_special_characters_in_base64(self):
        """Test base64 with special characters that should be invalid."""
        is_valid, error = validate_base64_image("YWJj@#$%^&*")
        assert is_valid is False
        assert "Invalid base64 encoding" in error

    def test_webp_format_detection(self):
        """Test proper WebP format detection."""
        # Minimal valid WebP header
        webp_header = b"RIFF\x00\x00\x00\x00WEBPVP8 "
        webp_base64 = base64.b64encode(webp_header).decode()

        # Should pass basic validation but might fail on complete image validation
        decoded = base64.b64decode(webp_base64)
        format_type = detect_image_format(decoded)
        assert format_type == "image/webp"

    def test_process_image_data_with_mixed_types(self):
        """Test processing image data with mixed valid types."""
        valid_png = create_test_image()
        valid_images, errors = validate_and_process_image_data(
            [
                valid_png,
                123,  # Invalid type
                None,  # Invalid type
                "",  # Empty string
                valid_png,  # Another valid one
            ]
        )

        assert len(valid_images) == 2
        assert len(errors) == 3
        assert any("must be string" in e for e in errors)
        assert any("non-empty string" in e for e in errors)

    def test_safe_extract_with_multiple_semicolons(self):
        """Test media type extraction with complex data URI."""
        complex_uri = "data:image/png;charset=utf-8;base64,iVBORw0KGgo="
        media_type, data = safe_extract_media_type_and_data(complex_uri)
        assert media_type == "image/png"
        assert data == "iVBORw0KGgo="

    def test_safe_extract_fallback_behavior(self):
        """Test fallback behavior in safe_extract_media_type_and_data."""
        # Test with completely invalid input
        media_type, data = safe_extract_media_type_and_data("not_base64_at_all")
        assert media_type == "image/jpeg"  # Fallback
        assert data == "not_base64_at_all"
