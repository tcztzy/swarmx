import pytest
from mcp.types import CallToolResult, EmbeddedResource, ImageContent

from swarmx.conversion import _image_to_md, _resource_to_md, result_to_content

pytestmark = pytest.mark.anyio


async def test_resource_to_md_filename_from_host():
    """Test _resource_to_md filename determination from host."""
    # Create a resource with host but no path
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "example://example.txt",
                "text": "Test content",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "example.txt" in result
    assert "Test content" in result


async def test_resource_to_md_filename_from_mimetype():
    """Test _resource_to_md with text/plain mimeType uses HTML comments."""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "data://test.txt",
                "text": "Test content",
                "mimeType": "text/plain",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "<!-- begin data://test.txt -->" in result
    assert "Test content" in result
    assert "<!-- end data://test.txt -->" in result


async def test_resource_to_md_markdown_with_nested_code_blocks():
    """Test _resource_to_md with markdown containing nested code blocks."""
    markdown_content = """# Example

```python
def hello():
    print("world")
```

More text here."""

    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///example.md",
                "text": markdown_content,
                "mimeType": "text/markdown",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "<!-- begin file:///example.md -->" in result
    assert markdown_content in result
    assert "<!-- end file:///example.md -->" in result
    # Verify the nested code block is preserved
    assert "```python" in result


async def test_resource_to_md_filename_error():
    """Test _resource_to_md filename determination error."""

    # Create a resource with no path, host, or mimeType
    resource = EmbeddedResource.model_validate(
        {"type": "resource", "resource": {"uri": "data://", "text": "Test content"}}
    )

    with pytest.raises(ValueError, match="Cannot determine filename for resource"):
        _resource_to_md(resource)


async def test_resource_to_md_lexer_not_found():
    """Test _resource_to_md with lexer not found"""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///test.unknown",
                "text": "Test content",
                "mimeType": "application/unknown",
            },
        }
    )

    result = _resource_to_md(resource)
    # Should default to "text" when lexer is not found
    assert "```text" in result
    assert "Test content" in result


async def test_resource_to_md_blob_content():
    """Test _resource_to_md with blob content."""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///test.png",
                "blob": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "mimeType": "image/png",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "<embed" in result
    assert 'type="image/png"' in result
    assert "data:image/png;base64," in result
    assert 'title="file:///test.png"' in result


async def test_result_to_content_audio():
    """Test result_to_content with audio content."""

    result = CallToolResult.model_validate(
        {
            "content": [
                {"type": "audio", "data": "base64audiodata", "mimeType": "audio/wav"}
            ]
        }
    )

    content = result_to_content(result)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert '<audio src="data:audio/wav;base64,base64audiodata"' in content[0]["text"]


async def test_result_to_content_resource_link():
    """Test result_to_content with resource_link content."""

    result = CallToolResult.model_validate(
        {
            "content": [
                {
                    "type": "resource_link",
                    "name": "Test Resource",
                    "uri": "file:///test.txt",
                }
            ]
        }
    )

    content = result_to_content(result)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "[Test Resource](file:///test.txt)" in content[0]["text"]


async def test_image_to_md_and_result_to_content_image():
    image = ImageContent.model_validate(
        {
            "type": "image",
            "data": "base64",
            "mimeType": "image/png",
            "meta": {"alt": "alt"},
        }
    )
    md = _image_to_md(image)
    assert md.startswith("![") and "data:image/png" in md

    result = CallToolResult.model_validate({"content": [image.model_dump()]})
    parts = result_to_content(result)
    assert md in parts[0]["text"]
