"""Tests for client validation and serialization."""

from openai import AsyncOpenAI

from swarmx.clients import (
    PydanticAsyncOpenAI,
    serialize_openai_client,
    validate_openai_client,
)


def test_validate_openai_client_existing_instance():
    """Test validation with existing AsyncOpenAI instance."""
    client = AsyncOpenAI(api_key="test-key")
    result = validate_openai_client(client)
    assert result is client


def test_validate_openai_client_from_dict():
    """Test validation with dict input."""
    config = {
        "api_key": "test-key",
        "base_url": "http://test.com",
    }
    result = validate_openai_client(config)
    assert isinstance(result, AsyncOpenAI)
    assert str(result.base_url) == "http://test.com"


def test_validate_openai_client_with_timeout_dict():
    """Test validation with timeout as dict."""
    config = {
        "api_key": "test-key",
        "timeout": {
            "connect": 10.0,
            "read": 30.0,
            "write": 30.0,
            "pool": 30.0,
        },
    }
    result = validate_openai_client(config)
    assert isinstance(result, AsyncOpenAI)
    assert result.timeout.connect == 10.0
    assert result.timeout.read == 30.0


def test_serialize_openai_client_default_base_url():
    """Test serialization excludes default base_url."""
    import os

    base_url = os.environ["OPENAI_BASE_URL"]
    del os.environ["OPENAI_BASE_URL"]
    client = AsyncOpenAI(api_key="test-key")
    result = serialize_openai_client(client)
    assert "base_url" not in result
    os.environ["OPENAI_BASE_URL"] = base_url


def test_serialize_openai_client_custom_base_url():
    """Test serialization includes custom base_url."""
    client = AsyncOpenAI(api_key="test-key", base_url="http://test.com")
    result = serialize_openai_client(client)
    assert result["base_url"] == "http://test.com"


def test_serialize_openai_client_with_timeout():
    """Test serialization includes timeout configuration."""
    client = AsyncOpenAI(api_key="test-key", base_url="http://test.com")
    result = serialize_openai_client(client)
    assert "timeout" in result
    assert isinstance(result["timeout"], dict)


def test_serialize_openai_client_excludes_api_key():
    """Test that api_key is never serialized."""
    client = AsyncOpenAI(api_key="test-key", base_url="http://test.com")
    result = serialize_openai_client(client)
    assert "api_key" not in result


def test_pydantic_async_openai_type():
    """Test that PydanticAsyncOpenAI annotation works."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        client: PydanticAsyncOpenAI = None

    # Test with None
    model = TestModel()
    assert model.client is None

    # Test with AsyncOpenAI instance
    client = AsyncOpenAI(api_key="test-key")
    model = TestModel(client=client)
    assert model.client is client

    # Test with dict
    model = TestModel(client={"api_key": "test-key", "base_url": "http://test.com"})
    assert isinstance(model.client, AsyncOpenAI)
    assert str(model.client.base_url) == "http://test.com"

    # Test serialization
    serialized = model.model_dump(mode="json")
    assert "base_url" in serialized["client"]
    assert "api_key" not in serialized["client"]
