"""Client validation and serialization for AsyncOpenAI."""

from typing import Annotated, Any

from httpx import Timeout
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI
from pydantic import PlainSerializer, PlainValidator


def validate_openai_client(v: Any) -> AsyncOpenAI:
    """Validate and construct AsyncOpenAI client.

    Args:
        v: Either an existing AsyncOpenAI instance or a dict with client config

    Returns:
        Validated AsyncOpenAI client

    """
    if isinstance(v, AsyncOpenAI):
        return v
    if isinstance(timeout_dict := v.get("timeout"), dict):
        v["timeout"] = Timeout(**timeout_dict)
    return AsyncOpenAI(**v)


def serialize_openai_client(v: AsyncOpenAI) -> dict[str, Any]:
    """Serialize AsyncOpenAI client to dict.

    Only serializes non-default parameters. api_key is excluded for security.

    Args:
        v: AsyncOpenAI client instance

    Returns:
        Dict of non-default client parameters

    """
    client: dict[str, Any] = {}
    if str(v.base_url) != "https://api.openai.com/v1/":
        client["base_url"] = str(v.base_url)
    for key in (
        "organization",
        "project",
        "websocket_base_url",
    ):
        if (attr := getattr(v, key, None)) is not None:
            client[key] = attr
    if isinstance(v.timeout, float | None):
        client["timeout"] = v.timeout
    elif isinstance(v.timeout, Timeout):
        client["timeout"] = v.timeout.as_dict()
    if v.max_retries != DEFAULT_MAX_RETRIES:
        client["max_retries"] = v.max_retries
    if bool(v._custom_headers):
        client["default_headers"] = v._custom_headers
    if bool(v._custom_query):
        client["default_query"] = v._custom_query
    return client


PydanticAsyncOpenAI = Annotated[
    AsyncOpenAI,
    PlainValidator(validate_openai_client),
    PlainSerializer(serialize_openai_client, return_type=dict[str, Any]),
]
"""Pydantic-compatible AsyncOpenAI type with validation and serialization."""
