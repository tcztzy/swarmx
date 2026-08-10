"""Strict input validation at the RSI MCP process boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REQUEST_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
KNOWN_REQUEST_KEYS = {
    "schemaVersion",
    "skillId",
    "variantId",
    "parentRevisionId",
    "parentRevisionDigest",
    "baselineContentRef",
    "baselineContentDigest",
    "targetAgentId",
    "targetModelFingerprint",
    "trainDataset",
    "devDataset",
    "optimizer",
    "budget",
    "requestedBy",
    "requestId",
    "proposer",
}


def validate_optimization_request(request: object) -> dict[str, Any]:
    if not isinstance(request, dict) or not all(
        isinstance(key, str) for key in request
    ):
        raise ValueError("request must be an object")
    if _integer(request, "schemaVersion") != 1 or set(request) - KNOWN_REQUEST_KEYS:
        raise ValueError("request fields are invalid")
    if not all(REQUEST_KEY_PATTERN.fullmatch(key) for key in request):
        raise ValueError("request key is unsafe")
    if request.get("proposer", "none") not in {"none", "gateway", "deterministic"}:
        raise ValueError("proposer is unsupported")
    for key in (
        "skillId",
        "variantId",
        "parentRevisionId",
        "parentRevisionDigest",
        "baselineContentRef",
        "baselineContentDigest",
        "targetAgentId",
        "targetModelFingerprint",
    ):
        _string(request, key)
    train = _mapping(request, "trainDataset")
    dev = _mapping(request, "devDataset")
    if _string(train, "role") != "train" or _string(dev, "role") != "dev":
        raise ValueError("dataset role is invalid")
    for dataset in (train, dev):
        _string(dataset, "contentRef")
        _string(dataset, "contentDigest")
    optimizer = _mapping(request, "optimizer")
    if _string(optimizer, "optimizerId") != "dspy.gepa.v1":
        raise ValueError("optimizer is unsupported")
    for key in ("optimizerVersion", "environmentDigest", "configDigest"):
        _string(optimizer, key)
    _integer(optimizer, "seed")
    budget = _mapping(request, "budget")
    _optional_bounded_integer(budget, "maxWallTimeMs", 1, 7 * 24 * 3600 * 1000)
    _optional_bounded_integer(budget, "maxModelCalls", 1, 1_000_000)
    _optional_bounded_integer(budget, "maxTokens", 0, 1_000_000_000)
    _optional_bounded_integer(budget, "maxArtifactBytes", 1, 16 * 1024 * 1024)
    return request


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    field = value.get(key)
    if not isinstance(field, dict):
        raise ValueError(f"{key} must be an object")
    return field


def _string(value: Mapping[str, Any], key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field or len(field) > 4_096 or "\x00" in field:
        raise ValueError(f"{key} must be a bounded string")
    return field


def _integer(value: Mapping[str, Any], key: str) -> int:
    field = value.get(key)
    if isinstance(field, bool) or not isinstance(field, int):
        raise ValueError(f"{key} must be an integer")
    return field


def _optional_bounded_integer(
    value: Mapping[str, Any], key: str, minimum: int, maximum: int
) -> None:
    field = value.get(key)
    if field is None:
        return
    if (
        isinstance(field, bool)
        or not isinstance(field, int)
        or not minimum <= field <= maximum
    ):
        raise ValueError(f"{key} is out of bounds")
