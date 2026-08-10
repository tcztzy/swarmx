"""DSPy language-model adapters that talk to the SwarmX capability gateway.

`CapabilityLm` is the production adapter: every model call crosses the task
worker protocol as a grant-checked `model.generate` capability call, so the
Python side never holds Provider credentials. `DeterministicLm` is a test-only
adapter that answers from the instruction text without any host call.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from dspy.clients.base_lm import BaseLM


class CapabilityLm(BaseLM):
    """BaseLM whose forward() relays messages through the capability gateway."""

    def __init__(self, capability_client: Any, model: str | None = None) -> None:
        super().__init__(model=model or "swarmx-gateway")
        self._capability_client = capability_client
        self._lock = threading.Lock()
        self._calls = 0
        self._tokens = 0

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        call_messages = messages
        if call_messages is None and prompt is not None:
            call_messages = [{"role": "user", "content": prompt}]
        if not call_messages:
            call_messages = [{"role": "user", "content": ""}]
        timeout_ms = int(kwargs.get("timeout", 60) * 1_000)
        outcome = self._capability_client.call(
            "skill_evolution",
            "model.generate",
            {
                "model": self.model,
                "messages": call_messages,
                "temperature": kwargs.get("temperature"),
                "maxTokens": kwargs.get("max_tokens"),
            },
            timeout_ms=timeout_ms,
        )
        if outcome.get("status") != "succeeded":
            error = outcome.get("error") or {}
            message = error.get("message", "model generation failed")
            raise RuntimeError(f"swarmx model.generate failed: {message}")
        value = outcome.get("value")
        if not isinstance(value, dict) or not isinstance(value.get("content"), str):
            raise RuntimeError("swarmx model.generate returned no text content")
        usage = value.get("usage") or {}
        with self._lock:
            self._calls += 1
            self._tokens += int(usage.get("totalTokens", 0) or 0)
        return chat_completion(value["content"], model=self.model)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"calls": self._calls, "tokens": self._tokens}


class DeterministicLm(BaseLM):
    """Test-only LM: answers from the instruction text without any host call.

    When the system instruction contains the mandatory rule "the final answer
    must be exactly `X`", the prediction is the chat-adapter field format
    `[[ ## answer ## ]]\\nX`; otherwise it is `[[ ## answer ## ]]\\nwrong`.
    This makes instruction edits observable in the prediction, so metrics and
    GEPA can run deterministically without any network or credentials.
    """

    def __init__(self) -> None:
        super().__init__(model="deterministic")
        self._lock = threading.Lock()
        self._calls = 0

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        instruction = ""
        for message in messages or []:
            if message.get("role") == "system" and isinstance(
                message.get("content"), str
            ):
                instruction = message["content"]
        keyword = mandated_keyword(instruction)
        with self._lock:
            self._calls += 1
        return chat_completion(
            f"[[ ## answer ## ]]\n{keyword or 'wrong'}", model=self.model
        )

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"calls": self._calls}


def mandated_keyword(instruction: str) -> str | None:
    match = re.search(r"exactly\s+`([A-Za-z0-9][A-Za-z0-9._-]*)`", instruction)
    return match.group(1) if match else None


def chat_completion(content: str, model: str) -> Any:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    return ChatCompletion(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        model=model,
        created=int(time.time()),
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
    )
