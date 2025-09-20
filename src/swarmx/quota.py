"""Quota for tokens and other resource."""

import asyncio


class QuotaExceededError(Exception):
    """When quota exceed, raise this exception."""

    def __init__(self, message, used, requested, limit):  # noqa: D107
        super().__init__(message)
        self.used = used
        self.requested = requested
        self.limit = limit


class QuotaManager:
    """Manager for quota."""

    def __init__(self, max_tokens: int | None = None):  # noqa: D107
        self._max_tokens = max_tokens
        self._used_tokens = 0
        self._lock = asyncio.Lock()

    @property
    def used_tokens(self) -> int:
        """Used tokens."""
        return self._used_tokens

    @property
    def max_tokens(self) -> int | float:
        """Max tokens."""
        return self._max_tokens or float("inf")

    async def consume(self, agent_name: str, total_tokens: int):
        """Report usage after task finished. It is a atomic operation."""
        async with self._lock:
            self._used_tokens += total_tokens
            if self.used_tokens >= self.max_tokens:
                raise QuotaExceededError(
                    f"Agent {agent_name} consume {total_tokens} tokens. Quota exceeded.",
                    used=self.used_tokens,
                    requested=total_tokens,
                    limit=self._max_tokens,
                )
