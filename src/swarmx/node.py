"""Shared node definitions for agents and tools."""

from collections.abc import Hashable
from typing import Annotated, Protocol, runtime_checkable

from pydantic import Field


@runtime_checkable
class Node[I, O](Hashable, Protocol):
    """Base node element in the execution graph."""

    name: Annotated[str, Field(frozen=True)]

    async def __call__(
        self,
        arguments: I,
        *args,
        **kwargs,
    ) -> O:
        """Callable."""
        ...
