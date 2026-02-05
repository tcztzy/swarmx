"""Shared node definitions for agents and tools."""

from abc import ABCMeta, abstractmethod
from collections.abc import Hashable
from typing import Annotated, Any

from pydantic import BaseModel, Field

from .hook import Hook


class Node(Hashable, BaseModel, use_attribute_docstrings=True, metaclass=ABCMeta):
    """Base node element in the execution graph."""

    name: Annotated[
        str, Field(pattern=r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)", frozen=True)
    ]
    """User-friendly name for the display.

    The name is unique among all nodes.
    """
    description: Annotated[str | None, Field(frozen=True)] = None
    """
    A description of what the node does, used by the model to choose when and how to call the node.

    Here is a template for description.
    ```
    Using this agent when <condition or necessity for this role>.

    Examples:
    - Context: <introduce the background of the real world case>
        user: "<user's query>"
        assistant: "<assistant's response>"
    - <more examples like the first one>
    ```
    """

    parameters: Annotated[dict[str, Any], Field(frozen=True)]
    """The parameters the node accepts, described as a JSON Schema object.

    Omitting `parameters` defines a node with an empty parameter list.
    """
    returns: Annotated[dict[str, Any] | None, Field(frozen=True)] = None
    """The returns schema of the node, described as a JSON Schema object.

    Omitting `returns` defines a node with a string returns.
    """

    hooks: list[Hook] = Field(default_factory=list)
    """Hooks to execute at various points in the agent lifecycle"""

    def __hash__(self):
        """Node name is unique among all nodes."""
        return hash(self.name)

    @abstractmethod
    async def __call__(
        self,
        arguments: Any,
        *,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,
        progress_callable: Any = None,
    ) -> Any:
        """Callable."""
        ...
