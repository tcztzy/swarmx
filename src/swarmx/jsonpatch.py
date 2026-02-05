"""RFC 6902 JSON Patch implementation.

This module implements JSON Patch (RFC 6902), providing operations to apply
and generate patches for JSON documents.
"""

import copy
from collections.abc import MutableMapping, MutableSequence
from functools import reduce
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .jsonpointer import JsonPointer, getitem


def _set_value(
    obj: Any, pointer: JsonPointer, value: Any, *, insert: bool = True
) -> Any:
    """Set a value at the location specified by a JSON Pointer.

    Args:
        obj: The document to modify.
        pointer: The JSON Pointer specifying the location.
        value: The value to set.
        insert: insert or not

    Returns:
        The modified document.

    Raises:
        KeyError: If a key is not found in a mapping.
        IndexError: If a sequence index is out of range.
        ValueError: If a sequence index is invalid.
        TypeError: If setting value on object that doesn't support item assignment.

    """
    if not pointer.reference_tokens:
        # Replace root document
        return value

    parent_pointer = pointer.parent
    if not parent_pointer.reference_tokens or parent_pointer.reference_tokens == ("",):
        parent = obj
    else:
        parent = getitem(obj, parent_pointer)
    last_token = pointer.reference_tokens[-1]

    # Set value on parent
    if isinstance(parent, MutableMapping):
        parent[last_token] = value
    elif isinstance(parent, MutableSequence):
        try:
            if insert and last_token == "-":
                index = len(parent)
            else:
                index = int(last_token)
            if insert:
                if index < 0 or index > len(parent):
                    raise IndexError(
                        f"Index {index} out of range for sequence of length {len(parent)}"
                    )
                parent.insert(index, value)
            else:
                if index < 0 or index >= len(parent):
                    raise IndexError(
                        f"Index {index} out of range for sequence of length {len(parent)}"
                    )
                parent[index] = value
        except ValueError as e:
            raise ValueError(f"Invalid sequence index '{last_token}'") from e
    else:
        raise TypeError("Cannot set value on immutable or non-indexable parent")

    return obj


def _remove_value(obj: Any, pointer: JsonPointer) -> Any:
    """Remove a value at the location specified by a JSON Pointer.

    Args:
        obj: The document to modify.
        pointer: The JSON Pointer specifying the location.

    Returns:
        The modified document.

    Raises:
        ValueError: If attempting to remove root document.
        KeyError: If a key is not found in a mapping.
        IndexError: If a sequence index is out of range.
        TypeError: If removing from immutable or non-indexable object.

    """
    if not pointer.reference_tokens:
        raise ValueError("Cannot remove root document")

    parent_pointer = pointer.parent
    if not parent_pointer.reference_tokens or parent_pointer.reference_tokens == ("",):
        parent = obj
    else:
        parent = getitem(obj, parent_pointer)
    last_token = pointer.reference_tokens[-1]

    # Remove from parent
    if isinstance(parent, MutableMapping):
        if last_token not in parent:
            raise KeyError(f"Key '{last_token}' not found")
        del parent[last_token]
    elif isinstance(parent, MutableSequence):
        try:
            index = int(last_token)
            if index < 0 or index >= len(parent):
                raise IndexError(
                    f"Index {index} out of range for sequence of length {len(parent)}"
                )
            parent.pop(index)
        except ValueError as e:
            raise ValueError(f"Invalid sequence index '{last_token}'") from e
    else:
        raise TypeError("Cannot remove from immutable or non-indexable parent")

    return obj


class AddOperation(BaseModel, frozen=True):
    """Add operation - adds a value to an object or inserts into an array."""

    op: Literal["add"] = "add"
    path: JsonPointer
    value: Any

    def apply(self, obj: Any) -> Any:
        """Apply add operation.

        Args:
            obj: Object to modify.

        Returns:
            Modified object.

        """
        obj = copy.deepcopy(obj)
        return _set_value(obj, self.path, self.value, insert=True)


class RemoveOperation(BaseModel, frozen=True):
    """Remove operation - removes a value from an object or array."""

    op: Literal["remove"] = "remove"
    path: JsonPointer

    def apply(self, obj: Any) -> Any:
        """Apply remove operation.

        Args:
            obj: Object to modify.

        Returns:
            Modified object.

        """
        obj = copy.deepcopy(obj)
        return _remove_value(obj, self.path)


class ReplaceOperation(BaseModel, frozen=True):
    """Replace operation - replaces a value."""

    op: Literal["replace"] = "replace"
    path: JsonPointer
    value: Any

    def apply(self, obj: Any) -> Any:
        """Apply replace operation.

        Args:
            obj: Object to modify.

        Returns:
            Modified object.

        """
        obj = copy.deepcopy(obj)
        # First verify the path exists
        getitem(obj, self.path)
        # Then replace the value
        return _set_value(obj, self.path, self.value, insert=False)


class MoveOperation(
    BaseModel, frozen=True, validate_by_alias=True, serialize_by_alias=True
):
    """Move operation - removes value at 'from' location and adds it to 'path'."""

    op: Literal["move"] = "move"
    path: JsonPointer
    from_: JsonPointer = Field(alias="from")

    @field_validator("from_")
    @classmethod
    def validate_from_not_prefix_of_path(cls, v: JsonPointer, info) -> JsonPointer:
        """Validate that 'from' is not a prefix of 'path'."""
        if "path" in info.data:
            path = info.data["path"]
            if v in path:
                raise ValueError("'from' location cannot be a prefix of 'path'")
        return v

    def to_atomic_operations(self, obj: Any) -> list["PatchOperation"]:
        """Return the equivalent atomic operations for this move."""
        value = copy.deepcopy(getitem(obj, self.from_))
        return [
            RemoveOperation(path=self.from_),
            AddOperation(path=self.path, value=value),
        ]

    def apply(self, obj: Any) -> Any:
        """Apply move operation.

        Args:
            obj: Object to modify.

        Returns:
            Modified object.

        """
        operations = self.to_atomic_operations(obj)
        result = obj
        for operation in operations:
            result = operation.apply(result)
        return result


class CopyOperation(
    BaseModel, frozen=True, validate_by_alias=True, serialize_by_alias=True
):
    """Copy operation - copies value at 'from' location to 'path'."""

    op: Literal["copy"] = "copy"
    path: JsonPointer
    from_: JsonPointer = Field(alias="from")

    def to_atomic_operations(self, obj: Any) -> list["PatchOperation"]:
        """Return the equivalent atomic operations for this copy."""
        value = copy.deepcopy(getitem(obj, self.from_))
        return [AddOperation(path=self.path, value=value)]

    def apply(self, obj: Any) -> Any:
        """Apply copy operation.

        Args:
            obj: Object to modify.

        Returns:
            Modified object.

        """
        operations = self.to_atomic_operations(obj)
        result = obj
        for operation in operations:
            result = operation.apply(result)
        return result


class TestOperation(BaseModel, frozen=True):
    """Test operation - tests that a value at the location equals the specified value."""

    __test__ = False
    op: Literal["test"] = "test"
    path: JsonPointer
    value: Any

    def apply(self, obj: Any) -> Any:
        """Apply test operation.

        Args:
            obj: Object to test.

        Returns:
            The unchanged object if test passes.

        Raises:
            AssertionError: If the test fails.

        """
        try:
            actual = getitem(obj, self.path)
            if actual != self.value:
                raise AssertionError(
                    f"Test failed: expected {self.value!r}, got {actual!r}"
                )
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise AssertionError(f"Test failed: {e}") from e
        return obj


# Union type for all operations
PatchOperation = (
    AddOperation
    | RemoveOperation
    | ReplaceOperation
    | MoveOperation
    | CopyOperation
    | TestOperation
)


class JsonPatch(BaseModel):
    """A JSON Patch document - a sequence of operations to apply to a JSON document."""

    patch: list[PatchOperation] = Field(default_factory=list)

    def apply(self, obj: Any) -> Any:
        """Apply all patch operations in sequence.

        Args:
            obj: The document to patch.

        Returns:
            The patched document.

        Raises:
            KeyError: If a key is not found.
            IndexError: If an array index is out of range.
            ValueError: If a value is invalid.
            TypeError: If an operation on an incompatible type.
            AssertionError: If a test operation fails.

        """
        return reduce(lambda d, op: op.apply(d), self.patch, obj)
