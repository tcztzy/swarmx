"""RFC6901."""

import re
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any, TypeGuard

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic.fields import FieldInfo
from pydantic_core import core_schema


class JsonPointer(str):
    """A JSON Pointer that can reference parts of a JSON document."""

    __slots__ = ("reference_tokens",)
    reference_tokens: tuple[str, ...]

    def __new__(cls, pointer: str):
        """Validate before new JsonPointer object."""
        if invalid_escape := re.search(r"(~[^01]|~$)", pointer):
            raise ValueError(f"Found invalid escape {invalid_escape.group()}")

        reference_tokens = pointer.split("/")
        if reference_tokens.pop(0) != "":
            raise ValueError("JSON Pointer must leading with slash")

        self = super().__new__(cls, pointer)
        self.reference_tokens = tuple(
            [token.replace("~1", "/").replace("~0", "~") for token in reference_tokens]
        )
        return self

    def __contains__(self, key: object):
        if isinstance(key, JsonPointer):
            return (
                self.reference_tokens[: len(key.reference_tokens)]
                == key.reference_tokens
            )
        if not isinstance(key, str):
            raise TypeError('Unsupported operand types for in ("object" and "str")')
        return super().__contains__(key)

    def __truediv__(self, reference_token: str):
        return JsonPointer(
            "/".join((str(self), reference_token.replace("~", "~0").replace("/", "~1")))
        )

    def __eq__(self, other: Any):
        if isinstance(other, JsonPointer):
            return self.reference_tokens == other.reference_tokens
        if isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self):
        return hash(tuple(self.reference_tokens))

    def __repr__(self):
        return f'{self.__class__.__name__}("{self}")'

    @property
    def parent(self) -> "JsonPointer":
        """Return the parent pointer (one token shorter)."""
        if not self.reference_tokens or len(self.reference_tokens) == 1:
            return self.__class__("")
        return self.__class__("/" + "/".join(self.reference_tokens[:-1]))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Provide Pydantic validation schema for JsonPointer."""
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema()
            ),
        )


def _token_matches_field(token: str, field_name: str, field_info: FieldInfo) -> bool:
    """Check whether a token matches a field by name or alias."""
    if token == field_name:
        return True
    alias_candidates = (
        field_info.alias,
        getattr(field_info, "validation_alias", None),
        getattr(field_info, "serialization_alias", None),
    )
    for alias in alias_candidates:
        if isinstance(alias, str) and token == alias:
            return True
    return False


def _resolve_field_name(model_cls: type[BaseModel], token: str) -> str | None:
    """Resolve the underlying field name for a token, considering aliases."""
    for field_name, field_info in model_cls.model_fields.items():
        if _token_matches_field(token, field_name, field_info):
            return field_name
    return None


def _get_from_model(model: BaseModel, token: str) -> Any:
    """Retrieve a field or extra value from a Pydantic model."""
    if field_name := _resolve_field_name(model.__class__, token):
        return getattr(model, field_name)
    if model.model_extra is not None and token in model.model_extra:
        return model.model_extra[token]
    raise KeyError(f"Key '{token}' not found in object")


def getitem(a: Any, b: Any) -> Any:
    """Retrieve value using a single token or full JsonPointer."""
    if isinstance(b, JsonPointer):
        return reduce(getitem, b.reference_tokens, a)

    token = b
    if isinstance(a, Mapping):
        if token not in a:
            raise KeyError(f"Key '{token}' not found in object")
        return a[token]

    if _is_sequence(a):
        index = _sequence_index(token, len(a), allow_end=False)
        return a[index]

    if isinstance(a, BaseModel):
        return _get_from_model(a, token)

    if hasattr(a, token):
        return getattr(a, token)

    raise TypeError(
        "Cannot traverse object: not a Mapping, Sequence, or attribute host"
    )


def _is_sequence(obj: Any) -> TypeGuard[Sequence]:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def _sequence_index(token: str, length: int, *, allow_end: bool) -> int:
    """Convert token to sequence index with bounds checking."""
    try:
        index = int(token)
    except ValueError as e:
        raise ValueError(f"Invalid sequence index '{token}'") from e

    upper_bound = length if allow_end else length - 1
    if index < 0 or index > upper_bound:
        verb = "insert" if allow_end else "access"
        raise IndexError(
            f"Index {index} out of range to {verb} sequence of length {length}"
        )
    return index


__all__ = ("JsonPointer", "getitem")
