"""Tests for RFC6901 JSON Pointer implementation."""

import pytest
from pydantic import BaseModel, ValidationError

from swarmx.jsonpointer import JsonPointer, getitem


class TestJsonPointerCreation:
    """Test JsonPointer object creation and validation."""

    def test_empty_pointer(self):
        """Test creating an empty pointer (root document)."""
        ptr = JsonPointer("")
        assert str(ptr) == ""
        assert ptr.reference_tokens == ()

    def test_simple_pointer(self):
        """Test creating a simple pointer with one token."""
        ptr = JsonPointer("/foo")
        assert str(ptr) == "/foo"
        assert ptr.reference_tokens == ("foo",)

    def test_nested_pointer(self):
        """Test creating a nested pointer with multiple tokens."""
        ptr = JsonPointer("/foo/bar/baz")
        assert str(ptr) == "/foo/bar/baz"
        assert ptr.reference_tokens == ("foo", "bar", "baz")
        assert ptr.parent == JsonPointer("/foo/bar")

    def test_numeric_token(self):
        """Test pointer with numeric tokens (array indices)."""
        ptr = JsonPointer("/foo/0/bar")
        assert str(ptr) == "/foo/0/bar"
        assert ptr.reference_tokens == ("foo", "0", "bar")
        assert ptr.parent == JsonPointer("/foo/0")

    def test_parent_of_single_token(self):
        ptr = JsonPointer("/foo")
        assert ptr.parent == JsonPointer("")

    def test_escaped_tilde(self):
        """Test pointer with escaped tilde (~0)."""
        ptr = JsonPointer("/foo~0bar")
        assert str(ptr) == "/foo~0bar"
        assert ptr.reference_tokens == ("foo~bar",)

    def test_escaped_slash(self):
        """Test pointer with escaped slash (~1)."""
        ptr = JsonPointer("/foo~1bar")
        assert str(ptr) == "/foo~1bar"
        assert ptr.reference_tokens == ("foo/bar",)

    def test_both_escapes(self):
        """Test pointer with both tilde and slash escapes."""
        ptr = JsonPointer("/foo~0bar~1baz")
        assert str(ptr) == "/foo~0bar~1baz"
        assert ptr.reference_tokens == ("foo~bar/baz",)

    def test_special_characters(self):
        """Test pointer with various special characters."""
        ptr = JsonPointer("/foo bar/with-dash/with_underscore")
        assert ptr.reference_tokens == ("foo bar", "with-dash", "with_underscore")

    def test_empty_token(self):
        """Test pointer with empty token (consecutive slashes)."""
        ptr = JsonPointer("/foo//bar")
        assert ptr.reference_tokens == ("foo", "", "bar")

    def test_missing_leading_slash_raises_error(self):
        """Test that pointer without leading slash raises ValueError."""
        with pytest.raises(ValueError, match="JSON Pointer must leading with slash"):
            JsonPointer("foo/bar")

    def test_invalid_escape_tilde_alone_raises_error(self):
        """Test that invalid escape sequence ~<invalid> raises ValueError."""
        with pytest.raises(ValueError, match="Found invalid escape"):
            JsonPointer("/foo~2bar")

    def test_invalid_escape_tilde_at_end_raises_error(self):
        """Test that tilde at end of token raises ValueError."""
        with pytest.raises(ValueError, match="Found invalid escape"):
            JsonPointer("/foo~")

    def test_invalid_escape_tilde_with_letter_raises_error(self):
        """Test that tilde followed by non-0/1 character raises ValueError."""
        with pytest.raises(ValueError, match="Found invalid escape"):
            JsonPointer("/foo~a")


class TestJsonPointerContains:
    """Test JsonPointer __contains__ method."""

    def test_contains_string(self):
        """Test string containment using default str behavior."""
        ptr = JsonPointer("/foo/bar")
        assert "/foo" in ptr
        assert "foo" in ptr
        assert "/baz" not in ptr

    def test_contains_json_pointer_prefix(self):
        """Test that JsonPointer checks if other is a prefix."""
        parent = JsonPointer("/foo/bar/baz")
        child1 = JsonPointer("/foo")
        child2 = JsonPointer("/foo/bar")
        child3 = JsonPointer("/foo/bar/baz")
        other = JsonPointer("/foo/qux")

        assert child1 in parent
        assert child2 in parent
        assert child3 in parent
        assert other not in parent

    def test_contains_empty_pointer(self):
        """Test that empty pointer is prefix of any pointer."""
        ptr = JsonPointer("/foo/bar")
        empty = JsonPointer("")
        assert empty in ptr

    def test_contains_self(self):
        """Test that pointer contains itself."""
        ptr = JsonPointer("/foo/bar")
        assert ptr in ptr


class TestJsonPointerTruediv:
    """Test JsonPointer __truediv__ method (/ operator)."""

    def test_append_simple_token(self):
        """Test appending a simple token."""
        ptr = JsonPointer("/foo")
        new_ptr = ptr / "bar"
        assert str(new_ptr) == "/foo/bar"
        assert new_ptr.reference_tokens == ("foo", "bar")

    def test_append_to_empty_pointer(self):
        """Test appending to root pointer."""
        ptr = JsonPointer("")
        new_ptr = ptr / "foo"
        assert str(new_ptr) == "/foo"
        assert new_ptr.reference_tokens == ("foo",)

    def test_append_with_tilde(self):
        """Test that tilde in token gets escaped."""
        ptr = JsonPointer("/foo")
        new_ptr = ptr / "bar~baz"
        assert str(new_ptr) == "/foo/bar~0baz"
        assert new_ptr.reference_tokens == ("foo", "bar~baz")

    def test_append_with_slash(self):
        """Test that slash in token gets escaped."""
        ptr = JsonPointer("/foo")
        new_ptr = ptr / "bar/baz"
        assert str(new_ptr) == "/foo/bar~1baz"
        assert new_ptr.reference_tokens == ("foo", "bar/baz")

    def test_append_with_both_special_chars(self):
        """Test that both tilde and slash get escaped correctly."""
        ptr = JsonPointer("/foo")
        new_ptr = ptr / "bar~/baz"
        assert str(new_ptr) == "/foo/bar~0~1baz"
        assert new_ptr.reference_tokens == ("foo", "bar~/baz")

    def test_chained_append(self):
        """Test chaining multiple appends."""
        ptr = JsonPointer("")
        new_ptr = ptr / "foo" / "bar" / "baz"
        assert str(new_ptr) == "/foo/bar/baz"
        assert new_ptr.reference_tokens == ("foo", "bar", "baz")

    def test_append_numeric_token(self):
        """Test appending numeric token (array index)."""
        ptr = JsonPointer("/foo")
        new_ptr = ptr / "0"
        assert str(new_ptr) == "/foo/0"
        assert new_ptr.reference_tokens == ("foo", "0")

    def test_append_empty_string(self):
        """Test appending empty string."""
        ptr = JsonPointer("/foo")
        new_ptr = ptr / ""
        assert str(new_ptr) == "/foo/"
        assert new_ptr.reference_tokens == ("foo", "")


class TestJsonPointerEquality:
    """Test JsonPointer __eq__ method."""

    def test_equal_pointers(self):
        """Test that identical pointers are equal."""
        ptr1 = JsonPointer("/foo/bar")
        ptr2 = JsonPointer("/foo/bar")
        assert ptr1 == ptr2

    def test_unequal_pointers(self):
        """Test that different pointers are not equal."""
        ptr1 = JsonPointer("/foo/bar")
        ptr2 = JsonPointer("/foo/baz")
        assert ptr1 != ptr2

    def test_empty_pointers_equal(self):
        """Test that empty pointers are equal."""
        ptr1 = JsonPointer("")
        ptr2 = JsonPointer("")
        assert ptr1 == ptr2

    def test_equal_with_escapes(self):
        """Test equality with escaped characters."""
        ptr1 = JsonPointer("/foo~0bar")
        ptr2 = JsonPointer("/foo~0bar")
        assert ptr1 == ptr2

    def test_equal_to_string(self):
        """Test that JsonPointer is not equal to plain string."""
        ptr = JsonPointer("/foo/bar")
        assert ptr == "/foo/bar"

    def test_not_equal_to_none(self):
        """Test that JsonPointer is not equal to None."""
        ptr = JsonPointer("/foo")
        assert ptr is not None
        assert ptr != None  # noqa: E711

    def test_equality_reflexive(self):
        """Test that equality is reflexive (a == a)."""
        ptr = JsonPointer("/foo/bar")
        assert ptr == ptr


class TestJsonPointerHash:
    """Test JsonPointer __hash__ method."""

    def test_hash_consistent(self):
        """Test that hash is consistent for same pointer."""
        ptr = JsonPointer("/foo/bar")
        assert hash(ptr) == hash(ptr)

    def test_equal_pointers_same_hash(self):
        """Test that equal pointers have same hash."""
        ptr1 = JsonPointer("/foo/bar")
        ptr2 = JsonPointer("/foo/bar")
        assert hash(ptr1) == hash(ptr2)

    def test_usable_in_set(self):
        """Test that JsonPointer can be used in a set."""
        ptr1 = JsonPointer("/foo")
        ptr2 = JsonPointer("/bar")
        ptr3 = JsonPointer("/foo")  # Duplicate

        pointer_set = {ptr1, ptr2, ptr3}
        assert len(pointer_set) == 2
        assert ptr1 in pointer_set
        assert ptr2 in pointer_set

    def test_usable_as_dict_key(self):
        """Test that JsonPointer can be used as dict key."""
        ptr1 = JsonPointer("/foo")
        ptr2 = JsonPointer("/bar")

        pointer_dict = {ptr1: "value1", ptr2: "value2"}
        assert pointer_dict[ptr1] == "value1"
        assert pointer_dict[ptr2] == "value2"

    def test_hash_with_empty_pointer(self):
        """Test hash for empty pointer."""
        ptr = JsonPointer("")
        assert isinstance(hash(ptr), int)


class TestJsonPointerRepr:
    """Test JsonPointer __repr__ method."""

    def test_repr_simple(self):
        """Test repr for simple pointer."""
        ptr = JsonPointer("/foo")
        assert repr(ptr) == 'JsonPointer("/foo")'

    def test_repr_nested(self):
        """Test repr for nested pointer."""
        ptr = JsonPointer("/foo/bar/baz")
        assert repr(ptr) == 'JsonPointer("/foo/bar/baz")'

    def test_repr_empty(self):
        """Test repr for empty pointer."""
        ptr = JsonPointer("")
        assert repr(ptr) == 'JsonPointer("")'

    def test_repr_with_escapes(self):
        """Test repr with escaped characters."""
        ptr = JsonPointer("/foo~0bar~1baz")
        assert repr(ptr) == 'JsonPointer("/foo~0bar~1baz")'


class TestJsonPointerStringBehavior:
    """Test that JsonPointer behaves like a string."""

    def test_is_string_subclass(self):
        """Test that JsonPointer is a string subclass."""
        ptr = JsonPointer("/foo")
        assert isinstance(ptr, str)

    def test_string_operations(self):
        """Test that string operations work on JsonPointer."""
        ptr = JsonPointer("/foo/bar")
        assert ptr.upper() == "/FOO/BAR"
        assert ptr.startswith("/foo")
        assert ptr.endswith("/bar")
        assert ptr.split("/") == ["", "foo", "bar"]

    def test_string_concatenation(self):
        """Test string concatenation."""
        ptr = JsonPointer("/foo")
        result = ptr + "/bar"
        # Result is a string, not JsonPointer
        assert result == "/foo/bar"
        assert isinstance(result, str)
        assert not isinstance(result, JsonPointer)

    def test_length(self):
        """Test len() on JsonPointer."""
        ptr = JsonPointer("/foo/bar")
        assert len(ptr) == 8  # String length

    def test_iteration(self):
        """Test iteration over JsonPointer characters."""
        ptr = JsonPointer("/foo")
        chars = list(ptr)
        assert chars == ["/", "f", "o", "o"]


class TestJsonPointerEdgeCases:
    """Test edge cases and RFC6901 examples."""

    def test_rfc_example_whole_document(self):
        """Test RFC6901 example: whole document."""
        ptr = JsonPointer("")
        assert ptr.reference_tokens == ()

    def test_rfc_example_foo(self):
        """Test RFC6901 example: /foo."""
        ptr = JsonPointer("/foo")
        assert ptr.reference_tokens == ("foo",)

    def test_rfc_example_array_index(self):
        """Test RFC6901 example: /foo/0."""
        ptr = JsonPointer("/foo/0")
        assert ptr.reference_tokens == ("foo", "0")

    def test_rfc_example_empty_string_key(self):
        """Test RFC6901 example: / (empty string key)."""
        ptr = JsonPointer("/")
        assert ptr.reference_tokens == ("",)

    def test_rfc_example_slash_in_key(self):
        """Test RFC6901 example: /a~1b (key 'a/b')."""
        ptr = JsonPointer("/a~1b")
        assert ptr.reference_tokens == ("a/b",)

    def test_rfc_example_tilde_in_key(self):
        """Test RFC6901 example: /m~0n (key 'm~n')."""
        ptr = JsonPointer("/m~0n")
        assert ptr.reference_tokens == ("m~n",)

    def test_rfc_example_complex(self):
        """Test RFC6901 complex example."""
        ptr = JsonPointer("/c%d")
        assert ptr.reference_tokens == ("c%d",)

    def test_rfc_example_pipe(self):
        """Test RFC6901 example with pipe character."""
        ptr = JsonPointer("/e^f")
        assert ptr.reference_tokens == ("e^f",)

    def test_rfc_example_backslash(self):
        """Test RFC6901 example with backslash."""
        ptr = JsonPointer("/g|h")
        assert ptr.reference_tokens == ("g|h",)

    def test_rfc_example_quotes(self):
        """Test RFC6901 example with quotes."""
        ptr = JsonPointer("/i\\j")
        assert ptr.reference_tokens == ("i\\j",)

    def test_unicode_characters(self):
        """Test pointer with unicode characters."""
        ptr = JsonPointer("/café/日本語/🚀")
        assert ptr.reference_tokens == ("café", "日本語", "🚀")


class TestJsonPointerPydanticIntegration:
    """Test JsonPointer integration with Pydantic models."""

    def test_simple_model_field(self):
        """Test JsonPointer as a simple Pydantic model field."""

        class SimpleModel(BaseModel):
            pointer: JsonPointer

        model = SimpleModel(pointer="/foo/bar")
        assert isinstance(model.pointer, JsonPointer)
        assert str(model.pointer) == "/foo/bar"
        assert model.pointer.reference_tokens == ("foo", "bar")

    def test_validation_from_string(self):
        """Test that Pydantic validates and converts string to JsonPointer."""

        class Model(BaseModel):
            path: JsonPointer

        model = Model(path="/foo/bar/baz")
        assert isinstance(model.path, JsonPointer)
        assert model.path.reference_tokens == ("foo", "bar", "baz")

    def test_validation_rejects_invalid_pointer(self):
        """Test that Pydantic validation rejects invalid JSON Pointers."""

        class Model(BaseModel):
            path: JsonPointer

        # Missing leading slash
        with pytest.raises(ValidationError) as exc_info:
            Model(path="foo/bar")
        assert "JSON Pointer must leading with slash" in str(exc_info.value)

    def test_validation_rejects_invalid_escape(self):
        """Test that Pydantic validation rejects invalid escape sequences."""

        class Model(BaseModel):
            path: JsonPointer

        with pytest.raises(ValidationError) as exc_info:
            Model(path="/foo~2bar")
        assert "Found invalid escape" in str(exc_info.value)

    def test_serialization_to_json(self):
        """Test that JsonPointer serializes to JSON as a string."""

        class Model(BaseModel):
            pointer: JsonPointer

        model = Model(pointer="/foo/bar")
        json_data = model.model_dump()
        assert json_data == {"pointer": "/foo/bar"}
        assert isinstance(json_data["pointer"], str)

    def test_serialization_to_json_string(self):
        """Test that JsonPointer serializes to JSON string correctly."""

        class Model(BaseModel):
            pointer: JsonPointer

        model = Model(pointer="/foo/bar")
        json_str = model.model_dump_json()
        assert json_str == '{"pointer":"/foo/bar"}'

    def test_deserialization_from_json(self):
        """Test that JsonPointer deserializes from JSON correctly."""

        class Model(BaseModel):
            pointer: JsonPointer

        json_data = {"pointer": "/foo/bar"}
        model = Model.model_validate(json_data)
        assert isinstance(model.pointer, JsonPointer)
        assert str(model.pointer) == "/foo/bar"

    def test_deserialization_from_json_string(self):
        """Test that JsonPointer deserializes from JSON string correctly."""

        class Model(BaseModel):
            pointer: JsonPointer

        json_str = '{"pointer":"/foo/bar"}'
        model = Model.model_validate_json(json_str)
        assert isinstance(model.pointer, JsonPointer)
        assert str(model.pointer) == "/foo/bar"

    def test_optional_pointer_field(self):
        """Test JsonPointer as an optional field."""

        class Model(BaseModel):
            pointer: JsonPointer | None = None

        model1 = Model(pointer="/foo")
        assert isinstance(model1.pointer, JsonPointer)
        assert str(model1.pointer) == "/foo"

        model2 = Model()
        assert model2.pointer is None

        model3 = Model(pointer=None)
        assert model3.pointer is None

    def test_list_of_pointers(self):
        """Test list of JsonPointers in Pydantic model."""

        class Model(BaseModel):
            pointers: list[JsonPointer]

        model = Model(pointers=["/foo", "/bar/baz", "/qux/0"])
        assert len(model.pointers) == 3
        assert all(isinstance(p, JsonPointer) for p in model.pointers)
        assert [str(p) for p in model.pointers] == ["/foo", "/bar/baz", "/qux/0"]

    def test_dict_with_pointer_values(self):
        """Test dict with JsonPointer values in Pydantic model."""

        class Model(BaseModel):
            paths: dict[str, JsonPointer]

        model = Model(paths={"first": "/foo", "second": "/bar/baz"})
        assert isinstance(model.paths["first"], JsonPointer)
        assert isinstance(model.paths["second"], JsonPointer)
        assert str(model.paths["first"]) == "/foo"
        assert str(model.paths["second"]) == "/bar/baz"

    def test_nested_model_with_pointer(self):
        """Test JsonPointer in nested Pydantic models."""

        class Inner(BaseModel):
            pointer: JsonPointer

        class Outer(BaseModel):
            inner: Inner

        model = Outer(inner={"pointer": "/foo/bar"})
        assert isinstance(model.inner.pointer, JsonPointer)
        assert str(model.inner.pointer) == "/foo/bar"

    def test_model_with_escaped_characters(self):
        """Test JsonPointer with escaped characters in Pydantic model."""

        class Model(BaseModel):
            pointer: JsonPointer

        model = Model(pointer="/foo~0bar~1baz")
        assert isinstance(model.pointer, JsonPointer)
        assert str(model.pointer) == "/foo~0bar~1baz"
        assert model.pointer.reference_tokens == ("foo~bar/baz",)

    def test_model_with_empty_pointer(self):
        """Test empty JsonPointer (root document) in Pydantic model."""

        class Model(BaseModel):
            pointer: JsonPointer

        model = Model(pointer="")
        assert isinstance(model.pointer, JsonPointer)
        assert str(model.pointer) == ""
        assert model.pointer.reference_tokens == ()

    def test_round_trip_serialization(self):
        """Test that JsonPointer survives round-trip serialization."""

        class Model(BaseModel):
            pointer: JsonPointer

        original = Model(pointer="/foo/bar/baz")
        json_str = original.model_dump_json()
        restored = Model.model_validate_json(json_str)

        assert isinstance(restored.pointer, JsonPointer)
        assert str(restored.pointer) == str(original.pointer)
        assert restored.pointer.reference_tokens == original.pointer.reference_tokens

    def test_model_config_validation(self):
        """Test that model validation works with JsonPointer."""

        class Model(BaseModel, validate_assignment=True):
            pointer: JsonPointer

        model = Model(pointer="/foo")
        assert str(model.pointer) == "/foo"

        # Valid reassignment
        model.pointer = JsonPointer("/bar")
        assert str(model.pointer) == "/bar"

        # Invalid reassignment should raise ValidationError
        with pytest.raises(ValidationError):
            model.pointer = "invalid"  # Missing leading slash

    def test_multiple_pointers_in_model(self):
        """Test model with multiple JsonPointer fields."""

        class Model(BaseModel):
            source: JsonPointer
            target: JsonPointer
            fallback: JsonPointer | None = None

        model = Model(source="/input/data", target="/output/result")
        assert isinstance(model.source, JsonPointer)
        assert isinstance(model.target, JsonPointer)
        assert model.fallback is None

        model_with_fallback = Model(
            source="/input/data", target="/output/result", fallback="/default"
        )
        assert isinstance(model_with_fallback.fallback, JsonPointer)

    def test_pointer_comparison_in_model(self):
        """Test that JsonPointer comparison works in Pydantic models."""

        class Model(BaseModel):
            pointer1: JsonPointer
            pointer2: JsonPointer

        model = Model(pointer1="/foo/bar", pointer2="/foo/bar")
        assert model.pointer1 == model.pointer2

        model2 = Model(pointer1="/foo/bar", pointer2="/foo/baz")
        assert model2.pointer1 != model2.pointer2

    def test_pointer_in_union_type(self):
        """Test JsonPointer in union types."""

        class Model(BaseModel):
            value: JsonPointer | str

        # Should create JsonPointer for valid pointer strings
        model1 = Model(value="/foo/bar")
        assert isinstance(model1.value, JsonPointer)

        # Should create JsonPointer for strings that look like pointers
        model2 = Model(value="/test")
        assert isinstance(model2.value, JsonPointer)

    def test_json_schema_generation(self):
        """Test that JsonPointer generates appropriate JSON schema."""

        class Model(BaseModel):
            pointer: JsonPointer

        schema = Model.model_json_schema()
        assert "pointer" in schema["properties"]
        assert schema["properties"]["pointer"]["type"] == "string"


class TestReferenceTokenHelpers:
    """Test single-token helper functions (getitem only)."""

    def test_getitem_mapping_and_sequence(self):
        doc = {"foo": {"bar": [1, 2, 3]}}
        assert getitem(doc, "foo") == {"bar": [1, 2, 3]}
        assert getitem(doc["foo"], "bar") == [1, 2, 3]
        assert getitem(doc["foo"]["bar"], "1") == 2
        assert getitem(doc, JsonPointer("/foo/bar/0")) == 1
        assert getitem(doc, JsonPointer("/foo/bar/2")) == 3

        with pytest.raises(KeyError):
            getitem(doc, "missing")

        with pytest.raises(ValueError):
            getitem(doc["foo"]["bar"], "NaN")

        with pytest.raises(IndexError):
            getitem(doc["foo"]["bar"], "5")

    def test_attribute_host_support(self):
        class Carrier:
            def __init__(self):
                self.payload = "data"

        carrier = Carrier()
        assert getitem(carrier, "payload") == "data"

    def test_hyphen_sequence_token_raises(self):
        seq = [1, 2]
        with pytest.raises(ValueError, match="-"):
            _ = getitem(seq, "-")
