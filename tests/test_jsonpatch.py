"""Tests for RFC 6902 JSON Patch operations."""

import pytest

from swarmx.jsonpatch import (
    AddOperation,
    CopyOperation,
    JsonPatch,
    MoveOperation,
    RemoveOperation,
    ReplaceOperation,
    TestOperation,
)
from swarmx.jsonpointer import JsonPointer


def test_add_operation_inserts_into_mapping_and_sequence():
    doc = {"foo": {"bar": [1, 2]}}
    op1 = AddOperation(path=JsonPointer("/foo/baz"), value=3)
    patched = op1.apply(doc)
    assert patched["foo"]["baz"] == 3
    assert doc["foo"].get("baz") is None  # original unchanged

    op2 = AddOperation(path=JsonPointer("/foo/bar/1"), value="X")
    patched2 = op2.apply(patched)
    assert patched2["foo"]["bar"] == [1, "X", 2]

    with pytest.raises(IndexError):
        AddOperation(path=JsonPointer("/foo/bar/5"), value=0).apply(doc)


def test_add_operation_appends_with_hyphen():
    doc = {"items": [1, 2]}
    op = AddOperation(path=JsonPointer("/items/-"), value=3)
    patched = op.apply(doc)
    assert patched["items"] == [1, 2, 3]


def test_remove_operation_removes_values():
    doc = {"foo": {"bar": [1, 2, 3]}}
    op = RemoveOperation(path=JsonPointer("/foo/bar/1"))
    patched = op.apply(doc)
    assert patched["foo"]["bar"] == [1, 3]
    assert doc["foo"]["bar"] == [1, 2, 3]

    with pytest.raises(ValueError):
        RemoveOperation(path=JsonPointer("")).apply(doc)

    with pytest.raises(KeyError):
        RemoveOperation(path=JsonPointer("/foo/missing")).apply(doc)


def test_replace_operation_overwrites_value():
    doc = {"a": 1}
    op = ReplaceOperation(path=JsonPointer("/a"), value=99)
    patched = op.apply(doc)
    assert patched["a"] == 99
    assert doc["a"] == 1

    with pytest.raises(KeyError):
        ReplaceOperation(path=JsonPointer("/missing"), value=1).apply(doc)


def test_move_operation_transfers_value():
    doc = {"a": {"b": [1, 2]}, "c": {}}
    op = MoveOperation.model_validate({"path": "/c/num", "from": "/a/b/0"})
    patched = op.apply(doc)
    assert patched["c"]["num"] == 1
    assert patched["a"]["b"] == [2]

    with pytest.raises(ValueError):
        MoveOperation.model_validate({"path": "/a/b", "from": "/a"}).apply(doc)


def test_copy_operation_copies_by_value():
    doc = {"a": {"b": [1, {"x": 2}]}}
    op = CopyOperation.model_validate({"path": "/a/copy", "from": "/a/b/1"})
    patched = op.apply(doc)
    assert patched["a"]["copy"] == {"x": 2}
    # mutate original to confirm deep copy
    patched["a"]["copy"]["x"] = 5
    assert doc["a"]["b"][1]["x"] == 2


def test_test_operation_pass_and_fail():
    doc = {"a": {"b": 2}}
    op = TestOperation(path=JsonPointer("/a/b"), value=2)
    assert op.apply(doc) == doc

    with pytest.raises(AssertionError):
        TestOperation(path=JsonPointer("/a/b"), value=3).apply(doc)

    with pytest.raises(AssertionError):
        TestOperation(path=JsonPointer("/missing"), value=1).apply(doc)


def test_json_patch_sequence_application():
    doc = {"items": [1, 2]}
    patch = JsonPatch.model_validate(
        {
            "patch": [
                {"op": "add", "path": "/items/2", "value": 3},
                {"op": "replace", "path": "/items/0", "value": 9},
                {"op": "test", "path": "/items/1", "value": 2},
            ]
        }
    )
    result = patch.apply(doc)
    assert result["items"] == [9, 2, 3]
    assert doc["items"] == [1, 2]


def test_move_and_copy_can_be_rewritten_to_atomic_operations():
    doc = {"items": [1, {"n": 2}], "dest": {}}
    move_op = MoveOperation.model_validate({"path": "/dest/moved", "from": "/items/1"})
    copy_op = CopyOperation.model_validate({"path": "/dest/copied", "from": "/items/0"})

    move_atomic = move_op.to_atomic_operations(doc)
    copy_atomic = copy_op.to_atomic_operations(doc)

    assert [op.__class__ for op in move_atomic] == [
        RemoveOperation,
        AddOperation,
    ]
    assert move_atomic[0].op == "remove" and move_atomic[0].path == "/items/1"
    assert (
        move_atomic[1].op == "add"
        and move_atomic[1].path == "/dest/moved"
        and move_atomic[1].value == {"n": 2}
    )

    assert [op.__class__ for op in copy_atomic] == [AddOperation]
    assert (
        copy_atomic[0].op == "add"
        and copy_atomic[0].path == "/dest/copied"
        and copy_atomic[0].value == 1
    )

    # Apply through the generated atomic operations to confirm equivalence.
    patched_via_atomic = JsonPatch(patch=[*move_atomic, *copy_atomic]).apply(doc)
    patched_via_apply = copy_op.apply(move_op.apply(doc))
    assert patched_via_atomic == patched_via_apply
