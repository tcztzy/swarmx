"""Tests for the utils module."""

import asyncio
from datetime import datetime
from unittest.mock import patch

import pytest

from swarmx.utils import get_random_string, join, now, read_into_queue

pytestmark = pytest.mark.anyio


def test_now():
    """Test the now function returns a timestamp."""
    timestamp = now()
    assert isinstance(timestamp, int)
    assert timestamp > 0

    # Test that it's close to current time (within 1 second)
    current_time = int(datetime.now().timestamp())
    assert abs(timestamp - current_time) <= 1


def test_get_random_string_default():
    """Test get_random_string with default parameters."""
    result = get_random_string(10)
    assert len(result) == 10
    assert all(c.isalnum() for c in result)


def test_get_random_string_custom_length():
    """Test get_random_string with custom length."""
    result = get_random_string(5)
    assert len(result) == 5

    result = get_random_string(20)
    assert len(result) == 20


def test_get_random_string_custom_chars():
    """Test get_random_string with custom allowed characters."""
    custom_chars = "abc123"
    result = get_random_string(10, custom_chars)
    assert len(result) == 10
    assert all(c in custom_chars for c in result)


def test_get_random_string_uniqueness():
    """Test that get_random_string generates unique strings."""
    results = [get_random_string(10) for _ in range(100)]
    # All results should be unique
    assert len(set(results)) == len(results)


async def test_read_into_queue():
    """Test read_into_queue function."""

    async def test_generator():
        for i in range(3):
            yield i

    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(1)

    await read_into_queue(test_generator(), queue, semaphore)

    # Check that all items are in the queue
    items = []
    while not queue.empty():
        items.append(await queue.get())

    assert items == [0, 1, 2]
    # Semaphore should be acquired (locked)
    assert semaphore.locked()


async def test_join_single_generator():
    """Test join function with a single generator."""

    async def test_generator():
        for i in range(3):
            yield i

    result = []
    async for item in join(test_generator()):
        result.append(item)

    assert result == [0, 1, 2]


async def test_join_multiple_generators():
    """Test join function with multiple generators."""

    async def gen1():
        for i in [1, 3, 5]:
            yield i
            await asyncio.sleep(0.001)  # Small delay

    async def gen2():
        for i in [2, 4, 6]:
            yield i
            await asyncio.sleep(0.001)  # Small delay

    result = []
    async for item in join(gen1(), gen2()):
        result.append(item)

    # All items should be present (order may vary due to async nature)
    assert sorted(result) == [1, 2, 3, 4, 5, 6]
    assert len(result) == 6


async def test_join_empty_generators():
    """Test join function with empty generators."""

    async def empty_gen():
        return
        yield  # This line will never be reached

    result = []
    async for item in join(empty_gen(), empty_gen()):
        result.append(item)

    assert result == []


async def test_join_timeout_error_handling():
    """Test join function handles timeout errors in gather."""

    async def slow_gen():
        yield 1
        await asyncio.sleep(0.1)  # Short delay

    # Mock asyncio.gather to raise TimeoutError
    with patch("asyncio.gather") as mock_gather:
        mock_gather.side_effect = asyncio.TimeoutError()

        with pytest.raises(NotImplementedError, match="Impossible state"):
            result = []
            async for item in join(slow_gen()):
                result.append(item)
