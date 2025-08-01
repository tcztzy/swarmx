"""SwarmX Utils module."""

import asyncio
import secrets
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TypeVar

T = TypeVar("T")


def now():
    """OpenAI compatible timestamp in integer."""
    return int(datetime.now().timestamp())


RANDOM_STRING_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def get_random_string(length, allowed_chars=RANDOM_STRING_CHARS):
    """Return a securely generated random string.

    The bit length of the returned value can be calculated with the formula:
        log_2(len(allowed_chars)^length)

    For example, with default `allowed_chars` (26+26+10), this gives:
      * length: 12, bit length =~ 71 bits
      * length: 22, bit length =~ 131 bits
    """
    return "".join(secrets.choice(allowed_chars) for i in range(length))


async def read_into_queue(
    task: AsyncGenerator[T, None],
    queue: asyncio.Queue[T],
    done: asyncio.Semaphore,
) -> None:
    """Read items from the task and put them into the queue."""
    async for item in task:
        await queue.put(item)
    # All items from this task are in the queue, decrease semaphore by one.
    await done.acquire()


async def join(*generators: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    """Join multiple async generators into one."""
    queue: asyncio.Queue[T] = asyncio.Queue(maxsize=1)
    done_semaphore = asyncio.Semaphore(len(generators))

    # Read from each given generator into the shared queue.
    produce_tasks = [
        asyncio.create_task(read_into_queue(task, queue, done_semaphore))
        for task in generators
    ]

    # Read items off the queue until it is empty and the semaphore value is down to zero.
    while not done_semaphore.locked() or not queue.empty():
        try:
            yield await asyncio.wait_for(queue.get(), 0.001)
        except TimeoutError:
            continue

    # Not strictly needed, but usually a good idea to await tasks, they are already finished here.
    try:
        await asyncio.wait_for(asyncio.gather(*produce_tasks), 0)
    except TimeoutError:
        raise NotImplementedError(
            "Impossible state: expected all tasks to be exhausted"
        )
