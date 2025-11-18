"""Extended coverage tests for swarmx.agent internals."""

import pytest

from swarmx.agent import QuotaManager
from swarmx.quota import QuotaExceededError
from swarmx.utils import join

pytestmark = pytest.mark.anyio


async def test_quota_manager_integration():
    manager = QuotaManager(max_tokens=5)
    await manager.consume("agent", 3)
    assert manager.used_tokens == 3
    with pytest.raises(QuotaExceededError):
        await manager.consume("agent", 3)


async def test_join_combines_generators():
    async def gen(prefix: str):
        for ch in "ab":
            yield f"{prefix}-{ch}"

    combined = [item async for item in join(gen("x"), gen("y"))]
    assert set(combined) == {"x-a", "x-b", "y-a", "y-b"}
