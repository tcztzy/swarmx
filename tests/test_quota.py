import pytest

from swarmx.quota import QuotaExceededError, QuotaManager

pytestmark = pytest.mark.anyio


async def test_quota_manager_integration():
    manager = QuotaManager(max_tokens=5)
    await manager.consume("agent", 3)
    assert manager.used_tokens == 3
    with pytest.raises(QuotaExceededError):
        await manager.consume("agent", 3)
