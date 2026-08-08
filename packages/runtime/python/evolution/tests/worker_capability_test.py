"""Unit tests for the worker capability client budget, cancel, and timeout rules.

The capability client lives in the dependency-free `swarmx_worker.py`; the
tests import it directly with a fake runtime so no subprocess is needed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from swarmx_worker import (  # noqa: E402
    CapabilityBudgetError,
    CapabilityCancelledError,
    CapabilityClient,
    CapabilityTimeoutError,
    WorkerRuntime,
)


class FakeRuntime(WorkerRuntime):
    def __init__(self) -> None:
        super().__init__("sha256:" + "0" * 64)
        self.emitted: list[dict] = []

    def _emit(self, message_type: str, payload: dict) -> None:
        self.emitted.append({"type": message_type, **payload})


def outcome(content: str = "x", tokens: int = 1) -> dict:
    return {
        "status": "succeeded",
        "value": {"content": content, "usage": {"totalTokens": tokens}},
        "artifactIds": [],
    }


def auto_deliver(client: CapabilityClient) -> None:
    """Patch runtime._emit so each capability_call is answered asynchronously."""
    runtime = client._runtime
    original_emit = runtime._emit

    def patched_emit(message_type: str, payload: dict) -> None:
        original_emit(message_type, payload)
        if message_type == "capability_call":
            call_id = payload["callId"]

            def deliver() -> None:
                time.sleep(0.01)
                client.deliver_result(
                    {"callId": call_id, "outcome": outcome(content=call_id[-4:])}
                )

            threading.Thread(target=deliver, daemon=True).start()

    runtime._emit = patched_emit  # type: ignore[method-assign]


class CapabilityClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()

    def tearDown(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)

    def test_budget_is_enforced_per_capability_operation(self) -> None:
        runtime = FakeRuntime()
        client = CapabilityClient(
            runtime,
            ["gnt_evolve_1"],
            {"skill_evolution:read_artifact": 1, "skill_evolution:model.generate": 5},
            "run_test",
            self.loop,
        )
        auto_deliver(client)
        result = client.call(
            "skill_evolution", "read_artifact", {"ref": "sha256:" + "a" * 64}
        )
        self.assertEqual(result["status"], "succeeded")
        with self.assertRaises(CapabilityBudgetError):
            client.call(
                "skill_evolution", "read_artifact", {"ref": "sha256:" + "a" * 64}
            )
        # A different operation keeps its own budget.
        result = client.call("skill_evolution", "model.generate", {"messages": []})
        self.assertEqual(result["status"], "succeeded")

    def test_token_budget_is_charged_from_model_generate_outcomes(self) -> None:
        runtime = FakeRuntime()
        client = CapabilityClient(
            runtime,
            ["gnt_evolve_1"],
            {"skill_evolution:model.generate": 10},
            "run_test",
            self.loop,
            token_budget=5,
        )

        def patched_emit(message_type: str, payload: dict) -> None:
            FakeRuntime._emit(runtime, message_type, payload)
            if message_type == "capability_call":
                call_id = payload["callId"]

                def deliver() -> None:
                    time.sleep(0.01)
                    client.deliver_result(
                        {"callId": call_id, "outcome": outcome(tokens=3)}
                    )

                threading.Thread(target=deliver, daemon=True).start()

        runtime._emit = patched_emit  # type: ignore[method-assign]
        client.call("skill_evolution", "model.generate", {"messages": []})
        with self.assertRaises(CapabilityBudgetError):
            client.call("skill_evolution", "model.generate", {"messages": []})

    def test_zero_token_budget_denies_model_calls_before_dispatch(self) -> None:
        runtime = FakeRuntime()
        client = CapabilityClient(
            runtime,
            ["gnt_evolve_1"],
            {"skill_evolution:model.generate": 10},
            "run_test",
            self.loop,
            token_budget=0,
        )
        with self.assertRaises(CapabilityBudgetError):
            client.call("skill_evolution", "model.generate", {"messages": []})
        self.assertEqual(runtime.emitted, [])

    def test_exhausted_token_budget_denies_before_the_next_call(self) -> None:
        runtime = FakeRuntime()
        client = CapabilityClient(
            runtime,
            ["gnt_evolve_1"],
            {"skill_evolution:model.generate": 10},
            "run_test",
            self.loop,
            token_budget=5,
        )
        client._used_tokens = 5
        with self.assertRaises(CapabilityBudgetError):
            client.call("skill_evolution", "model.generate", {"messages": []})

    def test_timeout_and_cancel(self) -> None:
        runtime = FakeRuntime()
        client = CapabilityClient(
            runtime,
            ["gnt_evolve_1"],
            {"skill_evolution:read_artifact": 10},
            "run_test",
            self.loop,
        )
        with self.assertRaises(CapabilityTimeoutError):
            client.call(
                "skill_evolution",
                "read_artifact",
                {"ref": "sha256:" + "a" * 64},
                timeout_ms=50,
            )
        runtime.cancel_event.set()
        with self.assertRaises(CapabilityCancelledError):
            client.call(
                "skill_evolution",
                "read_artifact",
                {"ref": "sha256:" + "a" * 64},
                timeout_ms=50,
            )


if __name__ == "__main__":
    unittest.main()
