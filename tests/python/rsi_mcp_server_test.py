"""Acceptance tests for the private RSI MCP server boundary."""

from __future__ import annotations

import sys
import unittest

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from swarmx.rsi import optimize
from swarmx.rsi import client as rsi_client
from swarmx.worker import deterministic_config_digest
from roundtrip_test import (
    BASELINE,
    DEV_TEXT,
    TRAIN_TEXT,
    build_request,
)


class RsiMcpServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_dspy_gepa_is_exposed_only_as_mcp_tool(self) -> None:
        request, artifacts = fixture()
        async with stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=["-I", "-B", "-u", "-m", "swarmx.rsi.server"],
            )
        ) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                initialized = await session.initialize()
                self.assertEqual(initialized.serverInfo.name, "swarmx-rsi")
                tools = await session.list_tools()
                self.assertEqual(
                    [tool.name for tool in tools.tools], ["swarmx_rsi_optimize"]
                )
                result = await session.call_tool(
                    "swarmx_rsi_optimize",
                    {"request": request, "artifacts": artifacts},
                )
                self.assertFalse(result.isError)
                self.assertIsNotNone(result.structuredContent)
                output = result.structuredContent or {}
                self.assertIn("parrot", output["candidateMarkdown"])
                self.assertEqual(
                    output["optimizerReport"]["actualOptimizer"], "dspy.gepa.v1"
                )

    async def test_unknown_input_fails_without_result(self) -> None:
        request, artifacts = fixture()
        request["unauthorized"] = "value"
        async with stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=["-I", "-B", "-u", "-m", "swarmx.rsi.server"],
            )
        ) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                result = await session.call_tool(
                    "swarmx_rsi_optimize",
                    {"request": request, "artifacts": artifacts},
                )
                self.assertTrue(result.isError)
                self.assertIsNone(result.structuredContent)

    async def test_artifact_content_must_match_its_granted_digest(self) -> None:
        request, artifacts = fixture()
        artifacts[request["baselineContentRef"]] = "tampered"
        async with stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=["-I", "-B", "-u", "-m", "swarmx.rsi.server"],
            )
        ) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                result = await session.call_tool(
                    "swarmx_rsi_optimize",
                    {"request": request, "artifacts": artifacts},
                )
                self.assertTrue(result.isError)
                self.assertIsNone(result.structuredContent)

    async def test_dependency_free_optimizer_stays_in_the_worker(self) -> None:
        request, artifacts = fixture()
        request["optimizer"]["optimizerId"] = "deterministic.v1"
        request["optimizer"]["configDigest"] = deterministic_config_digest(request)
        request["proposer"] = "none"
        async with stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=["-I", "-B", "-u", "-m", "swarmx.rsi.server"],
            )
        ) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                result = await session.call_tool(
                    "swarmx_rsi_optimize",
                    {"request": request, "artifacts": artifacts},
                )
                self.assertTrue(result.isError)
                self.assertIsNone(result.structuredContent)


class RsiMcpClientTest(unittest.TestCase):
    def test_dspy_gepa_runs_through_the_worker_side_mcp_client(self) -> None:
        request, artifacts = fixture()
        progress: list[tuple[str, float]] = []
        report, candidate = rsi_client.run_rsi_optimizer(
            request=request,
            artifacts=artifacts,
            capability_client=NoModelCapabilityClient(),
            cancel_check=lambda: False,
            progress=lambda message, fraction: progress.append((message, fraction)),
        )

        self.assertEqual(report["actualOptimizer"], "dspy.gepa.v1")
        self.assertIn("parrot", candidate)
        self.assertTrue(progress)


class NoModelCapabilityClient:
    def call(self, *_args, **_kwargs):
        raise AssertionError("deterministic GEPA must not request MCP sampling")


def fixture() -> tuple[dict, dict[str, str]]:
    request = build_request()
    request["optimizer"]["configDigest"] = optimize.canonical_config_digest(request)
    return request, {
        request["baselineContentRef"]: BASELINE,
        request["trainDataset"]["contentRef"]: TRAIN_TEXT,
        request["devDataset"]["contentRef"]: DEV_TEXT,
    }


if __name__ == "__main__":
    unittest.main()
