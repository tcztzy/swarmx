"""Real stdio-MCP acceptance test for the offline Reference server."""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from libzim.writer import Creator, Hint, Item, StringProvider
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class FixtureItem(Item):
    def __init__(self, title: str, path: str, content: str) -> None:
        super().__init__()
        self._title = title
        self._path = path
        self._content = content

    def get_path(self) -> str:
        return self._path

    def get_title(self) -> str:
        return self._title

    def get_mimetype(self) -> str:
        return "text/html"

    def get_contentprovider(self) -> StringProvider:
        return StringProvider(self._content)

    def get_hints(self) -> dict[Hint, bool]:
        return {Hint.FRONT_ARTICLE: True}


class ReferenceMcpServerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="swarmx-ref-")
        self.zim_path = Path(self._temporary.name) / "fixture.zim"
        illustration = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwAQMAAABtzGvEAAAAGXRFWHRTb2Z0d2FyZQBB"
            "ZG9iZSBJbWFnZVYWR5ccllPAAAAANQTFRFR3BMgvrS0gAAAAF0Uk5TAEDm2GYAAAAN"
            "SURBVBjTY2AYBdQEAAFQAAGn4toWAAAAAElFTkSuQmCC=="
        )
        with Creator(str(self.zim_path)).config_indexing(True, "zh") as creator:
            creator.set_mainpath("home")
            creator.add_item(
                FixtureItem(
                    "SwarmX Reference",
                    "home",
                    "<h1>SwarmX</h1><script>ignored()</script><p>Objective source.</p>",
                )
            )
            creator.add_illustration(48, illustration)
            for name, value in {
                "Creator": "SwarmX tests",
                "Description": "Reference fixture",
                "Name": "swarmx-ref-fixture",
                "Publisher": "SwarmX",
                "Title": "SwarmX Reference Fixture",
                "Language": "zho",
                "Date": "2026-08-10",
            }.items():
                creator.add_metadata(name, value)

    async def asyncTearDown(self) -> None:
        self._temporary.cleanup()

    async def test_exposes_one_read_only_tool_over_real_stdio_mcp(self) -> None:
        server = StdioServerParameters(
            command="python",
            args=[
                "-I",
                "-B",
                "-u",
                "-m",
                "swarmx.ref.server",
                "--zim",
                str(self.zim_path),
                "--stdio",
            ],
        )
        with self.assertNoLogs("mcp.client.stdio", level="ERROR"):
            async with stdio_client(server) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    initialized = await session.initialize()
                    self.assertEqual(initialized.serverInfo.name, "swarmx-ref")
                    tools = await session.list_tools()
                    self.assertEqual(
                        [tool.name for tool in tools.tools], ["swarmx_reference"]
                    )
                    status = await session.call_tool(
                        "swarmx_reference", {"request": {"operation": "status"}}
                    )
                    self.assertEqual(
                        status.structuredContent["sources"][0]["id"], "zim"
                    )
                    self.assertEqual(
                        status.structuredContent["source"]["fileName"], "fixture.zim"
                    )
                    page = await session.call_tool(
                        "swarmx_reference",
                        {"request": {"operation": "get", "path": "home"}},
                    )
                    self.assertFalse(page.isError)
                    self.assertEqual(page.structuredContent["operation"], "get")
                    self.assertIn("Objective source.", page.structuredContent["text"])
                    self.assertNotIn("ignored()", page.structuredContent["text"])
                    search = await session.call_tool(
                        "swarmx_reference",
                        {
                            "request": {
                                "operation": "search",
                                "query": "Objective",
                                "limit": 5,
                            }
                        },
                    )
                    self.assertFalse(search.isError)
                    self.assertEqual(
                        search.structuredContent["matches"][0]["path"], "home"
                    )
                    mutation = await session.call_tool(
                        "swarmx_reference",
                        {"request": {"operation": "create", "title": "Opinion"}},
                    )
                    self.assertTrue(mutation.isError)


if __name__ == "__main__":
    unittest.main()
