import datetime
import json

import pytest
from mcp.client.stdio import StdioServerParameters

from swarmx import AsyncSwarm


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_mcp_server(anyio_backend):
    async with AsyncSwarm(
        mcp_servers={
            "time": StdioServerParameters(
                command="uvx",
                args=["mcp-server-time", "--local-timezone", "UTC"],
            )
        }
    ) as client:
        assert client.mcp_servers["time"] == StdioServerParameters(
            command="uvx",
            args=["mcp-server-time", "--local-timezone", "UTC"],
        )
        server, tool = client.tool_registry["get_current_time"]
        assert tool == {
            "function": {
                "description": "Get current time in a specific timezones",
                "name": "get_current_time",
                "parameters": {
                    "properties": {
                        "timezone": {
                            "description": "IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use 'UTC' as local timezone if no timezone provided by the user.",
                            "type": "string",
                        }
                    },
                    "required": ["timezone"],
                    "type": "object",
                },
            },
            "type": "function",
        }
        result = await client.mcp_clients[server].call_tool(
            "get_current_time", {"timezone": "UTC"}
        )
        assert result.content[0].type == "text"
        json_result = json.loads(result.content[0].text)
        assert (
            "timezone" in json_result
            and "datetime" in json_result
            and "is_dst" in json_result
        )
        assert datetime.datetime.fromisoformat(
            json_result["datetime"]
        ) - datetime.datetime.now(datetime.timezone.utc) < datetime.timedelta(seconds=1)
