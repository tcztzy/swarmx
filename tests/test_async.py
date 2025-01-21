import datetime
import json
import re

import pytest
from mcp.client.stdio import StdioServerParameters

from swarmx import Agent, AsyncSwarm

pytestmark = pytest.mark.anyio


async def test_mcp_server():
    async with AsyncSwarm(
        mcp_servers={
            "time": StdioServerParameters(
                command="uv",
                args=["run", "mcp-server-time", "--local-timezone", "UTC"],
            )
        }
    ) as client:
        assert client.mcp_servers["time"] == StdioServerParameters(
            command="uv",
            args=["run", "mcp-server-time", "--local-timezone", "UTC"],
        )
        server, tool = client._tool_registry["get_current_time"]
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
        result = await client._mcp_clients[server].call_tool(
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


async def test_mcp_tool_call(mcp_tool_call_async_client: AsyncSwarm):
    client = mcp_tool_call_async_client
    client.mcp_servers = {
        "time": StdioServerParameters(
            command="uv",
            args=["run", "mcp-server-time", "--local-timezone", "UTC"],
        )
    }
    agent = Agent()
    response = await client.run(
        agent=agent,
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": "What time is it now? UTC time is okay. "
                "You should only answer time in %H:%M:%S format without "
                "any other characters, e.g. 12:34:56",
            }
        ],
    )
    message = response.messages[-1]
    assert message.get("name") == "Agent"
    now = datetime.datetime.now(datetime.timezone.utc)
    content = message.get("content")
    assert isinstance(content, str)
    mo = re.search(r"\d{2}:\d{2}:\d{2}", content)
    assert mo is not None
    answer_time = datetime.datetime.strptime(mo.group(), "%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )
    assert answer_time - now < datetime.timedelta(seconds=1)
