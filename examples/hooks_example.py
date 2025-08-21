"""Example demonstrating Hook functionality in SwarmX.

This example shows how to use hooks to execute MCP tools at various points
in the agent lifecycle. Hook tools can modify both messages and context
through structured output.
"""

import asyncio
import json

from swarmx import Agent, Hook


async def main():
    """Demonstrate hook functionality."""
    # Create hooks that reference MCP tool names
    # These tools would need to be available in your MCP server
    logging_hook = Hook(
        on_start="log_agent_start",
        on_end="log_agent_end",
        on_tool_start="log_tool_start",
        on_tool_end="log_tool_end",
        on_subagents_start="log_subagents_start",
        on_subagents_end="log_subagents_end",
    )

    # Create another hook for metrics collection
    metrics_hook = Hook(
        on_start="start_timer",
        on_end="end_timer_and_record_metrics",
    )

    # Create an agent with hooks
    agent = Agent(
        name="HookedAgent",
        instructions="You are a helpful assistant with lifecycle hooks.",
        hooks=[logging_hook, metrics_hook],
    )

    # Example of hook serialization
    print("Hook serialization example:")
    print(json.dumps(logging_hook.model_dump(), indent=2))
    print()

    # Example of agent with hooks serialization
    print("Agent with hooks serialization:")
    agent_dict = agent.model_dump()
    print(f"Agent has {len(agent_dict['hooks'])} hooks")
    print(json.dumps(agent_dict["hooks"], indent=2))
    print()

    # Example of deserializing agent with hooks
    print("Deserializing agent with hooks:")
    restored_agent = Agent.model_validate(agent_dict)
    print(f"Restored agent has {len(restored_agent.hooks)} hooks")
    print(f"First hook on_start: {restored_agent.hooks[0].on_start}")
    print()

    # Note: To actually run the agent, you would need:
    # 1. MCP servers configured with the hook tools that accept input format:
    #    {"messages": [...], "context": {...}}
    # 2. Hook tools that return structured output:
    #    {"messages": [...], "context": {...}}
    # 3. Proper OpenAI API configuration
    #
    # Example hook tool implementation:
    # @server.call_tool()
    # async def log_agent_start(input_data: dict) -> dict:
    #     messages = input_data["messages"]
    #     context = input_data["context"]
    #     logger.info(f"Agent started with {len(messages)} messages")
    #     return {"messages": messages, "context": context}
    #
    # Example run (commented out):
    # messages = [{"role": "user", "content": "Hello!"}]
    # response = await agent.run(messages=messages)
    # print("Agent response:", response)


if __name__ == "__main__":
    asyncio.run(main())
