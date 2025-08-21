# SwarmX Hooks

SwarmX supports lifecycle hooks that allow you to execute MCP tools at specific points during agent execution. This enables powerful capabilities like logging, metrics collection, debugging, and custom workflow orchestration.

## Hook Overview

A `Hook` is a Pydantic BaseModel that defines which MCP tools to execute at various lifecycle events:

- `on_start`: Executed when the agent begins processing
- `on_end`: Executed when the agent finishes processing  
- `on_tool_start`: Executed before any tool call
- `on_tool_end`: Executed after any tool call
- `on_subagents_start`: Executed before subagent processing begins
- `on_subagents_end`: Executed after subagent processing ends

## Creating Hooks

```python
from swarmx import Hook

# Create a logging hook
logging_hook = Hook(
    on_start="log_agent_start",
    on_end="log_agent_end",
    on_tool_start="log_tool_start", 
    on_tool_end="log_tool_end"
)

# Create a metrics hook
metrics_hook = Hook(
    on_start="start_timer",
    on_end="record_execution_time"
)
```

## Adding Hooks to Agents

```python
from swarmx import Agent, Hook

# Create hooks
hook1 = Hook(on_start="initialize_session")
hook2 = Hook(on_end="cleanup_session")

# Add hooks to agent
agent = Agent(
    name="MyAgent",
    hooks=[hook1, hook2]
)
```

## Hook Execution

Hooks are executed automatically during agent lifecycle:

1. **on_start** - Called at the beginning of `_run()` and `_run_stream()`
2. **on_tool_start** - Called before executing any tool calls
3. **on_tool_end** - Called after executing tool calls
4. **on_subagents_start** - Called before running subagents
5. **on_subagents_end** - Called after running subagents  
6. **on_end** - Called at the end of processing

## Hook Tool Requirements

Hook tools must be available in your MCP server configuration. The tools receive both the current messages and context as input parameters, and can return modified versions through structured output.

### Input Format

Hook tools receive input in this format:
```python
{
    "messages": [...],  # Current conversation messages
    "context": {...}    # Agent context (can be None)
}
```

### Output Format

Hook tools can return structured output to modify messages and context:
```python
{
    "messages": [...],  # Modified messages
    "context": {...}    # Modified context (can be None)
}
```

If no modifications are needed, return the input unchanged.

### Example MCP Tools

```python
# In your MCP server
@server.call_tool()
async def log_agent_start(input_data: dict) -> dict:
    """Log when an agent starts processing."""
    messages = input_data["messages"]
    context = input_data["context"]

    logger.info(f"Agent started with {len(messages)} messages and context: {context}")

    # Return unchanged if no modifications needed
    return {
        "messages": messages,
        "context": context
    }

@server.call_tool()
async def add_system_context(input_data: dict) -> dict:
    """Add system information to context."""
    messages = input_data["messages"]
    context = input_data.get("context") or {}

    # Modify context
    context["system_info"] = {
        "timestamp": datetime.now().isoformat(),
        "message_count": len(messages)
    }

    return {
        "messages": messages,
        "context": context
    }

@server.call_tool()
async def filter_messages(input_data: dict) -> dict:
    """Filter out certain message types."""
    messages = input_data["messages"]
    context = input_data["context"]

    # Filter messages
    filtered_messages = [
        msg for msg in messages
        if msg.get("role") != "system" or "important" in msg.get("content", "")
    ]

    return {
        "messages": filtered_messages,
        "context": context
    }
```

## Error Handling

If a hook tool fails, the error is logged but does not stop agent execution. This ensures that hook failures don't break your main workflow.

## Serialization

Hooks are fully serializable since they only store tool names (strings) rather than function references:

```python
# Serialize
hook_dict = hook.model_dump()

# Deserialize  
restored_hook = Hook.model_validate(hook_dict)
```

## Use Cases

- **Logging**: Track agent execution flow
- **Metrics**: Measure performance and usage
- **Debugging**: Inspect agent state at key points
- **Auditing**: Record all agent actions
- **Integration**: Trigger external systems
- **Workflow**: Coordinate complex multi-agent processes

## Example

See `examples/hooks_example.py` for a complete working example.
