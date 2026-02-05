# SwarmX Hooks

SwarmX supports lifecycle hooks that allow you to execute MCP tools at specific points during agent execution. This enables powerful capabilities like logging, metrics collection, debugging, and custom workflow orchestration.

## Hook Overview

A `Hook` is a Pydantic BaseModel that defines which MCP tools to execute at various lifecycle events:

- `on_start`: Executed when the agent begins processing
- `on_end`: Executed when the agent finishes processing
- `on_handoff`: Reserved for handoff events (not triggered by the current runtime)
- `on_chunk`: Executed after each streamed chunk

## Creating Hooks

```python
from swarmx import Hook

# Create a logging hook
logging_hook = Hook(
    on_start="log_agent_start",
    on_chunk="log_stream_chunk",
    on_end="log_agent_end"
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

1. **on_start** - Called at the beginning of processing
2. **on_chunk** - Called for each streamed chunk
3. **on_end** - Called at the end of processing
4. **on_handoff** - Reserved for future handoff events

## Hook Tool Requirements

Hook tools must be available in your MCP server configuration. The tools receive the current messages, context, and agent metadata. Structured output can update context.

### Input Format

Hook tools receive input in this format:
```python
{
    "messages": [...],  # Current conversation messages
    "context": {...},   # Agent context (can be None)
    "agent": {...}      # Agent metadata (name)
}
```

### Output Format

Hook tools can return structured output that is merged directly into context:
```python
{
    "system_info": {...}  # Added to context
}
```

If no modifications are needed, return an empty structured output or omit structured output.

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
    return {}

@server.call_tool()
async def add_system_context(input_data: dict) -> dict:
    """Add system information to context."""
    messages = input_data["messages"]
    context = input_data.get("context") or {}

    # Modify context
    return {
        "system_info": {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
        }
    }

@server.call_tool()
async def filter_messages(input_data: dict) -> dict:
    """Filter out certain message types."""
    messages = input_data["messages"]
    context = input_data["context"]
    logger.info("Received %d messages", len(messages))
    return {}
```

## Error Handling

If a hook tool raises, the exception propagates and will stop agent execution. Handle errors inside your hook tools if you want a best-effort behavior.

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
