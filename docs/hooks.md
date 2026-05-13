# SwarmX Hooks

SwarmX supports lifecycle hooks that allow you to execute MCP tools at specific points during agent execution. This enables powerful capabilities like logging, metrics collection, debugging, and custom workflow orchestration.

## Hook Overview

A `Hook` is a serializable struct that defines which MCP tools to execute at various lifecycle events:

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
```json
{
    "messages": [...],
    "context": {...},
    "agent": {"name": "AgentName"}
}
```

### Output Format

Hook tools can return structured output that is merged directly into context:
```json
{
    "system_info": {...}
}
```

If no modifications are needed, return an empty structured output or omit structured output.

### Example MCP Tools

In your MCP server, implement tools that accept the hook input format:

```python
# In your MCP server (Python MCP SDK still commonly used for server implementation)
@server.call_tool()
async def log_agent_start(input_data: dict) -> dict:
    messages = input_data["messages"]
    context = input_data["context"]
    logger.info(f"Agent started with {len(messages)} messages")
    return {}

@server.call_tool()
async def add_system_context(input_data: dict) -> dict:
    messages = input_data["messages"]
    context = input_data.get("context") or {}
    return {
        "system_info": {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
        }
    }
```

## Error Handling

If a hook tool returns an error, the error propagates and will stop agent execution. Handle errors inside your hook tools if you want best-effort behavior.

## Serialization

Hooks are fully serializable since they only store tool names (strings) rather than function references:

```rust
use swarmx_core::Hook;

let hook = Hook {
    on_start: Some("log_start".to_string()),
    ..Default::default()
};

// Serialize
let json = serde_json::to_string(&hook).unwrap();

// Deserialize
let restored: Hook = serde_json::from_str(&json).unwrap();
```

## Use Cases

- **Logging**: Track agent execution flow
- **Metrics**: Measure performance and usage
- **Debugging**: Inspect agent state at key points
- **Auditing**: Record all agent actions
- **Integration**: Trigger external systems
- **Workflow**: Coordinate complex multi-agent processes

## Example

See `examples/hooks_example.rs` for a complete working example.
