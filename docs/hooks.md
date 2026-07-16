# SwarmX Hooks

SwarmX models lifecycle hooks as portable configuration metadata on agents and
swarms. A hook value names a tool or host capability that an embedding runtime
can call at a lifecycle point.

The TypeScript core validates and preserves hook references. It does not start
MCP servers or execute hook tools as a side effect of parsing a config.

## Hook Shape

Hooks use camelCase fields in `SwarmConfig` JSON:

- `onStart` - host-defined capability to call before an agent or swarm starts.
- `onEnd` - host-defined capability to call after an agent or swarm completes.
- `onHandoff` - reserved for handoff-style host integrations.
- `onChunk` - host-defined capability to call for streamed chunks.

Example:

```json
{
  "onStart": "log_agent_start",
  "onChunk": "log_stream_chunk",
  "onEnd": "log_agent_end"
}
```

## Agent Configuration

```ts
import { Agent } from "@swarmx/core";

const agent = new Agent({
  name: "agent",
  instructions: "You are a helpful assistant.",
  hooks: [
    {
      onStart: "initialize_session",
      onEnd: "cleanup_session",
    },
  ],
});

console.log(agent.hooks[0].onStart);
```

## Swarm Configuration

```ts
import { Swarm } from "@swarmx/core";

const swarm = new Swarm({
  name: "hooked_workflow",
  root: "agent",
  hooks: [{ onStart: "workflow_started", onEnd: "workflow_finished" }],
  nodes: {
    agent: {
      kind: "agent",
      agent: {
        name: "agent",
        instructions: "Answer concisely.",
        hooks: [{ onChunk: "record_agent_chunk" }],
      },
    },
  },
  edges: [],
});

console.log(swarm.hooks[0].onEnd);
```

## Host Execution

If a desktop app, ACP adapter, or downstream product chooses to execute hooks,
it should resolve hook names against explicit host-owned capabilities such as
MCP tools. A typical hook input can include:

```json
{
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "context": {},
  "agent": { "name": "agent" }
}
```

Hook outputs should be explicit and sanitized before being merged into runtime
context. Keep secret values out of hook metadata and logs; use secret references
or request-scoped runtime injection instead.

## Serialization

Hooks are plain JSON-compatible records:

```ts
import { Hook } from "@swarmx/core";

const hook = new Hook({ onStart: "log_start" });
const json = JSON.stringify(hook);
const restored = new Hook(JSON.parse(json));

console.log(restored.onStart);
```

## Use Cases

- Logging: record host-level lifecycle events.
- Metrics: measure execution timing in a host-owned telemetry pipeline.
- Debugging: capture selected sanitized state at lifecycle points.
- Auditing: link external audit records to agent or swarm runs.
- Integration: notify explicit host systems without baking product behavior into core.
