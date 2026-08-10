# SwarmX Hooks

SwarmX executes portable lifecycle hooks on Agents and Swarms. Hook values are
names of host-owned capabilities. They are never interpreted as shell commands.
The host must inject an executor that resolves each name to an already
authorized command, MCP tool, HTTP integration, or in-process capability.

The design follows the event, matcher/handler, structured JSON, denial, and
timeout ideas used by [Claude Code hooks](https://code.claude.com/docs/en/hooks)
and [Codex hooks](https://learn.chatgpt.com/codex/hooks), while keeping process,
network, filesystem, and trust authority outside Core.

## Hook Shape

Hooks retain the existing camelCase `SwarmConfig` fields:

- `onStart`: before an Agent or Swarm starts. It may deny execution or add
  model-visible context.
- `onChunk`: once for each streamed chunk, in stream order.
- `onHandoff`: before a Swarm schedules an edge target. Both the source Agent
  and containing Swarm receive the event. It may deny the transition or add
  model-visible context for subsequent nodes.
- `onEnd`: after success or failure.

Each non-empty string is a capability name:

```json
{
  "onStart": "policy.check_start",
  "onChunk": "telemetry.observe_chunk",
  "onEnd": "telemetry.record_end"
}
```

Multiple hook objects can name handlers for the same event. Matching handlers
start concurrently, so one handler cannot prevent another from starting. Each
Agent or Swarm accepts at most 64 hook records.

## Runtime Execution

Construction validates and preserves hook references. Execution requires an
explicit host executor:

```ts
import { Swarm } from "@swarmx/core";

const swarm = new Swarm(config, {
  hook: {
    timeoutMs: 5_000,
    execute: async (capability, input, { signal }) => {
      return hostCapabilities.call(capability, input, { signal });
    },
  },
});

await swarm.execute({ messages: [{ role: "user", content: "Hello" }] });
```

If a configured event has no executor, execution fails closed. Core does not
silently skip the hook or spawn the target string.

The default timeout is 10 seconds per handler. A timeout aborts the handler's
signal and fails the lifecycle event. Hosts must pass that signal through to
their underlying process, network, or MCP operation.

## Structured Input

Every invocation includes:

```ts
interface HookInvocation {
  event: "onStart" | "onChunk" | "onHandoff" | "onEnd";
  scope: "agent" | "swarm";
  target: { name: string };
  arguments: Record<string, unknown>;
  context: Record<string, unknown>;
  chunk?: MessageChunk;
  handoff?: { source: string; target: string };
  outcome?: {
    status: "completed" | "failed";
    messages?: MessageChunk[];
    error?: string;
  };
}
```

Inputs are request-scoped and are not persisted by the dispatcher. Executors
must not put raw prompts, responses, credentials, or hook payloads in audit or
telemetry records.

## Structured Output

Handlers return nothing or this strict JSON-compatible shape:

```ts
interface HookResult {
  continue?: boolean;
  stopReason?: string;
  additionalContext?: string;
}
```

`continue: false` denies `onStart` or `onHandoff`; if several handlers return a
decision, any denial wins. `additionalContext` is accepted only for those two
events, is limited to 20,000 characters per handler and 50,000 characters per
event, and is inserted as a system message for the remaining execution.
`stopReason` is valid only with a denial.

`onChunk` and `onEnd` are observational: they must return no decision or
additional context. A malformed or event-incompatible result fails closed.

`onEnd` runs after a successful result and after execution errors. Agent
`onEnd` reports Agent start/chunk/run failures; Swarm `onEnd` additionally
reports handoff failures. If both the main operation and `onEnd` fail, the
surfaced error preserves both failures.

## Agent and Swarm Configuration

```json
{
  "name": "hooked_workflow",
  "root": "agent",
  "hooks": [
    {
      "onStart": "workflow.started",
      "onHandoff": "workflow.handoff",
      "onEnd": "workflow.finished"
    }
  ],
  "nodes": {
    "agent": {
      "kind": "agent",
      "agent": {
        "name": "agent",
        "instructions": "Answer concisely.",
        "hooks": [{ "onChunk": "agent.chunk" }]
      }
    }
  },
  "edges": []
}
```

Extension hook capabilities remain passive inventory. Loading an extension
does not trust, enable, or execute its declared command. A host may resolve an
enabled and explicitly authorized capability through the same executor
contract.
