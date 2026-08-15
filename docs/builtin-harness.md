# Built-in Harness plugins

The direct `swarmx` Harness has a small ordered plugin kernel for changing its
prompt, local tool set, or Agent Loop without modifying `Agent`. This seam is
only for the built-in Harness. External ACP Harnesses keep their own tools,
permissions, sessions, and loops, and `SwarmConfig` remains the separate
workflow/meta-Harness authority.

This kernel is a loop-level extension point. It does not define ablation
profiles, replace whole services, or measure subsystem contribution; those
responsibilities belong to the built-in Harness service registry and evaluation
profile.

## Contract

- Mount order determines prompt-section, tool, and middleware order.
- Plugin, prompt-section, tool, and middleware ids are unique. Conflicts fail
  before the Provider request starts.
- A plugin mount returns an idempotent unmount function. Unmounting removes all
  registrations made during setup and then makes the same plugin id reusable.
- Agent-loop middleware is a waterfall. It may call `next()` once, return its
  own result instead, or throw to reject the turn. A second `next()` call fails.
- Prompt sections and tools are resolved once at turn admission. Stream
  middleware may replace `emitChunk` to observe or transform live output.
- Plugin-private state is not Session authority. Model-visible prompt content
  and output must still flow through the admitted turn and emitted chunks.

## TypeScript example

```ts
import { Agent, BuiltinHarness } from "@swarmx/core";

const harness = new BuiltinHarness();
const unmount = harness.mount({
  id: "review-mode",
  setup(context) {
    context.addPromptSection("review-rules", () =>
      "Review the change before proposing an implementation.",
    );
    context.useAgentLoop("timing", async (turn, next) => {
      const startedAt = performance.now();
      try {
        return await next();
      } finally {
        console.info(turn.agentName, performance.now() - startedAt);
      }
    });
  },
});

const agent = new Agent(
  { name: "reviewer", model: "gpt-5" },
  { builtinHarness: harness },
);

// Later: removes both registrations made by review-mode.
unmount();
```

Register local tools with `context.addTool(tool)`. Tools use the existing
Provider-independent `LocalTool` contract and therefore keep the same MCP
adaptation, cancellation, progress, and result boundaries as host-provided
tools.
