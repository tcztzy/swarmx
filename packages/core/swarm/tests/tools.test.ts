import type { Agent } from "@deepseek-ai/dsh-agent";
import type { ToolRunContext } from "@deepseek-ai/dsh-tools";
import { describe, expect, it, vi } from "vitest";
import { createSwarmToolDefinition, SWARM_ACTIONS } from "../src/tools.js";

function execution(agent: Agent): ToolRunContext {
  return {
    agent,
    callId: "call-swarm",
    signal: new AbortController().signal,
  } as unknown as ToolRunContext;
}

describe("swarm aggregate tool", () => {
  it("V162/V170/V173: exposes one bounded aggregate surface with archive and no delete", async () => {
    const root = { id: "session-lead" } as Agent;
    const snapshot = vi.fn(() => Promise.resolve({ kind: "inactive", revision: 0 }));
    const tool = createSwarmToolDefinition({ snapshot } as never);
    const actions = (tool.parameters.properties.action as { enum: string[] }).enum;

    expect(actions).toEqual(SWARM_ACTIONS);
    expect(actions).toContain("archive");
    expect(actions).toContain("admit_knowledge");
    expect(actions).toContain("resolve_effect");
    expect(actions).toContain("submit_task");
    expect(actions).toContain("start_verification");
    expect(actions).toContain("record_verdict");
    expect(actions).toContain("record_monitor_finding");
    expect(actions).not.toContain("delete");
    const privateArgs = {
      action: "send_message",
      request: { content: "private coordination detail", delivery: "quiet", target: "alpha" },
    };
    expect(JSON.stringify(tool.presentCall?.(privateArgs))).not.toContain(
      "private coordination detail",
    );
    expect(
      JSON.stringify(
        tool.presentResult?.(privateArgs, {
          content: [{ type: "text", text: "private coordination detail" }],
          isError: false,
        }),
      ),
    ).not.toContain("private coordination detail");
    await expect(
      tool.execute({ action: "status", request: {} }, execution(root)),
    ).resolves.toMatchObject({ action: "status", data: { kind: "inactive" } });
    expect(snapshot).toHaveBeenCalledWith(root);
  });

  it("V162: rejects calls without an exact Agent carrier before service dispatch", async () => {
    const snapshot = vi.fn();
    const tool = createSwarmToolDefinition({ snapshot } as never);
    await expect(
      tool.execute({ action: "status", request: {} }, {
        callId: "call",
        signal: new AbortController().signal,
      } as ToolRunContext),
    ).rejects.toMatchObject({ code: "SWARM_UNAUTHORIZED" });
    expect(snapshot).not.toHaveBeenCalled();
  });
});
