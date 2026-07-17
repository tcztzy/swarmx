import type { MessageChunk, ModelTokenUsage } from "@swarmx/core";
import { describe, expect, it, vi } from "vitest";
import { ClaudeChildAgentHost } from "./child-agent-host.js";

describe("ClaudeChildAgentHost", () => {
  it("V417-V420 executes at the dynamic root and resumes request-scoped history", async () => {
    let root = "/project/main";
    const execute = vi.fn().mockImplementation(async ({ messages }) => ({
      messages: [
        { role: "assistant", content: "Inspecting", kind: "tool_call", toolName: "Read" },
        {
          role: "assistant",
          content: messages.at(-1)?.content === "Follow up" ? "Follow-up result" : "Initial result",
          kind: "message",
        },
      ] satisfies MessageChunk[],
      usages: [usage(20, 5, 2)],
    }));
    const host = new ClaudeChildAgentHost({
      parentModel: "claude-sonnet-4-6",
      root: () => root,
      systemContext: (currentRoot) => `Project root: ${currentRoot}`,
      execute,
    });

    const first = await host.run({
      description: "Inspect runtime",
      prompt: "Inspect",
      subagentType: "general-purpose",
      model: "sonnet",
    });
    expect(first).toMatchObject({
      status: "completed",
      prompt: "Inspect",
      agentId: expect.any(String),
      content: [{ type: "text", text: "Initial result" }],
      totalToolUseCount: 1,
      totalTokens: 25,
      usage: {
        input_tokens: 20,
        output_tokens: 5,
        cache_read_input_tokens: 2,
      },
    });
    expect(execute).toHaveBeenNthCalledWith(1, {
      agentId: first.agentId,
      root: "/project/main",
      messages: [
        { role: "system", content: "Project root: /project/main" },
        { role: "user", content: "Inspect" },
      ],
    });

    root = "/project/.claude/worktrees/feature";
    const resumed = await host.run({
      description: "Continue inspection",
      prompt: "Follow up",
      resume: first.agentId,
    });
    expect(resumed).toMatchObject({ agentId: first.agentId, prompt: "Follow up" });
    expect(execute).toHaveBeenNthCalledWith(2, {
      agentId: first.agentId,
      root: "/project/.claude/worktrees/feature",
      messages: [
        { role: "system", content: "Project root: /project/.claude/worktrees/feature" },
        { role: "user", content: "Inspect" },
        { role: "assistant", content: "Initial result" },
        { role: "user", content: "Follow up" },
      ],
    });

    await expect(
      host.run({ description: "Missing", prompt: "Continue", resume: "missing" }),
    ).rejects.toThrow(/No child agent missing/i);
    host.close();
    await expect(
      host.run({ description: "Closed", prompt: "Continue", resume: first.agentId }),
    ).rejects.toThrow(/No child agent/i);
  });

  it("V419 rejects unavailable model and specialized-agent routes", async () => {
    const execute = vi.fn();
    const host = new ClaudeChildAgentHost({
      parentModel: "claude-sonnet-4-6",
      root: () => "/project",
      systemContext: () => "Project",
      execute,
    });

    await expect(
      host.run({ description: "Use opus", prompt: "Inspect", model: "opus" }),
    ).rejects.toThrow(/no configured route/i);
    await expect(
      host.run({ description: "Use reviewer", prompt: "Inspect", subagentType: "reviewer" }),
    ).rejects.toThrow(/not configured/i);
    expect(execute).not.toHaveBeenCalled();
  });
});

function usage(
  inputTokens: number,
  outputTokens: number,
  cachedInputTokens: number,
): ModelTokenUsage {
  return {
    inputTokens,
    outputTokens,
    reasoningTokens: 0,
    cachedInputTokens,
    totalTokens: inputTokens + outputTokens,
    estimated: false,
  };
}
