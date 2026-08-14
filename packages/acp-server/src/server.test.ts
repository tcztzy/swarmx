import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  type AuditInput,
  listSessionSummaries,
  loadSession as loadCoreSession,
} from "@swarmx/core";
import { currentRequestSignal } from "@swarmx/core/request-scope";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SwarmXAgent } from "./server.js";

function captureAudit(): {
  audit: { append: (input: AuditInput) => never };
  events: AuditInput[];
} {
  const events: AuditInput[] = [];
  return {
    audit: {
      append: (input) => {
        events.push(input);
        return {} as never;
      },
    },
    events,
  };
}

describe("SwarmXAgent", () => {
  let sessionsDir: string;
  let cwd: string;
  const originalSessionsDir = process.env.SWARMX_SESSIONS_DIR;

  beforeEach(() => {
    sessionsDir = mkdtempSync(path.join(tmpdir(), "swarmx-acp-server-"));
    cwd = path.join(sessionsDir, "workspace");
    mkdirSync(cwd);
    process.env.SWARMX_SESSIONS_DIR = sessionsDir;
  });

  afterEach(() => {
    if (originalSessionsDir === undefined) {
      Reflect.deleteProperty(process.env, "SWARMX_SESSIONS_DIR");
    } else process.env.SWARMX_SESSIONS_DIR = originalSessionsDir;
    rmSync(sessionsDir, { recursive: true, force: true });
  });

  it("persists history and projects truthful cwd, resources, and MCP state", async () => {
    const execute = vi.fn(async (arguments_: Record<string, unknown>) => {
      const messages = arguments_.messages as Array<{ content: string }>;
      return [
        {
          role: "assistant",
          kind: "message" as const,
          content: `observed:${messages.length}`,
        },
      ];
    });
    const createSwarm = vi.fn(() => ({ execute }));
    const updates = vi.fn(async () => undefined);
    const agent = new SwarmXAgent({ createSwarm } as never);
    agent.setConnection({ sessionUpdate: updates } as never);

    const initialized = await agent.initialize({ protocolVersion: 1 } as never);
    expect(initialized.agentCapabilities).toMatchObject({
      loadSession: true,
      promptCapabilities: {
        image: false,
        audio: false,
        embeddedContext: false,
      },
      sessionCapabilities: {
        close: {},
        list: {},
        resume: {},
      },
    });

    const created = await agent.newSession({
      cwd,
      mcpServers: [
        {
          name: "workspace",
          command: "workspace-mcp",
          args: ["--stdio"],
          env: [{ name: "MCP_MODE", value: "read-only" }],
        },
      ],
    });
    expect(loadCoreSession(created.sessionId)).toMatchObject({ cwd, messages: [] });

    await agent.prompt({
      sessionId: created.sessionId,
      prompt: [
        { type: "text", text: "Inspect the docs. " },
        {
          type: "resource_link",
          name: "Architecture",
          uri: "file:///workspace/DESIGNS.md",
          mimeType: "text/markdown",
        },
      ],
    });
    await agent.prompt({
      sessionId: created.sessionId,
      prompt: [{ type: "text", text: "Summarize the prior turn." }],
    });

    const persisted = loadCoreSession(created.sessionId);
    expect(persisted?.messages.map((message) => [message.role, message.content])).toEqual([
      [
        "user",
        expect.stringContaining(
          "[Resource: Architecture](file:///workspace/DESIGNS.md; text/markdown)",
        ),
      ],
      ["assistant", "observed:1"],
      ["user", "Summarize the prior turn."],
      ["assistant", "observed:3"],
    ]);
    expect(execute.mock.calls[1]?.[0]).toMatchObject({
      messages: [
        expect.objectContaining({ role: "user" }),
        expect.objectContaining({ role: "assistant" }),
        expect.objectContaining({ role: "user" }),
      ],
    });
    expect(createSwarm).toHaveBeenLastCalledWith(
      expect.objectContaining({
        nodes: {
          agent: expect.objectContaining({
            agent: expect.objectContaining({
              process: expect.objectContaining({ currentDir: cwd }),
              mcpServers: {
                workspace: {
                  type: "stdio",
                  command: "workspace-mcp",
                  args: ["--stdio"],
                  env: { MCP_MODE: "read-only" },
                  cwd,
                },
              },
            }),
          }),
        },
      }),
    );

    await expect(agent.listSessions({ cwd })).resolves.toMatchObject({
      sessions: [expect.objectContaining({ sessionId: created.sessionId, cwd })],
    });
    await expect(agent.listSessions({ cwd: path.join(sessionsDir, "other") })).resolves.toEqual({
      sessions: [],
    });

    const replayUpdates = vi.fn(async () => undefined);
    const restored = new SwarmXAgent({ createSwarm } as never);
    restored.setConnection({ sessionUpdate: replayUpdates } as never);
    await restored.loadSession({
      sessionId: created.sessionId,
      cwd,
      mcpServers: [],
    });
    expect(replayUpdates).toHaveBeenCalledTimes(4);
    await expect(
      restored.loadSession({
        sessionId: created.sessionId,
        cwd: path.join(sessionsDir, "other"),
        mcpServers: [],
      }),
    ).rejects.toThrow("working directory");
    await expect(
      restored.resumeSession({
        sessionId: created.sessionId,
        cwd,
        mcpServers: [],
      }),
    ).resolves.toEqual({});
    await restored.closeSession({ sessionId: created.sessionId });
    await expect(
      restored.prompt({
        sessionId: created.sessionId,
        prompt: [{ type: "text", text: "closed" }],
      }),
    ).resolves.toMatchObject({ stopReason: "cancelled" });
    expect(listSessionSummaries()).toHaveLength(1);
  });

  it("rejects relative cwd and unsupported MCP transports instead of ignoring them", async () => {
    const agent = new SwarmXAgent();

    await expect(agent.newSession({ cwd: "relative", mcpServers: [] })).rejects.toThrow("absolute");
    await expect(
      agent.newSession({
        cwd,
        mcpServers: [
          {
            type: "http",
            name: "remote",
            url: "https://mcp.example.com",
            headers: [],
          },
        ],
      }),
    ).rejects.toThrow("Unsupported ACP MCP transport");
  });

  it("rejects invalid session operations and MCP definitions explicitly", async () => {
    const agent = new SwarmXAgent();

    await expect(agent.authenticate({} as never)).rejects.toThrow("not supported");
    await expect(agent.setSessionMode({} as never)).resolves.toEqual({});
    await expect(agent.loadSession({ sessionId: "missing", cwd, mcpServers: [] })).rejects.toThrow(
      "not found",
    );
    await expect(
      agent.resumeSession({ sessionId: "missing", cwd, mcpServers: [] }),
    ).rejects.toThrow("not found");
    await expect(
      agent.newSession({
        cwd,
        mcpServers: [
          { name: "duplicate", command: "one", args: [], env: [] },
          { name: "duplicate", command: "two", args: [], env: [] },
        ],
      }),
    ).rejects.toThrow("Duplicate ACP MCP server name");
    await expect(
      agent.newSession({ cwd, mcpServers: [{ name: "", command: "one", args: [], env: [] }] }),
    ).rejects.toThrow("name is required");
    await expect(
      agent.newSession({ cwd, mcpServers: [{ name: "empty", command: "", args: [], env: [] }] }),
    ).rejects.toThrow("command is required");
    await expect(
      agent.newSession({
        cwd,
        mcpServers: [
          {
            name: "env-duplicate",
            command: "one",
            args: [],
            env: [
              { name: "DUPLICATE", value: "one" },
              { name: "DUPLICATE", value: "two" },
            ],
          },
        ],
      }),
    ).rejects.toThrow("Duplicate environment variable");
  });

  it("projects every ACP message update and reports prompt failures", async () => {
    const updates = vi.fn(async (_call: { update: { sessionUpdate: string } }) => undefined);
    const chunks = [
      { role: "assistant", kind: "thinking" as const, content: "thinking" },
      {
        role: "assistant",
        kind: "tool_call" as const,
        content: '{"query":"docs"}',
        toolName: "search",
        render: { invocationId: "call-1" },
      },
      {
        role: "assistant",
        kind: "tool_progress" as const,
        content: "working",
        toolName: "search",
        render: { invocationId: "call-1" },
      },
      {
        role: "tool",
        kind: "tool_result" as const,
        content: "not-json",
        toolName: "search",
        render: { invocationId: "call-1" },
      },
      {
        role: "assistant",
        kind: "message" as const,
        content: "finished",
        agent: "worker",
        swarmEvent: "done",
      },
    ];
    const agent = new SwarmXAgent({
      createSwarm: () => ({ execute: async () => chunks }),
    } as never);
    agent.setConnection({ sessionUpdate: updates } as never);
    const created = await agent.newSession({ cwd, mcpServers: [] });

    await expect(
      agent.prompt({ sessionId: created.sessionId, prompt: [{ type: "text", text: "run" }] }),
    ).resolves.toEqual({ stopReason: "end_turn" });
    expect(updates.mock.calls.map(([call]) => call.update.sessionUpdate)).toEqual([
      "agent_thought_chunk",
      "tool_call",
      "tool_call_update",
      "tool_call_update",
      "agent_message_chunk",
    ]);
    expect(updates.mock.calls[1]?.[0]).toMatchObject({
      update: { rawInput: { query: "docs" }, toolCallId: "call-1" },
    });
    expect(updates.mock.calls[3]?.[0]).toMatchObject({
      update: { rawOutput: "not-json", status: "completed" },
    });

    const noConnection = new SwarmXAgent();
    await expect(
      noConnection.prompt({
        sessionId: created.sessionId,
        prompt: [{ type: "text", text: "run" }],
      }),
    ).resolves.toEqual({ stopReason: "end_turn" });
    await expect(
      agent.prompt({ sessionId: "missing", prompt: [{ type: "text", text: "run" }] }),
    ).resolves.toEqual({ stopReason: "cancelled" });
    await expect(
      agent.prompt({ sessionId: created.sessionId, prompt: [{ type: "text", text: "  " }] }),
    ).resolves.toEqual({ stopReason: "end_turn" });

    const failedUpdates = vi.fn(async () => undefined);
    const failed = new SwarmXAgent({
      createSwarm: () => ({
        execute: async () => {
          throw new Error("backend failed");
        },
      }),
    } as never);
    failed.setConnection({ sessionUpdate: failedUpdates } as never);
    const failedSession = await failed.newSession({ cwd, mcpServers: [] });
    await expect(
      failed.prompt({
        sessionId: failedSession.sessionId,
        prompt: [{ type: "text", text: "fail" }],
      }),
    ).resolves.toEqual({ stopReason: "refusal" });
    expect(failedUpdates).toHaveBeenCalledWith(
      expect.objectContaining({
        update: expect.objectContaining({
          content: expect.objectContaining({ text: "[error] backend failed" }),
        }),
      }),
    );
  });

  it("uses the default Swarm factory and preserves requested runtime config", async () => {
    const updates = vi.fn(async () => undefined);
    const agent = new SwarmXAgent();
    agent.setConnection({ sessionUpdate: updates } as never);
    const created = await agent.newSession({ cwd, mcpServers: [] });

    await expect(
      agent.prompt({
        sessionId: created.sessionId,
        prompt: [
          {
            type: "text",
            text: "run the requested swarm",
            _meta: {
              swarmConfig: { name: "broken", root: "missing", nodes: {}, edges: [] },
            },
          },
        ],
      } as never),
    ).resolves.toEqual({ stopReason: "refusal" });
    expect(updates).toHaveBeenCalledWith(
      expect.objectContaining({
        update: expect.objectContaining({
          content: expect.objectContaining({
            text: expect.stringContaining('Root node "missing"'),
          }),
        }),
      }),
    );
  });

  it("exposes the Core request signal to injected Swarm execution", async () => {
    let observedSignal: AbortSignal | undefined;
    const agent = new SwarmXAgent({
      createSwarm: () => ({
        execute: async () => {
          observedSignal = currentRequestSignal();
          return [];
        },
      }),
    } as never);
    agent.setConnection({ sessionUpdate: vi.fn(async () => undefined) } as never);
    const created = await agent.newSession({ cwd, mcpServers: [] });

    await agent.prompt({
      sessionId: created.sessionId,
      prompt: [{ type: "text", text: "signal" }],
    });

    expect(observedSignal).toBeDefined();
  });

  it("records correlated ACP lifecycle events without prompt, result, or error content", async () => {
    const { audit, events } = captureAudit();
    const secretPrompt = "audit-secret-prompt-sk-test-123456";
    const secretResult = "audit-secret-result-password=hunter2";
    const updates = vi.fn(async () => undefined);
    const agent = new SwarmXAgent({
      audit,
      createSwarm: () => ({
        execute: async () => [
          { role: "assistant", kind: "message" as const, content: secretResult },
        ],
      }),
    });
    agent.setConnection({ sessionUpdate: updates } as never);

    const created = await agent.newSession({ cwd, mcpServers: [] });
    await expect(
      agent.prompt({
        sessionId: created.sessionId,
        prompt: [{ type: "text", text: secretPrompt }],
      }),
    ).resolves.toEqual({ stopReason: "end_turn" });
    await expect(
      agent.loadSession({ sessionId: created.sessionId, cwd, mcpServers: [] }),
    ).resolves.toEqual({});
    await expect(agent.closeSession({ sessionId: created.sessionId })).resolves.toEqual({});

    expect(events.map(({ action, outcome }) => [action, outcome])).toEqual([
      ["acp.session.new", "attempted"],
      ["acp.session.new", "completed"],
      ["acp.prompt", "attempted"],
      ["acp.prompt", "completed"],
      ["acp.session.load", "attempted"],
      ["acp.session.load", "completed"],
      ["acp.session.close", "attempted"],
      ["acp.session.close", "completed"],
    ]);
    for (const pairStart of [0, 2, 4, 6]) {
      expect(events[pairStart]?.requestId).toBe(events[pairStart + 1]?.requestId);
    }
    expect(events.slice(1).every((event) => event.sessionId === created.sessionId)).toBe(true);
    const serialized = JSON.stringify(events);
    expect(serialized).not.toContain(secretPrompt);
    expect(serialized).not.toContain(secretResult);
    expect(serialized).not.toContain(cwd);
  });

  it("audits effectful ACP boundary methods without copying no-op mode payloads", async () => {
    const { audit, events } = captureAudit();
    const agent = new SwarmXAgent({ audit });

    await agent.initialize({ protocolVersion: 1 } as never);
    const created = await agent.newSession({ cwd, mcpServers: [] });
    await agent.listSessions({ cwd });
    await agent.resumeSession({ sessionId: created.sessionId, cwd, mcpServers: [] });
    const eventCountBeforeNoOp = events.length;
    await agent.setSessionMode({
      sessionId: created.sessionId,
      modeId: "private-mode-payload",
    } as never);
    expect(events).toHaveLength(eventCountBeforeNoOp);
    await expect(agent.authenticate({ token: "sk-private-auth" } as never)).rejects.toThrow(
      "not supported",
    );

    expect(events.map(({ action, outcome }) => [action, outcome])).toEqual([
      ["acp.initialize", "attempted"],
      ["acp.initialize", "completed"],
      ["acp.session.new", "attempted"],
      ["acp.session.new", "completed"],
      ["acp.session.list", "attempted"],
      ["acp.session.list", "completed"],
      ["acp.session.resume", "attempted"],
      ["acp.session.resume", "completed"],
      ["acp.authenticate", "attempted"],
      ["acp.authenticate", "denied"],
    ]);
    const serialized = JSON.stringify(events);
    expect(serialized).not.toContain(cwd);
    expect(serialized).not.toContain("private-mode-payload");
    expect(serialized).not.toContain("sk-private-auth");
  });

  it("fails closed before prompt execution and exposes a failed result audit", async () => {
    const execute = vi.fn(async () => []);
    let promptAuditFailure: "attempted" | "completed" | undefined;
    const audit = {
      append: (input: AuditInput) => {
        if (input.action === "acp.prompt" && input.outcome === promptAuditFailure) {
          throw new Error(`audit ${promptAuditFailure} failed`);
        }
        return {} as never;
      },
    };
    const agent = new SwarmXAgent({ audit, createSwarm: () => ({ execute }) });
    agent.setConnection({ sessionUpdate: vi.fn(async () => undefined) } as never);
    const created = await agent.newSession({ cwd, mcpServers: [] });

    promptAuditFailure = "attempted";
    await expect(
      agent.prompt({
        sessionId: created.sessionId,
        prompt: [{ type: "text", text: "must not execute" }],
      }),
    ).rejects.toThrow("audit attempted failed");
    expect(execute).not.toHaveBeenCalled();

    promptAuditFailure = "completed";
    await expect(
      agent.prompt({
        sessionId: created.sessionId,
        prompt: [{ type: "text", text: "execute once" }],
      }),
    ).rejects.toThrow("audit completed failed");
    expect(execute).toHaveBeenCalledTimes(1);
  });

  it("does not copy backend error details into failed audit events", async () => {
    const { audit, events } = captureAudit();
    const secretError = "backend credential=never-log-this";
    const untrustedSessionId = "credential-never-log-session-id";
    const agent = new SwarmXAgent({
      audit,
      createSwarm: () => ({
        execute: async () => {
          throw new Error(secretError);
        },
      }),
    });
    agent.setConnection({ sessionUpdate: vi.fn(async () => undefined) } as never);
    const created = await agent.newSession({ cwd, mcpServers: [] });

    await expect(
      agent.prompt({ sessionId: created.sessionId, prompt: [{ type: "text", text: "run" }] }),
    ).resolves.toEqual({ stopReason: "refusal" });
    await expect(
      agent.loadSession({ sessionId: untrustedSessionId, cwd, mcpServers: [] }),
    ).rejects.toThrow("not found");
    expect(events).toContainEqual(
      expect.objectContaining({ action: "acp.prompt", outcome: "failed" }),
    );
    expect(JSON.stringify(events)).not.toContain(secretError);
    expect(JSON.stringify(events)).not.toContain(untrustedSessionId);
  });

  it("cancels the active prompt through the Core request signal", async () => {
    let observedSignal: AbortSignal | undefined;
    let markStarted: (() => void) | undefined;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const { audit, events } = captureAudit();
    const agent = new SwarmXAgent({
      audit,
      createSwarm: () => ({
        execute: async () => {
          observedSignal = currentRequestSignal();
          markStarted?.();
          await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(resolve, 100);
            observedSignal?.addEventListener(
              "abort",
              () => {
                clearTimeout(timer);
                reject(observedSignal?.reason);
              },
              { once: true },
            );
          });
          return [];
        },
      }),
    } as never);
    agent.setConnection({ sessionUpdate: vi.fn(async () => undefined) } as never);
    const created = await agent.newSession({ cwd, mcpServers: [] });

    const pendingPrompt = agent.prompt({
      sessionId: created.sessionId,
      prompt: [{ type: "text", text: "cancel-secret-prompt-password=hunter2" }],
    });
    await started;
    await agent.cancel({ sessionId: created.sessionId });

    await expect(pendingPrompt).resolves.toMatchObject({ stopReason: "cancelled" });
    expect(observedSignal?.aborted).toBe(true);
    expect(events.map(({ action, outcome }) => [action, outcome])).toEqual([
      ["acp.session.new", "attempted"],
      ["acp.session.new", "completed"],
      ["acp.prompt", "attempted"],
      ["acp.prompt", "cancel_requested"],
      ["acp.prompt", "cancelled"],
      ["acp.prompt", "cancelled"],
    ]);
    expect(events[3]?.requestId).not.toBe(events[2]?.requestId);
    expect(events[4]?.requestId).toBe(events[3]?.requestId);
    expect(events[5]?.requestId).toBe(events[2]?.requestId);
    expect(events[3]?.metadata).toMatchObject({ promptRequestId: events[2]?.requestId });
    expect(events[4]?.metadata).toMatchObject({ promptRequestId: events[2]?.requestId });
    expect(events.every((event) => event.action !== "acp.prompt.cancel")).toBe(true);
    expect(JSON.stringify(events)).not.toContain("cancel-secret-prompt-password=hunter2");
  });

  it("records a compact denied cancellation lifecycle when no prompt is active", async () => {
    const { audit, events } = captureAudit();
    const agent = new SwarmXAgent({ audit });

    await agent.cancel({ sessionId: "inactive-secret-session" });

    expect(events.map(({ action, outcome }) => [action, outcome])).toEqual([
      ["acp.prompt", "cancel_requested"],
      ["acp.prompt", "denied"],
    ]);
    expect(events[1]?.requestId).toBe(events[0]?.requestId);
    expect(events.every((event) => event.action !== "acp.prompt.cancel")).toBe(true);
    expect(JSON.stringify(events)).not.toContain("inactive-secret-session");
  });
});
