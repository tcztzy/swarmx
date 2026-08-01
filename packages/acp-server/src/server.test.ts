import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  currentRequestSignal,
  listSessionSummaries,
  loadSession as loadCoreSession,
} from "@swarmx/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SwarmXAgent } from "./server.js";

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
    if (originalSessionsDir === undefined) process.env.SWARMX_SESSIONS_DIR = undefined;
    else process.env.SWARMX_SESSIONS_DIR = originalSessionsDir;
    rmSync(sessionsDir, { recursive: true, force: true });
  });

  it("V551-V554 persists history and projects truthful cwd, resources, and MCP state", async () => {
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

  it("V553 rejects relative cwd and unsupported MCP transports instead of ignoring them", async () => {
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

  it("V555 cancels the active prompt through the Core request signal", async () => {
    let observedSignal: AbortSignal | undefined;
    let markStarted: (() => void) | undefined;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const agent = new SwarmXAgent({
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
      prompt: [{ type: "text", text: "cancel me" }],
    });
    await started;
    await agent.cancel({ sessionId: created.sessionId });

    await expect(pendingPrompt).resolves.toMatchObject({ stopReason: "cancelled" });
    expect(observedSignal?.aborted).toBe(true);
  });
});
