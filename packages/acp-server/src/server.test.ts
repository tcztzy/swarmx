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
});
