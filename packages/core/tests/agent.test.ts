import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";
import type { AcpPromptInput } from "../src/acp.js";
import {
  AcpSessionUnavailableError,
  cancelAcpRequest,
  RequestCancelledError,
  withAcpRequest,
} from "../src/acp.js";
import { Agent, HookRef } from "../src/agent.js";
import {
  compileContext,
  createContextEvent,
  createContextHistorySnapshot,
  parseContextEngineConfig,
} from "../src/context-engine.js";
import type { LocalToolCallContext, McpConnectionResult } from "../src/mcp.js";
import { localToolResult, McpManager } from "../src/mcp.js";
import type { AgentConfig, MessageChunk } from "../src/types.js";

describe("Agent", () => {
  it("uses composition-scoped provider environment for the native client", () => {
    const agent = new Agent({
      name: "provider_agent",
      process: {
        env: {
          OPENAI_API_KEY: "scoped-key",
          OPENAI_BASE_URL: "https://provider.example/v1",
        },
      },
    });

    expect(agent.client.apiKey).toBe("scoped-key");
    expect(agent.client.baseURL).toBe("https://provider.example/v1");
  });

  it("never sends an ambient API key to an explicit third-party runtime", () => {
    const previousKey = process.env.OPENAI_API_KEY;
    const previousBaseUrl = process.env.OPENAI_BASE_URL;
    try {
      process.env.OPENAI_API_KEY = "ambient-openai-key";
      process.env.OPENAI_BASE_URL = "https://api.openai.example/v1";
      const agent = new Agent({
        name: "third_party_agent",
        process: {
          env: { OPENAI_BASE_URL: "https://third-party.example/v1" },
          clearEnv: true,
        },
      });

      expect(agent.client.apiKey).toBe("sk-no-key");
      expect(agent.client.baseURL).toBe("https://third-party.example/v1");
    } finally {
      if (previousKey === undefined) Reflect.deleteProperty(process.env, "OPENAI_API_KEY");
      else process.env.OPENAI_API_KEY = previousKey;
      if (previousBaseUrl === undefined) Reflect.deleteProperty(process.env, "OPENAI_BASE_URL");
      else process.env.OPENAI_BASE_URL = previousBaseUrl;
    }
  });

  it("constructs with minimal config", () => {
    const agent = new Agent({ name: "test" });
    expect(agent.name).toBe("test");
    expect(agent.description).toBeUndefined();
    expect(agent.model).toBe("gpt-4o");
    expect(agent.instructions).toBe("");
    expect(agent instanceof Agent).toBe(true);
  });

  it("creates OpenAI client from config", () => {
    const agent = new Agent({
      name: "test",
      client: { apiKey: "sk-test", baseUrl: "https://api.test.com/v1" },
    });
    expect(agent.client).toBeInstanceOf(OpenAI);
  });

  it("generates swarm config", () => {
    const agent = new Agent({
      name: "helper",
      description: "A helper agent",
      model: "claude-3",
      instructions: "Be helpful",
    });
    const config = agent.toSwarmConfig();
    expect(config.name).toBe("helper");
    expect(config.root).toBe("helper");
    expect(config.nodes).toHaveProperty("helper");
    expect(config.edges).toEqual([]);
  });

  it("rejects invalid agent name", () => {
    expect(() => new Agent({ name: "123bad" })).toThrow();
    expect(() => new Agent({ name: "bad-name" })).toThrow();
    expect(() => new Agent({ name: "" })).toThrow();
  });

  it("validates McpServer discriminated union", () => {
    const agent = new Agent({
      name: "test",
      mcpServers: {
        fs: { type: "stdio", command: "npx", args: ["-y", "server"] },
        web: { type: "sse", url: "http://localhost:8080" },
      },
    });
    expect(agent.mcpServers.size).toBe(2);
  });

  it("rejects invalid McpServer missing required fields", () => {
    const invalidMcpServers = {
      bad: { type: "stdio" },
    } as unknown as AgentConfig["mcpServers"];

    expect(
      () =>
        new Agent({
          name: "test",
          mcpServers: invalidMcpServers,
        }),
    ).toThrow();
  });

  it("accepts MCP servers and hooks", () => {
    const agent = new Agent({
      name: "test",
      mcpServers: {
        filesystem: {
          type: "stdio",
          command: "npx",
          args: ["-y", "@modelcontextprotocol/server-filesystem"],
        },
      },
      hooks: [{ onStart: "echo start" }],
    });
    expect(agent.mcpServers.size).toBe(1);
    expect(agent.hooks).toHaveLength(1);
    expect(agent.hooks[0].onStart).toBe("echo start");
  });

  it("uses model from config over default", () => {
    const agent = new Agent({ name: "test", model: "gpt-5-mini" });
    expect(agent.model).toBe("gpt-5-mini");
  });

  it("uses OPENAI_MODEL env var", () => {
    const previousModel = process.env.OPENAI_MODEL;
    try {
      process.env.OPENAI_MODEL = "env-model";
      const agent = new Agent({ name: "test" });
      expect(agent.model).toBe("env-model");
    } finally {
      if (previousModel === undefined) Reflect.deleteProperty(process.env, "OPENAI_MODEL");
      else process.env.OPENAI_MODEL = previousModel;
    }
  });

  it("echo backend returns the latest user message without a model call", async () => {
    const agent = new Agent({ name: "test", backend: { type: "echo" } });

    const result = await agent.call({
      messages: [
        { role: "system", content: "ignore" },
        { role: "user", content: "first" },
        { role: "assistant", content: "middle" },
        { role: "user", content: "last" },
      ],
    });

    expect(result.messages).toEqual([
      {
        role: "assistant",
        content: "last",
        kind: "message",
        agent: "test",
      },
    ]);
  });

  it("runs streamed agent lifecycle hooks with structured input", async () => {
    const invocations: Array<{
      capability: string;
      event: string;
      arguments: Record<string, unknown>;
      outcome?: { status: string };
    }> = [];
    const agent = new Agent(
      {
        name: "hooked_agent",
        backend: { type: "echo" },
        hooks: [
          {
            onStart: "agent.start",
            onChunk: "agent.chunk",
            onEnd: "agent.end",
          },
        ],
      },
      {
        hook: {
          execute: async (capability, input) => {
            invocations.push({
              capability,
              event: input.event,
              arguments: input.arguments,
              outcome: input.outcome,
            });
            if (input.event === "onStart") {
              (input.arguments.messages as Array<Record<string, unknown>>).push({
                role: "user",
                content: "mutated hook copy",
              });
              return { additionalContext: "Follow repository policy." };
            }
          },
        },
      },
    );
    const chunks: MessageChunk[] = [];

    const result = await agent.callStream(
      { messages: [{ role: "user", content: "hello" }] },
      (chunk) => chunks.push(chunk),
    );

    expect(result.messages).toEqual(chunks);
    expect(result.messages[0]?.content).toBe("hello");
    expect(invocations.map(({ capability }) => capability)).toEqual([
      "agent.start",
      "agent.chunk",
      "agent.end",
    ]);
    expect(invocations[1]?.arguments.messages).toEqual([
      { role: "system", content: "Follow repository policy." },
      { role: "user", content: "hello" },
    ]);
    expect(invocations[2]?.outcome).toEqual({ status: "completed", messages: result.messages });
  });

  it("starts matching hooks concurrently and any denial wins", async () => {
    const started: string[] = [];
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const agent = new Agent(
      {
        name: "hooked_agent",
        backend: { type: "echo" },
        hooks: [{ onStart: "first" }, { onStart: "second" }],
      },
      {
        hook: {
          execute: async (capability) => {
            started.push(capability);
            await gate;
            return capability === "second"
              ? { continue: false, stopReason: "blocked by policy" }
              : undefined;
          },
        },
      },
    );

    const call = agent.call({ messages: [{ role: "user", content: "hello" }] });
    await Promise.resolve();
    expect(started).toEqual(["first", "second"]);
    release?.();

    await expect(call).rejects.toThrow(/blocked by policy/);
  });

  it("fails closed when a configured hook has no executor", async () => {
    const agent = new Agent({
      name: "hooked_agent",
      backend: { type: "echo" },
      hooks: [{ onStart: "policy.check" }],
    });

    await expect(agent.call({ messages: [{ role: "user", content: "hello" }] })).rejects.toThrow(
      /hook executor/i,
    );
  });

  it("aborts and fails a hook that exceeds its timeout", async () => {
    const agent = new Agent(
      {
        name: "hooked_agent",
        backend: { type: "echo" },
        hooks: [{ onStart: "slow.policy" }],
      },
      {
        hook: {
          timeoutMs: 5,
          execute: async (_capability, _input, { signal }) =>
            new Promise((_, reject) => {
              signal.addEventListener("abort", () => reject(signal.reason), { once: true });
            }),
        },
      },
    );

    await expect(agent.call({ messages: [{ role: "user", content: "hello" }] })).rejects.toThrow(
      /timed out/i,
    );
  });

  it("rejects control output from observational hooks", async () => {
    const agent = new Agent(
      {
        name: "hooked_agent",
        backend: { type: "echo" },
        hooks: [{ onEnd: "bad.observer" }],
      },
      {
        hook: {
          execute: async () => ({ additionalContext: "too late" }),
        },
      },
    );

    await expect(agent.call({ messages: [{ role: "user", content: "hello" }] })).rejects.toThrow(
      /observational event onEnd/i,
    );
  });

  it("custom backend delegates prompts to the ACP client", async () => {
    let seen:
      | {
          opts: {
            command: string;
            args: string[];
            cwd?: string;
            env?: Record<string, string>;
            clearEnv?: boolean;
            preferredMode?: string;
          };
          prompt: AcpPromptInput;
          sessionId?: string;
        }
      | undefined;
    const streamed: MessageChunk[] = [];
    const agent = new Agent(
      {
        name: "codex_agent",
        instructions: "Plan with evidence.",
        backend: { type: "custom", program: "codex", args: ["acp"] },
        process: {
          currentDir: "/tmp/project",
          env: { OPENAI_MODEL: "gpt-5" },
          clearEnv: true,
        },
      },
      {
        acpMode: "plan",
        createAcpClient: () => ({
          async prompt(opts, prompt, _swarmConfig, sessionId, onChunk) {
            seen = { opts, prompt, sessionId };
            const chunk: MessageChunk = {
              role: "assistant",
              content: "working",
              kind: "thinking",
              agent: "codex_agent",
            };
            onChunk?.(chunk);
            return {
              messages: [
                {
                  role: "assistant",
                  content: "done",
                  kind: "message",
                  agent: "codex_agent",
                },
              ],
            };
          },
        }),
      },
    );

    const result = await agent.callStream(
      {
        messages: [
          { role: "user", content: "first request" },
          { role: "assistant", content: "middle" },
          { role: "user", content: "latest request" },
        ],
      },
      (chunk) => streamed.push(chunk),
    );

    expect(seen).toEqual({
      opts: {
        command: "codex",
        args: ["acp"],
        cwd: "/tmp/project",
        env: { OPENAI_MODEL: "gpt-5" },
        clearEnv: true,
        preferredMode: "plan",
      },
      prompt: {
        text:
          "Agent instructions:\nPlan with evidence.\n\n" +
          "Conversation history from the canonical SwarmX Session:\n" +
          "[user]\nfirst request\n\n[assistant]\nmiddle\n\n" +
          "Current user request:\nlatest request",
      },
      sessionId: undefined,
    });
    expect(agent.model).toBeUndefined();
    expect(
      (agent.toSwarmConfig().nodes as Record<string, { model?: string }>).codex_agent?.model,
    ).toBeUndefined();
    expect(streamed).toEqual([
      { role: "assistant", content: "working", kind: "thinking", agent: "codex_agent" },
    ]);
    expect(result.messages).toEqual([
      { role: "assistant", content: "done", kind: "message", agent: "codex_agent" },
    ]);
  });

  it("reuses a bound ACP Session and sends only the latest user turn", async () => {
    const sessionIds: Array<string | undefined> = [];
    const prompts: AcpPromptInput[] = [];
    const agent = new Agent(
      {
        name: "bound_agent",
        backend: { type: "custom", program: "test-acp" },
      },
      {
        acpSessionId: "external-session-1",
        onAcpSessionId: (sessionId) => sessionIds.push(sessionId),
        createAcpClient: () => ({
          async prompt(_opts, prompt, _swarmConfig, sessionId) {
            sessionIds.push(sessionId);
            prompts.push(prompt);
            return {
              messages: [{ role: "assistant", content: "continued", kind: "message" }],
            };
          },
        }),
      },
    );

    await expect(
      agent.call({
        messages: [
          { role: "user", content: "old request" },
          { role: "assistant", content: "old answer" },
          { role: "user", content: "new request" },
        ],
      }),
    ).resolves.toMatchObject({
      messages: [{ role: "assistant", content: "continued", kind: "message" }],
    });
    expect(sessionIds).toEqual(["external-session-1"]);
    expect(prompts).toEqual([{ text: "new request" }]);
  });

  it("replaces an unavailable binding before retrying with text-only history", async () => {
    const bindingChanges: Array<string | undefined> = [];
    const attempts: Array<{ sessionId?: string; prompt: AcpPromptInput }> = [];
    let clientCount = 0;
    const agent = new Agent(
      {
        name: "recovering_agent",
        backend: { type: "custom", program: "test-acp" },
      },
      {
        acpSessionId: "missing-session",
        onAcpSessionId: (sessionId) => bindingChanges.push(sessionId),
        createAcpClient: () => {
          clientCount += 1;
          return {
            async prompt(opts, prompt, _swarmConfig, sessionId) {
              attempts.push({ prompt, sessionId });
              if (sessionId) throw new AcpSessionUnavailableError(sessionId);
              await opts.onSessionId?.("replacement-session");
              return {
                messages: [{ role: "assistant", content: "recovered", kind: "message" }],
              };
            },
          };
        },
      },
    );

    await expect(
      agent.call({
        messages: [
          {
            role: "user",
            content: "inspect image",
            attachments: [
              {
                id: "media-1",
                name: "evidence.png",
                kind: "image",
                mimeType: "image/png",
                sizeBytes: 128,
                uri: "file:///tmp/evidence.png",
                source: "user",
              },
            ],
          },
          { role: "assistant", content: "old answer" },
          { role: "user", content: "continue without replaying bytes" },
        ],
      }),
    ).resolves.toMatchObject({
      messages: [{ role: "assistant", content: "recovered", kind: "message" }],
    });
    expect(clientCount).toBe(2);
    expect(bindingChanges).toEqual([undefined, "replacement-session"]);
    expect(attempts[0]).toEqual({
      sessionId: "missing-session",
      prompt: { text: "continue without replaying bytes" },
    });
    expect(attempts[1]?.sessionId).toBeUndefined();
    expect(attempts[1]?.prompt.attachments).toBeUndefined();
    expect(attempts[1]?.prompt.text).toContain(
      "Attachments (metadata only): evidence.png (image/png, 128 bytes)",
    );
    expect(attempts[1]?.prompt.text).not.toContain("base64");
  });

  it("does not continue after a custom backend confirms cancellation", async () => {
    const agent = new Agent(
      {
        name: "cancelled_agent",
        backend: { type: "custom", program: "test-acp" },
      },
      {
        createAcpClient: () => ({
          async prompt() {
            await cancelAcpRequest("cooperative-agent-cancel");
            return { messages: [] };
          },
        }),
      },
    );

    await expect(
      withAcpRequest("cooperative-agent-cancel", () =>
        agent.call({ messages: [{ role: "user", content: "stop" }] }),
      ),
    ).rejects.toBeInstanceOf(RequestCancelledError);
  });

  it("passes the request AbortSignal to native OpenAI calls", async () => {
    const agent = new Agent({ name: "native_agent", model: "gpt-5" });
    let receivedSignal: AbortSignal | undefined;
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });

    const create = vi.fn(
      (_body: unknown, options?: { signal?: AbortSignal }) =>
        new Promise<never>((_resolve, reject) => {
          receivedSignal = options?.signal;
          markStarted();
          options?.signal?.addEventListener("abort", () => reject(new Error("OpenAI aborted")), {
            once: true,
          });
        }),
    );
    Object.defineProperty(agent.client.chat.completions, "create", { value: create });

    const run = withAcpRequest("native-openai-cancel", () =>
      agent.call({ messages: [{ role: "user", content: "wait" }] }),
    );
    await started;

    await expect(cancelAcpRequest("native-openai-cancel")).resolves.toBe(true);
    expect(receivedSignal?.aborted).toBe(true);
    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
  });

  it("compiles a fixed context snapshot before the Provider request", async () => {
    const historicalEvent = createContextEvent({
      id: "evt_history",
      seq: 1,
      sessionId: "session_context",
      turnId: "turn_history",
      timestamp: "2026-08-11T00:00:00.000Z",
      kind: "assistant_message",
      payload: "Historical repository state: tests pass.",
      causalParents: [],
      labels: [],
      metadata: {},
    });
    const snapshot = createContextHistorySnapshot([historicalEvent]);
    const config = parseContextEngineConfig({
      components: {
        eventStore: "sqlite_wal",
        artifactStore: "local_cas",
        normalizer: "deterministic_atomic",
        masker: "deterministic_capsule",
        stateProjector: "sourced_state_v1",
        evidenceProvider: "bm25",
        assembler: "priority_quota",
        verifier: "deterministic",
      },
      assembler: {
        inputTokenBudget: 30,
        reservedOutputTokens: 10,
        slotTokenBudgets: {
          system: 10,
          taskContract: 0,
          state: 10,
          recent: 10,
          evidence: 0,
          summary: 0,
          capsules: 0,
        },
      },
    });
    const onCompiled = vi.fn();
    const contextEngine = {
      compile: vi.fn(
        (input: {
          requestId: string;
          modelVersion: string;
          runtimeContext: Record<string, unknown>;
        }) =>
          compileContext({
            requestId: input.requestId,
            snapshot,
            config,
            modelVersion: input.modelVersion,
            requestedMode: "retrieval",
            effectiveMode: "retrieval",
            items: [
              {
                itemId: "historical_state",
                slot: "state",
                content: "Historical repository state: tests pass.",
                tokenCount: 6,
                priority: 1,
                mandatory: false,
                trust: "untrusted",
                sourceEventIds: [historicalEvent.id],
              },
              {
                itemId: "live_state",
                slot: "state",
                content: `Live repository state: ${String(input.runtimeContext.repoState)}`,
                tokenCount: 6,
                priority: 100,
                mandatory: true,
                trust: "trusted",
                sourceEventIds: [],
              },
            ],
          }),
      ),
      onCompiled,
    };
    const agent = new Agent(
      {
        name: "context_agent",
        model: "gpt-context",
        instructions: "Follow the task contract.",
        client: { apiProtocol: "openai_responses" },
      },
      { contextEngine },
    );
    const create = vi.fn().mockResolvedValue({
      id: "resp_context",
      status: "completed",
      error: null,
      output: [
        {
          id: "msg_context",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "Context applied.", annotations: [] }],
        },
      ],
    });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    await agent.call(
      {
        requestId: "request_context",
        messages: [
          { role: "user", content: "Old request" },
          { role: "assistant", content: "Old answer" },
          { role: "user", content: "Current request" },
        ],
      },
      { repoState: "tests failing" },
    );

    expect(contextEngine.compile).toHaveBeenCalledWith(
      expect.objectContaining({
        requestId: "request_context",
        agentName: "context_agent",
        modelVersion: "gpt-context",
        runtimeContext: { repoState: "tests failing" },
      }),
    );
    expect(create.mock.calls[0]?.[0]?.instructions).toContain(
      "Live repository state: tests failing",
    );
    expect(create.mock.calls[0]?.[0]?.instructions).not.toContain(
      "Historical repository state: tests pass",
    );
    expect(JSON.stringify(create.mock.calls[0]?.[0]?.input)).not.toContain("Old request");
    expect(JSON.stringify(create.mock.calls[0]?.[0]?.input)).toContain("Current request");
    expect(onCompiled).toHaveBeenCalledWith(
      expect.objectContaining({ requestId: "request_context", snapshotId: snapshot.snapshotId }),
    );
  });

  it("rejects context overflow before MCP startup or a Provider request", async () => {
    const snapshot = createContextHistorySnapshot([]);
    const createMcpManager = vi.fn(() => new McpManager());
    const agent = new Agent(
      {
        name: "overflow_agent",
        model: "gpt-context",
        client: { apiProtocol: "openai_responses" },
      },
      {
        createMcpManager,
        contextEngine: {
          compile: ({ requestId, modelVersion }: { requestId: string; modelVersion: string }) =>
            compileContext({
              requestId,
              snapshot,
              config: parseContextEngineConfig({
                components: {
                  eventStore: "sqlite_wal",
                  artifactStore: "local_cas",
                  normalizer: "deterministic_atomic",
                  masker: "deterministic_capsule",
                  stateProjector: "sourced_state_v1",
                  evidenceProvider: "bm25",
                  assembler: "priority_quota",
                  verifier: "deterministic",
                },
                assembler: {
                  inputTokenBudget: 1,
                  reservedOutputTokens: 1,
                  slotTokenBudgets: {
                    system: 1,
                    taskContract: 0,
                    state: 0,
                    recent: 0,
                    evidence: 0,
                    summary: 0,
                    capsules: 0,
                  },
                },
              }),
              modelVersion,
              requestedMode: "retrieval",
              effectiveMode: "retrieval",
              items: [
                {
                  itemId: "mandatory_system",
                  slot: "system",
                  content: "mandatory context",
                  tokenCount: 2,
                  priority: 100,
                  mandatory: true,
                  trust: "trusted",
                  sourceEventIds: [],
                },
              ],
            }),
        },
      },
    );
    const create = vi.fn();
    Object.defineProperty(agent.client.responses, "create", { value: create });

    await expect(
      agent.call({ requestId: "request_overflow", messages: [{ role: "user", content: "Go" }] }),
    ).rejects.toThrow(/Mandatory system context/i);
    expect(createMcpManager).not.toHaveBeenCalled();
    expect(create).not.toHaveBeenCalled();
  });

  it("executes OpenAI Responses with legacy MCP-result normalization", async () => {
    const agent = new Agent({
      name: "responses_agent",
      model: "gpt-5",
      client: { apiProtocol: "openai_responses" },
      parameters: {
        reasoning: {
          control: "effort_enum",
          effort: "high",
          parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
        },
      },
    });
    const callTool = vi.fn().mockResolvedValue({ forecast: "sunny" });
    installMockMcp(agent, callTool);
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        id: "resp_1",
        status: "completed",
        error: null,
        output: [
          {
            id: "reason_1",
            type: "reasoning",
            summary: [{ type: "summary_text", text: "Check the weather tool." }],
          },
          {
            id: "msg_1",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: "I will check.", annotations: [] }],
          },
          {
            id: "fc_1",
            type: "function_call",
            call_id: "call_1",
            name: "weather",
            arguments: '{"city":"Shanghai"}',
            status: "completed",
          },
        ],
      })
      .mockResolvedValueOnce({
        id: "resp_2",
        status: "completed",
        error: null,
        output: [
          {
            id: "msg_2",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: "It is sunny.", annotations: [] }],
          },
        ],
      });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Weather?" }] });

    expect(agent.apiProtocol).toBe("openai_responses");
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      model: "gpt-5",
      reasoning: { effort: "high" },
      tools: [
        {
          type: "function",
          name: "weather",
          parameters: { type: "object" },
          strict: false,
        },
      ],
    });
    expect(create.mock.calls[1]?.[0]?.input).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "function_call", call_id: "call_1" }),
        {
          type: "function_call_output",
          call_id: "call_1",
          output: '{"forecast":"sunny"}',
        },
      ]),
    );
    expect(callTool).toHaveBeenCalledWith(
      "weather",
      { city: "Shanghai" },
      expect.objectContaining({ invocationId: "call_1" }),
    );
    expect(result.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "thinking", content: "Check the weather tool." }),
        expect.objectContaining({ kind: "tool_call", toolName: "weather" }),
        expect.objectContaining({ kind: "tool_result", content: '{"forecast":"sunny"}' }),
        expect.objectContaining({ kind: "message", content: "It is sunny." }),
      ]),
    );
  });

  it("executes host-injected Project tools through the native Responses loop", async () => {
    const readProjectFile = vi.fn().mockResolvedValue({
      path: "README.md",
      content: "# Project-aware SwarmX",
      truncated: false,
    });
    const agent = new Agent(
      {
        name: "project_agent",
        model: "gpt-5.6-luna",
        client: { apiProtocol: "openai_responses" },
      },
      {
        localTools: [
          {
            name: "workspace_read_file",
            description: "Read a Project file.",
            inputSchema: {
              type: "object",
              properties: { path: { type: "string" } },
              required: ["path"],
            },
            call: readProjectFile,
          },
        ],
      },
    );
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        id: "resp_project_tool",
        status: "completed",
        error: null,
        output: [
          {
            id: "fc_project_read",
            type: "function_call",
            call_id: "call_project_read",
            name: "workspace_read_file",
            arguments: '{"path":"README.md"}',
            status: "completed",
          },
        ],
      })
      .mockResolvedValueOnce({
        id: "resp_project_answer",
        status: "completed",
        error: null,
        output: [
          {
            id: "msg_project_answer",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [
              { type: "output_text", text: "This is Project-aware SwarmX.", annotations: [] },
            ],
          },
        ],
      });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({
      messages: [{ role: "user", content: "Introduce this Project." }],
    });

    expect(create.mock.calls[0]?.[0]?.tools).toContainEqual(
      expect.objectContaining({ name: "workspace_read_file" }),
    );
    expect(readProjectFile).toHaveBeenCalledWith(
      { path: "README.md" },
      expect.objectContaining({ invocationId: "call_project_read" }),
    );
    expect(create.mock.calls[1]?.[0]?.input).toContainEqual({
      type: "function_call_output",
      call_id: "call_project_read",
      output: expect.stringContaining("Project-aware SwarmX"),
    });
    expect(result.messages).toContainEqual(
      expect.objectContaining({ kind: "message", content: "This is Project-aware SwarmX." }),
    );
  });

  it("continues OpenAI Responses freeform custom tool calls", async () => {
    const applyPatch = vi
      .fn()
      .mockResolvedValue({ operations: [{ type: "add", path: "new.txt" }] });
    const agent = new Agent(
      {
        name: "codex_custom_tool_agent",
        model: "gpt-5.4",
        client: { apiProtocol: "openai_responses" },
      },
      {
        localTools: [
          {
            kind: "text",
            name: "apply_patch",
            description: "Apply a Codex patch.",
            format: {
              type: "grammar",
              syntax: "lark",
              definition: 'start: "*** Begin Patch"',
            },
            call: applyPatch,
          },
        ],
      },
    );
    const patch = "*** Begin Patch\n*** Add File: new.txt\n+new\n*** End Patch\n";
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        id: "resp_custom_tool",
        status: "completed",
        error: null,
        output: [
          {
            id: "ctc_apply_patch",
            type: "custom_tool_call",
            call_id: "call_apply_patch",
            name: "apply_patch",
            input: patch,
          },
        ],
      })
      .mockResolvedValueOnce({
        id: "resp_custom_answer",
        status: "completed",
        error: null,
        output: [
          {
            id: "msg_custom_answer",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: "Patch applied.", annotations: [] }],
          },
        ],
      });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Create new.txt" }] });

    expect(create.mock.calls[0]?.[0]?.tools).toContainEqual({
      type: "custom",
      name: "apply_patch",
      description: "Apply a Codex patch.",
      format: {
        type: "grammar",
        syntax: "lark",
        definition: 'start: "*** Begin Patch"',
      },
    });
    expect(applyPatch).toHaveBeenCalledWith(
      patch,
      expect.objectContaining({ invocationId: "call_apply_patch" }),
    );
    expect(result.messages).toContainEqual(
      expect.objectContaining({
        kind: "tool_result",
        toolName: "apply_patch",
        render: { invocationId: "call_apply_patch", status: "succeeded" },
      }),
    );
    expect(create.mock.calls[1]?.[0]?.input).toContainEqual({
      type: "custom_tool_call_output",
      call_id: "call_apply_patch",
      output: expect.stringContaining("new.txt"),
    });
    expect(result.messages).toContainEqual(
      expect.objectContaining({ kind: "message", content: "Patch applied." }),
    );
  });

  it("recovers a streamed Project tool call omitted from response.completed", async () => {
    const readProjectFile = vi.fn(
      async (_input: Record<string, unknown>, context?: LocalToolCallContext) => {
        context?.onProgress?.({
          content: "reading README\n",
          structuredContent: { output: "reading README\n", stream: "stdout", mode: "append" },
        });
        return localToolResult("# Streamed Project", {
          type: "text",
          file: { filePath: "README.md", content: "# Streamed Project" },
        });
      },
    );
    const agent = new Agent(
      {
        name: "streamed_project_agent",
        model: "gpt-5.4",
        client: { apiProtocol: "openai_responses" },
      },
      {
        localTools: [
          {
            name: "workspace_read_file",
            description: "Read a Project file.",
            inputSchema: {
              type: "object",
              properties: { path: { type: "string" } },
              required: ["path"],
            },
            call: readProjectFile,
          },
        ],
      },
    );
    const functionCall = {
      id: "fc_streamed_read",
      type: "function_call",
      call_id: "call_streamed_read",
      name: "workspace_read_file",
      arguments: '{"path":"README.md"}',
      status: "completed",
    };
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        async *[Symbol.asyncIterator]() {
          yield {
            type: "response.reasoning_summary_text.delta",
            delta: "**Inspecting README**",
          };
          yield { type: "response.output_item.done", output_index: 1, item: functionCall };
          yield {
            type: "response.completed",
            response: {
              id: "resp_streamed_tool",
              status: "completed",
              error: null,
              output: [
                {
                  id: "reason_streamed_tool",
                  type: "reasoning",
                  summary: [{ type: "summary_text", text: "**Inspecting README**" }],
                },
              ],
            },
          };
        },
      })
      .mockResolvedValueOnce({
        async *[Symbol.asyncIterator]() {
          yield {
            type: "response.reasoning_summary_text.delta",
            delta: "**Preparing final summary**",
          };
          yield {
            type: "response.completed",
            response: {
              id: "resp_streamed_reasoning_only",
              status: "completed",
              error: null,
              output: [
                {
                  id: "reason_streamed_summary",
                  type: "reasoning",
                  summary: [{ type: "summary_text", text: "**Preparing final summary**" }],
                },
              ],
            },
          };
        },
      })
      .mockResolvedValueOnce({
        async *[Symbol.asyncIterator]() {
          yield { type: "response.output_text.delta", delta: "This is the streamed Project." };
          yield {
            type: "response.completed",
            response: {
              id: "resp_streamed_answer",
              status: "completed",
              error: null,
              output: [
                {
                  id: "msg_streamed_answer",
                  type: "message",
                  role: "assistant",
                  status: "completed",
                  content: [
                    {
                      type: "output_text",
                      text: "This is the streamed Project.",
                      annotations: [],
                    },
                  ],
                },
              ],
            },
          };
        },
      });
    Object.defineProperty(agent.client.responses, "create", { value: create });
    const streamed: MessageChunk[] = [];

    const result = await agent.callStream(
      { messages: [{ role: "user", content: "Introduce this Project." }] },
      (chunk) => streamed.push(chunk),
    );

    expect(readProjectFile).toHaveBeenCalledWith(
      { path: "README.md" },
      expect.objectContaining({ invocationId: "call_streamed_read" }),
    );
    expect(create.mock.calls[1]?.[0]?.input).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "function_call_output",
          call_id: "call_streamed_read",
          output: expect.stringContaining("Streamed Project"),
        }),
      ]),
    );
    expect(create).toHaveBeenCalledTimes(3);
    expect(streamed).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "thinking", content: "**Inspecting README**" }),
        expect.objectContaining({ kind: "tool_call", toolName: "workspace_read_file" }),
        expect.objectContaining({
          kind: "tool_progress",
          toolName: "workspace_read_file",
          content: "reading README\n",
          render: { invocationId: "call_streamed_read", status: "running" },
        }),
        expect.objectContaining({
          kind: "tool_result",
          toolName: "workspace_read_file",
          content: "# Streamed Project",
          structuredContent: {
            type: "text",
            file: { filePath: "README.md", content: "# Streamed Project" },
          },
        }),
        expect.objectContaining({ kind: "message", content: "This is the streamed Project." }),
      ]),
    );
    expect(result.messages.at(-1)).toEqual(
      expect.objectContaining({ kind: "message", content: "This is the streamed Project." }),
    );
    expect(result.messages.some((chunk) => chunk.kind === "tool_progress")).toBe(false);
  });

  it("executes Codex Responses with SwarmX context and stateless encrypted continuation", async () => {
    const accessToken = fakeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "account-123" },
    });
    const agent = new Agent({
      name: "codex_responses_agent",
      model: "gpt-5.4",
      instructions: "Use only the SwarmX agent instructions.",
      client: { api_mode: "codex_responses", access_token: accessToken },
    });
    const callTool = vi.fn().mockResolvedValue({ forecast: "sunny" });
    installMockMcp(agent, callTool);
    const create = vi
      .fn()
      .mockResolvedValueOnce(
        openAIResponseStream({
          id: "resp_1",
          status: "completed",
          error: null,
          output: [
            {
              id: "reason_1",
              type: "reasoning",
              encrypted_content: "encrypted-state",
              summary: [{ type: "summary_text", text: "Use the weather tool." }],
              status: "completed",
            },
            {
              id: "fc_1",
              type: "function_call",
              call_id: "call_1",
              name: "weather",
              arguments: '{"city":"Shanghai"}',
              status: "completed",
            },
          ],
        }),
      )
      .mockResolvedValueOnce(
        openAIResponseStream({
          id: "resp_2",
          status: "completed",
          error: null,
          output: [
            {
              id: "msg_2",
              type: "message",
              role: "assistant",
              status: "completed",
              content: [{ type: "output_text", text: "It is sunny.", annotations: [] }],
            },
          ],
        }),
      );
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Weather?" }] });

    expect(agent.apiMode).toBe("codex_responses");
    expect(agent.apiProtocol).toBe("openai_responses");
    expect(agent.client.apiKey).toBe(accessToken);
    expect(agent.client.baseURL).toBe("https://chatgpt.com/backend-api/codex");
    const clientOptions = agent.client as unknown as {
      _options: { defaultHeaders?: Record<string, string> };
    };
    expect(clientOptions._options.defaultHeaders).toMatchObject({
      originator: "swarmx",
      "ChatGPT-Account-ID": "account-123",
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      model: "gpt-5.4",
      instructions: "Use only the SwarmX agent instructions.",
      store: false,
      include: ["reasoning.encrypted_content"],
      reasoning: { summary: "auto" },
      tool_choice: "auto",
      parallel_tool_calls: true,
      stream: true,
    });
    const replayInput = create.mock.calls[1]?.[0]?.input;
    expect(replayInput).toEqual(
      expect.arrayContaining([
        {
          type: "reasoning",
          encrypted_content: "encrypted-state",
          summary: [{ type: "summary_text", text: "Use the weather tool." }],
        },
        {
          type: "function_call",
          call_id: "call_1",
          name: "weather",
          arguments: '{"city":"Shanghai"}',
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: '{"forecast":"sunny"}',
        },
      ]),
    );
    expect(JSON.stringify(replayInput)).not.toContain("reason_1");
    expect(JSON.stringify(replayInput)).not.toContain("fc_1");
    expect(callTool).toHaveBeenCalledWith(
      "weather",
      { city: "Shanghai" },
      expect.objectContaining({ invocationId: "call_1" }),
    );
    expect(result.messages).toContainEqual(
      expect.objectContaining({ kind: "message", content: "It is sunny." }),
    );
  });

  it("streams Codex subscription Responses without a consumer callback", async () => {
    const agent = new Agent({
      name: "codex_unary_caller",
      model: "gpt-5.6-luna",
      client: { apiMode: "codex_responses", accessToken: fakeJwt({}) },
    });
    const response = {
      id: "resp_codex_stream",
      status: "completed",
      error: null,
      output: [],
    };
    const stream = {
      async *[Symbol.asyncIterator]() {
        yield { type: "response.output_text.delta", delta: "OK" };
        yield { type: "response.completed", response };
      },
    };
    const create = vi.fn((request: { stream?: boolean }) => {
      if (request.stream !== true) throw new Error("Codex subscription requires streaming.");
      return Promise.resolve(stream);
    });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Reply OK" }] });

    expect(create.mock.calls[0]?.[0]).toMatchObject({ model: "gpt-5.6-luna", stream: true });
    expect(result.messages).toContainEqual(
      expect.objectContaining({ kind: "message", content: "OK" }),
    );
  });

  it("rejects codex_responses with a non-Responses protocol", () => {
    expect(
      () =>
        new Agent({
          name: "invalid_codex_agent",
          client: { apiProtocol: "openai_chat", apiMode: "codex_responses" },
        }),
    ).toThrow(/requires apiProtocol "openai_responses"/);
  });

  it("executes Anthropic Messages natively with auth-token and tool-result blocks", async () => {
    const agent = new Agent({
      name: "anthropic_agent",
      model: "claude-sonnet-4-6",
      instructions: "Be concise.",
      client: { apiProtocol: "anthropic" },
      process: {
        env: {
          ANTHROPIC_AUTH_TOKEN: "scoped-token",
          ANTHROPIC_BASE_URL: "https://gateway.example/anthropic",
          ANTHROPIC_MODEL: "claude-sonnet-4-6",
        },
      },
      parameters: {
        reasoning: {
          control: "effort_enum",
          effort: "high",
          parameterMapping: { api: "anthropic.messages", path: "output_config.effort" },
        },
      },
    });
    const callTool = vi
      .fn()
      .mockResolvedValue(localToolResult("cloudy model text", { forecast: "cloudy" }));
    installMockMcp(agent, callTool);
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        id: "msg_1",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-6",
        stop_reason: "tool_use",
        stop_sequence: null,
        container: null,
        usage: {},
        content: [
          { type: "thinking", thinking: "Use the weather tool.", signature: "sig" },
          { type: "text", text: "Checking." },
          {
            type: "tool_use",
            id: "toolu_1",
            name: "weather",
            input: { city: "Shanghai" },
          },
        ],
      })
      .mockResolvedValueOnce({
        id: "msg_2",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-6",
        stop_reason: "end_turn",
        stop_sequence: null,
        container: null,
        usage: {},
        content: [{ type: "text", text: "It is cloudy." }],
      });
    Object.defineProperty(agent.anthropicClient.messages, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Weather?" }] });

    expect(agent.apiProtocol).toBe("anthropic");
    expect(agent.anthropicClient.authToken).toBe("scoped-token");
    expect(agent.anthropicClient.baseURL).toBe("https://gateway.example/anthropic");
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      model: "claude-sonnet-4-6",
      system: "Be concise.",
      output_config: { effort: "high" },
      stream: false,
      tools: [
        {
          name: "weather",
          description: "Weather lookup",
          input_schema: { type: "object" },
        },
      ],
    });
    expect(create.mock.calls[1]?.[0]?.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ role: "assistant" }),
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_1",
              content: "cloudy model text",
            },
          ],
        },
      ]),
    );
    expect(callTool).toHaveBeenCalledWith(
      "weather",
      { city: "Shanghai" },
      expect.objectContaining({ invocationId: "toolu_1" }),
    );
    expect(result.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "thinking", content: "Use the weather tool." }),
        expect.objectContaining({ kind: "tool_call", toolName: "weather" }),
        expect.objectContaining({
          kind: "tool_result",
          toolName: "weather",
          content: "cloudy model text",
          structuredContent: { forecast: "cloudy" },
        }),
        expect.objectContaining({ kind: "message", content: "It is cloudy." }),
      ]),
    );
  });

  it("streams typed OpenAI Responses events without Chat conversion", async () => {
    const agent = new Agent({
      name: "responses_stream_agent",
      model: "gpt-5",
      client: { apiProtocol: "openai_responses" },
    });
    const response = {
      id: "resp_stream",
      status: "completed",
      error: null,
      output: [
        {
          id: "reason_stream",
          type: "reasoning",
          summary: [{ type: "summary_text", text: "Brief thought." }],
        },
        {
          id: "msg_stream",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "Hello", annotations: [] }],
        },
      ],
    };
    const stream = {
      async *[Symbol.asyncIterator]() {
        yield { type: "response.reasoning_summary_text.delta", delta: "Brief thought." };
        yield { type: "response.output_text.delta", delta: "Hello" };
        yield { type: "response.completed", response };
      },
    };
    const create = vi.fn().mockResolvedValue(stream);
    Object.defineProperty(agent.client.responses, "create", { value: create });
    const streamed: MessageChunk[] = [];

    const result = await agent.callStream(
      { messages: [{ role: "user", content: "Hello" }] },
      (chunk) => streamed.push(chunk),
    );

    expect(create.mock.calls[0]?.[0]).toMatchObject({ model: "gpt-5", stream: true });
    expect(streamed).toEqual([
      expect.objectContaining({ kind: "thinking", content: "Brief thought." }),
      expect.objectContaining({ kind: "message", content: "Hello" }),
    ]);
    expect(result.messages).toEqual([
      expect.objectContaining({ kind: "thinking", content: "Brief thought." }),
      expect.objectContaining({ kind: "message", content: "Hello" }),
    ]);
  });

  it("streams Anthropic content blocks without OpenAI conversion", async () => {
    const agent = new Agent({
      name: "anthropic_stream_agent",
      model: "claude-sonnet-4-6",
      client: { apiProtocol: "anthropic" },
      process: { env: { ANTHROPIC_API_KEY: "scoped-key" } },
    });
    const finalMessage = {
      id: "msg_stream",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-6",
      stop_reason: "end_turn",
      stop_sequence: null,
      container: null,
      usage: {},
      content: [
        { type: "thinking", thinking: "Brief thought.", signature: "sig" },
        { type: "text", text: "Hello" },
      ],
    };
    const stream = {
      async *[Symbol.asyncIterator]() {
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "Brief thought." },
        };
        yield {
          type: "content_block_delta",
          index: 1,
          delta: { type: "text_delta", text: "Hello" },
        };
      },
      finalMessage: vi.fn().mockResolvedValue(finalMessage),
    };
    const createStream = vi.fn().mockReturnValue(stream);
    Object.defineProperty(agent.anthropicClient.messages, "stream", { value: createStream });
    const streamed: MessageChunk[] = [];

    const result = await agent.callStream(
      { messages: [{ role: "user", content: "Hello" }] },
      (chunk) => streamed.push(chunk),
    );

    expect(createStream.mock.calls[0]?.[0]).toMatchObject({
      model: "claude-sonnet-4-6",
      max_tokens: 8192,
    });
    expect(streamed).toEqual([
      expect.objectContaining({ kind: "thinking", content: "Brief thought." }),
      expect.objectContaining({ kind: "message", content: "Hello" }),
    ]);
    expect(result.messages).toEqual([
      expect.objectContaining({ kind: "thinking", content: "Brief thought." }),
      expect.objectContaining({ kind: "message", content: "Hello" }),
    ]);
  });

  it.each(["anthropic", "openai_responses"] as const)(
    "refreshes native %s tools after ToolSearch activates a deferred schema",
    async (apiProtocol) => {
      const agent = new Agent({
        name: `${apiProtocol}_dynamic_tools`,
        model: apiProtocol === "anthropic" ? "claude-sonnet-4-6" : "gpt-5",
        client: { apiProtocol },
        process: {
          env:
            apiProtocol === "anthropic"
              ? { ANTHROPIC_API_KEY: "scoped-key" }
              : { OPENAI_API_KEY: "scoped-key" },
        },
      });
      const toolSearch = nativeFunctionTool("ToolSearch");
      const deferred = nativeFunctionTool("mcp__github__list_issues");
      let activated = false;
      const callTool = vi.fn(async (name: string) => {
        if (name !== "ToolSearch") throw new Error(`Unexpected tool call: ${name}`);
        activated = true;
        return localToolResult("Loaded deferred tools: mcp__github__list_issues", {
          matches: ["mcp__github__list_issues"],
          query: "issues",
          total_deferred_tools: 1,
        });
      });
      Object.defineProperty(agent, "mcp", {
        configurable: true,
        writable: true,
        value: {
          toolsForOpenai: () => [toolSearch, ...(activated ? [deferred] : [])],
          toolsForNative: () => [toolSearch, ...(activated ? [deferred] : [])],
          callTool,
          close: vi.fn().mockResolvedValue(undefined),
        },
      });

      const create =
        apiProtocol === "anthropic"
          ? vi
              .fn()
              .mockResolvedValueOnce({
                id: "msg_search",
                type: "message",
                role: "assistant",
                model: "claude-sonnet-4-6",
                stop_reason: "tool_use",
                stop_sequence: null,
                usage: {},
                content: [
                  {
                    type: "tool_use",
                    id: "toolu_search",
                    name: "ToolSearch",
                    input: { query: "issues" },
                  },
                ],
              })
              .mockResolvedValueOnce({
                id: "msg_done",
                type: "message",
                role: "assistant",
                model: "claude-sonnet-4-6",
                stop_reason: "end_turn",
                stop_sequence: null,
                usage: {},
                content: [{ type: "text", text: "done" }],
              })
          : vi
              .fn()
              .mockResolvedValueOnce({
                id: "resp_search",
                status: "completed",
                error: null,
                output: [
                  {
                    id: "fc_search",
                    type: "function_call",
                    call_id: "call_search",
                    name: "ToolSearch",
                    arguments: '{"query":"issues"}',
                    status: "completed",
                  },
                ],
              })
              .mockResolvedValueOnce({
                id: "resp_done",
                status: "completed",
                error: null,
                output: [
                  {
                    id: "msg_done",
                    type: "message",
                    role: "assistant",
                    status: "completed",
                    content: [{ type: "output_text", text: "done", annotations: [] }],
                  },
                ],
              });
      if (apiProtocol === "anthropic") {
        Object.defineProperty(agent.anthropicClient.messages, "create", { value: create });
      } else {
        Object.defineProperty(agent.client.responses, "create", { value: create });
      }

      await agent.call({ messages: [{ role: "user", content: "List issues" }] });

      const firstNames = create.mock.calls[0]?.[0]?.tools.map(
        (tool: { name: string }) => tool.name,
      );
      const secondNames = create.mock.calls[1]?.[0]?.tools.map(
        (tool: { name: string }) => tool.name,
      );
      expect(firstNames).toEqual(["ToolSearch"]);
      expect(secondNames).toEqual(["ToolSearch", "mcp__github__list_issues"]);
      expect(callTool).toHaveBeenCalledWith(
        "ToolSearch",
        { query: "issues" },
        expect.objectContaining({
          invocationId: apiProtocol === "anthropic" ? "toolu_search" : "call_search",
        }),
      );
    },
  );

  it("refreshes OpenAI Chat tools after ToolSearch activates a deferred schema", async () => {
    const agent = new Agent({
      name: "chat_dynamic_tools",
      model: "claude-sonnet-4-6",
      process: { env: { OPENAI_API_KEY: "scoped-key" } },
    });
    const toolSearch = nativeFunctionTool("ToolSearch");
    const deferred = nativeFunctionTool("mcp__github__list_issues");
    let activated = false;
    const callTool = vi.fn(async () => {
      activated = true;
      return localToolResult("Loaded deferred tools: mcp__github__list_issues", {
        matches: ["mcp__github__list_issues"],
        query: "issues",
        total_deferred_tools: 1,
      });
    });
    Object.defineProperty(agent, "mcp", {
      configurable: true,
      writable: true,
      value: {
        toolsForOpenai: () => [toolSearch, ...(activated ? [deferred] : [])],
        toolsForNative: () => [toolSearch, ...(activated ? [deferred] : [])],
        callTool,
        close: vi.fn().mockResolvedValue(undefined),
      },
    });
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_search",
                  type: "function",
                  function: { name: "ToolSearch", arguments: '{"query":"issues"}' },
                },
              ],
            },
          },
        ],
      })
      .mockResolvedValueOnce({
        choices: [{ message: { role: "assistant", content: "done" } }],
      });
    Object.defineProperty(agent.client.chat.completions, "create", { value: create });

    await agent.call({ messages: [{ role: "user", content: "List issues" }] });

    const firstNames = create.mock.calls[0]?.[0]?.tools.map(
      (tool: { function: { name: string } }) => tool.function.name,
    );
    const secondNames = create.mock.calls[1]?.[0]?.tools.map(
      (tool: { function: { name: string } }) => tool.function.name,
    );
    expect(firstNames).toEqual(["ToolSearch"]);
    expect(secondNames).toEqual(["ToolSearch", "mcp__github__list_issues"]);
  });

  it("does not block a Claude profile's first model request on pending MCP startup", async () => {
    const connectServer = vi.fn(
      (_name: string, _config: unknown, signal: AbortSignal) =>
        new Promise<McpConnectionResult>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason), { once: true });
        }),
    );
    const manager = new McpManager({ connectServer });
    const agent = new Agent(
      {
        name: "claude_pending_mcp",
        model: "claude-sonnet-4-6",
        client: { apiProtocol: "openai_responses" },
        process: { env: { OPENAI_API_KEY: "scoped-key" } },
        mcpServers: { slow: { type: "stdio", command: "unused" } },
      },
      {
        createMcpManager: () => manager,
        localTools: [
          {
            name: "Bash",
            inputSchema: { type: "object" },
            call: async () => ({ ok: true }),
          },
        ],
      },
    );
    const create = vi.fn().mockResolvedValue({
      id: "resp_done",
      status: "completed",
      error: null,
      output: [
        {
          id: "msg_done",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "done", annotations: [] }],
        },
      ],
    });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    await agent.call({ messages: [{ role: "user", content: "Continue without MCP" }] });

    expect(connectServer).toHaveBeenCalledTimes(1);
    expect(create).toHaveBeenCalledTimes(1);
    expect(create.mock.calls[0]?.[0]?.tools.map((tool: { name: string }) => tool.name)).toEqual([
      "Bash",
      "ToolSearch",
      "WaitForMcpServers",
    ]);
  });

  it.each(["anthropic", "openai_responses"] as const)(
    "passes request cancellation to native %s calls",
    async (apiProtocol) => {
      const agent = new Agent({
        name: `${apiProtocol}_cancel_agent`,
        model: apiProtocol === "anthropic" ? "claude-sonnet-4-6" : "gpt-5",
        client: { apiProtocol },
        process: {
          env:
            apiProtocol === "anthropic"
              ? { ANTHROPIC_API_KEY: "scoped-key" }
              : { OPENAI_API_KEY: "scoped-key" },
        },
      });
      let receivedSignal: AbortSignal | undefined;
      let markStarted!: () => void;
      const started = new Promise<void>((resolve) => {
        markStarted = resolve;
      });
      const create = vi.fn(
        (_body: unknown, options?: { signal?: AbortSignal }) =>
          new Promise<never>((_resolve, reject) => {
            receivedSignal = options?.signal;
            markStarted();
            options?.signal?.addEventListener("abort", () => reject(new Error("aborted")), {
              once: true,
            });
          }),
      );
      if (apiProtocol === "anthropic") {
        Object.defineProperty(agent.anthropicClient.messages, "create", { value: create });
      } else {
        Object.defineProperty(agent.client.responses, "create", { value: create });
      }
      const requestId = `native-${apiProtocol}-cancel`;
      const run = withAcpRequest(requestId, () =>
        agent.call({ messages: [{ role: "user", content: "wait" }] }),
      );
      await started;

      await expect(cancelAcpRequest(requestId)).resolves.toBe(true);
      expect(receivedSignal?.aborted).toBe(true);
      await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    },
  );

  it("preserves provider reasoning content across tool-call continuation", async () => {
    const agent = new Agent({ name: "deepseek_agent", model: "deepseek-v4-pro" });
    const callTool = vi
      .fn()
      .mockResolvedValue(localToolResult("chat model text", { result: "structured" }));
    installMockMcp(agent, callTool);
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              reasoning_content: "verified reasoning state",
              tool_calls: [
                {
                  id: "tool-1",
                  type: "function",
                  function: { name: "weather", arguments: "{}" },
                },
              ],
            },
          },
        ],
      })
      .mockResolvedValueOnce({
        choices: [{ message: { role: "assistant", content: "done" } }],
      });
    Object.defineProperty(agent.client.chat.completions, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "reason" }] });

    expect(result.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "thinking", content: "verified reasoning state" }),
      ]),
    );
    expect(create.mock.calls[1]?.[0]?.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          role: "assistant",
          reasoning_content: "verified reasoning state",
          tool_calls: expect.any(Array),
        }),
        expect.objectContaining({ role: "tool", content: "chat model text" }),
      ]),
    );
    expect(result.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "tool_result",
          content: "chat model text",
          structuredContent: { result: "structured" },
          render: { invocationId: "tool-1", status: "succeeded" },
        }),
      ]),
    );
  });
});

function installMockMcp(agent: Agent, callTool: ReturnType<typeof vi.fn>): void {
  const tool = nativeFunctionTool("weather", "Weather lookup");
  Object.defineProperty(agent, "mcp", {
    configurable: true,
    writable: true,
    value: {
      toolsForOpenai: () => [tool],
      toolsForNative: () => [tool],
      callTool,
      close: vi.fn().mockResolvedValue(undefined),
    },
  });
}

function nativeFunctionTool(name: string, description = name) {
  return {
    type: "function" as const,
    function: {
      name,
      description,
      parameters: { type: "object" },
    },
  };
}

function fakeJwt(claims: Record<string, unknown>): string {
  return `header.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.signature`;
}

function openAIResponseStream(response: unknown) {
  return {
    async *[Symbol.asyncIterator]() {
      yield { type: "response.completed", response };
    },
  };
}

describe("HookRef", () => {
  it("constructs with hook config", () => {
    const hook = new HookRef({
      onStart: "start",
      onEnd: "end",
    });
    expect(hook.onStart).toBe("start");
    expect(hook.onEnd).toBe("end");
    expect(hook.onHandoff).toBeUndefined();
    expect(hook.onChunk).toBeUndefined();
  });
});
