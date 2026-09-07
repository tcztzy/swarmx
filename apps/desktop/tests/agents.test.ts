import type { NewSessionMeta } from "@agentclientprotocol/claude-agent-acp";
import * as acp from "@agentclientprotocol/sdk";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { loadAgent, selectedAgent } from "../src/agent.js";
import type { Observer } from "../src/agents/types.js";
import { DEFAULT_POLICY } from "../src/settings.js";

const state = vi.hoisted(() => ({
  launches: [] as { command: string; args: string[]; env: NodeJS.ProcessEnv }[],
  calls: [] as { method: string; params: unknown }[],
  prompt: undefined as
    | ((context: acp.AgentRequestContext<acp.PromptRequest>) => Promise<acp.PromptResponse>)
    | undefined,
  cancel: vi.fn(),
  closes: vi.fn(),
  next: 0,
  unavailable: false,
}));
const configOptions: acp.SessionConfigOption[] = [
  {
    id: "upstream-model",
    category: "model",
    name: "Model",
    type: "select",
    currentValue: "model-a",
    options: [
      { value: "model-a", name: "A" },
      { value: "model-b", name: "B" },
    ],
  },
  {
    id: "upstream-thinking",
    category: "thought_level",
    name: "Thinking",
    type: "select",
    currentValue: "high",
    options: [{ value: "high", name: "High" }],
  },
];
vi.mock("../src/agents/acp-process.js", () => ({
  connectAcpProcess: async (
    client: acp.ClientApp,
    command: string,
    args: string[],
    cwd: string,
    env: NodeJS.ProcessEnv,
  ) => {
    state.launches.push({ command, args, env });
    if (state.unavailable) throw new Error("adapter unavailable");
    const record = (method: string, params: unknown) => state.calls.push({ method, params });
    const configuration = () => ({
      configOptions: structuredClone(configOptions),
      modes: {
        currentModeId:
          JSON.parse(env.CODEX_CONFIG ?? "{}").sandbox_mode === "read-only"
            ? "read-only"
            : "default",
        availableModes: [
          { id: "default", name: "Default" },
          { id: "plan", name: "Plan" },
        ],
      },
    });
    const server = acp
      .agent()
      .onRequest(acp.methods.agent.initialize, ({ params }) => {
        record("initialize", params);
        return {
          protocolVersion: acp.PROTOCOL_VERSION,
          agentCapabilities: {
            loadSession: true,
            sessionCapabilities: { list: {}, resume: {} },
            mcpCapabilities: { http: true },
          },
          _meta: { steering: { supported: true } },
        };
      })
      .onRequest(acp.methods.agent.session.list, ({ params }) => {
        record("list", params);
        return {
          sessions: params.cursor
            ? [{ sessionId: "second", cwd }]
            : [
                { sessionId: "saved", cwd },
                { sessionId: "foreign", cwd: "/other-project" },
              ],
          ...(params.cursor ? {} : { nextCursor: "next" }),
        };
      })
      .onRequest(acp.methods.agent.session.new, ({ params }) => {
        record("new", params);
        const reserved = (params._meta as NewSessionMeta)?.claudeCode?.options?.sessionId;
        return { sessionId: reserved ?? `created-${++state.next}`, ...configuration() };
      })
      .onRequest(acp.methods.agent.session.resume, ({ params }) => {
        record("resume", params);
        return configuration();
      })
      .onRequest(acp.methods.agent.session.load, async ({ params, client }) => {
        record("load", params);
        await client.notify(acp.methods.client.session.update, {
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "history" },
          },
        });
        return configuration();
      })
      .onRequest(acp.methods.agent.session.setMode, ({ params }) => {
        record("mode", params);
        return {};
      })
      .onRequest(acp.methods.agent.session.setConfigOption, ({ params }) => {
        record("config", params);
        return { configOptions: structuredClone(configOptions) };
      })
      .onRequest(acp.methods.agent.session.prompt, async (context) => {
        record("prompt", context.params);
        return state.prompt?.(context) ?? { stopReason: "end_turn" };
      })
      .onNotification(acp.methods.agent.session.cancel, ({ params }) => {
        record("cancel", params);
        state.cancel();
      })
      .onRequest("_session/steering", { parse: (value: unknown) => value }, ({ params }) => {
        record("steer", params);
        return {};
      });
    const connection = client.connect(server);
    return {
      connection,
      close: async () => {
        state.closes();
        connection.close();
      },
    };
  },
}));

const options = {
  cwd: process.cwd(),
  mcp: { url: "http://localhost/mcp", headers: { authorization: "Bearer test" } },
};
const view = (): Observer => ({
  executionId: "run-1",
  text: vi.fn(),
  raw: vi.fn(),
  tool: vi.fn(),
  execution: vi.fn(),
  interact: vi.fn(async () => ({ optionId: "allow" })),
});
beforeEach(() => {
  state.calls = [];
  state.launches = [];
  state.prompt = undefined;
  state.unavailable = false;
  state.next = 0;
});
afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllEnvs();
});

it.each(["codex", "claude", "hermes", "openclaw"] as const)(
  "uses upstream %s ACP without requiring a SwarmX extension",
  async (id) => {
    const agent = await loadAgent(id, options);
    try {
      expect((await agent.list()).map((session) => session.sessionId)).toEqual([
        `${id}:saved`,
        `${id}:second`,
      ]);
      expect((await agent.models(`${id}:saved`)).current).toEqual({
        model: "model-a",
        effort: "high",
        mode: "default",
      });
      const observer = view();
      await agent.read(`${id}:saved`, observer);
      expect(observer.text).toHaveBeenCalledWith("saved", "history", "assistant");
      await expect(
        agent.start(`${id}:saved`, "hello", observer, { model: "model-b", effort: "high" }),
      ).resolves.toMatchObject({ stopReason: "end_turn" });
      expect(
        state.calls
          .filter((call) => call.method === "config")
          .map((call) => (call.params as acp.SetSessionConfigOptionRequest).configId),
      ).toEqual(["upstream-model", "upstream-thinking"]);
      expect(
        state.calls
          .filter((call) => call.method === "initialize")
          .every(
            (call) =>
              (call.params as acp.InitializeRequest).clientCapabilities?._meta === undefined,
          ),
      ).toBe(true);
      expect(state.launches[0]?.args).toEqual(
        id === "codex" || id === "claude"
          ? [
              expect.stringContaining(
                `${id === "codex" ? "codex-acp" : "claude-agent-acp"}/dist/index.js`,
              ),
            ]
          : ["acp"],
      );
    } finally {
      await agent.dispose();
    }
  },
);

it("never falls back when the upstream process fails", async () => {
  state.unavailable = true;
  expect(selectedAgent()).toBe("codex");
  expect(() => selectedAgent("unknown")).toThrow("Unknown Agent");
  await expect(loadAgent("codex", options)).rejects.toThrow("adapter unavailable");
  expect(state.launches).toHaveLength(1);
});

it.each(["/model forbidden", "/review", "/goal hidden work"])(
  "preserves native command dispatch through %s",
  async (prompt) => {
    const agent = await loadAgent("codex", { ...options, executionPolicy: () => DEFAULT_POLICY });
    await agent.start("codex:saved", prompt, view());
    expect(state.calls.findLast((call) => call.method === "prompt")?.params).toMatchObject({
      prompt: [{ type: "text", text: prompt }],
    });
    await agent.dispose();
  },
);

it.each(["hermes", "openclaw"] as const)(
  "supports Host-authorized %s with native execution semantics",
  async (id) => {
    const agent = await loadAgent(id, { ...options, executionPolicy: () => DEFAULT_POLICY });
    await expect(agent.start(`${id}:saved`, "work", view())).resolves.toEqual({
      stopReason: "end_turn",
    });
    expect(state.launches[0]?.args).toEqual(["acp"]);
    await agent.dispose();
  },
);

it("preserves the configured Hermes Python entry point and OpenClaw Gateway URL", async () => {
  vi.stubEnv("SWARMX_HERMES_PYTHON", "/hermes/python");
  vi.stubEnv("OPENCLAW_GATEWAY_URL", "ws://localhost:18789");
  const hermes = await loadAgent("hermes", options);
  const openclaw = await loadAgent("openclaw", options);
  expect(state.launches[0]).toMatchObject({
    command: "/hermes/python",
    args: ["-m", "acp_adapter"],
  });
  expect(state.launches[1]).toMatchObject({
    command: "openclaw",
    args: ["acp", "--url", "ws://localhost:18789"],
  });
  await Promise.all([hermes.dispose(), openclaw.dispose()]);
});

it("preserves native configuration and binds distinct MCP endpoints per run", async () => {
  let policy = { ...DEFAULT_POLICY };
  const released = new Set<string>();
  const bound = vi.fn();
  const registerMcp = (token: string) => ({
    bind: (session: string, run: string) => bound(token, session, run),
    dispose: () => {
      released.add(token);
    },
  });
  const agent = await loadAgent("codex", {
    ...options,
    executionPolicy: () => policy,
    registerMcp,
  });
  try {
    await agent.start("codex:saved", "first", view());
    policy = { ...policy, tools: [] };
    await agent.start("codex:saved", "second", { ...view(), executionId: "run-2" });
    const configs = state.launches
      .slice(-2)
      .map((launch) => JSON.parse(launch.env.CODEX_CONFIG ?? "{}"));
    expect(configs).toEqual([{}, {}]);
    expect(bound.mock.calls.map((call) => call.slice(1))).toEqual([
      ["codex:saved", "run-1"],
      ["codex:saved", "run-2"],
    ]);
    expect(bound.mock.calls[0]?.[0]).not.toEqual(bound.mock.calls[1]?.[0]);
    expect(bound.mock.calls.every(([token]) => released.has(token))).toBe(true);
    const params = state.calls.findLast((call) => call.method === "resume")
      ?.params as acp.ResumeSessionRequest;
    const server = params.mcpServers?.[0];
    expect(server).toMatchObject({
      type: "http",
      headers: [{ name: "authorization", value: `Bearer ${bound.mock.calls[1]?.[0]}` }],
    });
  } finally {
    await agent.dispose();
  }
});

it("preserves stricter inherited Codex adapter configuration", async () => {
  vi.stubEnv(
    "CODEX_CONFIG",
    JSON.stringify({ sandbox_mode: "read-only", approval_policy: "untrusted" }),
  );
  const agent = await loadAgent("codex", { ...options, executionPolicy: () => DEFAULT_POLICY });
  expect(JSON.parse(state.launches[0]?.env.CODEX_CONFIG ?? "{}")).toMatchObject({
    sandbox_mode: "read-only",
    approval_policy: "untrusted",
  });
  await agent.dispose();
});

it("retains never-started Claude IDs and resumes only after dispatch", async () => {
  const first = await loadAgent("claude", options);
  const id = await first.create();
  await first.dispose();
  const restored = await loadAgent("claude", options);
  restored.restoreEmptySessions?.([id]);
  const observer = view();
  await restored.read(id, observer);
  expect(observer.text).not.toHaveBeenCalled();
  await restored.start(id, "first prompt", observer);
  expect(state.calls.find((call) => call.method === "new")?.params).toMatchObject({
    _meta: { claudeCode: { options: { sessionId: id.slice("claude:".length) } } },
  });
  await restored.start(id, "continue", observer);
  expect(state.calls.findLast((call) => call.method === "resume")?.params).toMatchObject({
    sessionId: id.slice("claude:".length),
  });
  await restored.dispose();
});

it("keeps a fresh session when Host tools narrow and applies the selected native mode", async () => {
  let policy = { ...DEFAULT_POLICY };
  const agent = await loadAgent("codex", {
    ...options,
    executionPolicy: () => policy,
    registerMcp: () => ({ bind() {}, dispose() {} }),
  });
  const session = await agent.create();
  policy = { ...policy, tools: [] };
  await agent.start(session, "inspect", view(), { mode: "plan" });
  expect(state.calls.some((call) => call.method === "resume")).toBe(false);
  expect(state.calls.find((call) => call.method === "mode")?.params).toEqual({
    sessionId: session.slice("codex:".length),
    modeId: "plan",
  });
  expect(state.launches.at(-1)?.env.CODEX_CONFIG).toBeUndefined();
  await agent.dispose();
});

it("uses advertised mode config IDs and rejects unknown modes before dispatch", async () => {
  configOptions.push({
    id: "native-permissions",
    category: "mode",
    name: "Permissions",
    type: "select",
    currentValue: "read-only",
    options: [
      { value: "read-only", name: "Ask for approval" },
      { value: "agent-full-access", name: "Full access" },
    ],
  });
  const agent = await loadAgent("codex", options);
  try {
    const catalog = await agent.models("codex:saved");
    expect(catalog.modes).toEqual([
      { id: "read-only", name: "Ask for approval" },
      { id: "agent-full-access", name: "Full access" },
    ]);
    await agent.start("codex:saved", "work", view(), { mode: "agent-full-access" });
    expect(state.calls.find((call) => call.method === "config")?.params).toEqual({
      sessionId: "saved",
      configId: "native-permissions",
      value: "agent-full-access",
    });
    await expect(agent.start("codex:second", "work", view(), { mode: "invented" })).rejects.toThrow(
      "Unsupported ACP mode",
    );
    expect(state.calls.filter((call) => call.method === "prompt")).toHaveLength(1);
  } finally {
    configOptions.pop();
    await agent.dispose();
  }
});

it("leaves ordinary Claude hooks, tools, delegation and permission mode to the harness", async () => {
  const agent = await loadAgent("claude", {
    ...options,
    executionPolicy: () => ({ ...DEFAULT_POLICY, tools: [], delegation: false }),
  });
  await agent.start(await agent.create(), "work", view(), { mode: "plan" });
  const params = state.calls.find((call) => call.method === "new")?.params as acp.NewSessionRequest;
  expect((params._meta as NewSessionMeta)?.claudeCode?.options).toEqual({
    sessionId: expect.any(String),
  });
  expect(state.launches.at(-1)?.env.ACP_DISABLE_TITLE_GENERATION).toBeUndefined();
  expect(state.launches.at(-1)?.env.CLAUDE_CODE_DISABLE_TERMINAL_TITLE).toBeUndefined();
  await agent.dispose();
});

it("keeps Claude background reviews tool-free even when ordinary tasks have full grants", async () => {
  const agent = await loadAgent("claude", {
    ...options,
    reviewOnly: true,
    executionPolicy: () => DEFAULT_POLICY,
  });
  await agent.start(await agent.create(), "review", view());
  const params = state.calls.find((call) => call.method === "new")?.params as acp.NewSessionRequest;
  expect(params.mcpServers).toEqual([]);
  expect(params._meta?.claudeCode).toMatchObject({
    options: {
      tools: [],
      mcpServers: {},
      persistSession: false,
      permissionMode: "dontAsk",
      settings: {
        disableAllHooks: true,
        env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" },
      },
      disallowedTools: expect.arrayContaining(["Agent", "Task"]),
      sandbox: {
        failIfUnavailable: true,
        allowUnsandboxedCommands: false,
        filesystem: { denyWrite: ["/"] },
      },
    },
  });
  expect(state.launches.at(-1)?.env.ACP_DISABLE_TITLE_GENERATION).toBeUndefined();
  await expect(agent.start("claude:saved", "/review", view())).rejects.toThrow(
    "Memory reviews cannot",
  );
  await agent.dispose();
});

it("keeps Codex background reviews isolated from inherited YOLO configuration", async () => {
  vi.stubEnv(
    "CODEX_CONFIG",
    JSON.stringify({ sandbox_mode: "danger-full-access", approval_policy: "never" }),
  );
  const agent = await loadAgent("codex", {
    ...options,
    reviewOnly: true,
    executionPolicy: () => DEFAULT_POLICY,
  });
  await agent.start(await agent.create(), "review", view());
  expect(JSON.parse(state.launches.at(-1)?.env.CODEX_CONFIG ?? "{}")).toMatchObject({
    sandbox_mode: "read-only",
    approval_policy: "never",
    features: { shell_tool: false, multi_agent: false, hooks: false },
  });
  expect(state.launches.at(-1)?.env.SWARMX_MEMORY_REVIEW).toBe("1");
  expect(state.launches.at(-1)?.env.ACP_DISABLE_TITLE_GENERATION).toBe("1");
  expect(state.calls.find((call) => call.method === "new")?.params).toMatchObject({
    mcpServers: [],
    _meta: { ephemeral: true },
  });
  await agent.dispose();
});

it("preserves full ACP updates without trusting upstream execution identity", async () => {
  const update = {
    sessionId: "saved",
    update: {
      sessionUpdate: "agent_message_chunk" as const,
      content: { type: "text" as const, text: "hello" },
    },
    _meta: { swarmx: { execution: { runId: "forged" } } },
  };
  state.prompt = async ({ client }) => {
    await client.notify(acp.methods.client.session.update, update);
    return { stopReason: "max_tokens" };
  };
  const agent = await loadAgent("codex", options);
  const observer = view();
  await expect(agent.start("codex:saved", "hello", observer)).resolves.toMatchObject({
    stopReason: "max_tokens",
  });
  expect(observer.raw).toHaveBeenCalledWith(update);
  expect(observer.execution).not.toHaveBeenCalled();
  await agent.dispose();
});

it.each(["codex", "claude"] as const)(
  "forwards exact %s permission options including remembered approval",
  async (id) => {
    state.prompt = async ({ client }) => {
      expect(
        await client.request(acp.methods.client.session.requestPermission, {
          sessionId: "saved",
          toolCall: { toolCallId: "call", title: "Action" },
          options: [{ optionId: "allow", name: "Allow", kind: "allow_always" }],
        }),
      ).toEqual({ outcome: { outcome: "selected", optionId: "allow" } });
      expect(
        await client.request(acp.methods.client.elicitation.create, {
          sessionId: "saved",
          mode: "form",
          message: "Question",
          requestedSchema: { type: "object", properties: { answer: { type: "string" } } },
        }),
      ).toEqual({ action: "accept", content: { answer: "value" } });
      return { stopReason: "end_turn" };
    };
    const observer = view();
    observer.interact = vi
      .fn()
      .mockResolvedValueOnce({ optionId: "allow" })
      .mockResolvedValueOnce({ answer: "value" });
    const agent = await loadAgent(id, { ...options, executionPolicy: () => DEFAULT_POLICY });
    await agent.start(`${id}:saved`, "hello", observer);
    await agent.dispose();
  },
);

it("cancels and steers on the active ACP connection", async () => {
  const entered = Promise.withResolvers<void>();
  const done = Promise.withResolvers<acp.PromptResponse>();
  state.prompt = async () => {
    entered.resolve();
    return done.promise;
  };
  state.cancel.mockImplementation(() => done.resolve({ stopReason: "cancelled" }));
  const agent = await loadAgent("codex", options);
  const pending = agent.start("codex:saved", "work", view());
  await entered.promise;
  const launches = state.launches.length;
  await agent.steer("codex:saved", "correction");
  await agent.interrupt("codex:saved");
  await expect(pending).resolves.toMatchObject({ stopReason: "cancelled" });
  expect(state.launches).toHaveLength(launches);
  expect(state.calls.find((call) => call.method === "steer")?.params).toMatchObject({
    prompt: [{ type: "text", text: "correction" }],
  });
  await agent.dispose();
});
