import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import type { Options, Query, SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { loadAgent } from "../src/agent.js";
import { createClaude } from "../src/agents/claude.js";
import type { Observer } from "../src/agents/types.js";

type Fixture = {
  options: Options;
  messages: SDKUserMessage[];
  output: PassThrough;
  process: EventEmitter;
  native: Query;
  emit(message: object): void;
};
const mock = vi.hoisted(() => ({
  queries: [] as Fixture[],
  sessions: [] as { sessionId: string; cwd: string; summary: string; lastModified: number }[],
  history: [] as object[],
  respond: undefined as ((fixture: Fixture, input: SDKUserMessage) => void) | undefined,
  closeProcess: true,
  initialize: undefined as Promise<unknown> | undefined,
  mcpErrors: {} as Record<string, string>,
}));
vi.mock("node:child_process", () => ({ spawn: vi.fn(() => new EventEmitter()) }));
vi.mock("@anthropic-ai/claude-agent-sdk", () => ({
  listSessions: vi.fn(async () => mock.sessions),
  getSessionInfo: vi.fn(async (id: string) =>
    mock.sessions.find((session) => session.sessionId === id),
  ),
  getSessionMessages: vi.fn(async () => mock.history),
  resolveSettings: vi.fn(async () => ({
    model: "native",
    effortLevel: "high",
    permissions: { defaultMode: "plan" },
  })),
  filterEscalatingDefaultMode: vi.fn((settings) => settings),
  query: vi.fn(
    ({ prompt, options }: { prompt: AsyncIterable<SDKUserMessage>; options: Options }) => {
      const output = new PassThrough({ objectMode: true });
      const process = options.spawnClaudeCodeProcess?.({
        command: "claude",
        args: [],
        env: {},
        signal: new AbortController().signal,
      }) as unknown as EventEmitter;
      const native = Object.assign(output, {
        initializationResult: vi.fn(async () => {
          await mock.initialize;
          return {};
        }),
        supportedModels: vi.fn(async () => [
          {
            value: "native",
            resolvedModel: "native-resolved",
            displayName: "Native",
            description: "SDK model",
            supportedEffortLevels: ["low", "high"],
          },
        ]),
        setModel: vi.fn(async () => {}),
        setPermissionMode: vi.fn(async () => {}),
        applyFlagSettings: vi.fn(async () => {}),
        setMcpServers: vi.fn(async () => ({ errors: mock.mcpErrors, added: [], removed: [] })),
        close: vi.fn(() => {
          output.end();
          if (mock.closeProcess) process.emit("close");
        }),
      }) as unknown as Query;
      const fixture: Fixture = {
        options,
        output,
        process,
        native,
        messages: [],
        emit: (message) => {
          output.write(message);
        },
      };
      mock.queries.push(fixture);
      void (async () => {
        for await (const message of prompt) {
          fixture.messages.push(message);
          mock.respond?.(fixture, message);
        }
      })();
      return native;
    },
  ),
}));
const options = {
  cwd: "/workspace",
  mcp: { command: "node", args: ["/bridge.js"], env: {} },
};
const observer = (): Observer => ({
  executionId: "run",
  text: vi.fn(),
  tool: vi.fn(),
  raw: vi.fn(),
  activity: vi.fn(),
  interact: vi.fn(),
});
const result = (extra = {}) => ({
  type: "result",
  subtype: "success",
  is_error: false,
  result: "Answer",
  stop_reason: "end_turn",
  ...extra,
});
const usageResult = (total: number, extra = {}) =>
  result({
    total_cost_usd: total / 1000,
    modelUsage: {
      "reported-model": {
        inputTokens: total,
        outputTokens: total / 10,
        cacheReadInputTokens: total / 10,
        cacheCreationInputTokens: 0,
        thinkingTokens: total / 10,
        costUSD: total / 1000,
      },
    },
    ...extra,
  });
const idle = { type: "system", subtype: "session_state_changed", state: "idle" };
const answer = {
  type: "assistant",
  uuid: "answer-uuid",
  message: { id: "answer", content: [{ type: "text", text: "Answer" }] },
};
const agents: Awaited<ReturnType<typeof createClaude>>[] = [];
async function open(extra = {}) {
  const agent = await createClaude({ ...options, ...extra });
  agents.push(agent);
  return agent;
}
async function submitted(index = 0) {
  await vi.waitFor(() => expect(mock.queries[index]?.messages.length).toBeGreaterThan(0));
  return mock.queries[index] as Fixture;
}
beforeEach(() => {
  mock.queries.length = 0;
  mock.sessions = [
    { sessionId: "saved", cwd: options.cwd, summary: "Native title", lastModified: 1000 },
  ];
  mock.history = [];
  mock.closeProcess = true;
  mock.initialize = undefined;
  mock.mcpErrors = {};
  mock.respond = (fixture) => {
    fixture.emit(answer);
    fixture.emit(result());
    fixture.emit(idle);
  };
});
afterEach(async () => {
  mock.closeProcess = true;
  for (const fixture of mock.queries) fixture.process.emit("close");
  await Promise.all(agents.splice(0).map((agent) => agent.dispose()));
  vi.clearAllMocks();
});

it("uses native metadata and history without creating a writer or issuing a model request", async () => {
  const agent = await open();
  mock.sessions.push({
    sessionId: "foreign",
    cwd: "/elsewhere",
    summary: "Other",
    lastModified: 1,
  });
  expect(await agent.list()).toEqual([
    { sessionId: "saved", title: "Native title", updatedAt: "1970-01-01T00:00:01.000Z" },
  ]);
  const fresh = await agent.create();
  const view = observer();
  await agent.read(fresh, view);
  expect(view.text).not.toHaveBeenCalled();
  mock.history = [answer];
  await agent.read("saved", view);
  expect(view.text).toHaveBeenCalledWith("answer:0", "Answer", "assistant");
  await expect(agent.read("foreign", view)).rejects.toThrow("does not belong");
  expect(mock.queries).toHaveLength(0);
});

it("loads Claude directly through the SDK and keeps public session IDs scoped", async () => {
  const agent = await loadAgent("claude", options);
  agents.push(agent);
  expect((await agent.list())[0]?.sessionId).toBe("claude:saved");
  await expect(agent.start("claude:saved", "hello", observer())).resolves.toEqual({
    stopReason: "end_turn",
  });
  expect(mock.queries[0]?.options.resume).toBe("saved");
  expect(() => agent.read("codex:saved", observer())).toThrow("does not belong");
});

it.each(["assistant", "stream_event"])(
  "rejects a conflicting root Claude model in %s before projecting tools",
  async (type) => {
    mock.respond = (fixture) => {
      fixture.emit(
        type === "assistant"
          ? {
              ...answer,
              parent_tool_use_id: null,
              message: {
                model: "other",
                content: [{ type: "tool_use", id: "tool", name: "Bash", input: {} }],
              },
            }
          : {
              type,
              parent_tool_use_id: null,
              event: { type: "message_start", message: { id: "message", model: "other" } },
            },
      );
      fixture.emit(result());
      fixture.emit(idle);
    };
    const agent = await open();
    const sink = observer();
    await expect(agent.start("saved", "Run", sink, { model: "native" })).rejects.toThrow(
      'Claude model changed from "native" to "other"',
    );
    expect(sink.tool).not.toHaveBeenCalled();
    expect(mock.queries[0]?.native.close).toHaveBeenCalled();
  },
);

it("matches Claude model aliases using the SDK catalog and ignores native child models", async () => {
  const agent = await open();
  mock.respond = undefined;
  const sink = observer();
  const running = agent.start("saved", "Run", sink, { model: "native" });
  const fixture = await submitted();
  fixture.emit({
    ...answer,
    parent_tool_use_id: "child-tool",
    message: {
      ...answer.message,
      model: "child-model",
      usage: { input_tokens: 200, output_tokens: 20 },
    },
  });
  fixture.emit({
    ...answer,
    parent_tool_use_id: null,
    message: { ...answer.message, model: "native-resolved" },
  });
  fixture.emit(
    usageResult(3000, {
      modelUsage: {
        "native-resolved": { ...usageResult(1000).modelUsage["reported-model"], costUSD: 1 },
        "child-model": { ...usageResult(2000).modelUsage["reported-model"], costUSD: 2 },
      },
    }),
  );
  fixture.emit(idle);
  await expect(running).resolves.toEqual({ stopReason: "end_turn" });
  expect(sink.raw).toHaveBeenCalledWith(
    expect.objectContaining({ type: "result", total_cost_usd: 3 }),
    expect.objectContaining({ "swarmx.usage.cost_usd": 3 }),
  );
  expect(
    vi
      .mocked(sink.raw)
      .mock.calls.filter(([, attributes]) => attributes?.["swarmx.usage.cost_usd"] !== undefined),
  ).toHaveLength(1);
});

it("reserves empty sessions across Host restart and uses the native session ID on first dispatch", async () => {
  const first = await open();
  const id = await first.create();
  await first.dispose();
  const agent = await open();
  agent.restoreEmptySessions?.([id]);
  await agent.start(id, "first", observer());
  await agent.start(id, "second", observer());
  expect(mock.queries).toHaveLength(1);
  expect(mock.queries[0]?.options).toMatchObject({ sessionId: id });
  expect(mock.queries[0]?.options.resume).toBeUndefined();
  expect(mock.queries[0]?.messages.map((message) => message.message.content)).toEqual([
    "first",
    "second",
  ]);
});

it("records Claude response identity without treating cumulative query usage as run totals", async () => {
  const agent = await open();
  let turn = 0;
  mock.respond = (fixture) => {
    fixture.emit({
      type: "system",
      subtype: "init",
      model: "reported-model",
      permissionMode: "default",
      claude_code_version: "native-version",
    });
    fixture.emit({ ...answer, message: { ...answer.message, model: "reported-model" } });
    for (const total of [++turn * 100, 0, undefined])
      fixture.emit(
        result({
          total_cost_usd: total,
          modelUsage:
            total === undefined
              ? undefined
              : { "reported-model": { inputTokens: total, outputTokens: total, costUSD: total } },
        }),
      );
    fixture.emit(idle);
  };
  for (const text of ["First", "Second"]) {
    const sink = observer();
    await agent.start("saved", text, sink, { model: "reported-model" });
    expect(sink.raw).toHaveBeenCalledWith(
      expect.objectContaining({ type: "system", subtype: "init" }),
      expect.objectContaining({ "swarmx.harness.version": "native-version" }),
    );
    expect(sink.raw).toHaveBeenCalledWith(
      expect.objectContaining({ type: "assistant" }),
      expect.objectContaining({ "gen_ai.response.model": "reported-model" }),
    );
    for (const [, attributes] of vi.mocked(sink.raw).mock.calls)
      for (const key of [
        "gen_ai.provider.name",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "swarmx.usage.cost_usd",
      ])
        expect(attributes?.[key] ?? null).toBeNull();
  }
  expect(mock.queries).toHaveLength(1);
});

it("differences cumulative query usage across turns and counts queued results once", async () => {
  const agent = await open();
  let turn = 0;
  mock.respond = (fixture) => {
    for (const total of [++turn * 100, turn * 100, turn * 100 + 10]) {
      fixture.emit(usageResult(total, { uuid: `result-${total}` }));
    }
    fixture.emit(idle);
  };
  for (const [text, expected] of [
    ["First", 110],
    ["Second", 100],
  ] as const) {
    const sink = observer();
    await agent.start("saved", text, sink);
    const usage = vi
      .mocked(sink.raw)
      .mock.calls.findLast(([event]) => (event as { type: string }).type === "result")?.[1];
    expect(usage).toMatchObject({
      "gen_ai.usage.input_tokens": expected + expected / 10,
      "gen_ai.usage.output_tokens": expected / 10,
      "swarmx.usage.cached_input_tokens": expected / 10,
      "swarmx.usage.reasoning_output_tokens": expected / 10,
      "swarmx.usage.cost_source": "native-estimate",
      "swarmx.usage.coverage": "partial",
    });
    expect(usage?.["swarmx.usage.cost_usd"]).toBeCloseTo(expected / 1000);
  }
  expect(mock.queries).toHaveLength(1);
});

it("marks a cumulative reset unknown and resumes measurement from the new baseline", async () => {
  const agent = await open();
  for (const [total, expected] of [
    [100, 110],
    [20, null],
    [50, 33],
  ] as const) {
    mock.respond = (fixture) => {
      fixture.emit(usageResult(total));
      fixture.emit(idle);
    };
    const sink = observer();
    await agent.start("saved", "Next", sink);
    expect(sink.raw).toHaveBeenCalledWith(
      expect.objectContaining({ type: "result" }),
      expect.objectContaining({ "gen_ai.usage.input_tokens": expected }),
    );
  }
});

it.each([
  [0, null],
  [100, 110],
])(
  "retains actual failed-run usage but leaves zeroed error reports unknown (%s)",
  async (total, expected) => {
    const agent = await open();
    const sink = observer();
    mock.respond = (fixture) =>
      fixture.emit(usageResult(total, { is_error: true, result: "Failed" }));
    await expect(agent.start("saved", "Work", sink)).rejects.toThrow("Failed");
    expect(sink.raw).toHaveBeenCalledWith(
      expect.objectContaining({ type: "result" }),
      expect.objectContaining({ "gen_ai.usage.input_tokens": expected }),
    );
  },
);

it("starts a fresh accounting baseline when cancellation closes and then resumes a query", async () => {
  const agent = await open();
  mock.respond = (fixture) => fixture.emit(usageResult(100, { terminal_reason: "aborted_tools" }));
  const cancelled = observer();
  await expect(agent.start("saved", "Cancelled", cancelled)).resolves.toEqual({
    stopReason: "cancelled",
  });
  expect(cancelled.raw).toHaveBeenCalledWith(
    expect.objectContaining({ type: "result" }),
    expect.objectContaining({ "gen_ai.usage.input_tokens": 110 }),
  );
  mock.respond = (fixture) => {
    fixture.emit(usageResult(20));
    fixture.emit(idle);
  };
  const resumed = observer();
  await agent.start("saved", "Resume", resumed);
  expect(resumed.raw).toHaveBeenCalledWith(
    expect.objectContaining({ type: "result" }),
    expect.objectContaining({ "gen_ai.usage.input_tokens": 22 }),
  );
  expect(mock.queries).toHaveLength(2);
});

it("bounds each managed execution with a fresh query and preserves ordinary query reuse", async () => {
  const agent = await open();
  await agent.start("saved", "Ordinary first turn", observer());
  await agent.start("saved", "Ordinary follow-up", observer());
  expect(mock.queries).toHaveLength(1);
  expect(mock.queries[0]?.options.maxBudgetUsd).toBeUndefined();
  expect(mock.queries[0]?.native.close).not.toHaveBeenCalled();
  for (const budgetUsd of [0.2, 0.4]) {
    const sink = observer();
    await agent.start("saved", "Managed work", sink, { budgetUsd });
    const fixture = mock.queries.at(-1);
    expect(fixture?.options).toMatchObject({ resume: "saved", maxBudgetUsd: budgetUsd });
    expect(fixture?.native.close).toHaveBeenCalledTimes(1);
    expect(sink.raw).toHaveBeenCalledWith(
      { type: "swarmx.native.budget", budgetUsd, scope: "query" },
      { "swarmx.budget.limit_usd": budgetUsd, "swarmx.budget.enforcement": "claude.maxBudgetUsd" },
    );
  }
  expect(mock.queries).toHaveLength(3);
  expect(mock.queries[0]?.native.close).toHaveBeenCalledTimes(1);
  await agent.start("saved", "Ordinary turn again", observer());
  expect(mock.queries).toHaveLength(4);
  expect(mock.queries[3]?.options.maxBudgetUsd).toBeUndefined();
  expect(mock.queries[3]?.native.close).not.toHaveBeenCalled();
});

it("the installed SDK forwards the query cap to the actual native budget flag", async () => {
  const sdk = await vi.importActual<typeof import("@anthropic-ai/claude-agent-sdk")>(
    "@anthropic-ai/claude-agent-sdk",
  );
  const spawn = vi.fn<NonNullable<Options["spawnClaudeCodeProcess"]>>(() => {
    throw new Error("Fixture prevents native process dispatch");
  });
  expect(() =>
    sdk.query({
      prompt: new PassThrough({ objectMode: true }),
      options: { maxBudgetUsd: 0.25, spawnClaudeCodeProcess: spawn },
    }),
  ).toThrow("Fixture prevents native process dispatch");
  const args = spawn.mock.calls[0]?.[0].args;
  const index = args?.indexOf("--max-budget-usd");
  expect(index).toBeGreaterThanOrEqual(0);
  expect(args?.slice(index, (index ?? 0) + 2)).toEqual(["--max-budget-usd", "0.25"]);
});

it("keeps queued steering under one native query budget through idle", async () => {
  mock.respond = undefined;
  const agent = await open();
  const sink = observer();
  let completed = false;
  const work = agent.start("saved", "Initial work", sink, { budgetUsd: 0.3 }).then((outcome) => {
    completed = true;
    return outcome;
  });
  const fixture = await submitted();
  await agent.steer("saved", "Accepted follow-up");
  await vi.waitFor(() => expect(fixture.messages).toHaveLength(2));
  fixture.emit(usageResult(100));
  await new Promise((resolve) => setImmediate(resolve));
  expect(completed).toBe(false);
  expect(fixture.native.close).not.toHaveBeenCalled();
  fixture.emit(usageResult(200));
  fixture.emit(idle);
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  expect(mock.queries).toHaveLength(1);
  expect(fixture.options.maxBudgetUsd).toBe(0.3);
  expect(fixture.native.close).toHaveBeenCalledTimes(1);
  expect(sink.raw).toHaveBeenCalledWith(
    expect.objectContaining({ type: "result", total_cost_usd: 0.2 }),
    expect.objectContaining({ "swarmx.usage.cost_usd": 0.2 }),
  );
});

it("cancels budgeted initialization before any prompt or credential dispatch", async () => {
  const ready = Promise.withResolvers<void>();
  mock.initialize = ready.promise;
  const registerMcp = vi.fn();
  const agent = await open({ registerMcp });
  const work = agent.start("saved", "Never dispatch", observer(), { budgetUsd: 0.1 });
  await vi.waitFor(() => expect(mock.queries).toHaveLength(1));
  expect(mock.queries[0]?.options.maxBudgetUsd).toBe(0.1);
  await agent.interrupt("saved");
  ready.resolve();
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
  expect(mock.queries[0]?.messages).toEqual([]);
  expect(registerMcp).not.toHaveBeenCalled();
});

it("cancels while closing a retained query without opening or dispatching a budgeted replacement", async () => {
  const agent = await open();
  await agent.start("saved", "Previous ordinary task", observer());
  const previous = mock.queries[0];
  expect(previous).toBeDefined();
  mock.closeProcess = false;
  const work = agent.start("saved", "Never dispatch", observer(), { budgetUsd: 0.1 });
  await vi.waitFor(() => expect(previous?.native.close).toHaveBeenCalledTimes(1));
  const stop = agent.interrupt("saved");
  previous?.process.emit("close");
  await stop;
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
  expect(mock.queries).toHaveLength(1);
  expect(previous?.messages).toHaveLength(1);
});

it("preserves native budget-exhaustion errors and incurred usage without accepting the work", async () => {
  const agent = await open();
  const sink = observer();
  mock.respond = (fixture) =>
    fixture.emit(
      usageResult(200, {
        subtype: "error_max_budget_usd",
        is_error: true,
        errors: ["Budget exceeded"],
        terminal_reason: "budget_exhausted",
      }),
    );
  await expect(agent.start("saved", "Exhaust budget", sink, { budgetUsd: 0.1 })).rejects.toThrow(
    "Budget exceeded",
  );
  expect(mock.queries[0]?.options.maxBudgetUsd).toBe(0.1);
  expect(mock.queries[0]?.native.close).toHaveBeenCalled();
  expect(sink.raw).toHaveBeenCalledWith(
    expect.objectContaining({ type: "result", subtype: "error_max_budget_usd" }),
    expect.objectContaining({ "swarmx.usage.cost_usd": 0.2 }),
  );
});

it("waits for idle after result, retains the runtime for native titles, and detaches the completed observer", async () => {
  mock.respond = undefined;
  const agent = await open();
  const view = observer();
  let finished = false;
  const work = agent.start("saved", "hello", view).then((value) => {
    finished = true;
    return value;
  });
  const fixture = await submitted();
  fixture.emit(result());
  await new Promise((resolve) => setImmediate(resolve));
  expect(finished).toBe(false);
  fixture.emit(idle);
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  expect(fixture.native.close).not.toHaveBeenCalled();
  const count = vi.mocked(view.raw).mock.calls.length;
  fixture.emit({ type: "system", subtype: "native_title_finished", title: "Later title" });
  const session = mock.sessions[0];
  if (!session) throw new Error("Missing session fixture");
  session.summary = "Later title";
  await new Promise((resolve) => setImmediate(resolve));
  expect(view.raw).toHaveBeenCalledTimes(count);
  expect((await agent.list())[0]?.title).toBe("Later title");
  expect(fixture.messages).toHaveLength(1);
});

it("rotates per-run MCP credentials without sharing the Host credential or overriding native tools and hooks", async () => {
  const endpoints: {
    token: string;
    bind: ReturnType<typeof vi.fn>;
    dispose: ReturnType<typeof vi.fn>;
  }[] = [];
  const agent = await open({
    registerMcp: (token: string) => {
      const endpoint = { token, bind: vi.fn(), dispose: vi.fn() };
      endpoints.push(endpoint);
      return endpoint;
    },
  });
  await agent.start("saved", "one", observer(), {
    mode: "plan",
    model: "native",
    effort: "high",
    instructions: "Memory",
  });
  await agent.start("saved", "two", { ...observer(), executionId: "second-run" });
  const fixture = mock.queries[0];
  if (!fixture) throw new Error("Missing query fixture");
  expect(fixture.options).toMatchObject({
    resume: "saved",
    settingSources: ["user", "project", "local"],
    systemPrompt: {
      type: "preset",
      preset: "claude_code",
      append: expect.stringContaining("Memory"),
    },
  });
  expect(fixture.options.tools).toBeUndefined();
  expect(fixture.options.settings).toBeUndefined();
  expect(fixture.options.permissionMode).toBeUndefined();
  expect(fixture.native.setPermissionMode).toHaveBeenCalledWith("plan");
  expect(fixture.native.setModel).toHaveBeenCalledWith("native");
  expect(fixture.native.applyFlagSettings).toHaveBeenCalledWith({ effortLevel: "high" });
  expect(endpoints).toHaveLength(2);
  expect(endpoints[0]?.token).not.toBe(endpoints[1]?.token);
  for (const [index, endpoint] of endpoints.entries()) {
    expect(endpoint.bind).toHaveBeenCalledWith("claude:saved", index === 0 ? "run" : "second-run");
    expect(endpoint.dispose).toHaveBeenCalledOnce();
    expect(fixture.native.setMcpServers).toHaveBeenNthCalledWith(index + 1, {
      swarmx: {
        type: "stdio",
        command: "node",
        args: ["/bridge.js"],
        env: { SWARMX_MCP_TOKEN: endpoint.token },
      },
    });
  }
});

it("projects streamed reasoning, text and native tool failures without duplicating final blocks", async () => {
  mock.respond = (fixture) => {
    fixture.emit({
      type: "stream_event",
      parent_tool_use_id: null,
      event: { type: "message_start", message: { id: "answer" } },
    });
    fixture.emit({
      type: "stream_event",
      parent_tool_use_id: null,
      event: {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "Answer" },
      },
    });
    fixture.emit(answer);
    fixture.emit({
      type: "assistant",
      uuid: "tool",
      message: {
        id: "tool-message",
        content: [
          { type: "thinking", thinking: "Reason" },
          { type: "tool_use", id: "call", name: "Bash", input: { command: "false" } },
        ],
      },
    });
    fixture.emit({
      type: "user",
      uuid: "output",
      message: {
        content: [{ type: "tool_result", tool_use_id: "call", content: "Failed", is_error: true }],
      },
      tool_use_result: { exitCode: 1 },
    });
    fixture.emit(result());
    fixture.emit(idle);
  };
  const agent = await open();
  const view = observer();
  await agent.start("saved", "work", view);
  expect(view.text).toHaveBeenCalledTimes(2);
  expect(view.text).toHaveBeenCalledWith("tool-message:0", "Reason", "reasoning");
  expect(view.tool).toHaveBeenLastCalledWith("call", "Bash", { command: "false" }, { exitCode: 1 });
  expect(view.activity).toHaveBeenLastCalledWith({
    type: "tool",
    toolCallId: "call",
    status: "failed",
  });
});

it.each([
  [result({ is_error: true, result: "Native API failure" }), "Native API failure"],
  [
    { type: "result", subtype: "error_max_budget_usd", errors: ["Budget exceeded"] },
    "Budget exceeded",
  ],
])("does not report native error results as success", async (message, error) => {
  mock.respond = (fixture) => {
    fixture.emit(message as object);
  };
  const agent = await open();
  await expect(agent.start("saved", "work", observer())).rejects.toThrow(error as string);
  expect(mock.queries[0]?.native.close).toHaveBeenCalled();
});

it("rejects output EOF even after a result if native idle never arrives", async () => {
  mock.respond = (fixture) => {
    fixture.emit(result());
    fixture.output.end();
  };
  const agent = await open();
  await expect(agent.start("saved", "work", observer())).rejects.toThrow("output ended");
});

it.each([
  [result({ stop_reason: "max_tokens" }), "max_tokens"],
  [{ type: "result", subtype: "error_max_turns" }, "max_turn_requests"],
  [result({ terminal_reason: "aborted_tools" }), "cancelled"],
])("settles native terminal stops without requiring a later idle", async (message, stopReason) => {
  mock.respond = (fixture) => {
    fixture.emit(message as object);
  };
  const agent = await open();
  await expect(agent.start("saved", "work", observer())).resolves.toEqual({ stopReason });
  expect(mock.queries[0]?.native.close).toHaveBeenCalled();
});

it("waits for owned process exit when cancelling and leaves concurrent sessions alive", async () => {
  mock.respond = undefined;
  const agent = await open();
  const first = agent.start("saved", "one", observer());
  const firstFixture = await submitted();
  const secondId = await agent.create();
  const second = agent.start(secondId, "two", observer());
  const secondFixture = await submitted(1);
  await agent.steer("saved", "queued guidance");
  mock.closeProcess = false;
  let stopped = false;
  const stop = agent.interrupt("saved").then(() => {
    stopped = true;
  });
  await new Promise((resolve) => setImmediate(resolve));
  expect(stopped).toBe(false);
  expect(secondFixture.native.close).not.toHaveBeenCalled();
  firstFixture.process.emit("close");
  await stop;
  await expect(first).resolves.toEqual({ stopReason: "cancelled" });
  await expect(agent.steer("saved", "late")).rejects.toThrow("No running");
  secondFixture.emit(result());
  secondFixture.emit(idle);
  await expect(second).resolves.toEqual({ stopReason: "end_turn" });
});

it("cancels preparation without dispatching input or leaking a run credential", async () => {
  const ready = Promise.withResolvers<void>();
  mock.initialize = ready.promise;
  const registerMcp = vi.fn();
  const agent = await open({ registerMcp });
  const work = agent.start("saved", "never run", observer());
  await vi.waitFor(() => expect(mock.queries).toHaveLength(1));
  await agent.interrupt("saved");
  ready.resolve();
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
  expect(registerMcp).not.toHaveBeenCalled();
  expect(mock.queries[0]?.messages).toEqual([]);
});

it("preserves exact native remembered permissions and validates offered choices", async () => {
  mock.respond = undefined;
  const agent = await open();
  const view = observer();
  vi.mocked(view.interact).mockResolvedValue({ optionId: "remember" });
  const work = agent.start("saved", "work", view);
  const fixture = await submitted();
  const suggestions = [
    {
      type: "addRules" as const,
      rules: [{ toolName: "Bash", ruleContent: "git status" }],
      behavior: "allow" as const,
      destination: "session" as const,
    },
  ];
  const context = {
    signal: new AbortController().signal,
    requestId: "permission-request",
    toolUseID: "tool-call",
    suggestions,
    title: "Native prompt",
  };
  const input = { command: "git status" };
  await expect(fixture.options.canUseTool?.("Bash", input, context)).resolves.toEqual({
    behavior: "allow",
    updatedInput: input,
    updatedPermissions: suggestions,
    toolUseID: "tool-call",
  });
  expect(view.interact).toHaveBeenCalledWith(
    expect.objectContaining({
      id: "permission-request",
      title: "Native prompt",
      approval: expect.objectContaining({ toolId: "tool-call" }),
    }),
    context.signal,
  );
  vi.mocked(view.interact).mockResolvedValue({ optionId: "invented" });
  await expect(fixture.options.canUseTool?.("Bash", input, context)).rejects.toThrow(
    "Unknown native approval",
  );
  fixture.emit(result());
  fixture.emit(idle);
  await work;
});

it("routes questions and MCP form elicitation through Host interaction", async () => {
  mock.respond = undefined;
  const agent = await open();
  const view = observer();
  const work = agent.start("saved", "work", view);
  const fixture = await submitted();
  const context = {
    signal: new AbortController().signal,
    requestId: "question",
    toolUseID: "call",
  };
  const input = {
    questions: [
      { question: "Which?", options: [{ label: "A" }, { label: "B" }], multiSelect: true },
    ],
  };
  vi.mocked(view.interact).mockResolvedValue({ answer_0: ["A", "B"], other_0: "Custom" });
  await expect(fixture.options.canUseTool?.("AskUserQuestion", input, context)).resolves.toEqual({
    behavior: "allow",
    updatedInput: { ...input, answers: { "Which?": "A, B, Custom" } },
  });
  vi.mocked(view.interact).mockResolvedValue({ value: "Answer" });
  await expect(
    fixture.options.onElicitation?.(
      {
        serverName: "mcp",
        mode: "form",
        message: "Form",
        requestedSchema: { type: "object", properties: { value: { type: "string" } } },
      },
      context,
    ),
  ).resolves.toEqual({ action: "accept", content: { value: "Answer" } });
  fixture.emit(result());
  fixture.emit(idle);
  await work;
});

it("discovers catalogs without input or a run credential and does not invent persisted settings", async () => {
  const registerMcp = vi.fn();
  const agent = await open({ registerMcp });
  const catalog = await agent.models();
  expect(catalog.current).toEqual({ model: "native", effort: "high", mode: "plan" });
  expect(catalog.models[0]?.efforts).toEqual([
    { id: "low", name: "low" },
    { id: "high", name: "high" },
  ]);
  expect((await agent.models("saved")).current).toEqual({});
  expect(registerMcp).not.toHaveBeenCalled();
  for (const fixture of mock.queries) {
    expect(fixture.options).toMatchObject({
      persistSession: false,
      tools: [],
      strictMcpConfig: true,
    });
    expect(fixture.messages).toEqual([]);
    expect(fixture.native.close).toHaveBeenCalled();
  }
});

it("keeps memory reviews tool-free and rejects permission changes and native commands", async () => {
  const registerMcp = vi.fn();
  const agent = await open({ reviewOnly: true, registerMcp });
  const id = await agent.create();
  await agent.start(id, "Review", observer());
  expect(mock.queries[0]?.options).toMatchObject({
    tools: [],
    strictMcpConfig: true,
    persistSession: false,
    permissionMode: "dontAsk",
    settings: { disableAllHooks: true },
    sandbox: { failIfUnavailable: true, filesystem: { denyWrite: ["/"] } },
  });
  expect(mock.queries[0]?.native.close).toHaveBeenCalled();
  expect(registerMcp).not.toHaveBeenCalled();
  await expect(agent.start(id, "/review", observer())).rejects.toThrow("Memory reviews cannot");
  await expect(
    agent.start(id, "Review", observer(), { mode: "bypassPermissions" }),
  ).rejects.toThrow("Memory reviews cannot");
});

it("fails before input and revokes the credential if native MCP setup fails", async () => {
  mock.mcpErrors = { swarmx: "Connection refused" };
  const endpoint = { bind: vi.fn(), dispose: vi.fn() };
  const agent = await open({ registerMcp: () => endpoint });
  await expect(agent.start("saved", "work", observer())).rejects.toThrow("Connection refused");
  expect(mock.queries[0]?.messages).toEqual([]);
  expect(endpoint.dispose).toHaveBeenCalledOnce();
});
