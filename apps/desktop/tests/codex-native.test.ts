import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { loadAgent, scopeSessions } from "../src/agent.js";
import { createCodex } from "../src/agents/codex.js";
import type { Observer } from "../src/agents/types.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { recordedAgent } from "../src/host/recorded-agent.js";

type Params = {
  threadId?: string;
  cursor?: string | null;
  includeTurns?: boolean;
  [key: string]: unknown;
};
type Peer = {
  signal: AbortSignal;
  receive(message: { method: string; params: object; id?: string }): Promise<unknown>;
  failed(error: Error): void;
  request: ReturnType<typeof vi.fn<(method: string, params: Params) => Promise<unknown>>>;
  notify: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn<() => Promise<void>>>;
};
type Handler = (peer: Peer, method: string, params: Params) => unknown;
const mock = vi.hoisted(() => ({
  peers: [] as Peer[],
  handler: (() => {
    throw new Error("Missing native fixture");
  }) as Handler,
}));
vi.mock("../src/agents/rpc-process.js", () => ({
  rpcProcess: vi.fn((_command, _args, _cwd, receive, failed) => {
    const lifetime = new AbortController();
    const peer: Peer = {
      signal: lifetime.signal,
      receive,
      failed,
      request: vi.fn(async (method, params) => mock.handler(peer, method, params)),
      notify: vi.fn(),
      dispose: vi.fn(async () => {
        lifetime.abort();
      }),
    };
    mock.peers.push(peer);
    return peer;
  }),
}));
function currentPeer() {
  const peer = mock.peers[0];
  if (!peer) throw new Error("Native fixture has no process.");
  return peer;
}
const cwd = "/workspace";
const turn = (status = "completed") => ({
  id: "turn",
  status,
  error: null,
  items: [],
  itemsView: "full",
  startedAt: 100,
  durationMs: 1500,
});
const thread = (id = "saved") => ({
  id,
  cwd,
  historyMode: "legacy",
  model: "model-b",
  reasoningEffort: "high",
  cliVersion: "native-version",
  turns: [turn()],
  name: "Native title",
  updatedAt: 100,
});
const agentMessage = { type: "agentMessage", id: "answer", text: "Answer", phase: "final_answer" };
const model = (id: string) => ({
  model: id,
  displayName: id,
  description: id,
  defaultReasoningEffort: "high",
  supportedReasoningEfforts: [{ reasoningEffort: "high", description: "High" }],
});
const options = {
  cwd,
  mcp: { command: "node", args: ["/bridge.js"], env: {} },
};
const tokenUsage = (inputTokens = 100, outputTokens = 20) => ({
  inputTokens,
  outputTokens,
  cachedInputTokens: inputTokens / 5,
  cacheWriteInputTokens: inputTokens / 10,
  reasoningOutputTokens: outputTokens / 2,
  totalTokens: inputTokens + outputTokens,
});
const responseUsage = (
  responseId: string,
  usage: unknown,
  turnId = "turn",
  threadId = "saved",
) => ({
  method: "rawResponse/completed",
  params: {
    threadId,
    turnId,
    responseId,
    usage,
    usageMetadata: { amount: "123", metadata: { unit: "unspecified" } },
  },
});
const observer = (): Observer => ({
  executionId: "execution",
  text: vi.fn(),
  tool: vi.fn(),
  raw: vi.fn(),
  activity: vi.fn(),
  interact: vi.fn(),
});
const agents: Awaited<ReturnType<typeof createCodex>>[] = [];
async function open(extra = {}) {
  const agent = await createCodex({ ...options, ...extra });
  agents.push(agent);
  return agent;
}

it("stops Codex when a selected model is rerouted before allowing another tool", async () => {
  const original = mock.handler;
  mock.handler = async (peer, method, params) => {
    if (method !== "turn/start") return original(peer, method, params);
    queueMicrotask(async () => {
      await peer.receive({
        method: "turn/started",
        params: { threadId: "saved", turn: turn("inProgress") },
      });
      try {
        await peer.receive({
          method: "model/rerouted",
          params: {
            threadId: "saved",
            turnId: "turn",
            fromModel: "model-b",
            toModel: "other",
            reason: "highRiskCyberActivity",
          },
        });
      } catch (error) {
        peer.failed(error as Error);
      }
      if (!peer.signal.aborted)
        await peer.receive({
          method: "item/commandExecution/requestApproval",
          id: "tool",
          params: { threadId: "saved", turnId: "turn", itemId: "tool", command: "must not run" },
        });
      if (!peer.signal.aborted)
        await peer.receive({
          method: "turn/completed",
          params: { threadId: "saved", turn: turn() },
        });
    });
    return { turn: turn("inProgress") };
  };
  const agent = await open();
  const sink = observer();
  await expect(agent.start("saved", "Run", sink, { model: "model-b" })).rejects.toThrow(
    'Codex model changed from "model-b" to "other"',
  );
  expect(sink.interact).not.toHaveBeenCalled();
  expect(currentPeer().dispose).toHaveBeenCalled();
});

beforeEach(() => {
  mock.peers.length = 0;
  mock.handler = async (peer, method, params) => {
    switch (method) {
      case "initialize":
        return { userAgent: "codex native" };
      case "config/read":
        return {
          config: {
            model: "model-a",
            model_reasoning_effort: "low",
            default_permissions: ":workspace",
            mcp_servers: { ambient: { url: "http://ambient/mcp" } },
          },
        };
      case "model/list":
        return {
          data: [model(params.cursor ? "model-b" : "model-a")],
          nextCursor: params.cursor ? null : "more",
        };
      case "permissionProfile/list":
        return {
          data: [
            { id: ":workspace", description: "Workspace", allowed: true },
            { id: "restricted", description: null, allowed: false },
          ],
          nextCursor: null,
        };
      case "thread/list":
        return { data: [thread()], nextCursor: null };
      case "thread/read":
        return {
          thread: {
            ...thread(params.threadId),
            turns: params.includeTurns ? [{ ...turn(), items: [agentMessage] }] : [],
          },
        };
      case "thread/start":
        return { thread: thread("fresh"), activePermissionProfile: { id: ":workspace" } };
      case "thread/resume":
        return { thread: thread(params.threadId), activePermissionProfile: { id: ":workspace" } };
      case "turn/start":
        queueMicrotask(async () => {
          await peer.receive({
            method: "turn/started",
            params: { threadId: params.threadId, turn: turn("inProgress") },
          });
          await peer.receive({
            method: "item/started",
            params: { threadId: params.threadId, turnId: "turn", item: agentMessage },
          });
          await peer.receive({
            method: "item/agentMessage/delta",
            params: {
              threadId: params.threadId,
              turnId: "turn",
              itemId: "answer",
              delta: "Answer",
            },
          });
          await peer.receive({
            method: "item/completed",
            params: { threadId: params.threadId, turnId: "turn", item: agentMessage },
          });
          await peer.receive({
            method: "turn/completed",
            params: { threadId: params.threadId, turn: turn() },
          });
        });
        return { turn: turn("inProgress") };
      default:
        throw new Error(`Unexpected native method: ${method}`);
    }
  };
});
afterEach(async () => {
  await Promise.all(agents.splice(0).map((agent) => agent.dispose()));
  vi.unstubAllEnvs();
});

it("meters exact Codex response usage once per turn without summing cumulative notifications or token subsets", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-codex-usage-"));
  const journal = new ExecutionJournal(root, cwd);
  const native = await open();
  const agent = recordedAgent(journal, "codex", native);
  const first = responseUsage("response-one", tokenUsage());
  const next = responseUsage("response-two", tokenUsage(50, 10));
  const normal = mock.handler;
  let execution = 0;
  mock.handler = async (peer, method, params) => {
    if (method !== "turn/start") return normal(peer, method, params);
    await peer.receive({
      method: "turn/started",
      params: { threadId: "saved", turn: turn("inProgress") },
    });
    await peer.receive(responseUsage("earlier-turn", tokenUsage(900), "previous"));
    await peer.receive(responseUsage("foreign-thread", tokenUsage(900), "turn", "foreign"));
    if (++execution === 1) {
      await peer.receive(first);
      for (let i = 0; i < 2; i++) {
        await peer.receive({
          method: "thread/tokenUsage/updated",
          params: {
            threadId: "saved",
            turnId: "turn",
            tokenUsage: { total: tokenUsage(900), last: tokenUsage(), modelContextWindow: 200_000 },
          },
        });
        await peer.receive(first);
      }
      await peer.receive(next);
    }
    await peer.receive({ method: "turn/completed", params: { threadId: "saved", turn: turn() } });
    return { turn: turn("inProgress") };
  };
  try {
    await agent.start("saved", "First", observer(), { model: "requested-model" });
    await agent.start("saved", "Next", observer());
    const records = journal.read({}).events;
    const sources = records
      .filter(({ event }) => event.type === EventType.RUN_STARTED)
      .map(({ id }) => `urn:swarmx:execution:${id}`);
    const evidence = journal.evidence(sources);
    expect(evidence.runs[0]).toMatchObject({
      inputTokens: 150,
      outputTokens: 30,
      cachedInputTokens: 45,
      reasoningOutputTokens: 15,
      costUsd: null,
      costSource: "unknown",
      usageCoverage: "partial",
      requestedModel: "requested-model",
    });
    expect(evidence.runs[1]).toMatchObject({
      inputTokens: null,
      outputTokens: null,
      costUsd: null,
      usageCoverage: "unknown",
    });
    expect(
      records.filter(
        ({ event }) =>
          event.type === EventType.RAW && JSON.stringify(event.event) === JSON.stringify(first),
      ),
    ).toHaveLength(3);
    expect(evidence.statistics.usage).toEqual({
      sampleCount: 1,
      inputTokens: 150,
      outputTokens: 30,
    });
  } finally {
    await agent.dispose();
    journal.close();
    await rm(root, { recursive: true, force: true });
  }
});

it.each(["failed", "interrupted"])(
  "retains observed Codex usage when the native turn is %s",
  async (status) => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-codex-terminal-usage-"));
    const journal = new ExecutionJournal(root, cwd);
    const native = await open();
    const agent = recordedAgent(journal, "codex", native);
    const normal = mock.handler;
    const dispatched = Promise.withResolvers<Peer>();
    mock.handler = async (peer, method, params) => {
      if (method === "turn/start") {
        await peer.receive({
          method: "turn/started",
          params: { threadId: "saved", turn: turn("inProgress") },
        });
        await peer.receive(responseUsage("billed-response", tokenUsage()));
        dispatched.resolve(peer);
        return { turn: turn("inProgress") };
      }
      if (method === "turn/interrupt") return {};
      return normal(peer, method, params);
    };
    try {
      const run = agent.start("saved", "Run", observer());
      const peer = await dispatched.promise;
      if (status === "interrupted") await agent.interrupt("saved");
      const settled =
        status === "failed"
          ? expect(run).rejects.toThrow("Native failure")
          : expect(run).resolves.toEqual({ stopReason: "cancelled" });
      await peer.receive({
        method: "turn/completed",
        params: {
          threadId: "saved",
          turn: {
            ...turn(status),
            error: status === "failed" ? { message: "Native failure" } : null,
          },
        },
      });
      await settled;
      const started = journal
        .read({})
        .events.find(({ event }) => event.type === EventType.RUN_STARTED);
      expect(journal.evidence([`urn:swarmx:execution:${started?.id}`]).runs[0]).toMatchObject({
        outcome: status === "failed" ? "error" : "cancelled",
        inputTokens: 100,
        outputTokens: 20,
        costUsd: null,
        costSource: "unknown",
        usageCoverage: "partial",
      });
    } finally {
      await agent.dispose();
      journal.close();
      await rm(root, { recursive: true, force: true });
    }
  },
);

it.each([
  ["missing", null],
  ["negative", { ...tokenUsage(), inputTokens: -1 }],
  ["inconsistent subset", { ...tokenUsage(), reasoningOutputTokens: 21 }],
  ["conflicting duplicate", tokenUsage(200)],
])("keeps Codex usage unknown for a %s report", async (_reason, usage) => {
  const normal = mock.handler;
  const unknown = responseUsage("response", usage);
  mock.handler = async (peer, method, params) => {
    if (method !== "turn/start") return normal(peer, method, params);
    await peer.receive({
      method: "turn/started",
      params: { threadId: "saved", turn: turn("inProgress") },
    });
    await peer.receive(responseUsage("response", tokenUsage()));
    await peer.receive(unknown);
    await peer.receive({ method: "turn/completed", params: { threadId: "saved", turn: turn() } });
    return { turn: turn("inProgress") };
  };
  const agent = await open();
  const output = observer();
  await agent.start("saved", "Run", output);
  expect(output.raw).toHaveBeenCalledWith(
    unknown,
    expect.objectContaining({
      "gen_ai.usage.input_tokens": null,
      "gen_ai.usage.output_tokens": null,
      "swarmx.usage.coverage": "unknown",
      "swarmx.usage.cost_usd": null,
    }),
  );
});

it("fills a missing Codex response usage report without adding the same response twice", async () => {
  const normal = mock.handler;
  mock.handler = async (peer, method, params) => {
    if (method !== "turn/start") return normal(peer, method, params);
    await peer.receive({
      method: "turn/started",
      params: { threadId: "saved", turn: turn("inProgress") },
    });
    await peer.receive(responseUsage("response", null));
    await peer.receive(responseUsage("response", tokenUsage()));
    await peer.receive(responseUsage("response", tokenUsage()));
    await peer.receive({ method: "turn/completed", params: { threadId: "saved", turn: turn() } });
    return { turn: turn("inProgress") };
  };
  const agent = await open();
  const output = observer();
  await agent.start("saved", "Run", output);
  const reports = vi
    .mocked(output.raw)
    .mock.calls.filter(
      ([, attributes]) => attributes?.["swarmx.usage.scope"] === "native-responses",
    );
  expect(reports.map(([, attributes]) => attributes?.["gen_ai.usage.input_tokens"])).toEqual([
    null,
    100,
    100,
  ]);
});

it("reads native history and catalog without acquiring a writer or MCP credential", async () => {
  const registerMcp = vi.fn();
  const agent = await open({ registerMcp });
  const output = observer();
  await agent.read("saved", output);
  const catalog = await agent.models("saved");
  expect(catalog.current).toMatchObject({ model: "model-b", effort: "high" });
  expect(catalog.models.map((model) => model.id)).toEqual(["model-a", "model-b"]);
  expect(catalog.modes?.map((mode) => mode.id)).toEqual([":workspace", "read-only", "agent"]);
  expect(output.text).toHaveBeenCalledWith("answer", "Answer", "assistant");
  expect(output.activity).toHaveBeenCalledWith(
    expect.objectContaining({
      type: "message",
      messageId: "answer",
      phase: "final_answer",
      turnId: "turn",
      durationMs: 1500,
    }),
  );
  expect(registerMcp).not.toHaveBeenCalled();
  expect(
    mock.peers.flatMap((peer) => peer.request.mock.calls).map(([method]) => method),
  ).not.toEqual(expect.arrayContaining(["thread/resume", "thread/start", "turn/start"]));
});

it("loads the native Codex entry and rejects foreign session IDs before dispatch", async () => {
  const agent = await loadAgent("codex", options);
  agents.push(agent);
  expect(await agent.list()).toEqual([
    { sessionId: "codex:saved", title: "Native title", updatedAt: new Date(100_000).toISOString() },
  ]);
  expect(() => agent.start("claude:saved", "Hello", observer())).toThrow("does not belong");
  await expect(agent.start("codex:saved", "Hello", observer())).resolves.toEqual({
    stopReason: "end_turn",
  });
});

it("keeps inherited configuration and native prompt text intact", async () => {
  vi.stubEnv(
    "CODEX_CONFIG",
    JSON.stringify({ sandbox_mode: "read-only", approval_policy: "untrusted" }),
  );
  const agent = await open();
  await agent.start("saved", "/review", observer());
  expect(currentPeer().request).toHaveBeenCalledWith(
    "thread/resume",
    expect.objectContaining({
      config: expect.objectContaining({ sandbox_mode: "read-only", approval_policy: "untrusted" }),
    }),
  );
  expect(currentPeer().request).toHaveBeenCalledWith(
    "turn/start",
    expect.objectContaining({ input: [{ type: "text", text: "/review", text_elements: [] }] }),
  );
});

it("keeps an empty session alive until its first turn, preserves phases, and returns native completion", async () => {
  const agent = await open();
  const id = await agent.create();
  const peer = mock.peers.at(-1);
  expect(peer.dispose).not.toHaveBeenCalled();
  const output = observer();
  await expect(
    agent.start(id, "Hello", output, { model: "model-b", effort: "high", mode: ":workspace" }),
  ).resolves.toEqual({ stopReason: "end_turn" });
  expect(peer.request).not.toHaveBeenCalledWith("thread/resume", expect.anything());
  expect(peer.request).toHaveBeenCalledWith(
    "turn/start",
    expect.objectContaining({ model: "model-b", effort: "high", permissions: ":workspace" }),
  );
  expect(output.text).toHaveBeenCalledTimes(1);
  expect(output.activity).toHaveBeenCalledWith(
    expect.objectContaining({ messageId: "answer", phase: "final_answer" }),
  );
  expect(output.activity).toHaveBeenCalledWith(
    expect.objectContaining({ turnId: "turn", durationMs: 1500 }),
  );
  expect(peer.dispose).toHaveBeenCalledOnce();
});

it("keeps the empty-thread runtime when the first turn is rejected before its rollout", async () => {
  let reject = true;
  const normal = mock.handler;
  mock.handler = async (peer, method, params) => {
    if (method === "turn/start" && reject) {
      reject = false;
      throw new Error("Rejected before rollout");
    }
    return normal(peer, method, params);
  };
  const agent = await open();
  const id = await agent.create();
  const peer = mock.peers.at(-1);
  if (!peer) throw new Error("Native fixture has no process.");
  await expect(agent.start(id, "first", observer())).rejects.toThrow("Rejected before rollout");
  expect(peer.dispose).not.toHaveBeenCalled();
  await expect(agent.start(id, "retry", observer())).resolves.toEqual({ stopReason: "end_turn" });
  expect(mock.peers).toHaveLength(1);
  expect(peer.request).not.toHaveBeenCalledWith("thread/resume", expect.anything());
  expect(peer.request).toHaveBeenCalledWith("turn/start", expect.anything());
});

it.each(["before the first prompt", "during the first prompt"])(
  "evicts a failed empty-thread runtime %s and propagates native resume failure",
  async (when) => {
    const endpoint = { bind: vi.fn(), dispose: vi.fn() };
    const agent = await open({ registerMcp: () => endpoint });
    const id = await agent.create();
    const peer = currentPeer();
    const error = new Error("Agent exited (7).");
    const normal = mock.handler;
    mock.handler = (runtime, method, params) => {
      if (method === "turn/start" && runtime === peer) {
        runtime.failed(error);
        throw error;
      }
      if (method === "thread/resume") throw new Error("No persisted rollout");
      return normal(runtime, method, params);
    };

    if (when === "before the first prompt") peer.failed(error);
    else await expect(agent.start(id, "first", observer())).rejects.toThrow("Agent exited (7).");

    expect(peer.dispose).toHaveBeenCalledOnce();
    expect(endpoint.dispose).toHaveBeenCalledOnce();
    peer.request.mockClear();
    await expect(agent.start(id, "retry", observer())).rejects.toThrow("No persisted rollout");
    expect(peer.request).not.toHaveBeenCalled();
    expect(mock.peers).toHaveLength(2);
    expect(mock.peers[1]?.request).toHaveBeenCalledWith(
      "thread/resume",
      expect.objectContaining({ threadId: id }),
    );
    await agent.dispose();
    expect(peer.dispose).toHaveBeenCalledOnce();
  },
);

it("settles controls after a rejected first prompt without affecting a retry", async () => {
  const normal = mock.handler;
  const submitted = Promise.withResolvers<void>();
  const firstTurn = Promise.withResolvers<unknown>();
  mock.handler = (peer, method, params) => {
    if (method === "turn/start") {
      submitted.resolve();
      return firstTurn.promise;
    }
    return normal(peer, method, params);
  };
  const agent = await open();
  const id = await agent.create();
  const peer = currentPeer();
  const work = agent.start(id, "first", observer());
  const failed = expect(work).rejects.toThrow("Rejected before rollout");
  await submitted.promise;
  const steerFinished = vi.fn();
  const stopFinished = vi.fn();
  const steering = expect(agent.steer(id, "old correction"))
    .rejects.toThrow("No running Codex turn.")
    .then(steerFinished);
  const stopping = agent.interrupt(id).then(stopFinished);
  firstTurn.reject(new Error("Rejected before rollout"));
  await failed;
  await vi.waitFor(() => {
    expect(steerFinished).toHaveBeenCalledOnce();
    expect(stopFinished).toHaveBeenCalledOnce();
  });
  await Promise.all([steering, stopping]);
  expect(peer.dispose).not.toHaveBeenCalled();
  expect(peer.request).not.toHaveBeenCalledWith("turn/steer", expect.anything());
  expect(peer.request).not.toHaveBeenCalledWith("turn/interrupt", expect.anything());

  mock.handler = (peer, method, params) => {
    if (method === "turn/start") return { turn: { ...turn("inProgress"), id: "retry" } };
    if (method === "turn/steer" || method === "turn/interrupt") return {};
    return normal(peer, method, params);
  };
  const retry = agent.start(id, "retry", observer());
  await agent.steer(id, "new correction");
  expect(peer.request).toHaveBeenCalledWith(
    "turn/steer",
    expect.objectContaining({ expectedTurnId: "retry" }),
  );
  expect(peer.request).not.toHaveBeenCalledWith("turn/interrupt", expect.anything());
  await agent.interrupt(id);
  expect(peer.request).toHaveBeenCalledWith("turn/interrupt", { threadId: id, turnId: "retry" });
  await peer.receive({
    method: "turn/completed",
    params: { threadId: id, turn: { ...turn("interrupted"), id: "retry" } },
  });
  await expect(retry).resolves.toEqual({ stopReason: "cancelled" });
  expect(mock.peers).toHaveLength(1);
});

it("rejects steering after the native interrupt was requested", async () => {
  const normal = mock.handler;
  mock.handler = async (peer, method, params) => {
    if (method === "turn/start") return { turn: turn("inProgress") };
    if (method === "turn/interrupt" || method === "turn/steer") return {};
    return normal(peer, method, params);
  };
  const agent = await open();
  const work = agent.start("saved", "first", observer());
  await vi.waitFor(() =>
    expect(currentPeer().request).toHaveBeenCalledWith("turn/start", expect.anything()),
  );
  await agent.interrupt("saved");
  await expect(agent.steer("saved", "late input")).rejects.toThrow("No running Codex turn.");
  expect(currentPeer().request).not.toHaveBeenCalledWith("turn/steer", expect.anything());
  await currentPeer().receive({
    method: "turn/completed",
    params: { threadId: "saved", turn: turn("interrupted") },
  });
  await work;
});

it("binds a distinct revocable MCP credential to each execution without forwarding the Host bearer", async () => {
  const registrations: {
    token: string;
    bind: ReturnType<typeof vi.fn>;
    dispose: ReturnType<typeof vi.fn>;
  }[] = [];
  const agent = await open({
    registerMcp: (token: string) => {
      const registration = { token, bind: vi.fn(), dispose: vi.fn() };
      registrations.push(registration);
      return registration;
    },
  });
  await Promise.all([
    agent.start("one", "Hello", observer()),
    agent.start("two", "Hello", { ...observer(), executionId: "second" }),
  ]);
  expect(registrations).toHaveLength(2);
  expect(new Set(registrations.map((r) => r.token)).size).toBe(2);
  expect(registrations[0]?.bind).toHaveBeenCalledWith("codex:one", "execution");
  expect(registrations[1]?.bind).toHaveBeenCalledWith("codex:two", "second");
  for (const registration of registrations) expect(registration.dispose).toHaveBeenCalledOnce();
  const requests = mock.peers.flatMap((peer) => peer.request.mock.calls);
  expect(JSON.stringify(requests)).not.toContain("HOST_SECRET");
  expect(JSON.stringify(requests)).toContain("SWARMX_MCP_TOKEN");
});

it("rejects foreign history and propagates native writer errors without starting a turn", async () => {
  const normal = mock.handler;
  mock.handler = (peer, method, params) => {
    if (method === "thread/read") return { thread: { ...thread(), cwd: "/foreign" } };
    return normal(peer, method, params);
  };
  const agent = await open();
  await expect(agent.read("saved", observer())).rejects.toThrow("another directory");
  mock.handler = (peer, method, params) => {
    if (method === "thread/resume") throw new Error("already has an active writer");
    return normal(peer, method, params);
  };
  await expect(agent.start("saved", "Hello", observer())).rejects.toThrow("active writer");
  expect(
    mock.peers
      .flatMap((peer) => peer.request.mock.calls)
      .some(([method]) => method === "turn/start"),
  ).toBe(false);
});

it("steers the owned native turn and waits for a terminal cancellation", async () => {
  const normal = mock.handler;
  mock.handler = (peer, method, params) => {
    if (method === "turn/start") return { turn: turn("inProgress") };
    if (method === "turn/interrupt" || method === "turn/steer") return {};
    return normal(peer, method, params);
  };
  const agent = await open();
  const result = agent.start("saved", "Hello", observer());
  let finished = false;
  void result.then(() => {
    finished = true;
  });
  await vi.waitFor(() =>
    expect(mock.peers[0]?.request).toHaveBeenCalledWith("turn/start", expect.anything()),
  );
  const peer = currentPeer();
  await agent.steer("saved", "Correction");
  expect(peer.request).toHaveBeenCalledWith(
    "turn/steer",
    expect.objectContaining({
      expectedTurnId: "turn",
      input: [expect.objectContaining({ text: "Correction" })],
    }),
  );
  await agent.interrupt("saved");
  expect(peer.request).toHaveBeenCalledWith("turn/interrupt", {
    threadId: "saved",
    turnId: "turn",
  });
  expect(finished).toBe(false);
  await peer.receive({ method: "turn/completed", params: { threadId: "child", turn: turn() } });
  expect(finished).toBe(false);
  await peer.receive({
    method: "turn/completed",
    params: { threadId: "saved", turn: turn("interrupted") },
  });
  await expect(result).resolves.toEqual({ stopReason: "cancelled" });
});

it("never dispatches a cancelled preparation", async () => {
  const normal = mock.handler;
  const resume = Promise.withResolvers<unknown>();
  mock.handler = (peer, method, params) =>
    method === "thread/resume" ? resume.promise : normal(peer, method, params);
  const agent = await open();
  const result = agent.start("saved", "Hello", observer());
  await vi.waitFor(() =>
    expect(mock.peers[0]?.request).toHaveBeenCalledWith("thread/resume", expect.anything()),
  );
  const interrupted = agent.interrupt("saved");
  resume.resolve({ thread: thread() });
  await expect(result).resolves.toEqual({ stopReason: "cancelled" });
  await interrupted;
  expect(mock.peers[0]?.request.mock.calls.some(([method]) => method === "turn/start")).toBe(false);
});

it("returns the exact offered native approval and rejects an unoffered choice", async () => {
  const normal = mock.handler;
  mock.handler = (peer, method, params) =>
    method === "turn/start" ? { turn: turn("inProgress") } : normal(peer, method, params);
  const agent = await open();
  const output = observer();
  const result = agent.start("saved", "Hello", output);
  await vi.waitFor(() =>
    expect(mock.peers[0]?.request).toHaveBeenCalledWith("turn/start", expect.anything()),
  );
  const peer = currentPeer();
  const amendment = { acceptWithExecpolicyAmendment: { execpolicy_amendment: ["git", "status"] } };
  const request = {
    id: "approval",
    method: "item/commandExecution/requestApproval",
    params: {
      threadId: "saved",
      itemId: "command",
      command: "git status",
      reason: "Read state",
      availableDecisions: [amendment, "cancel"],
    },
  };
  vi.mocked(output.interact).mockResolvedValueOnce({ optionId: "0" });
  await expect(peer.receive(request)).resolves.toEqual({ decision: amendment });
  vi.mocked(output.interact).mockResolvedValueOnce({ optionId: "not-offered" });
  await expect(peer.receive(request)).rejects.toThrow("Unknown native approval choice");
  await peer.receive({ method: "turn/completed", params: { threadId: "saved", turn: turn() } });
  await result;
});

it("delivers Codex secret answers without persisting them in the Host journal", async () => {
  const directory = await mkdtemp(join(tmpdir(), "swarmx-codex-secret-"));
  const journal = new ExecutionJournal(directory, "workspace");
  const normal = mock.handler;
  mock.handler = (peer, method, params) =>
    method === "turn/start" ? { turn: turn("inProgress") } : normal(peer, method, params);
  const agent = recordedAgent(journal, "codex", scopeSessions("codex", await open()));
  try {
    const sessionId = await agent.create();
    const output = observer();
    const running = agent.start(sessionId, "Answer the native questions", output);
    await vi.waitFor(() =>
      expect(currentPeer().request).toHaveBeenCalledWith("turn/start", expect.anything()),
    );
    for (const isSecret of [false, true]) {
      const answer = {
        context: "Public context",
        value: isSecret ? "synthetic-secret" : "Plain answer",
      };
      vi.mocked(output.interact).mockResolvedValueOnce(answer);
      await expect(
        currentPeer().receive({
          id: `input-${isSecret}`,
          method: "item/tool/requestUserInput",
          params: {
            threadId: "fresh",
            turnId: "turn",
            itemId: "input",
            questions: [
              {
                id: "context",
                header: "Context",
                question: "Context",
                isOther: false,
                isSecret: false,
                options: null,
              },
              {
                id: "value",
                header: "Value",
                question: "Value",
                isOther: false,
                isSecret,
                options: null,
              },
            ],
          },
        }),
      ).resolves.toEqual({
        answers: { context: { answers: [answer.context] }, value: { answers: [answer.value] } },
      });
    }
    await currentPeer().receive({
      method: "turn/completed",
      params: { threadId: "fresh", turn: turn() },
    });
    await running;
    const records = journal.read({ session: sessionId, limit: 1000 }).events;
    expect(JSON.stringify(records)).not.toContain("synthetic-secret");
    expect(
      records
        .filter(
          ({ event }) => event.type === "CUSTOM" && event.name === "swarmx.interaction.answered",
        )
        .map(({ event }) => event),
    ).toEqual([
      expect.objectContaining({
        value: {
          id: "input-false",
          status: "answered",
          answer: { context: "Public context", value: "Plain answer" },
        },
      }),
      expect.objectContaining({ value: { id: "input-true", status: "answered", redacted: true } }),
    ]);
  } finally {
    await agent.dispose();
    journal.close();
    await rm(directory, { recursive: true, force: true });
  }
});

it("fails when a runtime exits without a terminal result", async () => {
  const normal = mock.handler;
  mock.handler = (peer, method, params) =>
    method === "turn/start" ? { turn: turn("inProgress") } : normal(peer, method, params);
  const agent = await open();
  const result = agent.start("saved", "Hello", observer());
  const rejected = expect(result).rejects.toThrow("runtime exited");
  await vi.waitFor(() =>
    expect(mock.peers[0]?.request).toHaveBeenCalledWith("turn/start", expect.anything()),
  );
  currentPeer().failed(new Error("runtime exited"));
  await rejected;
  expect(currentPeer().dispose).toHaveBeenCalledOnce();
});

it("uses native pagination for complete history, retaining tool failures and timing", async () => {
  const normal = mock.handler;
  mock.handler = (peer, method, params) => {
    if (method === "thread/read") return { thread: { ...thread(), historyMode: "paginated" } };
    if (method === "thread/turns/list") return { data: [turn()], nextCursor: null };
    if (method === "thread/items/list")
      return {
        data: [
          {
            turnId: "turn",
            item: params.cursor
              ? {
                  type: "commandExecution",
                  id: "shell",
                  command: "false",
                  aggregatedOutput: "Native stderr",
                  exitCode: 1,
                  status: "failed",
                }
              : agentMessage,
          },
        ],
        nextCursor: params.cursor ? null : "more",
      };
    return normal(peer, method, params);
  };
  const agent = await open();
  const output = observer();
  await agent.read("saved", output);
  expect(output.text).toHaveBeenCalledOnce();
  expect(output.tool).toHaveBeenCalledWith(
    "shell",
    "commandExecution",
    expect.anything(),
    expect.objectContaining({ aggregatedOutput: "Native stderr", exitCode: 1 }),
  );
  expect(output.activity).toHaveBeenCalledWith({
    type: "tool",
    toolCallId: "shell",
    kind: "execute",
    status: "failed",
  });
  expect(mock.peers[0]?.request).not.toHaveBeenCalledWith(
    "thread/read",
    expect.objectContaining({ includeTurns: true }),
  );
});

it("restricts memory review tools and overrides a permissive native mode", async () => {
  vi.stubEnv(
    "CODEX_CONFIG",
    JSON.stringify({ sandbox_mode: "danger-full-access", approval_policy: "never" }),
  );
  const agent = await open({
    reviewOnly: true,
    registerMcp: vi.fn(() => {
      throw new Error("Review requested MCP");
    }),
  });
  const id = await agent.create();
  const peer = currentPeer();
  expect(peer.request).toHaveBeenCalledWith(
    "thread/start",
    expect.objectContaining({
      sandbox: "read-only",
      approvalPolicy: "never",
      ephemeral: true,
      config: expect.objectContaining({
        "features.shell_tool": false,
        "features.hooks": false,
        "mcp_servers.ambient.enabled": false,
      }),
    }),
  );
  await agent.start(id, "Review only", observer(), { mode: ":danger-full-access" });
  const started = peer.request.mock.calls.find(([method]) => method === "turn/start");
  expect(started?.[1]).not.toHaveProperty("permissions");
  await expect(agent.start("other", "/run unsafe", observer())).rejects.toThrow("slash commands");
});

it.each([
  ["read-only", ":workspace", "on-request", "user"],
  ["agent", ":workspace", "on-request", "auto_review"],
  ["agent-full-access", ":danger-full-access", "never", "user"],
])(
  "retains the persisted public %s approval policy",
  async (mode, permissions, approvalPolicy, approvalsReviewer) => {
    const agent = await open();
    const output = observer();
    await agent.start("saved", "Continue", output, { mode });
    expect(currentPeer().request).toHaveBeenCalledWith(
      "turn/start",
      expect.objectContaining({ permissions, approvalPolicy, approvalsReviewer }),
    );
    expect(output.raw).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ "swarmx.native.mode": mode }),
    );
  },
);

it("closes the spawned process when run credential registration fails", async () => {
  const agent = await open({
    registerMcp: () => {
      throw new Error("Registration failed");
    },
  });
  await expect(agent.start("saved", "Hello", observer())).rejects.toThrow("Registration failed");
  expect(currentPeer().dispose).toHaveBeenCalledOnce();
});
