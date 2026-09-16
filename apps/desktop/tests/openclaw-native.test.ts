import type { GatewayClientOptions } from "@openclaw/gateway-client";
import type { QuestionRecord } from "@openclaw/gateway-protocol";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createOpenClaw } from "../src/agents/openclaw.js";
import type { Observer } from "../src/agents/types.js";
import { loadAgUiHistory } from "../src/host/ag-ui.js";

type Client = {
  options: GatewayClientOptions;
  request: ReturnType<typeof vi.fn>;
  stopAndWait: ReturnType<typeof vi.fn>;
};
const state = vi.hoisted(() => ({
  clients: [] as Client[],
  handler: undefined as
    | ((method: string, params: Record<string, unknown>) => Promise<unknown>)
    | undefined,
  error: undefined as Error | undefined,
}));
vi.mock("@openclaw/gateway-client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@openclaw/gateway-client")>()),
  GatewayClient: class {
    request = vi.fn(async (method: string, params: Record<string, unknown>) =>
      state.handler?.(method, params),
    );
    stopAndWait = vi.fn(async () => {});
    constructor(readonly options: GatewayClientOptions) {
      state.clients.push(this);
    }
    start() {
      queueMicrotask(() =>
        state.error
          ? this.options.onConnectError?.(state.error)
          : this.options.onHelloOk?.({} as never),
      );
    }
  },
}));
const view = (): Observer => ({
  text: vi.fn(),
  tool: vi.fn(),
  raw: vi.fn(),
  activity: vi.fn(),
  interact: vi.fn(),
});
const agents: Awaited<ReturnType<typeof createOpenClaw>>[] = [];
async function open() {
  const agent = await createOpenClaw({
    cwd: "/workspace",
    mcp: { command: "node", args: ["/bridge.js"], env: {} },
    registerMcp: () => {
      throw new Error("OpenClaw must keep native tools");
    },
  });
  agents.push(agent);
  return agent;
}
function client() {
  const found = state.clients[0];
  if (!found) throw new Error("Missing fixture client");
  return found;
}
async function submitted(key: string) {
  await vi.waitFor(() =>
    expect(client().request).toHaveBeenCalledWith(
      "chat.send",
      expect.objectContaining({ sessionKey: key }),
      { timeoutMs: null },
    ),
  );
  return client().request.mock.calls.find(
    ([method, params]) => method === "chat.send" && params.sessionKey === key,
  )?.[1].idempotencyKey as string;
}
function emit(event: string, payload: unknown) {
  client().options.onEvent?.({ type: "event", event, payload });
}
function chat(key: string, runId: string, seq: number, payload: object) {
  emit("chat", { sessionKey: key, runId, seq, ...payload });
}
function steeringId() {
  const call = client().request.mock.calls.findLast(
    ([method, params]) => method === "chat.send" && params.queueMode === "steer",
  );
  if (!call) throw new Error("Missing steering request");
  return call[1].idempotencyKey as string;
}
beforeEach(() => {
  state.clients.length = 0;
  state.error = undefined;
  state.handler = async (method, params) => {
    switch (method) {
      case "sessions.list":
        return {
          hasMore: false,
          nextOffset: null,
          sessions: [
            {
              key: "saved",
              derivedTitle: "Native title",
              updatedAt: 1000,
              model: "model",
              modelProvider: "provider",
              thinkingLevel: "high",
              permissionMode: "guarded",
            },
          ],
        };
      case "sessions.create":
        return { ok: true, key: "created" };
      case "models.list":
        return {
          models: [
            {
              id: "model",
              provider: "provider",
              name: "Model",
              thinkingLevels: [{ id: "high", label: "High" }],
            },
          ],
        };
      case "chat.history":
        return {
          hasMore: false,
          nextOffset: null,
          messages: [
            { id: "message", role: "assistant", content: [{ type: "text", text: "History" }] },
          ],
        };
      case "sessions.patch":
        return { ok: true };
      case "chat.send":
        return { status: "started", runId: params.idempotencyKey };
      case "chat.abort":
        return { ok: true, aborted: true, runIds: [params.runId] };
      default:
        throw new Error(`Unhandled fixture method: ${method}`);
    }
  };
});
afterEach(async () => {
  await Promise.all(agents.splice(0).map((agent) => agent.dispose()));
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

it("uses the official Gateway Client and native catalog without injecting Host MCP credentials", async () => {
  vi.stubEnv("OPENCLAW_GATEWAY_URL", "wss://gateway.example");
  vi.stubEnv("OPENCLAW_GATEWAY_TOKEN", "native token");
  const agent = await open();
  expect(client().options).toMatchObject({
    url: "wss://gateway.example",
    token: "native token",
    caps: ["tool-events", "exec-approvals"],
    minProtocol: 4,
    maxProtocol: 4,
  });
  expect(await agent.list()).toEqual([
    { sessionId: "saved", title: "Native title", updatedAt: "1970-01-01T00:00:01.000Z" },
  ]);
  expect(await agent.models("saved")).toMatchObject({
    models: [{ id: "provider/model", efforts: [{ id: "high", name: "High" }] }],
    current: { model: "provider/model", effort: "high", mode: "guarded" },
  });
  expect(await agent.create()).toBe("created");
  expect(client().request).toHaveBeenCalledWith("sessions.create", {
    cwd: "/workspace",
    idempotencyKey: expect.any(String),
  });
});

it("lists native sessions whose timestamp is absent or null", async () => {
  const handler = state.handler;
  state.handler = async (method, params) =>
    method === "sessions.list"
      ? { hasMore: false, nextOffset: null, sessions: [{ key: "unupdated", updatedAt: null }] }
      : handler?.(method, params);
  const agent = await open();
  expect(await agent.list()).toEqual([{ sessionId: "unupdated" }]);
});

it("replays paginated native history oldest first without opening or resuming a writer", async () => {
  const handler = state.handler;
  state.handler = async (method, params) =>
    method === "chat.history"
      ? {
          messages: [{ role: "assistant", content: params.offset ? "Old" : "New" }],
          hasMore: !params.offset,
          ...(params.offset ? {} : { nextOffset: 1 }),
        }
      : handler?.(method, params);
  const agent = await open();
  const observer = view();
  await agent.read("saved", observer);
  expect(vi.mocked(observer.text).mock.calls.map((call) => call[1])).toEqual(["Old", "New"]);
  expect(client().request.mock.calls.every(([method]) => method === "chat.history")).toBe(true);
});

it("preserves tool failures when native history is reloaded", async () => {
  const handler = state.handler;
  state.handler = async (method, params) =>
    method === "chat.history"
      ? {
          messages: [
            {
              role: "assistant",
              content: [
                {
                  type: "toolCall",
                  id: "failed-read",
                  name: "read",
                  arguments: { path: "missing" },
                },
              ],
            },
            {
              role: "toolResult",
              toolCallId: "failed-read",
              toolName: "read",
              isError: true,
              content: [{ type: "text", text: "Permission denied" }],
            },
          ],
          hasMore: false,
        }
      : handler?.(method, params);
  const history = await loadAgUiHistory(await open(), "saved");
  expect(history.find((message) => message.id === "call:failed-read")?._tool?.status).toBe(
    "failed",
  );
});

it("preserves settings, native run identity, tool events and the final text tail", async () => {
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "Work", observer, {
    model: "provider/model",
    mode: "guarded",
    effort: "high",
    instructions: "Memory",
  });
  const runId = await submitted("saved");
  expect(client().request).toHaveBeenCalledWith("sessions.patch", {
    key: "saved",
    model: "provider/model",
    permissionMode: "guarded",
  });
  expect(client().request).toHaveBeenCalledWith(
    "chat.send",
    expect.objectContaining({
      thinking: "high",
      deliver: false,
      message: expect.stringContaining("Work\n\n<swarmx-memory-context>\nMemory"),
    }),
    { timeoutMs: null },
  );
  chat("saved", runId, 1, {
    state: "delta",
    message: { role: "assistant", content: "An" },
    deltaText: "An",
  });
  chat("saved", runId, 1, {
    state: "delta",
    message: { role: "assistant", content: "An" },
    deltaText: "An",
  });
  emit("agent", {
    sessionKey: "saved",
    runId,
    seq: 1,
    stream: "tool",
    data: { toolCallId: "tool", phase: "start", name: "exec", args: { command: "ls" } },
  });
  emit("agent", {
    sessionKey: "saved",
    runId,
    seq: 2,
    stream: "tool",
    data: { toolCallId: "tool", phase: "result", result: "Denied", isError: true },
  });
  chat("saved", runId, 2, { state: "final", message: { role: "assistant", content: "Answer" } });
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  expect(vi.mocked(observer.text).mock.calls.map((call) => call[1])).toEqual(["An", "swer"]);
  expect(observer.tool).toHaveBeenLastCalledWith("tool", "exec", { command: "ls" }, "Denied");
  expect(observer.activity).toHaveBeenLastCalledWith({
    type: "tool",
    toolCallId: "tool",
    status: "failed",
  });
});

it("targets cancellation to one exact native run and waits for its terminal event", async () => {
  const agent = await open();
  let finished = false;
  const first = agent.start("one", "One", view()).then((value) => {
    finished = true;
    return value;
  });
  const run1 = await submitted("one");
  const second = agent.start("two", "Two", view());
  const run2 = await submitted("two");
  await agent.interrupt("one");
  expect(client().request).toHaveBeenCalledWith("chat.abort", {
    sessionKey: "one",
    runId: run1,
    preserveSideRuns: true,
  });
  expect(finished).toBe(false);
  chat("two", run1, 1, { state: "aborted" });
  expect(finished).toBe(false);
  chat("one", run1, 1, { state: "aborted" });
  await expect(first).resolves.toEqual({ stopReason: "cancelled" });
  chat("two", run2, 1, { state: "final" });
  await second;
});

it("does not dispatch after cancellation during native configuration", async () => {
  const configured = Promise.withResolvers<unknown>();
  const handler = state.handler;
  state.handler = (method, params) =>
    method === "sessions.patch" ? configured.promise : Promise.resolve(handler?.(method, params));
  const agent = await open();
  const work = agent.start("saved", "Never", view(), { model: "provider/model" });
  await vi.waitFor(() =>
    expect(client().request).toHaveBeenCalledWith("sessions.patch", expect.anything()),
  );
  await agent.interrupt("saved");
  configured.resolve({ ok: true });
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
  expect(client().request.mock.calls.some(([method]) => method === "chat.send")).toBe(false);
});

it.each(["error", "disconnect"])(
  "does not call a native %s successful completion",
  async (kind) => {
    const agent = await open();
    const work = agent.start("saved", "Work", view());
    const rejected = expect(work).rejects.toThrow();
    const runId = await submitted("saved");
    if (kind === "error")
      chat("saved", runId, 1, { state: "error", errorMessage: "Provider failed" });
    else client().options.onClose?.(1006, "Disconnected");
    await rejected;
  },
);

it("releases the SDK connection after handshake failure", async () => {
  state.error = new Error("Pairing required");
  await expect(open()).rejects.toThrow("Pairing required");
  expect(client().stopAndWait).toHaveBeenCalledOnce();
});

it("waits for cancellation completion before disposing the SDK connection", async () => {
  const agent = await open();
  const work = agent.start("saved", "Work", view());
  const runId = await submitted("saved");
  const closing = agent.dispose();
  await vi.waitFor(() =>
    expect(client().request).toHaveBeenCalledWith("chat.abort", expect.objectContaining({ runId })),
  );
  expect(client().stopAndWait).not.toHaveBeenCalled();
  await expect(agent.start("other", "Too late", view())).rejects.toThrow("disposed");
  chat("saved", runId, 1, { state: "aborted" });
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
  await closing;
  expect(client().stopAndWait).toHaveBeenCalledOnce();
});

it("requests native cancellation when an owned event is malformed", async () => {
  const agent = await open();
  const work = agent.start("saved", "Work", view());
  const failed = expect(work).rejects.toThrow();
  const runId = await submitted("saved");
  chat("saved", runId, 1, { state: "invented" });
  await failed;
  expect(client().request).toHaveBeenCalledWith("chat.abort", {
    sessionKey: "saved",
    runId,
    preserveSideRuns: true,
  });
});

it("uses native steer dispatch only while the owned turn is running", async () => {
  const agent = await open();
  const work = agent.start("saved", "Work", view());
  const runId = await submitted("saved");
  await agent.steer("saved", "Guidance");
  expect(client().request).toHaveBeenCalledWith(
    "chat.send",
    expect.objectContaining({ sessionKey: "saved", queueMode: "steer", message: "Guidance" }),
    { timeoutMs: null },
  );
  chat("saved", steeringId(), 1, { state: "final" });
  chat("saved", runId, 1, { state: "final" });
  await work;
  await expect(agent.steer("saved", "Late")).rejects.toThrow("No running");
});

it("observes steering output and stays busy until queued native follow-ups finish", async () => {
  const agent = await open();
  const observer = view();
  const settled = vi.fn();
  const work = agent.start("saved", "Work", observer).then((result) => {
    settled();
    return result;
  });
  const runId = await submitted("saved");
  chat("saved", runId, 1, { state: "delta", deltaText: "Main answer" });
  await agent.steer("saved", "Then this");
  const extra = steeringId();
  chat("saved", runId, 1, { state: "final" });
  await Promise.resolve();
  expect(settled).not.toHaveBeenCalled();
  await expect(agent.start("saved", "Another caller", view())).rejects.toThrow("busy");
  chat("saved", extra, 1, { state: "delta", deltaText: "Follow-up answer" });
  chat("saved", extra, 1, { state: "final" });
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  expect(observer.text).toHaveBeenCalledWith(runId, "Main answer", "assistant");
  expect(observer.text).toHaveBeenCalledWith(extra, "Follow-up answer", "assistant");
});

it("cancels the main run and owned steering without cancelling unrelated side runs", async () => {
  const agent = await open();
  const work = agent.start("saved", "Work", view());
  const runId = await submitted("saved");
  await agent.steer("saved", "More");
  const extra = steeringId();
  chat("saved", "foreign", 1, { state: "delta", deltaText: "Not ours" });
  await agent.interrupt("saved");
  expect(
    client()
      .request.mock.calls.filter(([method]) => method === "chat.abort")
      .map(([, params]) => params),
  ).toEqual([
    { sessionKey: "saved", runId, preserveSideRuns: true },
    { sessionKey: "saved", runId: extra, preserveSideRuns: true },
  ]);
  chat("saved", runId, 1, { state: "aborted" });
  chat("saved", extra, 1, { state: "aborted" });
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
});

it("fails a malformed steering acknowledgement and cancels only the submitted identities", async () => {
  const handler = state.handler;
  state.handler = async (method, params) =>
    method === "chat.send" && params.queueMode
      ? { status: "started", runId: "foreign" }
      : handler?.(method, params);
  const agent = await open();
  const work = agent.start("saved", "Work", view());
  const failed = expect(work).rejects.toThrow("identity");
  const runId = await submitted("saved");
  await expect(agent.steer("saved", "More")).rejects.toThrow("identity");
  const extra = steeringId();
  await failed;
  expect(
    client()
      .request.mock.calls.filter(([method]) => method === "chat.abort")
      .map(([, params]) => params.runId),
  ).toEqual([runId, extra]);
});

it("recognizes native pre-admission abort receipts without waiting for an absent chat event", async () => {
  const handler = state.handler;
  state.handler = async (method, params) =>
    method === "chat.send"
      ? { status: "timeout", summary: "aborted", runId: params.idempotencyKey, endedAt: Date.now() }
      : handler?.(method, params);
  const agent = await open();
  await expect(agent.start("saved", "Work", view())).resolves.toEqual({ stopReason: "cancelled" });
});

it.each(["unknown", "timeout"])(
  "rejects a %s acknowledgement instead of leaving a turn pending",
  async (status) => {
    const handler = state.handler;
    state.handler = async (method, params) =>
      method === "chat.send" ? { status, runId: params.idempotencyKey } : handler?.(method, params);
    const agent = await open();
    await expect(agent.start("saved", "Work", view())).rejects.toThrow();
    expect(client().request.mock.calls.filter(([method]) => method === "chat.abort")).toHaveLength(
      1,
    );
  },
);

it("settles a terminal snapshot that shares the last delta sequence", async () => {
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "Work", observer);
  const runId = await submitted("saved");
  chat("saved", runId, 8, { state: "delta", deltaText: "Answer" });
  chat("saved", runId, 8, { state: "final", message: { role: "assistant", content: "Answer" } });
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  expect(observer.text).toHaveBeenCalledExactlyOnceWith(runId, "Answer", "assistant");
});

it("keeps native commentary segments separate from the final answer and preserves reasoning", async () => {
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "Work", observer);
  const runId = await submitted("saved");
  chat("saved", runId, 1, {
    state: "delta",
    deltaText: "Checking",
    message: {
      role: "assistant",
      openclawStreamFallback: { itemId: "commentary" },
      content: [
        { type: "text", text: "Checking" },
        { type: "thinking", thinking: "Reason" },
      ],
    },
  });
  chat("saved", runId, 2, { state: "final", message: { role: "assistant", content: "Answer" } });
  await work;
  expect(observer.text).toHaveBeenCalledWith("commentary", "Checking", "assistant");
  expect(observer.text).toHaveBeenCalledWith("commentary:reasoning", "Reason", "reasoning");
  expect(observer.text).toHaveBeenCalledWith(runId, "Answer", "assistant");
});

it.each(["agent", "exec.approval.requested"])(
  "reads authoritative approval choices from %s",
  async (source) => {
    const handler = state.handler;
    const snapshot = {
      id: "approval",
      urlPath: "/approval/approval",
      createdAtMs: 1,
      expiresAtMs: Date.now() + 60000,
      status: "pending",
      presentation: {
        kind: "exec",
        commandText: "git status",
        allowedDecisions: ["allow-once", "deny"],
      },
    };
    state.handler = async (method, params) =>
      method === "approval.get"
        ? { approval: snapshot }
        : method === "approval.resolve"
          ? {
              applied: true,
              approval: {
                ...snapshot,
                status: "denied",
                decision: "deny",
                reason: "user",
                resolvedAtMs: Date.now(),
              },
            }
          : handler?.(method, params);
    const agent = await open();
    const observer = view();
    vi.mocked(observer.interact).mockResolvedValue({ optionId: "deny" });
    const work = agent.start("saved", "Work", observer);
    const runId = await submitted("saved");
    if (source === "agent")
      emit("agent", {
        runId,
        sessionKey: "saved",
        seq: 1,
        stream: "approval",
        data: { approvalId: "approval", phase: "requested", kind: "exec" },
      });
    else emit(source, { id: "approval", request: { runId, sessionKey: "saved" } });
    await vi.waitFor(() =>
      expect(client().request).toHaveBeenCalledWith("approval.resolve", {
        id: "approval",
        kind: "exec",
        decision: "deny",
      }),
    );
    expect(client().request).toHaveBeenCalledWith("approval.get", { id: "approval" });
    expect(
      vi.mocked(observer.interact).mock.calls[0]?.[0].approval?.choices.map((choice) => choice.id),
    ).toEqual(["allow-once", "deny"]);
    chat("saved", runId, 2, { state: "final" });
    await work;
  },
);

it.each(["resolution", "expiry", "stop"])(
  "does not submit an approval answer after %s",
  async (cause) => {
    const handler = state.handler;
    state.handler = async (method, params) =>
      method === "approval.get"
        ? {
            approval: {
              id: "approval",
              urlPath: "/approval/approval",
              createdAtMs: 1,
              expiresAtMs: Date.now() + 60000,
              status: "pending",
              presentation: {
                kind: "exec",
                commandText: "git status",
                allowedDecisions: ["allow-once", "deny"],
              },
            },
          }
        : handler?.(method, params);
    const agent = await open();
    const observer = view();
    const answer = Promise.withResolvers<unknown>();
    vi.mocked(observer.interact).mockReturnValue(answer.promise);
    const work = agent.start("saved", "Work", observer);
    const runId = await submitted("saved");
    const expiry = new AbortController();
    const timeout = vi.spyOn(AbortSignal, "timeout").mockReturnValue(expiry.signal);
    emit("agent", {
      runId,
      sessionKey: "saved",
      seq: 1,
      stream: "approval",
      data: { approvalId: "approval", phase: "requested", kind: "exec" },
    });
    await vi.waitFor(() => expect(observer.interact).toHaveBeenCalled());
    expect(timeout).toHaveBeenCalledWith(expect.any(Number));
    if (cause === "expiry") expiry.abort();
    else if (cause === "stop") await agent.interrupt("saved");
    else emit("exec.approval.resolved", { id: "approval" });
    expect(vi.mocked(observer.interact).mock.calls[0]?.[1]?.aborted).toBe(true);
    answer.resolve({ optionId: "allow-once" });
    await Promise.resolve();
    expect(client().request.mock.calls.some(([method]) => method === "approval.resolve")).toBe(
      false,
    );
    chat("saved", runId, 2, { state: cause === "stop" ? "aborted" : "final" });
    await work;
  },
);

it("ignores global approval requests without the owned run and matching session", async () => {
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "Work", observer);
  const runId = await submitted("saved");
  for (const request of [
    { runId: "other", sessionKey: "saved" },
    { runId, sessionKey: "other" },
    { sessionKey: "saved" },
  ])
    emit("exec.approval.requested", { id: "foreign", request });
  expect(observer.interact).not.toHaveBeenCalled();
  expect(client().request.mock.calls.some(([method]) => method === "approval.get")).toBe(false);
  chat("saved", runId, 1, { state: "final" });
  await work;
});

it("retains persisted ACP bridge IDs while routing history, turns and cancellation to native keys", async () => {
  const id = "3483bb38-2bbd-4a26-a7a3-d5e2178e6585";
  const key = `agent:main:acp-bridge:${id}`;
  const handler = state.handler;
  state.handler = async (method, params) =>
    method === "sessions.list"
      ? { sessions: [{ key, derivedTitle: "Old conversation" }] }
      : handler?.(method, params);
  const agent = await open();
  expect(await agent.list()).toEqual([
    { sessionId: key, title: "Old conversation" },
    { sessionId: id, title: "Old conversation" },
  ]);
  await agent.read(id, view());
  expect(client().request).toHaveBeenCalledWith("chat.history", {
    sessionKey: key,
    limit: 1000,
    offset: 0,
  });
  await expect(agent.models(id)).resolves.toHaveProperty("models");
  const work = agent.start(id, "Continue", view(), { mode: "guarded" });
  const runId = await submitted(key);
  await expect(agent.start(key, "Duplicate writer", view())).rejects.toThrow("busy");
  expect(client().request).toHaveBeenCalledWith("sessions.patch", {
    key,
    permissionMode: "guarded",
  });
  await agent.interrupt(id);
  expect(client().request).toHaveBeenCalledWith("chat.abort", {
    sessionKey: key,
    runId,
    preserveSideRuns: true,
  });
  chat(key, runId, 1, { state: "aborted" });
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
});

function questionFixture(runId: string, questions: QuestionRecord["questions"]): QuestionRecord {
  const record: QuestionRecord = {
    id: "question",
    runId,
    sessionKey: "saved",
    createdAtMs: Date.now(),
    expiresAtMs: Date.now() + 60000,
    status: "pending",
    questions,
  };
  const handler = state.handler;
  state.handler = async (method, params) =>
    method === "question.get"
      ? { question: record }
      : method === "question.resolve"
        ? params.cancel
          ? { status: "cancelled" }
          : { status: "answered", answers: params.answers }
        : handler?.(method, params);
  return record;
}

it("keeps another session running when an owned question fails validation", async () => {
  const agent = await open();
  const invalid = view();
  vi.mocked(invalid.interact).mockResolvedValue({ other_0: "" });
  const first = agent.start("saved", "First", invalid);
  const failed = expect(first).rejects.toBeInstanceOf(Error);
  const firstId = await submitted("saved");
  const observer = view();
  const second = agent.start("other", "Second", observer);
  const secondId = await submitted("other");
  emit(
    "question.requested",
    questionFixture(firstId, [
      { questionId: "input", header: "Input", question: "Input?", options: [] },
    ]),
  );
  await failed;
  expect(
    client()
      .request.mock.calls.filter(([method]) => method === "chat.abort")
      .map(([, params]) => params.runId),
  ).toEqual([firstId]);
  chat("other", secondId, 1, {
    state: "final",
    message: { role: "assistant", content: "Still working" },
  });
  await expect(second).resolves.toEqual({ stopReason: "end_turn" });
  expect(observer.text).toHaveBeenCalledWith(secondId, "Still working", "assistant");
});

it("answers native question batches with exact choices, custom input and native IDs", async () => {
  const agent = await open();
  const observer = view();
  vi.mocked(observer.interact).mockResolvedValue({
    choice_0: "Local",
    choice_1: ["Tests", "Build"],
    other_1: "Docs",
    other_2: "Extra context",
  });
  const work = agent.start("saved", "Work", observer);
  const runId = await submitted("saved");
  const record = questionFixture(runId, [
    {
      questionId: "where",
      header: "Location",
      question: "Where?",
      options: [{ label: "Local", description: "On this device" }, { label: "Remote" }],
    },
    {
      questionId: "checks",
      header: "Checks",
      question: "Which checks?",
      options: [{ label: "Tests" }, { label: "Build" }],
      multiSelect: true,
      isOther: true,
    },
    { questionId: "context", header: "Context", question: "More context?", options: [] },
  ]);
  emit("question.requested", record);
  emit("question.requested", record);
  await vi.waitFor(() =>
    expect(client().request).toHaveBeenCalledWith("question.resolve", {
      id: "question",
      answers: {
        answers: {
          where: ["Local"],
          checks: ["Tests", "Build", "Docs"],
          context: ["Extra context"],
        },
      },
    }),
  );
  expect(observer.interact).toHaveBeenCalledTimes(1);
  expect(client().options.scopes).toContain("operator.questions");
  expect(vi.mocked(observer.interact).mock.calls[0]?.[0].schema).toMatchObject({
    properties: {
      choice_0: {
        oneOf: [
          { const: "Local", title: "Local — On this device" },
          { const: "Remote", title: "Remote" },
        ],
      },
    },
  });
  chat("saved", runId, 1, { state: "final" });
  await work;
});

it("discloses native secret storage and masks the unchanged secret answer", async () => {
  const agent = await open();
  const observer = view();
  vi.mocked(observer.interact).mockResolvedValue({ other_0: "  private value  " });
  const work = agent.start("saved", "Work", observer);
  const runId = await submitted("saved");
  emit(
    "question.requested",
    questionFixture(runId, [
      {
        questionId: "token",
        header: "Token",
        question: "Enter the token",
        options: [],
        isSecret: true,
        url: "https://example.com/tokens",
        secretStore: {
          name: "API_TOKEN",
          kind: "secret",
          allowedHosts: ["api.example.com"],
          reason: "Read your repository",
        },
        secretStoreExisting: { updatedAtMs: 1 },
      },
    ]),
  );
  await vi.waitFor(() =>
    expect(client().request).toHaveBeenCalledWith("question.resolve", {
      id: "question",
      answers: { answers: { token: ["  private value  "] } },
    }),
  );
  const request = vi.mocked(observer.interact).mock.calls[0]?.[0];
  expect(request).toMatchObject({
    sensitive: true,
    schema: { properties: { other_0: { format: "password" } } },
  });
  const presented = JSON.stringify(request);
  for (const value of [
    "API_TOKEN",
    "api.example.com",
    "Read your repository",
    "Replace",
    "https://example.com/tokens",
  ])
    expect(presented).toContain(value);
  expect(presented).not.toContain("private value");
  chat("saved", runId, 1, { state: "final" });
  await work;
});

it("cancels the native question when its form is dismissed", async () => {
  const agent = await open();
  const work = agent.start("saved", "Work", view());
  const runId = await submitted("saved");
  emit(
    "question.requested",
    questionFixture(runId, [
      { questionId: "input", header: "Input", question: "Input?", options: [] },
    ]),
  );
  await vi.waitFor(() =>
    expect(client().request).toHaveBeenCalledWith("question.resolve", {
      id: "question",
      cancel: true,
    }),
  );
  chat("saved", runId, 1, { state: "final" });
  await work;
});

it.each(["resolution", "expiry", "stop", "completion"])(
  "does not answer a native question after %s",
  async (cause) => {
    const agent = await open();
    const observer = view();
    const answer = Promise.withResolvers<unknown>();
    vi.mocked(observer.interact).mockReturnValue(answer.promise);
    const work = agent.start("saved", "Work", observer);
    const runId = await submitted("saved");
    const expiry = new AbortController();
    vi.spyOn(AbortSignal, "timeout").mockReturnValue(expiry.signal);
    emit(
      "question.requested",
      questionFixture(runId, [
        { questionId: "input", header: "Input", question: "Input?", options: [] },
      ]),
    );
    await vi.waitFor(() => expect(observer.interact).toHaveBeenCalled());
    if (cause === "expiry") expiry.abort();
    else if (cause === "stop") await agent.interrupt("saved");
    else if (cause === "completion") chat("saved", runId, 1, { state: "final" });
    else emit("question.resolved", { id: "question", status: "answered" });
    expect(vi.mocked(observer.interact).mock.calls[0]?.[1]?.aborted).toBe(true);
    answer.resolve({ other_0: "Late answer" });
    await new Promise((resolve) => setImmediate(resolve));
    expect(client().request.mock.calls.some(([method]) => method === "question.resolve")).toBe(
      false,
    );
    if (cause !== "completion")
      chat("saved", runId, 1, { state: cause === "stop" ? "aborted" : "final" });
    await work;
  },
);

it("ignores questions without the owned run and matching native session", async () => {
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "Work", observer);
  const runId = await submitted("saved");
  for (const identity of [
    { runId: "foreign", sessionKey: "saved" },
    { runId, sessionKey: "foreign" },
    { sessionKey: "saved" },
  ])
    emit("question.requested", { id: "question", ...identity });
  expect(client().request.mock.calls.some(([method]) => method === "question.get")).toBe(false);
  expect(observer.interact).not.toHaveBeenCalled();
  chat("saved", runId, 1, { state: "final" });
  await work;
});

it.each(["runId", "sessionKey", "id"])(
  "rejects a question snapshot whose %s changed",
  async (field) => {
    const agent = await open();
    const observer = view();
    const work = agent.start("saved", "Work", observer);
    const failed = expect(work).rejects.toThrow("question identity changed");
    const runId = await submitted("saved");
    const record = questionFixture(runId, [
      { questionId: "input", header: "Input", question: "Input?", options: [] },
    ]);
    emit("question.requested", { ...record });
    Object.assign(record, { [field]: "foreign" });
    await failed;
    expect(observer.interact).not.toHaveBeenCalled();
    expect(client().request.mock.calls.some(([method]) => method === "question.resolve")).toBe(
      false,
    );
  },
);

it.each([{ choice_0: "Unknown" }, { choice_0: "A", other_0: "Other" }, {}, { other_0: "  " }])(
  "rejects invalid native question answers: %j",
  async (answer) => {
    const agent = await open();
    const observer = view();
    vi.mocked(observer.interact).mockResolvedValue(answer);
    const work = agent.start("saved", "Work", observer);
    const failed = expect(work).rejects.toBeInstanceOf(Error);
    const runId = await submitted("saved");
    emit(
      "question.requested",
      questionFixture(runId, [
        {
          questionId: "input",
          header: "Input",
          question: "Input?",
          options: [{ label: "A" }, { label: "B" }],
          isOther: true,
        },
      ]),
    );
    await failed;
    expect(client().request.mock.calls.some(([method]) => method === "question.resolve")).toBe(
      false,
    );
  },
);
