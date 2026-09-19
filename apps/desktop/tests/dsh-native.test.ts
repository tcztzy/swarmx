import assert from "node:assert/strict";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname } from "node:path";
import type { DeepSeekHarnessOptions, HarnessNotification } from "@deepseek-ai/dsh-sdk-client";
import { beforeEach, expect, it, vi } from "vitest";
import { createDsh, runDshSession } from "../src/agents/dsh.js";
import type { Observer } from "../src/agents/types.js";

const sdk = vi.hoisted(() => ({
  runtimes: [] as {
    options: DeepSeekHarnessOptions;
    close: ReturnType<typeof vi.fn>;
    run: ReturnType<typeof vi.fn>;
    finish: (error?: Error) => void;
  }[],
  hold: false,
}));
vi.mock("@deepseek-ai/dsh-sdk-client", () => ({
  DeepSeekHarness: class {
    private readonly done = Promise.withResolvers<void>();
    readonly close = vi.fn(async () => {
      this.done.reject(new Error("SDK closed"));
    });
    readonly run = vi.fn(async (_text, options, id) => {
      if (sdk.hold) await this.done.promise;
      options.onNotification({
        method: "session.event",
        params: {
          sessionId: id,
          event: { type: "turn/end", seq: 1, data: { turn: 1, reason: { kind: "completed" } } },
        },
      });
      return { sessionId: id, finalResponse: "", events: [], notifications: [] };
    });
    constructor(readonly options: DeepSeekHarnessOptions) {
      void this.done.promise.catch(() => {});
      sdk.runtimes.push(this);
    }
    finish = (error?: Error) => (error ? this.done.reject(error) : this.done.resolve());
    session(id: string) {
      return { id, run: (text: string, options: unknown) => this.run(text, options, id) };
    }
  },
}));
beforeEach(() => {
  sdk.runtimes.length = 0;
  sdk.hold = false;
});

it("owns one native runtime per task and removes its private MCP configuration and credential", async () => {
  const endpoint = { bind: vi.fn(), dispose: vi.fn() };
  const registerMcp = vi.fn((_token: string) => endpoint);
  const agent = await createDsh({
    cwd: "/workspace",
    mcp: { command: "node", args: ["/bridge.js"], env: {} },
    registerMcp,
  });
  const id = await agent.create();
  expect(sdk.runtimes).toHaveLength(0);
  sdk.hold = true;
  const work = agent.start(id, "work", { ...view(), executionId: "execution" });
  const runtime = sdk.runtimes[0];
  assert.ok(runtime);
  const path = runtime.options.patches?.[0];
  assert.ok(path);
  expect(statSync(path).mode & 0o777).toBe(0o600);
  const patch = JSON.parse(readFileSync(path, "utf8"));
  expect(patch[0].insert[0]).toMatchObject({
    name: "@deepseek-ai/dsh-mcp-client",
    config: {
      transport: "stdio",
      command: "node",
      args: ["/bridge.js"],
      failOnStartupError: true,
    },
  });
  expect(patch[0].insert[0].config.env.SWARMX_MCP_TOKEN).toBe(registerMcp.mock.calls[0]?.[0]);
  expect(endpoint.bind).toHaveBeenCalledWith(`dsh:${id}`, "execution");
  expect(runtime.options).toEqual({ cwd: "/workspace", processCwd: "/workspace", patches: [path] });
  runtime.finish();
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  expect(runtime.close).toHaveBeenCalledOnce();
  expect(endpoint.dispose).toHaveBeenCalledOnce();
  expect(existsSync(dirname(path))).toBe(false);
  await expect(agent.start(id, "again", view())).rejects.toThrow("execute once");
  expect(sdk.runtimes).toHaveLength(1);
  await agent.dispose();
});

it("stops only the owned DSH execution and preserves native failures in siblings", async () => {
  sdk.hold = true;
  const agent = await createDsh({
    cwd: "/workspace",
    mcp: { command: "node", args: ["/bridge.js"], env: {} },
  });
  const first = await agent.create();
  const second = await agent.create();
  const a = agent.start(first, "first", view());
  const b = agent.start(second, "second", view());
  await agent.interrupt(first);
  await expect(a).resolves.toEqual({ stopReason: "cancelled" });
  const sibling = sdk.runtimes[1];
  assert.ok(sibling);
  expect(sibling.close).not.toHaveBeenCalled();
  const failure = new Error("Native model failed");
  sibling.finish(failure);
  await expect(b).rejects.toBe(failure);
  for (const runtime of sdk.runtimes) {
    const patch = runtime.options.patches?.[0];
    assert.ok(patch);
    expect(existsSync(dirname(patch))).toBe(false);
  }
  await agent.dispose();
});

it.each(["sdk", "sdk-minimal"])(
  "passes the selected route, effort and %s profile into the owned runtime",
  async (profile) => {
    const agent = await createDsh({
      cwd: "/workspace",
      mcp: { command: "node", args: ["/bridge.js"], env: {} },
    });
    const id = await agent.create();
    await expect(
      agent.start(id, "work", view(), {
        model: "deepseek-official/deepseek/v4-pro-0813",
        effort: "high",
        profile,
      }),
    ).resolves.toEqual({ stopReason: "end_turn" });
    expect(sdk.runtimes).toHaveLength(1);
    expect(sdk.runtimes[0]?.options).toMatchObject({
      provider: "deepseek-official",
      model: "deepseek/v4-pro-0813",
      reasoningEffort: "high",
      profile,
    });
    await agent.dispose();
  },
);

it("can select effort and profile while preserving the SDK's default model route", async () => {
  const agent = await createDsh({
    cwd: "/workspace",
    mcp: { command: "node", args: ["/bridge.js"], env: {} },
  });
  await agent.start(await agent.create(), "work", view(), {
    effort: "native-adapter-effort",
    profile: "sdk-minimal",
  });
  expect(sdk.runtimes[0]?.options).toMatchObject({
    reasoningEffort: "native-adapter-effort",
    profile: "sdk-minimal",
  });
  expect(sdk.runtimes[0]?.options).not.toHaveProperty("provider");
  expect(sdk.runtimes[0]?.options).not.toHaveProperty("model");
  await agent.dispose();
});

it.each([
  { model: "deepseek-v4-pro-0813" },
  { model: "/deepseek-v4-pro-0813" },
  { model: "deepseek-official/" },
  { model: "deepseek official/deepseek-v4-pro-0813" },
  { model: "deepseek-official/deepseek v4-pro-0813" },
  { model: "" },
  { profile: "custom-profile" },
  { profile: "" },
  { mode: "sdk-minimal" },
])("rejects unsupported selection %j before creating native resources", async (selection) => {
  const registerMcp = vi.fn();
  const agent = await createDsh({
    cwd: "/workspace",
    mcp: { command: "node", args: ["/bridge.js"], env: {} },
    registerMcp,
  });
  const id = await agent.create();
  await expect(agent.start(id, "work", view(), selection)).rejects.toThrow();
  expect(registerMcp).not.toHaveBeenCalled();
  expect(sdk.runtimes).toHaveLength(0);
  expect(await agent.list()).toEqual([{ sessionId: id }]);
  await agent.dispose();
});

const view = (): Observer => ({
  text: vi.fn(),
  tool: vi.fn(),
  raw: vi.fn(),
  activity: vi.fn(),
  interact: vi.fn(),
});
const event = (
  type: string,
  data: object,
  seq: number,
  sessionId = "root",
): HarnessNotification => ({
  method: "session.event",
  params: { sessionId, event: { type, seq, time: 1, data } },
});
const end = (reason: object, turn = 1) => event("turn/end", { turn, reason }, 10 + turn);
function session(events: HarnessNotification[], idle?: Promise<void>) {
  return {
    id: "root",
    run: vi.fn(
      async (
        _input: string,
        options?: { onNotification?: (notification: HarnessNotification) => void },
      ) => {
        for (const item of events) options?.onNotification?.(item);
        await idle;
        return {
          sessionId: "root",
          finalResponse: "SDK summary must not duplicate streamed content",
          events: [],
          notifications: events,
        };
      },
    ),
  };
}

it("projects committed SDK text, reasoning and tool outcomes without duplicating finalResponse", async () => {
  const native = session([
    event("turn/start", { turn: 1 }, 1),
    event(
      "assistant/message",
      {
        message: {
          content: [
            { type: "reasoning", text: "Consider the change" },
            { type: "text", text: "Checking" },
            { type: "tool-call", id: "tool", name: "read", arguments: '{"path":"file"}' },
          ],
        },
      },
      2,
    ),
    event("tool/call", { callId: "tool", name: "read", arguments: '{"path":"file"}' }, 3),
    event(
      "tool/result",
      {
        message: {
          content: [
            {
              type: "tool-result",
              toolCallId: "tool",
              content: [{ type: "text", text: "Missing file" }],
              isError: true,
            },
          ],
        },
      },
      4,
    ),
    event(
      "assistant/message",
      { message: { content: [{ type: "text", text: "File is missing" }] } },
      5,
    ),
    end({ kind: "completed" }),
  ]);
  const observer = view();
  await expect(runDshSession(native, "Read file", observer)).resolves.toEqual({
    stopReason: "end_turn",
  });
  expect(native.run).toHaveBeenCalledWith("Read file", { onNotification: expect.any(Function) });
  expect(vi.mocked(observer.text).mock.calls).toEqual([
    ["root:2:reasoning", "Consider the change", "reasoning"],
    ["root:2", "Checking", "assistant"],
    ["root:5", "File is missing", "assistant"],
  ]);
  expect(observer.tool).toHaveBeenCalledTimes(2);
  expect(observer.tool).toHaveBeenLastCalledWith("tool", "read", '{"path":"file"}', [
    { type: "text", text: "Missing file" },
  ]);
  expect(observer.activity).toHaveBeenLastCalledWith({
    type: "tool",
    toolCallId: "tool",
    status: "failed",
  });
});

it("retains child diagnostics without mixing their output and failures into the root result", async () => {
  const events = [
    event(
      "assistant/message",
      { message: { content: [{ type: "text", text: "Child answer" }] } },
      1,
      "child",
    ),
    event(
      "turn/end",
      { turn: 1, reason: { kind: "error", error: { message: "Child failed" } } },
      2,
      "child",
    ),
    end({ kind: "completed" }),
  ];
  const observer = view();
  await expect(runDshSession(session(events), "Work", observer)).resolves.toEqual({
    stopReason: "end_turn",
  });
  expect(observer.text).not.toHaveBeenCalled();
  expect(observer.raw).toHaveBeenCalledWith(events[0], { "swarmx.native.session_id": "child" });
});

it("waits for SDK idle after a native terminal event", async () => {
  const idle = Promise.withResolvers<void>();
  const settled = vi.fn();
  const work = runDshSession(
    session([end({ kind: "completed" })], idle.promise),
    "Work",
    view(),
  ).then(settled);
  await Promise.resolve();
  expect(settled).not.toHaveBeenCalled();
  idle.resolve();
  await work;
  expect(settled).toHaveBeenCalledWith({ stopReason: "end_turn" });
});

it("rejects native errors even when the SDK reaches idle and a later queued turn completes", async () => {
  await expect(
    runDshSession(
      session([
        end({
          kind: "error",
          error: { message: "Local fixture rejection", code: "INVALID_REQUEST", status: 400 },
        }),
        end({ kind: "completed" }, 2),
      ]),
      "Work",
      view(),
    ),
  ).rejects.toThrow("Local fixture rejection");
});

it.each(["blocked", "interrupted", "unknown"])(
  "does not turn native %s into success",
  async (kind) => {
    await expect(runDshSession(session([end({ kind })]), "Work", view())).rejects.toThrow();
  },
);

it.each([
  [{ kind: "aborted", reason: { kind: "user" } }, "cancelled"],
  [{ kind: "max-tokens" }, "max_tokens"],
] as const)(
  "preserves terminal outcome %j across subsequent completion",
  async (reason, stopReason) => {
    await expect(
      runDshSession(session([end(reason), end({ kind: "completed" }, 2)]), "Work", view()),
    ).resolves.toEqual({ stopReason });
  },
);

it.each([
  { events: [] },
  { events: [event("turn/start", { turn: 1 }, 1)] },
  { events: [end({ kind: "completed" }), event("turn/start", { turn: 2 }, 12)] },
])("rejects SDK idle without a complete native terminal outcome: %j", async ({ events }) => {
  await expect(runDshSession(session(events), "Work", view())).rejects.toThrow("terminal outcome");
});

it("preserves SDK transport failures", async () => {
  const native = session([]);
  const failure = new Error("Native process exited");
  native.run.mockRejectedValue(failure);
  await expect(runDshSession(native, "Work", view())).rejects.toBe(failure);
});

it("rejects malformed native messages before projecting them", async () => {
  const observer = view();
  await expect(
    runDshSession(
      session([
        event("assistant/message", { message: { content: [{ type: "text", text: 7 }] } }, 1),
        end({ kind: "completed" }),
      ]),
      "Work",
      observer,
    ),
  ).rejects.toThrow();
  expect(observer.text).not.toHaveBeenCalled();
});
