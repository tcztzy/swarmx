import type { JSONRPCRequest } from "json-rpc-2.0";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createHermes } from "../src/agents/hermes.js";
import type { Observer } from "../src/agents/types.js";

type Peer = {
  command: string;
  args: string[];
  env: NodeJS.ProcessEnv;
  signal: AbortSignal;
  receive(request: JSONRPCRequest): Promise<unknown>;
  fail(error: Error): void;
  request: ReturnType<
    typeof vi.fn<(method: string, params: Record<string, unknown>) => Promise<unknown>>
  >;
  dispose: ReturnType<typeof vi.fn<() => Promise<void>>>;
};
const mock = vi.hoisted(() => ({
  peers: [] as Peer[],
  handler: undefined as
    | ((peer: Peer, method: string, params: Record<string, unknown>) => Promise<unknown>)
    | undefined,
  autoComplete: true,
}));
vi.mock("../src/agents/rpc-process.js", () => ({
  rpcProcess: vi.fn((command, args, _cwd, receive, fail, env) => {
    const lifetime = new AbortController();
    const peer: Peer = {
      command,
      args,
      env,
      receive,
      fail,
      signal: lifetime.signal,
      request: vi.fn(async (method, params) => {
        lifetime.signal.throwIfAborted();
        return mock.handler?.(peer, method, params);
      }),
      dispose: vi.fn(async () => {
        lifetime.abort();
      }),
    };
    mock.peers.push(peer);
    return peer;
  }),
}));
const options = {
  cwd: "/workspace",
  mcp: { command: "node", args: ["/bridge.js"], env: {} },
};
const view = (): Observer => ({
  executionId: "execution",
  text: vi.fn(),
  tool: vi.fn(),
  raw: vi.fn(),
  activity: vi.fn(),
  interact: vi.fn(),
});
const agents: Awaited<ReturnType<typeof createHermes>>[] = [];
async function open(extra = {}) {
  const agent = await createHermes({ ...options, ...extra });
  agents.push(agent);
  return agent;
}
async function event(peer: Peer, type: string, payload: object, sessionId = "live", settle = true) {
  await peer.receive({
    jsonrpc: "2.0",
    method: "event",
    params: { type, session_id: sessionId, payload },
  });
  if (type === "message.complete" && settle)
    await event(peer, "session.info", { running: false }, sessionId);
}
async function submitted(index = 0) {
  await vi.waitFor(() =>
    expect(mock.peers[index]?.request).toHaveBeenCalledWith("prompt.submit", expect.anything()),
  );
  const peer = mock.peers[index];
  if (!peer) throw new Error("No native fixture");
  return peer;
}
beforeEach(() => {
  vi.stubEnv("SWARMX_HERMES_PYTHON", "/native/python");
  mock.peers.length = 0;
  mock.autoComplete = true;
  mock.handler = async (peer, method, params) => {
    switch (method) {
      case "session.create":
        return {
          session_id: "live",
          stored_session_id: "stored",
          info: { cwd: options.cwd, model: "model", provider: "provider" },
        };
      case "session.resume":
        return {
          session_id: "live",
          session_key: params.session_id,
          info: { cwd: options.cwd, lazy: true },
        };
      case "session.list":
        return {
          sessions: [{ id: "saved", title: "Native title", preview: "hello", started_at: 1000 }],
        };
      case "session.history":
        return {
          messages: [
            { role: "assistant", text: "History", row_id: 5, reasoning_content: "Reason" },
            { role: "tool", name: "terminal", args: { command: "ls" } },
          ],
        };
      case "model.options":
        return {
          model: "model",
          provider: "provider",
          providers: [
            {
              slug: "provider",
              name: "Provider",
              models: ["model", "plain"],
              capabilities: {
                model: { reasoning: true, can_disable_reasoning: false },
                plain: { reasoning: false },
              },
            },
          ],
        };
      case "config.get":
        return { value: "high" };
      case "config.set":
        return { key: params.key, value: params.value };
      case "prompt.submit":
        if (mock.autoComplete)
          setImmediate(() => {
            void event(peer, "message.complete", { text: "Answer", status: "complete" }).catch(
              peer.fail,
            );
          });
        return { status: "streaming" };
      case "session.steer":
        return { status: "queued" };
      case "swarmx.session.wait":
        return { running: false };
      case "session.interrupt":
        return { status: "interrupted" };
      case "approval.respond":
        return { resolved: true };
      case "clarify.respond":
      case "secret.respond":
      case "sudo.respond":
        return { status: "ok" };
      default:
        throw new Error(`Unhandled fixture method: ${method}`);
    }
  };
});
afterEach(async () => {
  await Promise.all(agents.splice(0).map((agent) => agent.dispose()));
  vi.unstubAllEnvs();
  vi.clearAllMocks();
});

it("uses native catalog/history APIs without Host execution credentials or an eager resume", async () => {
  const registerMcp = vi.fn();
  const agent = await open({ registerMcp });
  expect(await agent.list()).toEqual([
    { sessionId: "saved", title: "Native title", updatedAt: "1970-01-01T00:16:40.000Z" },
  ]);
  const observer = view();
  await agent.read("saved", observer);
  expect(observer.text).toHaveBeenCalledWith("saved:5", "History", "assistant");
  expect(observer.text).toHaveBeenCalledWith("saved:5:reasoning", "Reason", "reasoning");
  expect(mock.peers[1]?.request).toHaveBeenCalledWith("session.resume", {
    session_id: "saved",
    lazy: true,
  });
  expect(mock.peers[0]?.request).toHaveBeenCalledWith("session.list", {
    limit: Number.MAX_SAFE_INTEGER,
  });
  const catalog = await agent.models("saved");
  expect(catalog.current).toEqual({});
  expect(catalog.models[0]?.id).toBe("provider:model");
  expect(catalog.models[0]?.efforts.some((effort) => effort.id === "none")).toBe(false);
  expect(catalog.models[1]?.efforts).toEqual([]);
  expect(registerMcp).not.toHaveBeenCalled();
  for (const peer of mock.peers) {
    expect(JSON.parse(peer.env.SWARMX_HERMES_MCP ?? "")).toEqual({});
    expect(peer.dispose).toHaveBeenCalledOnce();
  }
});

it("observes the native queued prompt after a resumed automatic continuation", async () => {
  mock.autoComplete = false;
  const normal = mock.handler;
  if (!normal) throw new Error("Missing fixture handler");
  mock.handler = async (peer, method, params) => {
    if (method === "prompt.submit") return { status: "queued" };
    return normal(peer, method, params);
  };
  const agent = await open();
  const observer = view();
  const settled = vi.fn();
  const work = agent.start("saved", "next", observer).then(settled);
  const peer = await submitted();
  expect(peer.request).toHaveBeenCalledWith("session.resume", { session_id: "saved" });
  await event(peer, "message.complete", { text: "Resumed work", status: "interrupted" });
  await Promise.resolve();
  expect(settled).not.toHaveBeenCalled();
  expect(peer.dispose).not.toHaveBeenCalled();
  await event(peer, "message.start", {});
  await event(peer, "message.complete", { text: "Queued answer", status: "complete" });
  await work;
  expect(settled).toHaveBeenCalledWith({ stopReason: "end_turn" });
  expect(observer.text).toHaveBeenCalledWith(expect.any(String), "Queued answer", "assistant");
});

it.each([true, false, undefined])(
  "waits for the new turn when a prior completion precedes acknowledgement (running: %s)",
  async (running) => {
    mock.autoComplete = false;
    let nativeRunning = true;
    const normal = mock.handler;
    if (!normal) throw new Error("Missing fixture handler");
    mock.handler = async (peer, method, params) => {
      if (method === "session.resume") {
        if (!params.lazy && running !== true) await event(peer, "message.start", {});
        return {
          session_id: "live",
          session_key: params.session_id,
          info: { cwd: options.cwd, running },
          running: nativeRunning,
        };
      }
      if (method === "prompt.submit") {
        await event(peer, "message.complete", { text: "Old automatic turn", status: "complete" });
        await event(peer, "message.start", {});
        return { status: "streaming" };
      }
      return normal(peer, method, params);
    };
    const agent = await open();
    const observer = view();
    const settled = vi.fn();
    const work = agent.start("saved", "new prompt", observer).then(settled);
    const peer = await submitted();
    await vi.waitFor(() =>
      expect(observer.text).toHaveBeenCalledWith(
        expect.any(String),
        "Old automatic turn",
        "assistant",
      ),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(settled).not.toHaveBeenCalled();
    expect(peer.dispose).not.toHaveBeenCalled();
    await event(peer, "session.info", { running: false });
    expect(settled).not.toHaveBeenCalled();
    nativeRunning = false;
    await event(peer, "message.complete", { text: "New answer", status: "complete" });
    await work;
    expect(settled).toHaveBeenCalledWith({ stopReason: "end_turn" });
    expect(observer.text).toHaveBeenCalledWith(expect.any(String), "New answer", "assistant");
  },
);

it.each([
  [false, "complete"],
  [false, "error"],
  [true, "complete"],
  [true, "error"],
] as const)(
  "settles a new terminal before submit acknowledgement (prior running: %s, status: %s)",
  async (priorRunning, status) => {
    mock.autoComplete = false;
    const normal = mock.handler;
    if (!normal) throw new Error("Missing fixture handler");
    mock.handler = async (peer, method, params) => {
      if (method === "session.resume")
        return {
          session_id: "live",
          session_key: params.session_id,
          info: { cwd: options.cwd, running: priorRunning },
          running: false,
        };
      if (method === "prompt.submit") {
        if (priorRunning)
          await event(peer, "message.complete", { text: "Old answer", status: "complete" });
        await event(peer, "message.complete", {
          text: "New answer",
          status,
          error: "Native turn failed",
        });
        return { status: "streaming" };
      }
      return normal(peer, method, params);
    };
    const endpoint = { bind: vi.fn(), dispose: vi.fn() };
    const agent = await open({ registerMcp: () => endpoint });
    const settled = vi.fn();
    const work = agent.start("saved", "new prompt", view()).then(settled, settled);
    const peer = await submitted();
    await vi.waitFor(() =>
      expect(settled).toHaveBeenCalledWith(
        status === "error" ? new Error("Native turn failed") : { stopReason: "end_turn" },
      ),
    );
    await work;
    expect(peer.dispose).toHaveBeenCalledOnce();
    expect(endpoint.dispose).toHaveBeenCalledOnce();
  },
);

it("preserves completion received while waiting for the native execution thread", async () => {
  mock.autoComplete = false;
  const normal = mock.handler;
  if (!normal) throw new Error("Missing fixture handler");
  mock.handler = async (peer, method, params) => {
    if (method === "swarmx.session.wait") {
      await event(peer, "message.start", {});
      await event(peer, "message.complete", { text: "New answer", status: "complete" });
      return { running: false };
    }
    if (method === "prompt.submit") {
      await event(peer, "message.complete", { text: "Old answer", status: "complete" });
      return { status: "streaming" };
    }
    return normal(peer, method, params);
  };
  const endpoint = { bind: vi.fn(), dispose: vi.fn() };
  const agent = await open({ registerMcp: () => endpoint });
  const observer = view();
  await expect(agent.start("saved", "new prompt", observer)).resolves.toEqual({
    stopReason: "end_turn",
  });
  expect(observer.text).toHaveBeenCalledWith(expect.any(String), "New answer", "assistant");
  expect(mock.peers[0]?.dispose).toHaveBeenCalledOnce();
  expect(endpoint.dispose).toHaveBeenCalledOnce();
});

it.each(["steered", "redirected"])(
  "waits for only the current turn when a busy submit is %s",
  async (status) => {
    mock.autoComplete = false;
    const normal = mock.handler;
    if (!normal) throw new Error("Missing fixture handler");
    mock.handler = async (peer, method, params) => {
      if (method === "session.resume")
        return {
          session_id: "live",
          session_key: params.session_id,
          info: { cwd: options.cwd, running: true },
        };
      if (method === "prompt.submit") return { status };
      return normal(peer, method, params);
    };
    const agent = await open();
    const work = agent.start("saved", "correct this turn", view());
    const peer = await submitted();
    await event(peer, "message.complete", { text: "Corrected answer", status: "complete" });
    await expect(work).resolves.toEqual({ stopReason: "end_turn" });
    expect(peer.dispose).toHaveBeenCalledOnce();
  },
);

it("preserves early native resume events and can stop a queued continuation", async () => {
  mock.autoComplete = false;
  const normal = mock.handler;
  if (!normal) throw new Error("Missing fixture handler");
  mock.handler = async (peer, method, params) => {
    if (method === "session.resume") {
      await peer.receive({
        jsonrpc: "2.0",
        method: "event",
        params: {
          type: "message.complete",
          session_id: "live",
          payload: { text: "Recovered output", status: "complete" },
        },
      });
      return normal(peer, method, params);
    }
    if (method === "prompt.submit") return { status: "queued" };
    return normal(peer, method, params);
  };
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "next", observer);
  const peer = await submitted();
  expect(observer.text).toHaveBeenCalledWith(expect.any(String), "Recovered output", "assistant");
  await agent.interrupt("saved");
  await event(peer, "message.complete", { text: "", status: "interrupted" });
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
  expect(peer.dispose).toHaveBeenCalledOnce();
});

it("retains native draft IDs until the first turn, then resumes in a separate execution process", async () => {
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
  const id = await agent.create();
  expect(id).toBe("stored");
  const observer = view();
  await agent.read(id, observer);
  expect(observer.text).not.toHaveBeenCalled();
  await agent.start(id, "first", observer);
  expect(mock.peers).toHaveLength(1);
  expect(mock.peers[0]?.request).not.toHaveBeenCalledWith("session.resume", expect.anything());
  await agent.start(id, "second", { ...view(), executionId: "second" });
  expect(mock.peers).toHaveLength(2);
  expect(mock.peers[1]?.request).toHaveBeenCalledWith("session.resume", {
    session_id: "stored",
  });
  expect(endpoints[0]?.token).not.toBe(endpoints[1]?.token);
  for (const [index, endpoint] of endpoints.entries()) {
    expect(endpoint.bind).toHaveBeenCalledWith(
      "hermes:stored",
      index === 0 ? "execution" : "second",
    );
    expect(endpoint.dispose).toHaveBeenCalledOnce();
    expect(JSON.parse(mock.peers[index]?.env.SWARMX_HERMES_MCP ?? "")).toEqual({
      swarmx: {
        command: "node",
        args: ["/bridge.js"],
        env: { SWARMX_MCP_TOKEN: endpoint.token },
      },
    });
  }
  expect(mock.peers[0]?.command).toBe("/native/python");
  expect(mock.peers[0]?.args).toEqual([expect.stringContaining("resources/hermes-native.py")]);
});

it("uses per-session model and effort settings and preserves native prompts", async () => {
  const agent = await open();
  await agent.start("saved", "hello", view(), {
    model: "provider:model",
    effort: "high",
    instructions: "Memory context",
  });
  const peer = mock.peers[0];
  expect(peer?.request).toHaveBeenCalledWith("config.set", {
    session_id: "live",
    key: "model",
    value: "provider:model",
    scope: "session",
  });
  expect(peer?.request).toHaveBeenCalledWith("config.set", {
    session_id: "live",
    key: "reasoning",
    value: "high",
    scope: "session",
  });
  expect(peer?.request).toHaveBeenCalledWith("prompt.submit", {
    session_id: "live",
    text: expect.stringContaining("hello\n\n<swarmx-memory-context>\nMemory context"),
  });
});

it("does not silently grant approval for an unsupported ACP edit mode", async () => {
  const agent = await open();
  await expect(agent.start("saved", "work", view(), { mode: "dont_ask" })).rejects.toThrow(
    "ACP edit modes are unsupported",
  );
  expect(mock.peers).toHaveLength(0);
});

it("preserves a native tool_error result as failed activity without failing the entire turn", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "Work", observer);
  const peer = await submitted();
  const result = { error: "file not found" };
  await event(peer, "tool.complete", {
    tool_id: "read",
    name: "read_file",
    args: { path: "missing" },
    result,
  });
  expect(observer.activity).toHaveBeenCalledWith({
    type: "tool",
    toolCallId: "read",
    status: "failed",
  });
  expect(observer.tool).toHaveBeenCalledWith(
    "read",
    "read_file",
    { path: "missing" },
    expect.objectContaining({ result }),
  );
  await event(peer, "message.complete", { text: "Handled missing file", status: "complete" });
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
});

it("retains interim text, reasoning, tools and the final tail without repeating streamed text", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "work", observer);
  const peer = await submitted();
  await event(peer, "message.delta", { text: "Checking" });
  await event(peer, "message.interim", { text: "Checking", already_streamed: true });
  await event(peer, "reasoning.delta", { text: "Reason" });
  await event(peer, "tool.start", { tool_id: "tool", name: "terminal", args: { command: "ls" } });
  await event(peer, "tool.complete", {
    tool_id: "tool",
    name: "terminal",
    args: { command: "ls" },
    result: { output: "file" },
  });
  await event(peer, "message.delta", { text: "An" });
  await event(peer, "message.complete", { text: "Answer", status: "complete" });
  await work;
  expect(vi.mocked(observer.text).mock.calls.map((call) => call[1])).toEqual([
    "Checking",
    "Reason",
    "An",
    "swer",
  ]);
  expect(observer.tool).toHaveBeenLastCalledWith(
    "tool",
    "terminal",
    { command: "ls" },
    expect.objectContaining({ result: { output: "file" } }),
  );
});

it("does not finish on interrupt acknowledgement and ignores other sessions' terminal events", async () => {
  mock.autoComplete = false;
  const agent = await open();
  let finished = false;
  const work = agent.start("saved", "work", view()).then((result) => {
    finished = true;
    return result;
  });
  const peer = await submitted();
  await agent.interrupt("saved");
  expect(finished).toBe(false);
  await event(peer, "message.complete", { text: "Other", status: "complete" }, "other");
  expect(finished).toBe(false);
  await event(peer, "message.complete", { text: "Stopped", status: "interrupted" });
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
});

it("closes only the cancelled runtime and leaves another session running", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const first = agent.start("one", "one", view());
  const p1 = await submitted();
  const second = agent.start("two", "two", view());
  const p2 = await submitted(1);
  await agent.interrupt("one");
  await event(p1, "message.complete", { text: "", status: "interrupted" });
  await first;
  expect(p2.dispose).not.toHaveBeenCalled();
  await event(p2, "message.complete", { text: "Two", status: "complete" });
  await second;
});

it("reports native errors and process loss instead of a completed turn", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const first = agent.start("saved", "work", view());
  const firstCheck = expect(first).rejects.toThrow("Provider unavailable");
  const peer = await submitted();
  await event(peer, "message.complete", {
    status: "error",
    text: "Partial",
    error: "Provider unavailable",
  });
  await firstCheck;
  const second = agent.start("saved", "retry", view());
  const secondCheck = expect(second).rejects.toThrow("Process lost");
  const next = await submitted(1);
  next.fail(new Error("Process lost"));
  await secondCheck;
});

it("rejects steering when Hermes does not accept it", async () => {
  const handler = mock.handler;
  mock.handler = async (peer, method, params) =>
    method === "session.steer" ? { status: "rejected" } : handler?.(peer, method, params);
  mock.autoComplete = false;
  const agent = await open();
  const work = agent.start("saved", "work", view());
  const peer = await submitted();
  await expect(agent.steer("saved", "guidance")).rejects.toThrow("rejected steering");
  await event(peer, "message.complete", { status: "complete", text: "Done" });
  await work;
});

it("finishes when accepted steering is consumed within the current turn", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const work = agent.start("saved", "work", view());
  const peer = await submitted();
  await agent.steer("saved", "check the citations");
  await event(peer, "tool.complete", {
    tool_id: "read",
    name: "read_file",
    args: { path: "citations.md" },
    result: { output: "References" },
  });
  await event(peer, "message.complete", { status: "complete", text: "Citations checked" });
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  expect(peer.dispose).toHaveBeenCalledOnce();
});

it.each(["complete", "error", "interrupted"])(
  "observes the native late-steer follow-up through its %s outcome",
  async (status) => {
    mock.autoComplete = false;
    const normal = mock.handler;
    const nativeDone = Promise.withResolvers<{ running: boolean }>();
    mock.handler = async (peer, method, params) =>
      method === "swarmx.session.wait" ? nativeDone.promise : normal?.(peer, method, params);
    const endpoint = { bind: vi.fn(), dispose: vi.fn() };
    const agent = await open({ registerMcp: () => endpoint });
    const observer = view();
    const work = agent.start("saved", "work", observer);
    const outcome = work.then(
      (result) => result,
      (error: unknown) => error,
    );
    const peer = await submitted();
    await agent.steer("saved", "check the citations");
    // Hermes emits the old idle before requeuing a steer left over from the final API call.
    await event(peer, "message.complete", { status: "complete", text: "Initial answer" });
    expect(peer.dispose).not.toHaveBeenCalled();
    expect(endpoint.dispose).not.toHaveBeenCalled();
    await expect(agent.steer("saved", "too late for that turn")).rejects.toThrow(
      "No running Hermes session",
    );
    await event(peer, "message.start", {});
    await agent.steer("saved", "include the publication year");
    if (status === "interrupted") await agent.interrupt("saved");
    await event(peer, "message.complete", {
      status,
      text: "Citations checked",
      error: "Follow-up failed",
    });
    expect(peer.dispose).not.toHaveBeenCalled();
    nativeDone.resolve({ running: false });
    expect(await outcome).toEqual(
      status === "error"
        ? new Error("Follow-up failed")
        : { stopReason: status === "interrupted" ? "cancelled" : "end_turn" },
    );
    expect(peer.request.mock.calls.filter(([method]) => method === "swarmx.session.wait")).toEqual([
      ["swarmx.session.wait", { session_id: "live" }],
    ]);
    if (status !== "error")
      expect(observer.text).toHaveBeenCalledWith(
        expect.any(String),
        "Citations checked",
        "assistant",
      );
    expect(peer.dispose).toHaveBeenCalledOnce();
    expect(endpoint.dispose).toHaveBeenCalledOnce();
  },
);

it("settles an in-flight steering acknowledgement before releasing the runtime", async () => {
  mock.autoComplete = false;
  const normal = mock.handler;
  const ack = Promise.withResolvers<{ status: string }>();
  mock.handler = async (peer, method, params) =>
    method === "session.steer" ? ack.promise : normal?.(peer, method, params);
  const agent = await open();
  const work = agent.start("saved", "work", view());
  const peer = await submitted();
  const steering = agent.steer("saved", "check the citations");
  await event(peer, "message.complete", { status: "complete", text: "Citations checked" });
  expect(peer.dispose).not.toHaveBeenCalled();
  ack.resolve({ status: "queued" });
  await steering;
  await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  expect(peer.dispose).toHaveBeenCalledOnce();
});

it("rejects a native execution that still reports running after its thread exits", async () => {
  const normal = mock.handler;
  mock.handler = async (peer, method, params) =>
    method === "swarmx.session.wait" ? { running: true } : normal?.(peer, method, params);
  const agent = await open();
  await expect(agent.start("saved", "work", view())).rejects.toThrow(
    "Hermes execution did not settle",
  );
  expect(mock.peers[0]?.dispose).toHaveBeenCalledOnce();
});

it("preserves every offered approval choice and never offers unavailable persistent approval", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const observer = view();
  vi.mocked(observer.interact).mockResolvedValue({ optionId: "session" });
  const work = agent.start("saved", "work", observer);
  const peer = await submitted();
  await event(peer, "approval.request", {
    request_id: "approval",
    command: "git status",
    choices: ["once", "session", "deny"],
  });
  expect(peer.request).toHaveBeenCalledWith("approval.respond", {
    session_id: "live",
    request_id: "approval",
    choice: "session",
  });
  const request = vi.mocked(observer.interact).mock.calls[0]?.[0];
  expect(request?.approval?.choices.map((choice) => choice.id)).toEqual([
    "once",
    "session",
    "deny",
  ]);
  await event(peer, "message.complete", { status: "complete", text: "Done" });
  await work;
});

it("aborts an expired prompt and does not send a late secret", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const observer = view();
  const answer = Promise.withResolvers<unknown>();
  vi.mocked(observer.interact).mockReturnValue(answer.promise);
  const work = agent.start("saved", "work", observer);
  const peer = await submitted();
  const request = event(peer, "secret.request", { request_id: "secret", prompt: "Credential" });
  await vi.waitFor(() => expect(observer.interact).toHaveBeenCalled());
  expect(vi.mocked(observer.interact).mock.calls[0]?.[0]).toMatchObject({
    sensitive: true,
    schema: { properties: { answer: { format: "password" } } },
  });
  const signal = vi.mocked(observer.interact).mock.calls[0]?.[1];
  await event(peer, "secret.expire", { request_id: "secret" });
  expect(signal?.aborted).toBe(true);
  answer.resolve({ answer: "late value" });
  await request;
  expect(peer.request).not.toHaveBeenCalledWith("secret.respond", expect.anything());
  await event(peer, "message.complete", { status: "complete", text: "Done" });
  await work;
});

it("preserves batched clarification, multiple selections and custom answers", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const observer = view();
  vi.mocked(observer.interact).mockResolvedValue({
    answer_0: ["A, B", "C"],
    other_0: "custom",
    other_1: "free text",
  });
  const work = agent.start("saved", "work", observer);
  const peer = await submitted();
  await event(peer, "clarify.request", {
    request_id: "questions",
    questions: [
      { qid: "first", question: "Select", choices: ["A, B", "C"], multi_select: true },
      { qid: "second", question: "Explain", choices: [], multi_select: false },
    ],
  });
  expect(vi.mocked(observer.interact).mock.calls[0]?.[0].schema).toMatchObject({
    properties: {
      answer_0: { type: "array", items: { enum: ["A, B", "C"] } },
      other_1: { type: "string", title: "Explain" },
    },
  });
  expect(peer.request).toHaveBeenCalledWith("clarify.respond", {
    session_id: "live",
    request_id: "questions",
    question_id: "first",
    answer: JSON.stringify(["A, B", "C", "custom"]),
  });
  expect(peer.request).toHaveBeenCalledWith("clarify.respond", {
    session_id: "live",
    request_id: "questions",
    question_id: "second",
    answer: "free text",
  });
  await event(peer, "message.complete", { status: "complete", text: "Done" });
  await work;
});

it("cancels all batched questions with the native empty response", async () => {
  mock.autoComplete = false;
  const agent = await open();
  const observer = view();
  const work = agent.start("saved", "work", observer);
  const peer = await submitted();
  await event(peer, "clarify.request", {
    request_id: "questions",
    questions: [{ qid: "one", question: "Choose", choices: ["A"] }],
  });
  expect(peer.request).toHaveBeenCalledWith("clarify.respond", {
    session_id: "live",
    request_id: "questions",
    answer: "",
  });
  await event(peer, "message.complete", { status: "complete", text: "Done" });
  await work;
});

it.each([true, false])("honors native model cost confirmation: %s", async (accept) => {
  const handler = mock.handler;
  mock.handler = async (peer, method, params) =>
    method === "config.set" && !params.confirm_expensive_model
      ? { confirm_required: true, confirm_message: "Use expensive model?" }
      : handler?.(peer, method, params);
  const agent = await open();
  const observer = view();
  vi.mocked(observer.interact).mockResolvedValue({ confirm: accept });
  const work = agent.start("saved", "work", observer, { model: "provider:expensive" });
  if (accept) await expect(work).resolves.toEqual({ stopReason: "end_turn" });
  else await expect(work).rejects.toThrow("model selection declined");
  expect(vi.mocked(observer.interact).mock.calls[0]?.[0].title).toBe("Use expensive model?");
  const calls = mock.peers[0]?.request.mock.calls ?? [];
  expect(
    calls.some(
      ([method, params]) => method === "config.set" && params.confirm_expensive_model === true,
    ),
  ).toBe(accept);
  expect(calls.some(([method]) => method === "prompt.submit")).toBe(accept);
});

it("cancels preparation without dispatching a prompt and revokes its credential", async () => {
  const handler = mock.handler;
  const resumed = Promise.withResolvers<void>();
  mock.handler = async (peer, method, params) => {
    if (method === "session.resume") await resumed.promise;
    return handler?.(peer, method, params);
  };
  const endpoint = { bind: vi.fn(), dispose: vi.fn() };
  const agent = await open({ registerMcp: () => endpoint });
  const work = agent.start("saved", "never dispatch", view());
  await vi.waitFor(() =>
    expect(mock.peers[0]?.request).toHaveBeenCalledWith("session.resume", expect.anything()),
  );
  await agent.interrupt("saved");
  resumed.resolve();
  await expect(work).resolves.toEqual({ stopReason: "cancelled" });
  expect(mock.peers[0]?.request).not.toHaveBeenCalledWith("prompt.submit", expect.anything());
  expect(endpoint.dispose).toHaveBeenCalledOnce();
});

it.each(["complete", "interrupted", "error"])(
  "waits for native title and cleanup work to settle after a %s terminal frame",
  async (status) => {
    mock.autoComplete = false;
    const agent = await open();
    const observer = view();
    const work = agent.start("saved", "Work", observer);
    const outcome =
      status === "error"
        ? expect(work).rejects.toThrow("Native failure")
        : expect(work).resolves.toEqual({
            stopReason: status === "interrupted" ? "cancelled" : "end_turn",
          });
    const peer = await submitted();
    await event(
      peer,
      "message.complete",
      { status, text: "Answer", error: "Native failure" },
      "live",
      false,
    );
    await event(peer, "session.info", { running: true });
    expect(peer.dispose).not.toHaveBeenCalled();
    await event(peer, "session.info", { running: false, title: "Native delayed title" });
    await outcome;
    expect(peer.dispose).toHaveBeenCalledOnce();
    expect(observer.raw).toHaveBeenCalledWith(
      expect.objectContaining({
        params: expect.objectContaining({
          payload: { running: false, title: "Native delayed title" },
        }),
      }),
    );
  },
);
