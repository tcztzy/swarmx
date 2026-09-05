import type { Options, SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import type { GatewayClientOptions } from "@openclaw/gateway-client";
import type { JSONRPCRequest } from "json-rpc-2.0";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadAgent, selectedAgent } from "../src/agent.js";
import type { Observer } from "../src/agents/types.js";

const state = vi.hoisted(() => ({
  imports: [] as string[],
  receive: undefined as ((message: JSONRPCRequest) => Promise<unknown>) | undefined,
  rpc: vi.fn<(method: string, params: object) => Promise<unknown>>(),
  nativeOptions: undefined as Options | undefined,
  gateway: undefined as GatewayClientOptions | undefined,
  gatewayRequest: vi.fn<(method: string, params: unknown) => Promise<unknown>>(),
  interrupted: vi.fn(),
  claudeMessages: [] as SDKMessage[],
}));
vi.mock("../src/agents/rpc-process.js", () => ({
  rpcProcess: (_command: string, _args: string[], _cwd: string, receive: typeof state.receive) => {
    state.receive = receive;
    return { request: state.rpc, notify: vi.fn(), dispose: vi.fn() };
  },
}));
vi.mock("@anthropic-ai/claude-agent-sdk", () => {
  state.imports.push("claude");
  return {
    listSessions: async () => [],
    getSessionMessages: async () => [],
    query: ({ options }: { options: Options }) => {
      state.nativeOptions = options;
      return Object.assign(
        (async function* () {
          yield* state.claudeMessages;
        })(),
        { close: vi.fn(), interrupt: state.interrupted },
      );
    },
  };
});
vi.mock("@openclaw/gateway-client", () => {
  state.imports.push("openclaw");
  return {
    GatewayClient: class {
      constructor(options: GatewayClientOptions) {
        state.gateway = options;
      }
      start() {
        state.gateway?.onHelloOk?.({} as never);
      }
      stop() {}
      async stopAndWait() {}
      request = state.gatewayRequest;
    },
  };
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.clearAllMocks();
});
const options = {
  cwd: process.cwd(),
  mcp: { url: "http://127.0.0.1:1234/mcp", headers: { authorization: "Bearer test" } },
};
function observer(): Observer {
  return {
    text: vi.fn(),
    tool: vi.fn(),
    raw: vi.fn(),
    interact: vi.fn(async () => ({ decision: "accept" })),
  };
}
async function event(method: string, params: object, id?: number) {
  return state.receive?.({ jsonrpc: "2.0", method, params, ...(id === undefined ? {} : { id }) });
}

describe("native Agents", () => {
  it("defaults to Codex, lazily loads only the selected native integration, and never falls back", async () => {
    state.rpc.mockResolvedValue({ data: [], nextCursor: null });
    expect(selectedAgent()).toBe("codex");
    expect(() => selectedAgent("dsh")).toThrow(/Unknown Agent/);
    const agent = await loadAgent("codex", options);
    expect(state.imports).toEqual([]);
    expect(state.rpc).toHaveBeenCalledWith("initialize", expect.any(Object));
    await agent.dispose();
    state.rpc.mockRejectedValueOnce(new Error("native unavailable"));
    await expect(loadAgent("codex", options)).rejects.toThrow("native unavailable");
    expect(state.imports).toEqual([]);
  });

  it("Codex uses native start, steer, approval, history and interrupt with native identifiers", async () => {
    state.rpc.mockImplementation(async (method) => {
      if (method === "thread/list") return { data: [], nextCursor: null };
      if (method === "thread/start") return { thread: { id: "thread-native" } };
      if (method === "thread/read")
        return {
          thread: {
            cwd: options.cwd,
            turns: [{ items: [{ type: "agentMessage", id: "answer-1", text: "history" }] }],
          },
        };
      if (method === "turn/start") return { turn: { id: "turn-native" } };
      return {};
    });
    const agent = await loadAgent("codex", options);
    const id = await agent.create();
    const view = observer();
    await agent.read(id, view);
    expect(view.text).not.toHaveBeenCalled();
    const pending = agent.start(id, "hello", view);
    await vi.waitFor(() =>
      expect(state.rpc).toHaveBeenCalledWith("turn/start", expect.any(Object)),
    );
    await event("item/agentMessage/delta", {
      threadId: "thread-native",
      itemId: "answer-2",
      delta: "live",
    });
    expect(view.text).toHaveBeenCalledWith("answer-2", "live", "assistant");
    await expect(
      event(
        "item/commandExecution/requestApproval",
        { threadId: "thread-native", reason: "Write result" },
        7,
      ),
    ).resolves.toEqual({ decision: "accept" });
    await agent.steer(id, "focus");
    expect(state.rpc).toHaveBeenCalledWith(
      "turn/steer",
      expect.objectContaining({ expectedTurnId: "turn-native" }),
    );
    await agent.interrupt(id);
    expect(state.rpc).toHaveBeenCalledWith("turn/interrupt", {
      threadId: "thread-native",
      turnId: "turn-native",
    });
    await event("turn/completed", {
      threadId: "thread-native",
      turn: { id: "turn-native", status: "interrupted", error: null },
    });
    await pending;
    await agent.read(id, view);
    expect(view.text).toHaveBeenCalledWith("answer-1", "history");
    await agent.dispose();
  });

  it("Codex Stop waits for a starting native turn instead of dropping cancellation", async () => {
    const acknowledged = Promise.withResolvers<unknown>();
    state.rpc.mockImplementation(async (method) => {
      if (method === "thread/list") return { data: [], nextCursor: null };
      if (method === "thread/start") return { thread: { id: "starting-thread" } };
      if (method === "turn/start") return acknowledged.promise;
      return {};
    });
    const agent = await loadAgent("codex", options);
    const id = await agent.create();
    const pending = agent.start(id, "hello", observer());
    const cancelling = agent.interrupt(id);
    acknowledged.resolve({ turn: { id: "starting-turn" } });
    await cancelling;
    expect(state.rpc).toHaveBeenCalledWith("turn/interrupt", {
      threadId: "starting-thread",
      turnId: "starting-turn",
    });
    await event("turn/completed", {
      threadId: "starting-thread",
      turn: { id: "starting-turn", status: "interrupted", error: null },
    });
    await pending;
    await agent.dispose();
  });

  it("Claude keeps SDK settings, native events, MCP and tool interaction callbacks", async () => {
    state.claudeMessages = [
      {
        type: "assistant",
        uuid: "row-1",
        session_id: "native",
        parent_tool_use_id: "parent-tool",
        message: { id: "message-native", content: [{ type: "text", text: "hello" }] },
      } as SDKMessage,
    ];
    const agent = await loadAgent("claude", options);
    const id = await agent.create();
    const view = observer();
    await agent.start(id, "hello", view);
    expect(state.nativeOptions).toMatchObject({
      settingSources: ["user", "project", "local"],
      includePartialMessages: true,
      sessionId: id.slice("claude:".length),
      mcpServers: { swarmx: { type: "http", ...options.mcp } },
    });
    expect(view.raw).toHaveBeenCalledWith(state.claudeMessages[0]);
    vi.mocked(view.interact).mockResolvedValue({ allow: true });
    await expect(
      state.nativeOptions?.canUseTool?.(
        "Bash",
        { command: "pwd" },
        {
          toolUseID: "tool-native",
          signal: new AbortController().signal,
        },
      ),
    ).resolves.toMatchObject({ behavior: "allow", updatedInput: { command: "pwd" } });
    await agent.dispose();
  });

  it("Claude consumes queued native output until the SDK reports idle", async () => {
    const reply = (text: string) =>
      ({
        type: "assistant",
        uuid: text,
        session_id: "native",
        parent_tool_use_id: null,
        message: { id: text, content: [{ type: "text", text }] },
      }) as SDKMessage;
    state.claudeMessages = [
      reply("first"),
      { type: "result", subtype: "success", is_error: false, result: "first" } as SDKMessage,
      reply("steered"),
      { type: "result", subtype: "success", is_error: false, result: "steered" } as SDKMessage,
      { type: "system", subtype: "session_state_changed", state: "idle" } as SDKMessage,
    ];
    const agent = await loadAgent("claude", options);
    const view = observer();
    await agent.start(await agent.create(), "hello", view);
    expect(view.text).toHaveBeenCalledWith("steered:0", "steered", "assistant");
    await agent.dispose();
  });

  it("Hermes uses its native gateway with durable session ids, approvals and Stop", async () => {
    vi.stubEnv("SWARMX_HERMES_PYTHON", "/installed/hermes/python");
    state.rpc.mockImplementation(async (method) => {
      if (method === "session.create") return { session_id: "live", stored_session_id: "stored" };
      return { sessions: [] };
    });
    const agent = await loadAgent("hermes", options);
    const id = await agent.create();
    expect(id).toBe("hermes:stored");
    const view = observer();
    vi.mocked(view.interact).mockResolvedValue({ choice: "once" });
    const pending = agent.start(id, "hello", view);
    await vi.waitFor(() =>
      expect(state.rpc).toHaveBeenCalledWith("prompt.submit", {
        session_id: "live",
        text: "hello",
      }),
    );
    await event("event", {
      type: "approval.request",
      session_id: "live",
      payload: { request_id: "approval", command: "pwd" },
    });
    expect(state.rpc).toHaveBeenCalledWith("approval.respond", {
      session_id: "live",
      request_id: "approval",
      choice: "once",
    });
    await agent.steer(id, "focus");
    await agent.interrupt(id);
    expect(state.rpc).toHaveBeenCalledWith("session.interrupt", { session_id: "live" });
    await event("event", {
      type: "message.complete",
      session_id: "live",
      payload: { text: "done" },
    });
    await pending;
    await agent.dispose();
  });

  it("OpenClaw uses the official native Gateway client, streams deltas and aborts runs", async () => {
    vi.stubEnv("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789");
    vi.stubEnv("OPENCLAW_GATEWAY_TOKEN", "test");
    state.gatewayRequest.mockImplementation(async (method) =>
      method === "sessions.create" ? { key: "native:key" } : { sessions: [] },
    );
    const agent = await loadAgent("openclaw", options);
    const id = await agent.create();
    const view = observer();
    const pending = agent.start(id, "hello", view);
    const request = state.gatewayRequest.mock.calls.find(
      ([method]) => method === "chat.send",
    )?.[1] as { idempotencyKey: string };
    const chat = (payload: object) =>
      state.gateway?.onEvent?.({ type: "event", event: "chat", payload });
    chat({
      sessionKey: "native:key",
      runId: request.idempotencyKey,
      state: "delta",
      deltaText: "hello",
      seq: 1,
    });
    expect(view.text).toHaveBeenCalledWith(request.idempotencyKey, "hello");
    await agent.interrupt(id);
    expect(state.gatewayRequest).toHaveBeenCalledWith("chat.abort", { sessionKey: "native:key" });
    chat({ sessionKey: "native:key", runId: request.idempotencyKey, state: "aborted", seq: 2 });
    await pending;
    await agent.dispose();
  });
});
