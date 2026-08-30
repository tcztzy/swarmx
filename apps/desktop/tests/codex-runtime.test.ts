import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { CodexRpcError } from "../src/runtime/codex/connection.js";
import { CodexConversationRuntime, type CodexRpcClient } from "../src/runtime/codex/runtime.js";
import type { ApprovalResponse, RuntimeEvent, WorkspaceScope } from "../src/runtime/index.js";

class FakeCodexRpc implements CodexRpcClient {
  readonly calls: Array<{ method: string; params: Record<string, unknown> }> = [];
  readonly notifications = new Map<string, (params: unknown) => void>();
  readonly requests = new Map<
    string,
    (params: Record<string, unknown>, requestId: string | number) => Promise<unknown>
  >();
  readonly responses = new Map<string, unknown[]>();
  dispose = vi.fn(async () => undefined);

  async request(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    this.calls.push({ method, params });
    const queue = this.responses.get(method);
    if (queue === undefined || queue.length === 0) {
      if (method === "thread/list" && params.archived === true) {
        return { data: [], nextCursor: null };
      }
      throw new Error(`No response for ${method}`);
    }
    const response = queue.shift();
    if (response instanceof Error) throw response;
    return response;
  }

  onNotification(method: string, handler: (params: unknown) => void): () => void {
    this.notifications.set(method, handler);
    return () => this.notifications.delete(method);
  }

  onRequest(
    method: string,
    handler: (params: Record<string, unknown>, requestId: string | number) => Promise<unknown>,
  ): () => void {
    this.requests.set(method, handler);
    return () => this.requests.delete(method);
  }

  respond(method: string, ...values: unknown[]): void {
    this.responses.set(method, [...values]);
  }
}

const workspace: WorkspaceScope = {
  id: "workspace-1",
  label: "swarmx",
  root: process.cwd(),
  token: "host-token",
};
const allSourceKinds = [
  "cli",
  "vscode",
  "exec",
  "appServer",
  "subAgent",
  "subAgentReview",
  "subAgentCompact",
  "subAgentThreadSpawn",
  "subAgentOther",
  "unknown",
];

function thread(turns: unknown[] = [], historyMode: "legacy" | "paginated" = "legacy") {
  return {
    id: "thread-1",
    cwd: workspace.root,
    name: "Codex thread",
    preview: "hello",
    archived: false,
    createdAt: 1,
    updatedAt: 2,
    historyMode,
    turns,
  };
}

function namedThread(id: string, updatedAt: number) {
  return { ...thread(), id, updatedAt };
}

async function within<Value>(promise: Promise<Value>, label: string): Promise<Value> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timeout = setTimeout(() => reject(new Error(`Timed out waiting for ${label}.`)), 50);
  });
  try {
    return await Promise.race([promise, deadline]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
}

const invalidElicitationProperties: ReadonlyArray<{
  name: string;
  schema: Record<string, unknown>;
}> = [
  { name: "fractional integer default", schema: { type: "integer", default: 1.5 } },
  {
    name: "default below minimum",
    schema: { type: "number", minimum: 2, default: 1 },
  },
  {
    name: "default above maximum",
    schema: { type: "number", maximum: 2, default: 3 },
  },
  {
    name: "default below minLength",
    schema: { type: "string", minLength: 2, default: "a" },
  },
  {
    name: "default above maxLength",
    schema: { type: "string", maxLength: 1, default: "ab" },
  },
  {
    name: "default violating format",
    schema: { type: "string", format: "email", default: "not-an-email" },
  },
  {
    name: "enum default outside options",
    schema: { type: "string", enum: ["fast", "safe"], default: "unknown" },
  },
  {
    name: "oneOf default outside options",
    schema: {
      type: "string",
      oneOf: [
        { const: "fast", title: "Fast" },
        { const: "safe", title: "Safe" },
      ],
      default: "unknown",
    },
  },
  {
    name: "array default outside enum options",
    schema: {
      type: "array",
      items: { type: "string", enum: ["a", "b"] },
      default: ["c"],
    },
  },
  {
    name: "array default outside anyOf options",
    schema: {
      type: "array",
      items: {
        anyOf: [
          { const: "a", title: "A" },
          { const: "b", title: "B" },
        ],
      },
      default: ["c"],
    },
  },
  {
    name: "array default below minItems",
    schema: {
      type: "array",
      minItems: 2,
      items: { type: "string", enum: ["a", "b"] },
      default: ["a"],
    },
  },
  {
    name: "array default above maxItems",
    schema: {
      type: "array",
      maxItems: 1,
      items: { type: "string", enum: ["a", "b"] },
      default: ["a", "b"],
    },
  },
  {
    name: "duplicate enum values",
    schema: { type: "string", enum: ["fast", "fast"] },
  },
  {
    name: "duplicate array enum values",
    schema: { type: "array", items: { type: "string", enum: ["a", "a"] } },
  },
  {
    name: "duplicate oneOf values",
    schema: {
      type: "string",
      oneOf: [
        { const: "fast", title: "Fast" },
        { const: "fast", title: "Still fast" },
      ],
    },
  },
  {
    name: "duplicate anyOf values",
    schema: {
      type: "array",
      items: {
        anyOf: [
          { const: "a", title: "A" },
          { const: "a", title: "Still A" },
        ],
      },
    },
  },
  {
    name: "enumNames length mismatch",
    schema: {
      type: "string",
      enum: ["fast", "safe"],
      enumNames: ["Fast"],
    },
  },
];

describe("Codex conversation adapter", () => {
  it("keeps new threads legacy unless experimental paginated history is explicit", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/start", { thread: thread() });
    const runtime = new CodexConversationRuntime(rpc, { paginatedHistory: false });

    await runtime.create({ workspace });

    expect(rpc.calls).toEqual([
      {
        method: "thread/start",
        params: { cwd: workspace.root, approvalPolicy: "on-request" },
      },
    ]);

    const paginatedRpc = new FakeCodexRpc();
    paginatedRpc.respond("thread/start", { thread: thread([], "paginated") });
    const paginated = new CodexConversationRuntime(paginatedRpc, { paginatedHistory: true });
    await paginated.create({ workspace });
    expect(paginatedRpc.calls).toEqual([
      {
        method: "thread/start",
        params: {
          cwd: workspace.root,
          approvalPolicy: "on-request",
          historyMode: "paginated",
        },
      },
    ]);
  });

  it("tags an unmaterialized member before publishing its exact handle", async () => {
    const rpc = new FakeCodexRpc();
    const memberId = "80000000-0000-4000-8000-000000000001";
    rpc.respond("thread/start", {
      thread: { ...thread(), threadSource: `swarmx-member:${memberId}` },
    });
    const runtime = new CodexConversationRuntime(rpc, { paginatedHistory: true });

    await expect(runtime.createProvisionedMember({ workspace }, memberId)).resolves.toMatchObject({
      conversationId: "codex:thread-1",
    });
    expect(rpc.calls[0]).toEqual({
      method: "thread/start",
      params: {
        cwd: workspace.root,
        approvalPolicy: "on-request",
        historyMode: "paginated",
        threadSource: `swarmx-member:${memberId}`,
      },
    });
  });

  it("deletes only the exact tagged blank provisioning Thread and treats deletion as idempotent", async () => {
    const rpc = new FakeCodexRpc();
    const memberId = "80000000-0000-4000-8000-000000000002";
    rpc.respond(
      "thread/read",
      {
        thread: { ...thread(), threadSource: `swarmx-member:${memberId}` },
      },
      new CodexRpcError("thread not loaded: thread-1", -32600),
    );
    rpc.respond("thread/delete", {});
    const runtime = new CodexConversationRuntime(rpc);

    await expect(
      runtime.retireProvisionedMember("codex:thread-1", memberId),
    ).resolves.toBeUndefined();
    await expect(
      runtime.retireProvisionedMember("codex:thread-1", memberId),
    ).resolves.toBeUndefined();
    expect(rpc.calls.filter((call) => call.method === "thread/delete")).toEqual([
      { method: "thread/delete", params: { threadId: "thread-1" } },
    ]);
  });

  it("refuses to delete an unmaterialized Thread without its exact provisioning tag", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/read", { thread: thread() });
    const runtime = new CodexConversationRuntime(rpc);

    await expect(
      runtime.retireProvisionedMember("codex:thread-1", "80000000-0000-4000-8000-000000000003"),
    ).rejects.toThrow("without its exact provisioning identity");
    expect(rpc.calls.some((call) => call.method === "thread/delete")).toBe(false);
  });

  it("lets the archived Thread list override a stale false read field", async () => {
    const rpc = new FakeCodexRpc();
    const stored = thread();
    rpc.respond("thread/read", { thread: stored });
    rpc.respond("thread/list", { data: [stored], nextCursor: null });
    const runtime = new CodexConversationRuntime(rpc);

    await expect(runtime.read("codex:thread-1")).resolves.toMatchObject({ archived: true });
    expect(rpc.calls).toContainEqual({
      method: "thread/list",
      params: {
        limit: 100,
        archived: true,
        sortKey: "updated_at",
        sortDirection: "desc",
        sourceKinds: allSourceKinds,
      },
    });
  });

  it("canonicalizes a native Thread workspace alias to the Host workspace identity", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-workspace-"));
    const canonical = join(root, "canonical");
    const alias = join(root, "alias");
    mkdirSync(canonical);
    symlinkSync(canonical, alias);
    const scoped: WorkspaceScope = {
      id: "canonical-workspace",
      label: "canonical",
      root: realpathSync(canonical),
      token: "host-token",
    };
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/start", { thread: { ...thread(), cwd: alias } });
    const runtime = new CodexConversationRuntime(rpc);
    try {
      await expect(runtime.create({ workspace: scoped })).resolves.toMatchObject({
        workspace: { id: "canonical-workspace", label: "canonical" },
      });
    } finally {
      await runtime.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects a created Thread whose canonical workspace differs from the request", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-foreign-create-"));
    const requestedRoot = join(root, "requested");
    const foreignRoot = join(root, "foreign");
    mkdirSync(requestedRoot);
    mkdirSync(foreignRoot);
    const scoped: WorkspaceScope = {
      id: "requested-workspace",
      label: "requested",
      root: requestedRoot,
      token: "host-token",
    };
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/start", { thread: { ...thread(), cwd: foreignRoot } });
    const runtime = new CodexConversationRuntime(rpc);
    try {
      await expect(runtime.create({ workspace: scoped })).rejects.toThrow(
        /workspace.*does not match/iu,
      );
    } finally {
      await runtime.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("pages native summaries and keeps an unmaterialized created thread visible until start", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/start", { thread: namedThread("empty", 30) });
    rpc.respond(
      "thread/list",
      { data: [namedThread("newer", 20)], nextCursor: "page-2" },
      { data: [namedThread("older", 10)], nextCursor: null },
      { data: [], nextCursor: null },
    );
    rpc.respond("turn/start", { turn: { id: "turn-1" } });
    const runtime = new CodexConversationRuntime(rpc);

    await runtime.create({ workspace });
    await expect(runtime.list()).resolves.toMatchObject([
      { conversationId: "codex:empty" },
      { conversationId: "codex:newer" },
      { conversationId: "codex:older" },
    ]);
    expect(rpc.calls.filter((call) => call.method === "thread/list")).toEqual([
      {
        method: "thread/list",
        params: {
          limit: 100,
          archived: false,
          sortKey: "updated_at",
          sortDirection: "desc",
          sourceKinds: ["cli", "vscode", "appServer"],
        },
      },
      {
        method: "thread/list",
        params: {
          limit: 100,
          archived: false,
          sortKey: "updated_at",
          sortDirection: "desc",
          sourceKinds: ["cli", "vscode", "appServer"],
          cursor: "page-2",
        },
      },
    ]);

    await runtime.start({ conversationId: "codex:empty", text: "materialize" });
    await expect(runtime.list()).resolves.toEqual([]);
  });

  it("fails closed when thread pagination keeps advancing without bounded results", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond(
      "thread/list",
      ...Array.from({ length: 10 }, (_, index) => ({
        data: [],
        nextCursor: `page-${String(index + 1)}`,
      })),
    );
    const runtime = new CodexConversationRuntime(rpc);

    await expect(runtime.list()).rejects.toThrow("page limit");
    expect(rpc.calls.filter((call) => call.method === "thread/list")).toHaveLength(10);
  });

  it("resumes an unknown stored thread once before starting turns", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/resume", { thread: thread() });
    rpc.respond("turn/start", { turn: { id: "turn-1" } }, { turn: { id: "turn-2" } });
    const runtime = new CodexConversationRuntime(rpc);

    await runtime.start({ conversationId: "codex:thread-1", text: "first" });
    await runtime.start({ conversationId: "codex:thread-1", text: "second" });

    expect(rpc.calls.map((call) => call.method)).toEqual([
      "thread/resume",
      "turn/start",
      "turn/start",
    ]);
  });

  it("shares one resume across concurrent starts for the same stored thread", async () => {
    const rpc = new FakeCodexRpc();
    let releaseResume: ((value: unknown) => void) | undefined;
    const resume = new Promise<unknown>((resolve) => {
      releaseResume = resolve;
    });
    rpc.respond("thread/resume", resume);
    rpc.respond("turn/start", { turn: { id: "turn-1" } }, { turn: { id: "turn-2" } });
    const runtime = new CodexConversationRuntime(rpc);

    const first = runtime.start({ conversationId: "codex:thread-1", text: "first" });
    const second = runtime.start({ conversationId: "codex:thread-1", text: "second" });
    await vi.waitFor(() =>
      expect(rpc.calls.filter((call) => call.method === "thread/resume")).toHaveLength(1),
    );
    releaseResume?.({ thread: thread() });

    await expect(Promise.all([first, second])).resolves.toHaveLength(2);
    expect(rpc.calls.filter((call) => call.method === "thread/resume")).toHaveLength(1);
  });

  it("maps native list/create/read/start/steer/interrupt/archive without leaking native ids", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/list", { data: [thread()], nextCursor: null });
    rpc.respond("thread/start", { thread: thread() });
    rpc.respond("thread/read", {
      thread: thread([
        {
          id: "turn-1",
          status: "completed",
          startedAt: 3,
          items: [
            {
              id: "user-1",
              type: "userMessage",
              content: [{ type: "text", text: "hello" }],
            },
            { id: "agent-1", type: "agentMessage", text: "hi" },
          ],
        },
      ]),
    });
    rpc.respond("turn/start", { turn: { id: "turn-2", status: "inProgress", items: [] } });
    rpc.respond("turn/steer", {});
    rpc.respond("turn/interrupt", {});
    rpc.respond("thread/archive", {});
    const runtime = new CodexConversationRuntime(rpc, { paginatedHistory: true });

    expect(await runtime.list()).toEqual([
      expect.objectContaining({ runtime: "codex", conversationId: "codex:thread-1" }),
    ]);
    expect(await runtime.create({ workspace })).toMatchObject({
      conversationId: "codex:thread-1",
      workspace: { id: "workspace-1", label: "swarmx" },
    });
    expect(await runtime.read("codex:thread-1")).toMatchObject({
      conversationId: "codex:thread-1",
      turns: [
        {
          id: "codex:turn-1",
          items: [
            { type: "user_message", id: "codex:user-1", text: "hello" },
            { type: "assistant_message", id: "codex:agent-1", text: "hi" },
          ],
        },
      ],
    });
    await expect(
      runtime.start({ conversationId: "codex:thread-1", text: "next" }),
    ).resolves.toEqual({ turnId: "codex:turn-2" });
    await runtime.steer({
      conversationId: "codex:thread-1",
      turnId: "codex:turn-2",
      text: "more",
    });
    await runtime.interrupt({ conversationId: "codex:thread-1", turnId: "codex:turn-2" });
    await runtime.archive("codex:thread-1");

    expect(rpc.calls).toContainEqual({
      method: "thread/start",
      params: {
        cwd: workspace.root,
        approvalPolicy: "on-request",
        historyMode: "paginated",
      },
    });
    expect(rpc.calls).toContainEqual({
      method: "turn/start",
      params: {
        threadId: "thread-1",
        input: [{ type: "text", text: "next" }],
        approvalPolicy: "on-request",
      },
    });
  });

  it("hydrates paginated thread history through stable bounded thread/read", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/read", {
      thread: thread(
        [
          { id: "turn-1", status: "completed", items: [] },
          { id: "turn-2", status: "failed", items: [] },
        ],
        "paginated",
      ),
    });
    const runtime = new CodexConversationRuntime(rpc);

    await expect(runtime.read("codex:thread-1")).resolves.toMatchObject({
      turns: [
        { id: "codex:turn-1", status: "completed" },
        { id: "codex:turn-2", status: "failed" },
      ],
    });
    expect(rpc.calls).toEqual([
      {
        method: "thread/read",
        params: { threadId: "thread-1", includeTurns: true },
      },
      {
        method: "thread/list",
        params: {
          limit: 100,
          archived: true,
          sortKey: "updated_at",
          sortDirection: "desc",
          sourceKinds: allSourceKinds,
        },
      },
    ]);
  });

  it("marks assistant content from a running native read as provisional", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/read", {
      thread: thread([
        {
          id: "turn-running",
          status: "inProgress",
          startedAt: 3,
          items: [{ id: "agent-running", type: "agentMessage", text: "partial" }],
        },
      ]),
    });
    const runtime = new CodexConversationRuntime(rpc);

    await expect(runtime.read("codex:thread-1")).resolves.toMatchObject({
      turns: [
        {
          status: "running",
          items: [
            {
              id: "codex:agent-running",
              text: "partial",
              provisional: true,
            },
          ],
        },
      ],
    });
  });

  it("reads a newly created unmaterialized thread as empty metadata", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond(
      "thread/read",
      new CodexRpcError(
        "thread thread-1 is not materialized yet; includeTurns is unavailable before first user message",
        -32600,
      ),
      { thread: thread() },
    );
    const runtime = new CodexConversationRuntime(rpc);

    await expect(runtime.read("codex:thread-1")).resolves.toMatchObject({ turns: [] });
    expect(rpc.calls).toEqual([
      {
        method: "thread/read",
        params: { threadId: "thread-1", includeTurns: true },
      },
      {
        method: "thread/read",
        params: { threadId: "thread-1", includeTurns: false },
      },
      {
        method: "thread/list",
        params: {
          limit: 100,
          archived: true,
          sortKey: "updated_at",
          sortDirection: "desc",
          sourceKinds: allSourceKinds,
        },
      },
    ]);
  });

  it("does not reinterpret a similar or differently coded thread/read failure", async () => {
    for (const failure of [
      new Error(
        "thread thread-1 is not materialized yet; includeTurns is unavailable before first user message",
      ),
      new CodexRpcError(
        "thread thread-1 is not materialized yet; includeTurns is unavailable before first user message",
        -32601,
      ),
      new CodexRpcError(
        "thread another-thread is not materialized yet; includeTurns is unavailable before first user message",
        -32600,
      ),
    ]) {
      const rpc = new FakeCodexRpc();
      rpc.respond("thread/read", failure);
      const runtime = new CodexConversationRuntime(rpc);

      await expect(runtime.read("codex:thread-1")).rejects.toBe(failure);
      expect(rpc.calls).toEqual([
        {
          method: "thread/read",
          params: { threadId: "thread-1", includeTurns: true },
        },
      ]);
    }
  });

  it("reverts the latest terminal paginated turn and starts its replacement in the same thread", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/read", {
      thread: thread(
        [
          { id: "turn-1", status: "completed", items: [] },
          { id: "turn-2", status: "failed", items: [] },
        ],
        "paginated",
      ),
    });
    rpc.respond("thread/resume", { thread: thread([], "paginated") });
    rpc.respond("thread/revert", { thread: thread([], "paginated") });
    rpc.respond("turn/start", { turn: { id: "turn-3", status: "inProgress", items: [] } });
    const runtime = new CodexConversationRuntime(rpc);

    await expect(
      runtime.revise({
        conversationId: "codex:thread-1",
        beforeTurnId: "codex:turn-2",
        text: "revised",
      }),
    ).resolves.toMatchObject({ conversationId: "codex:thread-1" });
    expect(rpc.calls).toContainEqual({
      method: "thread/revert",
      params: { threadId: "thread-1", beforeTurnId: "turn-2" },
    });
    expect(rpc.calls).toContainEqual({
      method: "turn/start",
      params: {
        threadId: "thread-1",
        input: [{ type: "text", text: "revised" }],
        approvalPolicy: "on-request",
      },
    });
  });

  it("branches revisions of older or legacy turns instead of pretending they were reverted", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/read", {
      thread: thread([
        { id: "turn-1", status: "completed", items: [] },
        { id: "turn-2", status: "completed", items: [] },
      ]),
    });
    rpc.respond("thread/fork", { thread: { ...thread(), id: "fresh" } });
    rpc.respond("turn/start", { turn: { id: "child-turn", status: "inProgress", items: [] } });
    const runtime = new CodexConversationRuntime(rpc);

    await expect(
      runtime.revise({
        conversationId: "codex:thread-1",
        beforeTurnId: "codex:turn-1",
        text: "replacement",
      }),
    ).resolves.toMatchObject({ conversationId: "codex:fresh" });
    expect(rpc.calls.some((call) => call.method === "thread/revert")).toBe(false);
    expect(rpc.calls).toContainEqual({
      method: "thread/fork",
      params: { threadId: "thread-1", beforeTurnId: "turn-1", ephemeral: false },
    });
    expect(rpc.calls).toContainEqual({
      method: "turn/start",
      params: {
        threadId: "fresh",
        input: [{ type: "text", text: "replacement" }],
        approvalPolicy: "on-request",
      },
    });
  });

  it("does not silently fork when native revert or replacement start fails", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond(
      "thread/read",
      {
        thread: thread([{ id: "turn-1", status: "completed", items: [] }], "paginated"),
      },
      {
        thread: thread([{ id: "turn-1", status: "completed", items: [] }], "paginated"),
      },
    );
    rpc.respond("thread/revert", new Error("method unavailable"), {
      thread: thread([], "paginated"),
    });
    rpc.respond("thread/resume", { thread: thread([], "paginated") });
    rpc.respond("turn/start", new Error("model unavailable"));
    const runtime = new CodexConversationRuntime(rpc);
    const request = {
      conversationId: "codex:thread-1",
      beforeTurnId: "codex:turn-1",
      text: "replacement",
    };

    await expect(runtime.revise(request)).rejects.toThrow("method unavailable");
    await expect(runtime.revise(request)).rejects.toThrow(
      "history was reverted, but the replacement turn could not start",
    );
    expect(rpc.calls.some((call) => call.method === "thread/fork")).toBe(false);
  });

  it("uses the exact native before-turn fork boundary for first and later turns", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond(
      "thread/read",
      { thread: thread([{ id: "turn-1", status: "completed", items: [] }]) },
      {
        thread: thread([
          { id: "turn-1", status: "completed", items: [] },
          { id: "turn-2", status: "completed", items: [] },
        ]),
      },
    );
    rpc.respond(
      "thread/fork",
      { thread: { ...thread(), id: "first-fork" } },
      { thread: { ...thread(), id: "later-fork" } },
    );
    const runtime = new CodexConversationRuntime(rpc);

    await expect(
      runtime.fork({ conversationId: "codex:thread-1", beforeTurnId: "codex:turn-1" }),
    ).resolves.toMatchObject({ conversationId: "codex:first-fork" });
    await expect(
      runtime.fork({ conversationId: "codex:thread-1", beforeTurnId: "codex:turn-2" }),
    ).resolves.toMatchObject({ conversationId: "codex:later-fork" });
    expect(rpc.calls.filter((call) => call.method === "thread/fork")).toEqual([
      {
        method: "thread/fork",
        params: { threadId: "thread-1", beforeTurnId: "turn-1", ephemeral: false },
      },
      {
        method: "thread/fork",
        params: { threadId: "thread-1", beforeTurnId: "turn-2", ephemeral: false },
      },
    ]);
    expect(rpc.calls.some((call) => call.method === "thread/start")).toBe(false);
  });

  it("rejects a forked Thread whose canonical workspace differs from its source", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-foreign-fork-"));
    const sourceRoot = join(root, "source");
    const foreignRoot = join(root, "foreign");
    mkdirSync(sourceRoot);
    mkdirSync(foreignRoot);
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/read", {
      thread: {
        ...thread([{ id: "turn-1", status: "completed", items: [] }]),
        cwd: sourceRoot,
      },
    });
    rpc.respond("thread/fork", {
      thread: { ...thread(), id: "foreign-fork", cwd: foreignRoot },
    });
    const runtime = new CodexConversationRuntime(rpc);
    try {
      await expect(
        runtime.fork({ conversationId: "codex:thread-1", beforeTurnId: "codex:turn-1" }),
      ).rejects.toThrow(/workspace.*does not match/iu);
    } finally {
      await runtime.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("does not replace a rejected exact fork boundary with a fresh thread", async () => {
    const rpc = new FakeCodexRpc();
    rpc.respond("thread/read", {
      thread: thread([{ id: "turn-1", status: "completed", items: [] }]),
    });
    rpc.respond("thread/fork", new Error("beforeTurnId unavailable"));
    const runtime = new CodexConversationRuntime(rpc);

    await expect(
      runtime.fork({ conversationId: "codex:thread-1", beforeTurnId: "codex:turn-1" }),
    ).rejects.toThrow("beforeTurnId unavailable");
    expect(rpc.calls.some((call) => call.method === "thread/start")).toBe(false);
  });

  it("projects ordered notifications and resolves scoped command approvals", async () => {
    const rpc = new FakeCodexRpc();
    const runtime = new CodexConversationRuntime(rpc);
    const events: RuntimeEvent[] = [];
    runtime.subscribe((event) => events.push(event));

    rpc.notifications.get("item/agentMessage/delta")?.({
      threadId: "thread-1",
      turnId: "turn-1",
      itemId: "item-1",
      delta: "hello",
    });
    rpc.notifications.get("turn/completed")?.({
      threadId: "thread-1",
      turn: { id: "turn-1", status: "completed", items: [] },
    });
    expect(events).toMatchObject([
      { seq: 1, type: "item_delta", conversationId: "codex:thread-1" },
      { seq: 2, type: "turn_status", status: "completed" },
    ]);

    const approval = rpc.requests.get("item/commandExecution/requestApproval")?.(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: "item-1",
        command: "pnpm test",
        reason: "Run tests",
      },
      "rpc-1",
    );
    await vi.waitFor(() => expect(events.at(-1)?.type).toBe("approval_requested"));
    const request = events.at(-1);
    if (request?.type !== "approval_requested") throw new Error("missing approval event");
    rpc.respond("thread/read", { thread: thread() });
    await expect(runtime.read("codex:thread-1")).resolves.toMatchObject({
      approvals: [expect.objectContaining({ approvalId: request.approvalId })],
    });
    const response: ApprovalResponse = { ...request, decision: "accept" };
    await runtime.respondToApproval(response);
    await expect(approval).resolves.toEqual({ decision: "accept" });
  });

  it("clears the exact server-resolved approval and honors available command decisions", async () => {
    const rpc = new FakeCodexRpc();
    const runtime = new CodexConversationRuntime(rpc);
    const events: RuntimeEvent[] = [];
    runtime.subscribe((event) => events.push(event));

    const approval = rpc.requests.get("item/commandExecution/requestApproval")?.(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: "item-1",
        command: "pnpm test",
        availableDecisions: ["acceptForSession", "decline", { unsupported: true }],
      },
      "rpc-resolved",
    );
    await vi.waitFor(() => expect(events.at(-1)?.type).toBe("approval_requested"));
    expect(events.at(-1)).toMatchObject({
      choices: ["accept_for_session", "decline"],
    });

    rpc.notifications.get("serverRequest/resolved")?.({
      threadId: "thread-1",
      requestId: "rpc-resolved",
    });
    await expect(approval).rejects.toThrow("resolved by Codex App Server");
    expect(events.at(-1)).toMatchObject({
      type: "approval_resolved",
      conversationId: "codex:thread-1",
      approvalId: "codex:item/commandExecution/requestApproval:rpc-resolved",
    });
    rpc.respond("thread/read", { thread: thread() });
    await expect(runtime.read("codex:thread-1")).resolves.toMatchObject({ approvals: [] });
  });

  it("clears descendant state when native archive cascades", async () => {
    const rpc = new FakeCodexRpc();
    const runtime = new CodexConversationRuntime(rpc);
    const events: RuntimeEvent[] = [];
    runtime.subscribe((event) => events.push(event));
    const approval = rpc.requests.get("item/fileChange/requestApproval")?.(
      {
        threadId: "child-thread",
        turnId: "turn-1",
        itemId: "item-1",
        reason: "Apply patch",
      },
      "rpc-child",
    );
    await vi.waitFor(() => expect(events.at(-1)?.type).toBe("approval_requested"));

    rpc.notifications.get("thread/archived")?.({ threadId: "child-thread" });

    await expect(approval).rejects.toThrow("archived");
    expect(events.at(-1)).toMatchObject({
      type: "approval_resolved",
      conversationId: "codex:child-thread",
    });
  });

  it("maps structured user input and MCP elicitation responses", async () => {
    const rpc = new FakeCodexRpc();
    const runtime = new CodexConversationRuntime(rpc);
    const events: RuntimeEvent[] = [];
    runtime.subscribe((event) => events.push(event));

    const userInput = rpc.requests.get("item/tool/requestUserInput")?.(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: "item-1",
        isBlocking: true,
        questions: [
          {
            id: "mode",
            header: "Mode",
            question: "Choose a mode",
            options: [{ label: "fast", description: "Fast" }],
          },
        ],
      },
      "rpc-input",
    );
    await vi.waitFor(() => expect(events.at(-1)?.type).toBe("approval_requested"));
    const question = events.at(-1);
    if (question?.type !== "approval_requested") throw new Error("missing input event");
    expect(question.questions).toEqual([
      { id: "mode", header: "Mode", prompt: "Choose a mode", options: ["fast"] },
    ]);
    await runtime.respondToApproval({
      ...question,
      decision: "submit",
      answers: { mode: ["fast"] },
    });
    await expect(userInput).resolves.toEqual({ answers: { mode: { answers: ["fast"] } } });

    const elicitation = rpc.requests.get("mcpServer/elicitation/request")?.(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        serverName: "swarmx",
        mode: "form",
        message: "Confirm write",
        requestedSchema: {
          $schema: "https://json-schema.org/draft/2020-12/schema",
          type: "object",
          properties: {
            confirm: {
              type: "boolean",
              title: "Confirm",
              description: "Allow this write",
              default: false,
            },
            reason: { type: "string", title: "Reason", minLength: 3 },
          },
          required: ["confirm", "reason"],
        },
      },
      "rpc-form",
    );
    await vi.waitFor(() => expect(events.at(-1)?.type).toBe("approval_requested"));
    const form = events.at(-1);
    if (form?.type !== "approval_requested") throw new Error("missing form event");
    expect(form).toMatchObject({
      prompt: "Confirm write",
      choices: ["accept", "decline", "cancel"],
      questions: [
        {
          id: "confirm",
          type: "boolean",
          prompt: "Allow this write",
          header: "Confirm",
          required: true,
          defaultValue: false,
        },
        {
          id: "reason",
          type: "string",
          prompt: "Reason",
          header: "Reason",
          required: true,
          minLength: 3,
        },
      ],
    });
    await expect(
      runtime.respondToApproval({
        ...form,
        decision: "accept",
        form: { confirm: "yes", reason: "no" },
      }),
    ).rejects.toThrow();
    await runtime.respondToApproval({
      ...form,
      decision: "accept",
      form: { confirm: true, reason: "needed" },
    });
    await expect(elicitation).resolves.toEqual({
      action: "accept",
      content: { confirm: true, reason: "needed" },
    });

    const eventCount = events.length;
    const missingSchema = rpc.requests.get("mcpServer/elicitation/request")?.(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        serverName: "swarmx",
        mode: "form",
        message: "Missing schema",
      },
      "rpc-form-missing-schema",
    );
    await expect(missingSchema).rejects.toThrow("requestedSchema");
    expect(events).toHaveLength(eventCount);

    const urlMode = rpc.requests.get("mcpServer/elicitation/request")?.(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        serverName: "swarmx",
        mode: "url",
        message: "Open an external approval URL",
        requestedSchema: {
          type: "object",
          properties: { confirm: { type: "boolean" } },
        },
      },
      "rpc-url-elicitation",
    );
    await expect(urlMode).rejects.toThrow("mode=form");
    expect(events).toHaveLength(eventCount);

    const unsupportedSchema = rpc.requests.get("mcpServer/elicitation/request")?.(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        serverName: "swarmx",
        mode: "form",
        message: "Unsupported form keyword",
        requestedSchema: {
          type: "object",
          properties: { reason: { type: "string", pattern: "^allowed$" } },
        },
      },
      "rpc-unsupported-form-schema",
    );
    await expect(unsupportedSchema).rejects.toThrow(/unsupported|unrecognized/iu);
    expect(events).toHaveLength(eventCount);

    const unknownTopLevelKeyword = rpc.requests.get("mcpServer/elicitation/request")?.(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        serverName: "swarmx",
        mode: "form",
        message: "Unknown top-level keyword",
        requestedSchema: {
          type: "object",
          properties: { confirm: { type: "boolean" } },
          additionalProperties: false,
        },
      },
      "rpc-unknown-top-level-schema",
    );
    await expect(unknownTopLevelKeyword).rejects.toThrow(/unsupported|unrecognized/iu);
    expect(events).toHaveLength(eventCount);

    for (const requestedSchema of [
      {
        type: "object",
        properties: {},
        required: ["toString"],
      },
      JSON.parse(
        '{"type":"object","properties":{"__proto__":{"type":"boolean"}},"required":["__proto__"]}',
      ),
    ]) {
      const unsafeProperty = rpc.requests.get("mcpServer/elicitation/request")?.(
        {
          threadId: "thread-1",
          turnId: "turn-1",
          serverName: "swarmx",
          mode: "form",
          message: "Unsafe property",
          requestedSchema,
        },
        "rpc-unsafe-property-schema",
      );
      await expect(unsafeProperty).rejects.toThrow(/Elicitation|unsafe/iu);
      expect(events).toHaveLength(eventCount);
    }
  });

  it("rejects MCP elicitation without an explicit form mode", async () => {
    const rpc = new FakeCodexRpc();
    const runtime = new CodexConversationRuntime(rpc);
    const events: RuntimeEvent[] = [];
    runtime.subscribe((event) => events.push(event));
    const handle = rpc.requests.get("mcpServer/elicitation/request");
    if (handle === undefined) throw new Error("missing elicitation handler");
    const pending = handle(
      {
        threadId: "thread-1",
        message: "Missing mode",
        requestedSchema: {
          type: "object",
          properties: { confirm: { type: "boolean" } },
        },
      },
      "rpc-missing-mode",
    );

    try {
      await expect(within(pending, "missing-mode rejection")).rejects.toThrow("mode=form");
      expect(events).toEqual([]);
    } finally {
      await runtime.dispose();
    }
  });

  it.each(invalidElicitationProperties)(
    "rejects inconsistent elicitation schema at entry: $name",
    async ({ schema }) => {
      const rpc = new FakeCodexRpc();
      const runtime = new CodexConversationRuntime(rpc);
      const events: RuntimeEvent[] = [];
      runtime.subscribe((event) => events.push(event));
      const handle = rpc.requests.get("mcpServer/elicitation/request");
      if (handle === undefined) throw new Error("missing elicitation handler");
      const pending = handle(
        {
          threadId: "thread-1",
          mode: "form",
          message: "Invalid form",
          requestedSchema: {
            type: "object",
            properties: { value: schema },
          },
        },
        `rpc-invalid-${String(schema.type)}`,
      );

      try {
        await expect(within(pending, "invalid-schema rejection")).rejects.toThrow("Elicitation");
        expect(events).toEqual([]);
      } finally {
        await runtime.dispose();
      }
    },
  );

  it("keeps delimiter-containing elicitation schema identities distinct", async () => {
    const rpc = new FakeCodexRpc();
    const runtime = new CodexConversationRuntime(rpc);
    const events: RuntimeEvent[] = [];
    runtime.subscribe((event) => events.push(event));
    const handle = rpc.requests.get("mcpServer/elicitation/request");
    if (handle === undefined) throw new Error("missing elicitation handler");
    const firstPending = handle(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: "item\u0000codex:approval",
        approvalId: "tail",
        mode: "form",
        message: "First form",
        requestedSchema: {
          type: "object",
          properties: { confirm: { type: "boolean" } },
          required: ["confirm"],
        },
      },
      "rpc-delimited-first",
    );
    const secondPending = handle(
      {
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: "item",
        approvalId: "approval\u0000codex:tail",
        mode: "form",
        message: "Second form",
        requestedSchema: {
          type: "object",
          properties: { reason: { type: "string", minLength: 2 } },
          required: ["reason"],
        },
      },
      "rpc-delimited-second",
    );
    void firstPending.catch(() => undefined);
    void secondPending.catch(() => undefined);

    await vi.waitFor(() =>
      expect(events.filter((event) => event.type === "approval_requested")).toHaveLength(2),
    );
    const approvals = events.filter(
      (event): event is Extract<RuntimeEvent, { type: "approval_requested" }> =>
        event.type === "approval_requested",
    );
    const first = approvals[0];
    const second = approvals[1];
    if (first === undefined || second === undefined) throw new Error("missing approval events");
    await runtime.respondToApproval({ ...first, decision: "accept", form: { confirm: true } });
    await runtime.respondToApproval({ ...second, decision: "accept", form: { reason: "ok" } });
    await expect(firstPending).resolves.toEqual({
      action: "accept",
      content: { confirm: true },
    });
    await expect(secondPending).resolves.toEqual({
      action: "accept",
      content: { reason: "ok" },
    });
  });
});
