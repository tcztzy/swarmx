import { describe, expect, it, vi } from "vitest";
import { DshConversationRuntime, type DshRuntimeHost } from "../src/runtime/dsh/runtime.js";
import type { ApprovalResponse, RuntimeEvent, WorkspaceScope } from "../src/runtime/index.js";

const workspace: WorkspaceScope = {
  id: "workspace-1",
  label: "swarmx",
  root: "/workspace/swarmx",
  token: "host-token",
};

function userMessage(id: string, text: string) {
  return {
    id,
    role: "user",
    content: [{ type: "text", text }],
    source: { kind: "user" },
  };
}

function assistantMessage(id: string, text: string) {
  return {
    id,
    role: "assistant",
    content: [{ type: "text", text }],
    source: { kind: "model", provider: "deepseek", model: "chat" },
  };
}

function events() {
  return [
    { seq: 0, time: 10, type: "turn/start", data: { turn: 1 } },
    {
      seq: 1,
      time: 11,
      type: "user/message",
      data: userMessage("user-1", "hello"),
      surfaceOp: "append",
      sourceEventSeqs: [],
    },
    {
      seq: 2,
      time: 12,
      type: "assistant/message",
      data: { turn: 1, step: 1, message: assistantMessage("assistant-1", "hi") },
      surfaceOp: "append",
      sourceEventSeqs: [],
    },
    {
      seq: 3,
      time: 13,
      type: "turn/end",
      data: { turn: 1, reason: { kind: "completed" } },
    },
    { seq: 4, time: 20, type: "turn/start", data: { turn: 2 } },
    {
      seq: 5,
      time: 21,
      type: "user/message",
      data: userMessage("user-2", "again"),
      surfaceOp: "append",
      sourceEventSeqs: [],
    },
    {
      seq: 6,
      time: 22,
      type: "turn/end",
      data: { turn: 2, reason: { kind: "completed" } },
    },
  ];
}

function createHost() {
  const listeners = new Map<string, (...args: unknown[]) => unknown>();
  const liveEvents = events();
  const followup = vi.fn(() => {
    liveEvents.push({ seq: 7, time: 30, type: "turn/start", data: { turn: 3 } });
  });
  const agent = {
    id: "session-1",
    status: "idle",
    session: { id: "session-1", events: liveEvents },
    followup,
    steer: vi.fn(),
    cancel: vi.fn(),
    whenIdle: vi.fn(async () => undefined),
  };
  const create = vi.fn(async (options: Record<string, unknown>) => {
    const id = String(options.sessionId);
    return {
      agent: {
        ...agent,
        id,
        session: { id, events: (options.seed as unknown[] | undefined) ?? [] },
      },
      dispose: vi.fn(async () => undefined),
    };
  });
  const host = {
    sessionQuery: {
      listSessions: vi.fn(async () => [
        {
          header: { id: "session-1", cwd: workspace.root, createdAt: 1 },
          live: true,
          persisted: true,
        },
      ]),
      readTitle: vi.fn(async () => ({ title: "DSH conversation", updatedAt: 30 })),
      readTitleSnapshots: vi.fn(async (ids: readonly string[]) =>
        ids.map((id) => ({
          sessionId: id,
          status: "fulfilled" as const,
          value: {
            session: { id, cwd: workspace.root, createdAt: 1 },
            title: { title: "DSH conversation", updatedAt: 30 },
          },
        })),
      ),
      readSession: vi.fn(async () => ({
        session: { id: "session-1", cwd: workspace.root, createdAt: 1 },
        events: liveEvents,
      })),
    },
    agents: {
      get: vi.fn(() => agent),
      create,
      resume: vi.fn(async () => ({ agent, dispose: vi.fn(async () => undefined) })),
    },
    agentPresets: {
      defaultId: "standard",
      mount: vi.fn(async () => undefined),
    },
    workspaceRegistry: {
      archivedSessionIds: [] as string[],
      archiveSession: vi.fn(async () => undefined),
      create: vi.fn(async () => ({ attachSession: vi.fn(async () => undefined) })),
      list: vi.fn(() => []),
    },
    userQuestions: {
      registerProvider: vi.fn(() => () => undefined),
    },
    on: vi.fn((name: string, listener: (...args: unknown[]) => unknown) => {
      listeners.set(name, listener);
      return () => listeners.delete(name);
    }),
  };
  return { agent, create, host: host as unknown as DshRuntimeHost, listeners };
}

describe("DSH conversation adapter", () => {
  it("projects logical persisted/live sessions and drives native agents", async () => {
    const { agent, host } = createHost();
    const runtime = new DshConversationRuntime(host);

    expect(await runtime.list()).toEqual([
      expect.objectContaining({
        runtime: "dsh",
        conversationId: "dsh:session-1",
        title: "DSH conversation",
      }),
    ]);
    expect(await runtime.read("dsh:session-1")).toMatchObject({
      conversationId: "dsh:session-1",
      turns: [
        {
          id: "dsh:session-1:turn:1",
          items: [
            { type: "user_message", id: "dsh:user-1", text: "hello" },
            {
              type: "assistant_message",
              id: "dsh:session-1:assistant:1:1",
              text: "hi",
            },
          ],
        },
        { id: "dsh:session-1:turn:2" },
      ],
    });
    await expect(runtime.start({ conversationId: "dsh:session-1", text: "next" })).resolves.toEqual(
      {
        turnId: "dsh:session-1:turn:3",
      },
    );
    expect(agent.followup).toHaveBeenCalledWith(
      expect.objectContaining({ role: "user", content: [{ type: "text", text: "next" }] }),
    );
    await runtime.steer({
      conversationId: "dsh:session-1",
      turnId: "dsh:session-1:turn:3",
      text: "more",
    });
    expect(agent.steer).toHaveBeenCalledOnce();
  });

  it("creates an empty child at the first turn and a native seeded child before later turns", async () => {
    const { create, host } = createHost();
    const runtime = new DshConversationRuntime(host);

    await runtime.fork({ conversationId: "dsh:session-1", beforeTurnId: "dsh:session-1:turn:1" });
    await runtime.fork({ conversationId: "dsh:session-1", beforeTurnId: "dsh:session-1:turn:2" });

    expect(create).toHaveBeenNthCalledWith(
      1,
      expect.not.objectContaining({ seed: expect.anything() }),
    );
    expect(create).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        seed: expect.arrayContaining([
          expect.objectContaining({
            type: "turn/end",
            data: { turn: 1, reason: { kind: "completed" } },
          }),
        ]),
      }),
    );
    const secondSeed = create.mock.calls[1]?.[0]?.seed as unknown[];
    expect(secondSeed).toHaveLength(4);
  });

  it("revises by branching before the selected turn and starting the replacement", async () => {
    const { agent, create, host } = createHost();
    const runtime = new DshConversationRuntime(host);

    await expect(
      runtime.revise({
        conversationId: "dsh:session-1",
        beforeTurnId: "dsh:session-1:turn:2",
        text: "replacement",
      }),
    ).resolves.toMatchObject({ runtime: "dsh" });
    expect(create).toHaveBeenCalledWith(expect.objectContaining({ seed: expect.any(Array) }));
    expect(agent.followup).toHaveBeenCalledWith(
      expect.objectContaining({ content: [{ type: "text", text: "replacement" }] }),
    );
  });

  it("keeps archived sessions readable but hidden and rejects later native mutation", async () => {
    const { agent, create, host } = createHost();
    (host.workspaceRegistry.archivedSessionIds as string[]).push("session-1");
    const runtime = new DshConversationRuntime(host);

    await expect(runtime.list()).resolves.toEqual([]);
    await expect(runtime.read("dsh:session-1")).resolves.toMatchObject({ archived: true });
    await expect(
      runtime.start({ conversationId: "dsh:session-1", text: "must not resume" }),
    ).rejects.toThrow(/archived/iu);
    await expect(
      runtime.fork({
        conversationId: "dsh:session-1",
        beforeTurnId: "dsh:session-1:turn:2",
      }),
    ).rejects.toThrow(/archived/iu);
    expect(agent.followup).not.toHaveBeenCalled();
    expect(create).not.toHaveBeenCalled();
  });

  it("rejects archive while a turn is running and preserves interrupt authority", async () => {
    const { agent, host } = createHost();
    const runtime = new DshConversationRuntime(host);
    const started = await runtime.start({ conversationId: "dsh:session-1", text: "keep running" });

    await expect(runtime.archive("dsh:session-1")).rejects.toThrow(
      "Cannot archive running DSH conversation",
    );
    expect(host.workspaceRegistry.archiveSession).not.toHaveBeenCalled();
    await runtime.interrupt({ conversationId: "dsh:session-1", turnId: started.turnId });
    expect(agent.cancel).toHaveBeenCalledOnce();
  });

  it("bounds title reads and summaries to the newest one thousand sessions", async () => {
    const { host } = createHost();
    vi.mocked(host.sessionQuery.listSessions).mockResolvedValue(
      Array.from({ length: 1_005 }, (_value, index) => ({
        header: {
          id: `session-${String(index)}` as never,
          cwd: workspace.root,
          createdAt: 2_000 - index,
        },
        live: false,
        persisted: true,
      })),
    );
    const runtime = new DshConversationRuntime(host);

    await expect(runtime.list()).resolves.toHaveLength(1_000);
    expect(host.sessionQuery.readTitleSnapshots).toHaveBeenCalledOnce();
    expect(host.sessionQuery.readTitleSnapshots).toHaveBeenCalledWith(
      expect.arrayContaining(["session-0", "session-999"]),
      undefined,
    );
    expect(
      vi.mocked(host.sessionQuery.readTitleSnapshots).mock.calls[0]?.[0].map(String),
    ).not.toContain("session-1000");
    expect(host.sessionQuery.readTitle).not.toHaveBeenCalled();
  });

  it("keeps peer summaries when one batched title observation is rejected", async () => {
    const { host } = createHost();
    vi.mocked(host.sessionQuery.listSessions).mockResolvedValue([
      {
        header: { id: "session-1" as never, cwd: workspace.root, createdAt: 1 },
        live: false,
        persisted: true,
      },
      {
        header: { id: "session-2" as never, cwd: workspace.root, createdAt: 2 },
        live: false,
        persisted: true,
      },
    ]);
    vi.mocked(host.sessionQuery.readTitleSnapshots).mockResolvedValue([
      { sessionId: "session-1" as never, status: "rejected", reason: new Error("invalid title") },
      {
        sessionId: "session-2" as never,
        status: "fulfilled",
        value: {
          session: { id: "session-2" as never, cwd: workspace.root, createdAt: 2 },
          title: { title: "Healthy title", updatedAt: 3 },
        },
      },
    ]);

    await expect(new DshConversationRuntime(host).list()).resolves.toEqual([
      expect.objectContaining({ conversationId: "dsh:session-2", title: "Healthy title" }),
      expect.objectContaining({ conversationId: "dsh:session-1", title: "New conversation" }),
    ]);
  });

  it("rechecks cancellation after ignored resume/read signals before native mutation", async () => {
    const resumed = createHost();
    const startController = new AbortController();
    const dispose = vi.fn(async () => undefined);
    vi.mocked(resumed.host.agents.get).mockReturnValue(undefined);
    vi.mocked(resumed.host.agents.resume).mockImplementation(async () => {
      startController.abort();
      return { agent: resumed.agent, dispose } as never;
    });
    const startRuntime = new DshConversationRuntime(resumed.host);

    await expect(
      startRuntime.start(
        { conversationId: "dsh:session-1", text: "must not send" },
        startController.signal,
      ),
    ).rejects.toThrow();
    expect(resumed.agent.followup).not.toHaveBeenCalled();
    expect(dispose).toHaveBeenCalledOnce();

    const forked = createHost();
    const forkController = new AbortController();
    vi.mocked(forked.host.sessionQuery.readSession).mockImplementation(async () => {
      forkController.abort();
      return {
        session: { id: "session-1", cwd: workspace.root, createdAt: 1 },
        events: events(),
      } as never;
    });
    const forkRuntime = new DshConversationRuntime(forked.host);
    await expect(
      forkRuntime.fork(
        { conversationId: "dsh:session-1", beforeTurnId: "dsh:session-1:turn:2" },
        forkController.signal,
      ),
    ).rejects.toThrow();
    expect(forked.create).not.toHaveBeenCalled();

    const created = createHost();
    const createController = new AbortController();
    const attachSession = vi.fn(async () => undefined);
    const createdHandleDispose = vi.fn(async () => undefined);
    vi.mocked(created.host.agents.create).mockImplementation(
      async (options) =>
        ({
          agent: {
            ...created.agent,
            id: options.sessionId,
            session: { id: options.sessionId, events: [] },
          },
          dispose: createdHandleDispose,
        }) as never,
    );
    vi.mocked(created.host.workspaceRegistry.create).mockImplementation(async () => {
      createController.abort();
      return { attachSession } as never;
    });
    const createRuntime = new DshConversationRuntime(created.host);
    await expect(createRuntime.create({ workspace }, createController.signal)).rejects.toThrow();
    expect(attachSession).not.toHaveBeenCalled();
    expect(createdHandleDispose).toHaveBeenCalledOnce();
  });

  it("emits ordered native events without registering a second DSH approval answerer", async () => {
    const { agent, host, listeners } = createHost();
    const runtime = new DshConversationRuntime(host);
    const emitted: RuntimeEvent[] = [];
    runtime.subscribe((event) => emitted.push(event));
    const sessionListener = listeners.get("session/event");
    sessionListener?.(agent.session, {
      seq: 7,
      time: 40,
      type: "assistant/chunk",
      data: { turn: 3, step: 1, chunk: { type: "text-delta", index: 0, text: "hello" } },
    });
    expect(emitted[0]).toMatchObject({
      seq: 1,
      type: "item_delta",
      conversationId: "dsh:session-1",
      turnId: "dsh:session-1:turn:3",
    });

    expect(listeners.has("approval/request")).toBe(false);
    const response: ApprovalResponse = {
      runtime: "dsh",
      conversationId: "dsh:session-1",
      turnId: "dsh:session-1:turn:3",
      itemId: "dsh:call-1",
      approvalId: "dsh:approval-1",
      decision: "accept",
    };
    await expect(runtime.respondToApproval(response)).rejects.toThrow(
      "DSH approvals are handled by the native DSH Web approval channel.",
    );
  });
});
