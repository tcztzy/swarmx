import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { ExecutionEventBus, RequestContext } from "@a2a-js/sdk/server";
import { EventType } from "@ag-ui/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";
import type { ExecutionRecord } from "../src/execution-record.js";
import { SwarmA2AExecutor } from "../src/host/a2a.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { recordedAgent } from "../src/host/recorded-agent.js";
import { projectPermissions } from "../src/permissions.js";

const cleanups: Array<() => Promise<void>> = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-log-"));
  const journal = new ExecutionJournal(root, "workspace-a");
  cleanups.push(async () => {
    journal.close();
    await rm(root, { recursive: true, force: true });
  });
  return { root, journal };
}
const sink: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
function native(start: NativeAgent["start"]): NativeAgent {
  return {
    name: "Test Harness",
    capabilities: HARNESS_CAPABILITIES.codex,
    models: async () => ({ models: [], current: {} }),
    list: async () => [],
    create: async () => "codex:session",
    read: async () => {},
    start,
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
}

describe("execution journal", () => {
  it("retains native mode choices and reported changes across restart without lending them to siblings", async () => {
    const { root, journal } = await fixture();
    const start = vi.fn<NativeAgent["start"]>(async () => ({ stopReason: "end_turn" }));
    const leaf = native(start);
    const agent = recordedAgent(journal, "codex", leaf);
    await agent.start("codex:one", "work", sink, { mode: "plan" });
    const reopened = new ExecutionJournal(root, "workspace-a");
    try {
      const resumed = recordedAgent(reopened, "codex", leaf);
      expect((await resumed.models("codex:one")).current.mode).toBe("plan");
      start.mockImplementationOnce(async (_id, _text, output) => {
        output.raw(
          { update: { sessionUpdate: "current_mode_update", currentModeId: "full" } },
          { "swarmx.native.mode": "full" },
        );
        return { stopReason: "end_turn" };
      });
      await resumed.start("codex:one", "continue", sink);
      expect(start.mock.lastCall?.[3]?.mode).toBe("plan");
      await resumed.start("codex:one", "continue again", sink);
      expect(start.mock.lastCall?.[3]?.mode).toBe("full");
      await resumed.start("codex:two", "independent", sink);
      expect(start.mock.lastCall?.[3]?.mode).toBeUndefined();
    } finally {
      reopened.close();
    }
  });
  it("restores only never-dispatched empty reservations within their workspace", async () => {
    const { root, journal } = await fixture();
    const leaf = native(async () => ({ stopReason: "end_turn" }));
    leaf.create = async () => "claude:empty";
    const agent = recordedAgent(journal, "claude", leaf);
    await agent.create();
    expect(journal.emptySessions("claude")).toEqual(["claude:empty"]);
    const reopened = new ExecutionJournal(root, "workspace-a");
    const other = new ExecutionJournal(root, "workspace-b");
    try {
      expect(reopened.emptySessions("claude")).toEqual(["claude:empty"]);
      expect(other.emptySessions("claude")).toEqual([]);
      await agent.start("claude:empty", "first message", sink);
      expect(reopened.emptySessions("claude")).toEqual([]);
    } finally {
      reopened.close();
      other.close();
    }
  });

  it("persists A2A conversation bindings across executor restart and isolates Agent aliases", async () => {
    const { root, journal } = await fixture();
    const start = vi.fn<NativeAgent["start"]>(async () => ({ stopReason: "end_turn" }));
    const leaf = native(start);
    leaf.create = vi.fn(async () => "codex:bound");
    const context = {
      contextId: "context",
      taskId: "task",
      userMessage: { parts: [{ content: { $case: "text", value: "work" } }] },
    } as RequestContext;
    const events = { publish: vi.fn(), finished: vi.fn() } as unknown as ExecutionEventBus;
    await new SwarmA2AExecutor(leaf, journal, "swarm").execute(context, events);
    const reopened = new ExecutionJournal(root, "workspace-a");
    try {
      await new SwarmA2AExecutor(leaf, reopened, "swarm").execute(
        { ...context, taskId: "next" },
        events,
      );
      expect(leaf.create).toHaveBeenCalledTimes(1);
      expect(start.mock.calls.map(([id]) => id)).toEqual(["codex:bound", "codex:bound"]);
      expect(reopened.conversationBindings("another").size).toBe(0);
    } finally {
      reopened.close();
    }
  });

  it("shares one A2A session allocation across concurrent requests for a context", async () => {
    const { journal } = await fixture();
    const created = Promise.withResolvers<string>();
    const leaf = native(async () => ({ stopReason: "end_turn" }));
    leaf.create = vi.fn(() => created.promise);
    const executor = new SwarmA2AExecutor(leaf, journal, "swarm");
    const context = {
      contextId: "same",
      taskId: "one",
      userMessage: { parts: [{ content: { $case: "text", value: "work" } }] },
    } as RequestContext;
    const events = { publish: vi.fn(), finished: vi.fn() } as unknown as ExecutionEventBus;
    const first = executor.execute(context, events);
    const second = executor.execute({ ...context, taskId: "two" }, events);
    expect(leaf.create).toHaveBeenCalledTimes(1);
    created.resolve("codex:shared");
    await Promise.all([first, second]);
    expect(journal.conversationBindings("swarm").get("same")).toBe("codex:shared");
  });
  it("intersects all persisted grants without a page limit and isolates identical session IDs by workspace", async () => {
    const { root, journal } = await fixture();
    const context = { sessionId: "codex:shared", runId: "run", causedBy: null, attributes: {} };
    const first = projectPermissions({ tools: ["memory.read", "science.read"] });
    journal.append(context, {
      type: EventType.CUSTOM,
      name: "swarmx.session.created",
      value: { permissions: first },
    });
    for (let index = 0; index < 205; index += 1)
      journal.append(context, {
        type: EventType.CUSTOM,
        name: "swarmx.session.created",
        value: { permissions: projectPermissions({ delegation: false }) },
      });
    journal.append(context, {
      type: EventType.RUN_STARTED,
      threadId: context.sessionId,
      runId: context.runId,
      input: {
        threadId: context.sessionId,
        runId: context.runId,
        messages: [],
        tools: [],
        context: [],
        state: {},
        forwardedProps: { permissions: projectPermissions({ harnesses: { codex: ["small"] } }) },
      },
    });
    expect(journal.sessionPermissions(context.sessionId)).toEqual({
      tools: ["memory.read", "science.read"],
      delegation: false,
      harnesses: { codex: ["small"] },
    });
    expect(journal.sessionPermissions("codex:unseen")).toBeUndefined();
    const other = new ExecutionJournal(root, "workspace-b");
    try {
      expect(other.sessionPermissions(context.sessionId)).toBeUndefined();
    } finally {
      other.close();
    }
  });

  it("rejects invalid recorded permissions instead of treating them as a legacy missing grant", async () => {
    const { journal } = await fixture();
    journal.append(
      { sessionId: "codex:invalid", runId: "run", causedBy: null, attributes: {} },
      {
        type: EventType.CUSTOM,
        name: "swarmx.session.created",
        value: { permissions: null },
      },
    );
    expect(() => journal.sessionPermissions("codex:invalid")).toThrow();
  });
  it("retains native control failures separately from requests", async () => {
    const { journal } = await fixture();
    const ready = Promise.withResolvers<void>();
    const done = Promise.withResolvers<void>();
    const leaf = native(async () => {
      ready.resolve();
      await done.promise;

      return { stopReason: "end_turn" as const };
    });
    vi.spyOn(leaf, "interrupt").mockImplementation(async () => {
      throw new Error("native stop rejected");
    });
    const agent = recordedAgent(journal, "codex", leaf);
    const operation = agent.start("codex:child", "work", sink);
    await ready.promise;
    try {
      await expect(agent.interrupt("codex:child")).rejects.toThrow("native stop rejected");
      const events = journal.read().events;
      expect(events.at(-2)?.event).toMatchObject({ name: "swarmx.run.interrupt_requested" });
      expect(events.at(-1)?.event).toMatchObject({
        name: "swarmx.control.failed",
        value: { message: "native stop rejected" },
      });
      expect(events.at(-1)?.causedBy).toBe(events.at(-2)?.id);
      expect(journal.activeSession("codex:child")).toBeDefined();
    } finally {
      done.resolve();
      await operation;
    }
  });

  it("preserves raw JSON across reopen, rejects rewrites and isolates workspace/cursor reads", async () => {
    const { root, journal } = await fixture();
    const raw = {
      future: { chunks: [null, "中文\n", { model: "reported-model" }] },
      id: "native-1",
    };
    const first = journal.append(null, { type: EventType.RAW, source: "codex", event: raw });
    const second = journal.append(null, { type: EventType.CUSTOM, name: "swarmx.test", value: {} });
    const db = new DatabaseSync(journal.databasePath);
    try {
      expect(() => db.exec("UPDATE execution_events SET record_json = '{}'")).toThrow(
        /append-only/,
      );
      expect(() => db.exec("DELETE FROM execution_events")).toThrow(/append-only/);
    } finally {
      db.close();
    }
    journal.close();
    const reopened = new ExecutionJournal(root, "workspace-a");
    const other = new ExecutionJournal(root, "workspace-b");
    try {
      expect(reopened.read({ limit: 1 }).events).toEqual([first]);
      expect(reopened.read({ after: first.seq }).events).toEqual([second]);
      expect(reopened.read().events[0]?.event).toMatchObject({ event: raw });
      expect(other.read().events).toEqual([]);
      other.append(null, { type: EventType.CUSTOM, name: "swarmx.other", value: {} });
      expect(reopened.read().events).toHaveLength(2);
    } finally {
      reopened.close();
      other.close();
    }
  });

  it("records requested settings and per-response reported models without rewriting earlier runs", async () => {
    const { journal } = await fixture();
    const agent = recordedAgent(
      journal,
      "claude",
      native(async (_id, _text, observer) => {
        observer.raw(
          { type: "assistant", message: { id: "native-answer", model: "resolved-model" } },
          {
            "gen_ai.response.id": "native-answer",
            "gen_ai.response.model": "resolved-model",
            "swarmx.harness.version": "native-version",
          },
        );
        observer.text("native-answer", "answer");

        return { stopReason: "end_turn" as const };
      }),
    );
    await agent.start("claude:s", "first", sink, { model: "alias-a", effort: "high" });
    const before = journal.read().events;
    await agent.start("claude:s", "second", sink, { model: "alias-b", effort: "low" });
    const events = journal.read().events;
    expect(events.slice(0, before.length)).toEqual(before);
    const starts = events.filter(({ event }) => event.type === EventType.RUN_STARTED);
    expect(starts.map(({ attributes }) => attributes["gen_ai.request.model"])).toEqual([
      "alias-a",
      "alias-b",
    ]);
    expect(new Set(starts.map(({ runId }) => runId)).size).toBe(2);
    expect(starts[0]?.event).toMatchObject({ input: { messages: [{ content: "first" }] } });
    const responses = events.filter(({ event }) => event.type === EventType.RAW);
    expect(responses[0]?.attributes).toMatchObject({
      "swarmx.harness.name": "claude",
      "gen_ai.response.model": "resolved-model",
    });
    expect(responses[0]?.runId).toBe(starts[0]?.runId);
    expect(journal.read({ run: starts[0]?.runId ?? "" }).events).toEqual(before);
  });

  it("commits before dispatch and delivery, and propagates logging failures", async () => {
    const { journal } = await fixture();
    const start = vi.fn<NativeAgent["start"]>(async (_id, _text, observer) => {
      expect(journal.read().events.at(-1)?.event.type).toBe(EventType.RUN_STARTED);
      observer.text("message", "visible only after commit");

      return { stopReason: "end_turn" as const };
    });
    const agent = recordedAgent(journal, "codex", native(start));
    const delivered = vi.fn(() => {
      expect(journal.read().events.at(-1)?.event).toMatchObject({
        delta: "visible only after commit",
      });
    });
    await agent.start("codex:s", "prompt", { ...sink, text: delivered });
    expect(delivered).toHaveBeenCalledOnce();
    journal.close();
    await expect(agent.start("codex:s", "must not run", sink)).rejects.toThrow();
    expect(start).toHaveBeenCalledOnce();
  });

  it("retains errors, interaction answers, steering and cancellation requests", async () => {
    const { journal } = await fixture();
    const started = Promise.withResolvers<void>();
    const stopped = Promise.withResolvers<void>();
    const leaf = native(async (_id, text, observer) => {
      if (text === "fail") throw new Error("native failure");
      const answer = await observer.interact({ id: "approval", title: "Write?", schema: {} });
      expect(answer).toEqual({ allow: false });
      expect(journal.read().events.at(-1)?.event).toMatchObject({
        name: "swarmx.interaction.answered",
      });
      started.resolve();
      await stopped.promise;

      return { stopReason: "end_turn" as const };
    });
    leaf.interrupt = async () => {
      stopped.resolve();
      return;
    };
    const agent = recordedAgent(journal, "codex", leaf);
    await expect(agent.start("codex:failed", "fail", sink)).rejects.toThrow("native failure");
    const pending = agent.start("codex:active", "wait", {
      ...sink,
      interact: async () => ({ allow: false }),
    });
    await started.promise;
    await agent.steer("codex:active", "focus");
    await agent.interrupt("codex:active");
    await pending;
    const events = journal.read().events;
    expect(
      events.some(
        ({ event }) => event.type === EventType.RUN_ERROR && event.message === "native failure",
      ),
    ).toBe(true);
    expect(
      events.map(({ event }) => (event.type === EventType.CUSTOM ? event.name : null)),
    ).toEqual(
      expect.arrayContaining([
        "swarmx.interaction.requested",
        "swarmx.interaction.answered",
        "swarmx.input.steered",
        "swarmx.run.interrupt_requested",
      ]),
    );
    expect(
      events.findLast(({ event }) => event.type === EventType.RUN_FINISHED)?.event,
    ).toMatchObject({
      type: EventType.RUN_FINISHED,
      result: { interruptionRequested: true },
    });
  });

  it("does not deliver an event that fails to persist or execute a tool with an unknown parent", async () => {
    const { journal } = await fixture();
    const append = journal.append.bind(journal);
    vi.spyOn(journal, "append").mockImplementation((context, event, attributes) => {
      if (event.type === EventType.TEXT_MESSAGE_CHUNK) throw new Error("disk full");
      return append(context, event, attributes);
    });
    const delivered = vi.fn();
    const agent = recordedAgent(
      journal,
      "codex",
      native(async (_id, _text, output) => {
        await output.text("answer", "must not appear");
        return { stopReason: "end_turn" as const };
      }),
    );
    await expect(agent.start("codex:s", "test", { ...sink, text: delivered })).rejects.toThrow(
      "disk full",
    );
    expect(delivered).not.toHaveBeenCalled();
    expect(journal.read().events.at(-1)?.event).toMatchObject({
      type: EventType.RUN_ERROR,
      message: "disk full",
    });
    const execute = vi.fn(async () => ({}));
    await expect(
      journal.tool(
        "science",
        {},
        { actorId: "agent", callId: "call", sessionId: "unknown" },
        execute,
      ),
    ).rejects.toThrow(/no active/);
    expect(execute).not.toHaveBeenCalled();
  });

  it("rejects stale MCP execution IDs and keeps unscoped callers unattributed", async () => {
    const { journal } = await fixture();
    const execute = vi.fn(async () => ({}));
    const agent = recordedAgent(
      journal,
      "codex",
      native(async (_id, _text, output) => {
        await expect(
          journal.tool(
            "science",
            {},
            {
              actorId: "codex:s",
              callId: "stale",
              sessionId: "codex:s",
              runId: "previous-run",
            },
            execute,
          ),
        ).rejects.toThrow(/does not match/);
        await journal.tool(
          "science",
          {},
          {
            actorId: "codex:s",
            callId: "current",
            sessionId: "codex:s",
            runId: output.executionId,
          },
          execute,
        );

        return { stopReason: "end_turn" as const };
      }),
    );
    await agent.start("codex:s", "test", sink);
    expect(execute).toHaveBeenCalledOnce();
    await journal.tool("external", {}, { actorId: "agent", callId: "external" }, execute);
    const external = journal
      .read()
      .events.find(
        ({ event }) => event.type === EventType.TOOL_CALL_START && event.toolCallId === "external",
      );
    expect(external).toMatchObject({ sessionId: null, causedBy: null });
    expect(external?.attributes["swarmx.harness.name"]).toBeUndefined();
  });

  it("links product results and delegated runs through the causing event", async () => {
    const { journal } = await fixture();
    const child = recordedAgent(
      journal,
      "claude",
      native(async (_id, _text, observer) => {
        await observer.text("child", "done");
        return { stopReason: "end_turn" as const };
      }),
    );
    const parent = recordedAgent(
      journal,
      "codex",
      native(async () => {
        await journal.tool(
          "swarm",
          { action: "send_message" },
          { actorId: "codex:parent", callId: "delegate" },
          async () => {
            await child.start("claude:child", "review", sink);
            return { sessionId: "claude:child", text: "done", locator: { journalSeq: 42 } };
          },
        );

        return { stopReason: "end_turn" as const };
      }),
    );
    await parent.start("codex:parent", "delegate", sink);
    const events = journal.read().events;
    const tool = events.find(({ event }) => event.type === EventType.TOOL_CALL_START);
    const childStart = events.find(
      ({ event, sessionId }) =>
        event.type === EventType.RUN_STARTED && sessionId === "claude:child",
    );
    expect(childStart?.causedBy).toBe(tool?.id);
    expect(childStart?.event).not.toHaveProperty("parentRunId");
    expect(
      events.find(({ event }) => event.type === EventType.TOOL_CALL_RESULT)?.event,
    ).toMatchObject({ content: expect.stringContaining('"journalSeq":42') });
    expect(journal.activeSession("codex:parent")).toBeUndefined();
    await child.start("claude:child", "unrelated later run", sink);
    expect(journal.read({ session: "codex:parent", descendants: true }).events).toEqual(events);
    const paged: ExecutionRecord[] = [];
    let after = 0;
    for (;;) {
      const page = journal.read({ session: "codex:parent", descendants: true, after, limit: 2 });
      if (page.events.length === 0) break;
      paged.push(...page.events);
      after = page.nextAfter;
    }
    expect(paged).toEqual(events);
  });

  it("does not fabricate completion when reopening an interrupted log", async () => {
    const { root, journal } = await fixture();
    journal.append(
      { sessionId: "codex:crash", runId: "run", causedBy: null, attributes: {} },
      {
        type: EventType.RUN_STARTED,
        threadId: "codex:crash",
        runId: "run",
      },
    );
    journal.close();
    const reopened = new ExecutionJournal(root, "workspace-a");
    try {
      expect(reopened.activeRuns()).toEqual([]);
      expect(reopened.read().events.map(({ event }) => event.type)).toEqual([
        EventType.RUN_STARTED,
      ]);
    } finally {
      reopened.close();
    }
  });
});
