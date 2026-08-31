import type { Context } from "@deepseek-ai/cordis";
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import { describe, expect, it, vi } from "vitest";
import { createSibling, RerunController } from "../src/client/controller.js";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

/** A snapshot with two closed turns and both user messages in window. */
function snapshot(overrides: Record<string, unknown> = {}) {
  const messages = new Map([
    ["user-1", { kind: "user", data: { content: [{ type: "text", text: "first" }] } }],
    ["user-2", { kind: "user", data: { content: [{ type: "text", text: "second" }] } }],
  ]);
  return {
    hasMore: false,
    chat: {
      nodes: {
        get: (key: string) => messages.get(key),
        values: () => [...messages.values()],
      },
      locations: {
        getTurn: (turn: number) => (turn === 1 ? ["user-1"] : turn === 2 ? ["user-2"] : []),
        getStep: () => [],
      },
      timeline: {
        turnOrder: [1, 2],
        turns: new Map([
          [1, { turn: 1, status: "closed", start: { seq: 0 }, end: { seq: 2 } }],
          [2, { turn: 2, status: "closed", start: { seq: 2 }, end: { seq: 4 } }],
        ]),
      },
    },
    ...overrides,
  };
}

/** Controller over a doubles-backed sessions service. */
function controller(snap: unknown = snapshot()) {
  const createSibling = vi.fn(() => Promise.resolve("fresh" as never));
  const fork = vi.fn(() => Promise.resolve("child" as never));
  const prompt = vi.fn(() => Promise.resolve());
  const setDraft = vi.fn();
  const deps = {
    sessions: {
      createSibling,
      fork,
      open: vi.fn(),
      prompt,
    },
    snapshot: () => snap,
    setDraft,
  };
  return {
    instance: new RerunController(deps as never, "source" as never),
    createSibling,
    fork,
    prompt,
    setDraft,
  };
}

describe("RerunController", () => {
  it("creates an addressable first-turn sibling while an older refresh is in flight", async () => {
    const sourceId = "source" as SessionId;
    const siblingId = "sibling" as SessionId;
    const staleRefresh = deferred();
    const refresh = vi.fn(() => staleRefresh.promise);
    const create = vi.fn(async () => siblingId);
    const ctx = {
      sessions: {
        create,
        refresh,
        binding: vi.fn(),
        list: {
          getSnapshot: () => ({
            byId: {
              [sourceId]: { projectionValues: { agentPreset: "dsh-science" } },
            },
          }),
        },
      },
      workspaces: {
        create: vi.fn(),
        list: {
          getSnapshot: () => ({
            items: [{ workspaceId: "workspace", sessionIds: [sourceId] }],
          }),
        },
      },
    } as unknown as Context;
    const inFlight = refresh();

    await expect(createSibling(ctx, sourceId)).resolves.toBe(siblingId);
    expect(create).toHaveBeenCalledWith({
      workspaceId: "workspace",
      agentPreset: "dsh-science",
    });
    expect(refresh).toHaveBeenCalledOnce();
    expect(ctx.sessions.binding).not.toHaveBeenCalled();

    staleRefresh.resolve();
    await inFlight;
  });

  it("adopts an ungrouped first-turn source working directory", async () => {
    const sourceId = "source" as SessionId;
    const create = vi.fn(async () => "sibling" as SessionId);
    const createWorkspace = vi.fn(async () => ({ workspaceId: "workspace" }));
    const ctx = {
      sessions: {
        create,
        list: {
          getSnapshot: () => ({ byId: { [sourceId]: { cwd: "/work" } } }),
        },
      },
      workspaces: {
        create: createWorkspace,
        list: { getSnapshot: () => ({ items: [] }) },
      },
    } as unknown as Context;

    await expect(createSibling(ctx, sourceId)).resolves.toBe("sibling");
    expect(createWorkspace).toHaveBeenCalledWith({ path: "/work" });
    expect(create).toHaveBeenCalledWith({ workspaceId: "workspace" });
  });

  it("offers a re-run for a turn whose text and boundary are available", () => {
    expect(controller().instance.canRerun(2)).toBe(true);
  });

  it("declines a message whose opening text is outside the window", () => {
    const snap = snapshot({
      hasMore: true,
      chat: {
        ...snapshot().chat,
        locations: {
          getTurn: () => [],
          getStep: () => [],
        },
      },
    });
    expect(controller(snap).instance.canRerun(2)).toBe(false);
  });

  it("retries by forking before the turn and sending the original text", async () => {
    const { instance, fork, prompt } = controller();
    await instance.rerun(2);
    expect(fork).toHaveBeenCalledWith({ sessionId: "source", atSeq: 2, increaseTitle: true });
    expect(prompt).toHaveBeenCalledWith("child", "second");
  });

  it("edits by forking before the turn and seeding the child composer", async () => {
    const { instance, fork, prompt, setDraft } = controller();
    await expect(instance.beginEdit(2, "second")).resolves.toBeUndefined();
    expect(fork).toHaveBeenCalledWith({ sessionId: "source", atSeq: 2, increaseTitle: true });
    expect(setDraft).toHaveBeenCalledWith("child", "second");
    expect(prompt).not.toHaveBeenCalled();
  });

  it("edits the first turn in a fresh sibling session", async () => {
    const { instance, createSibling, fork, setDraft } = controller();
    await expect(instance.beginEdit(1, "first")).resolves.toBeUndefined();
    expect(createSibling).toHaveBeenCalledWith("source");
    expect(fork).not.toHaveBeenCalled();
    expect(setDraft).toHaveBeenCalledWith("fresh", "first");
  });
});
