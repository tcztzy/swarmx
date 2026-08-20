import { describe, expect, it, vi } from "vitest";
import { RerunController } from "../src/client/controller.js";

/** A snapshot with two closed turns and both user messages in window. */
function snapshot(overrides: Record<string, unknown> = {}) {
  return {
    hasMore: false,
    nodes: [
      { kind: "user", seq: 1, content: [{ type: "text", text: "first" }] },
      { kind: "assistant", seq: 2, messageId: "m1", turn: 1 },
      { kind: "user", seq: 3, content: [{ type: "text", text: "second" }] },
      { kind: "assistant", seq: 4, messageId: "m2", turn: 2 },
    ],
    chat: {
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
      binding: () =>
        snap === undefined ? undefined : { session: { getSnapshot: () => snap } as never },
    },
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
  it("offers a re-run for a turn whose text and boundary are available", () => {
    expect(controller().instance.canRerun(2)).toBe(true);
  });

  it("declines a message whose opening text is outside the window", () => {
    const snap = snapshot({
      hasMore: true,
      nodes: [
        {
          kind: "assistant",
          seq: 4,
          messageId: "m2",
          turn: 2,
        },
      ],
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
