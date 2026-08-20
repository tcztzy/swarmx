import { describe, expect, it, vi } from "vitest";
import type { Timeline } from "../src/fork-boundary.js";
import type { RerunSessions } from "../src/rerun.js";
import { rerunTurn } from "../src/rerun.js";

const TIMELINE: Timeline = {
  turnOrder: [1, 2],
  turns: new Map([
    [1, { turn: 1, status: "closed", end: { seq: 10 } }],
    [2, { turn: 2, status: "closed", end: { seq: 20 } }],
  ]),
};

/** A sessions double recording calls and returning a fixed child id. */
function sessions(overrides: Partial<RerunSessions> = {}) {
  return {
    createSibling: vi.fn(() => Promise.resolve("fresh" as never)),
    fork: vi.fn(() => Promise.resolve("child" as never)),
    open: vi.fn(),
    prompt: vi.fn(() => Promise.resolve()),
    ...overrides,
  } satisfies RerunSessions;
}

const request = {
  sessionId: "source" as never,
  turn: 2,
  timeline: TIMELINE,
  windowReachesStart: true,
  text: "revised",
};

describe("rerunTurn", () => {
  it("forks at the preceding boundary, opens the child, and prompts it", async () => {
    const service = sessions();
    await expect(rerunTurn(service, request)).resolves.toBe("child");
    expect(service.fork).toHaveBeenCalledWith({
      sessionId: "source",
      atSeq: 10,
      increaseTitle: true,
    });
    expect(service.open).toHaveBeenCalledWith("child");
    expect(service.prompt).toHaveBeenCalledWith("child", "revised");
  });

  it("creates a fresh sibling instead of forking the first turn", async () => {
    const service = sessions();
    await expect(rerunTurn(service, { ...request, turn: 1 })).resolves.toBe("fresh");
    expect(service.createSibling).toHaveBeenCalledWith("source");
    expect(service.fork).not.toHaveBeenCalled();
    expect(service.open).toHaveBeenCalledWith("fresh");
    expect(service.prompt).toHaveBeenCalledWith("fresh", "revised");
  });

  it("does nothing when no boundary is resolvable", async () => {
    const service = sessions();
    await expect(
      rerunTurn(service, { ...request, turn: 1, windowReachesStart: false }),
    ).resolves.toBeUndefined();
    expect(service.createSibling).not.toHaveBeenCalled();
    expect(service.fork).not.toHaveBeenCalled();
    expect(service.prompt).not.toHaveBeenCalled();
  });

  it("never prompts when the fork fails", async () => {
    const service = sessions({ fork: vi.fn(() => Promise.reject(new Error("denied"))) });
    await expect(rerunTurn(service, request)).rejects.toThrow("denied");
    expect(service.prompt).not.toHaveBeenCalled();
  });

  it("never prompts when fresh-session creation fails", async () => {
    const service = sessions({
      createSibling: vi.fn(() => Promise.reject(new Error("create denied"))),
    });
    await expect(rerunTurn(service, { ...request, turn: 1 })).rejects.toThrow("create denied");
    expect(service.prompt).not.toHaveBeenCalled();
  });

  it("propagates a prompt failure with the child already open", async () => {
    const service = sessions({ prompt: vi.fn(() => Promise.reject(new Error("offline"))) });
    await expect(rerunTurn(service, request)).rejects.toThrow("offline");
    expect(service.open).toHaveBeenCalledWith("child");
  });
});
