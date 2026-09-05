import { describe, expect, it, vi } from "vitest";
import { type Agent, createSwarm } from "../src/index.js";

describe("native Swarm composition", () => {
  it("delegates to a leaf or nested Swarm through the same in-process interface", async () => {
    const observer = { token: vi.fn() };
    const leaf = fakeAgent();
    const nested = createSwarm("parent", createSwarm("child", createSwarm("grandchild", leaf)));
    const sessionId = await nested.create();
    await nested.start(sessionId, "research", observer);
    await nested.steer(sessionId, "verify sources");
    await nested.read(sessionId, observer);
    expect(nested.name).toBe("parent");
    expect(leaf.start).toHaveBeenCalledWith("native-session", "research", observer);
    expect(leaf.steer).toHaveBeenCalledWith("native-session", "verify sources");
    expect(leaf.read).toHaveBeenCalledWith("native-session", observer);
    expect(await nested.list()).toEqual([{ sessionId: "native-session", title: "Native" }]);
  });

  it("propagates cancellation and failures without retry or fallback", async () => {
    const leaf = fakeAgent();
    const swarm = createSwarm("parent", createSwarm("child", leaf));
    await swarm.interrupt("native-session");
    expect(leaf.interrupt).toHaveBeenCalledExactlyOnceWith("native-session");
    leaf.start.mockRejectedValueOnce(new Error("native failure"));
    await expect(swarm.start("native-session", "work", {})).rejects.toThrow("native failure");
    expect(leaf.start).toHaveBeenCalledOnce();
  });
});

function fakeAgent() {
  return {
    name: "native",
    list: vi.fn(async () => [{ sessionId: "native-session", title: "Native" }]),
    create: vi.fn(async () => "native-session"),
    read: vi.fn(async (_id: string, _observer: unknown) => {}),
    start: vi.fn(async (_id: string, _text: string, _observer: unknown) => {}),
    steer: vi.fn(async (_id: string, _text: string) => {}),
    interrupt: vi.fn(async (_id: string) => {}),
    dispose: vi.fn(async () => {}),
  } satisfies Agent<unknown>;
}
