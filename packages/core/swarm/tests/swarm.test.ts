import { expect, it, vi } from "vitest";
import { type Agent, createSwarm, type RunResult } from "../src/index.js";

interface Observer {
  text(value: string): void;
  interact(): Promise<string>;
}

function leaf(): Agent<Observer> {
  return {
    name: "native",
    capabilities: {
      history: true,
      list: true,
      resume: true,
      steer: true,
      emptySessionResume: false,
    },
    models: async () => ({ models: [], current: {} }),
    list: async () => [{ sessionId: "native" }],
    create: async () => "native",
    read: async (_id, observer) => observer.text("history"),
    start: async () => ({ stopReason: "end_turn" }),
    steer: async () => {},
    interrupt: async () => {},
    dispose: vi.fn(async () => {}),
  };
}

function nested(agent: Agent<Observer>) {
  return createSwarm("parent", createSwarm("child", createSwarm("grandchild", agent)));
}

it.each(["end_turn", "cancelled", "max_tokens", "max_turn_requests", "refusal"] as const)(
  "preserves %s, native objects and interactions through recursive composition",
  async (stopReason) => {
    const agent = leaf();
    const result = { stopReason };
    const observer = { text: vi.fn(), interact: vi.fn(async () => "allow") };
    const options = { model: "native-model", effort: "high", mode: "plan" };
    agent.start = vi.fn(async (_id, _text, view) => {
      view.text("answer");
      expect(await view.interact()).toBe("allow");
      return result;
    });
    const swarm = nested(agent);
    expect(swarm.name).toBe("parent");
    expect(swarm.capabilities).toBe(agent.capabilities);
    expect(await swarm.create()).toBe("native");
    expect(await swarm.start("native", "work", observer, options)).toBe(result);
    expect(agent.start).toHaveBeenCalledExactlyOnceWith("native", "work", observer, options);
    expect(observer.text).toHaveBeenCalledExactlyOnceWith("answer");
    await swarm.dispose();
    expect(agent.dispose).not.toHaveBeenCalled();
  },
);

it("forwards steering and cancellation while preserving native errors", async () => {
  const agent = leaf();
  const done = Promise.withResolvers<RunResult>();
  const failure = new Error("missing native session", { cause: { code: "native-error" } });
  agent.start = () => done.promise;
  agent.interrupt = vi.fn(async () => {
    done.resolve({ stopReason: "cancelled" });
  });
  agent.steer = vi.fn(async () => {});
  agent.read = async () => {
    throw failure;
  };
  const swarm = nested(agent);
  const observer = { text() {}, interact: async () => "allow" };
  const running = swarm.start("native", "work", observer);
  await swarm.steer("native", "follow up");
  expect(agent.steer).toHaveBeenCalledExactlyOnceWith("native", "follow up");
  await swarm.interrupt("native");
  expect(await running).toEqual({ stopReason: "cancelled" });
  await expect(swarm.read("native", observer)).rejects.toBe(failure);
});
