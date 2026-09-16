import { expect, it, vi } from "vitest";
import type { AgentOptions, NativeAgent } from "../src/agents/types.js";
import { AgentRegistry, bindAgent, currentAgentBinding } from "../src/host/agent-registry.js";

const observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };

function binding(cwd: string): AgentOptions {
  return { cwd, mcp: { command: "node", args: ["/bridge.js"], env: {} } };
}

function fake() {
  const starts: string[] = [];
  const dispose = vi.fn(async () => {});
  const agent = {
    name: "fake",
    capabilities: {
      history: true,
      list: true,
      resume: true,
      steer: true,
      emptySessionResume: true,
    },
    models: async () => ({ models: [], current: {} }),
    list: async () => {
      starts.push(currentAgentBinding().cwd);
      return [];
    },
    create: async () => "session",
    read: async () => {},
    start: async () => {
      starts.push(currentAgentBinding().cwd);
      return { stopReason: "end_turn" };
    },
    steer: async () => {},
    interrupt: async () => {},
    dispose,
  } as unknown as NativeAgent;
  return { agent, dispose, starts };
}

it("resolves the execution binding per call for one shared runtime", async () => {
  const { agent, dispose, starts } = fake();
  const first = bindAgent(agent, binding("/first"));
  const second = bindAgent(agent, binding("/second"));

  await first.start("s1", "hello", observer);
  await second.start("s2", "hello", observer);
  await first.list();

  expect(starts).toEqual(["/first", "/second", "/first"]);
  await first.dispose();
  expect(dispose).not.toHaveBeenCalled();
});

it("loads one shared runtime per harness and disposes it once at the host level", async () => {
  const { agent, dispose } = fake();
  const load = vi.fn(async () => agent);
  const registry = new AgentRegistry(load);

  const [one, two] = await Promise.all([
    registry.agent("codex", binding("/first")),
    registry.agent("codex", binding("/second")),
  ]);
  expect(one).toBe(two);
  expect(load).toHaveBeenCalledOnce();

  expect(await registry.agent("codex", binding("/third"))).toBe(one);
  expect(load).toHaveBeenCalledOnce();

  await registry.dispose();
  expect(dispose).toHaveBeenCalledOnce();
  expect(() => registry.agent("codex", binding("/fourth"))).toThrow("closed");
});

it("rejects shared runtimes used outside an execution binding", async () => {
  const { agent } = fake();
  let received: AgentOptions | undefined;
  const registry = new AgentRegistry(async (_id, options) => {
    received = options;
    return agent;
  });
  await registry.agent("codex", binding("/first"));

  expect(() => received?.cwd).toThrow("outside an execution binding");
  expect(() => currentAgentBinding()).toThrow("outside an execution binding");
});
