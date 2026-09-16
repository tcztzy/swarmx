import { beforeEach, expect, it, vi } from "vitest";
import { AGENT_IDS, loadAgent, scopeSessions, selectedAgent } from "../src/agent.js";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";

const factory = vi.hoisted(() => ({
  pi: vi.fn(),
  codex: vi.fn(),
  claude: vi.fn(),
  hermes: vi.fn(),
  openclaw: vi.fn(),
  dsh: vi.fn(),
}));
vi.mock("../src/agents/pi.js", () => ({ createPi: factory.pi }));
vi.mock("../src/agents/codex.js", () => ({ createCodex: factory.codex }));
vi.mock("../src/agents/claude.js", () => ({ createClaude: factory.claude }));
vi.mock("../src/agents/hermes.js", () => ({ createHermes: factory.hermes }));
vi.mock("../src/agents/openclaw.js", () => ({ createOpenClaw: factory.openclaw }));
vi.mock("../src/agents/dsh.js", () => ({ createDsh: factory.dsh }));
const options = { cwd: "/workspace", mcp: { command: "node", args: ["/bridge.js"], env: {} } };
const observer: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
function native(): NativeAgent {
  return {
    name: "native",
    capabilities: HARNESS_CAPABILITIES.codex,
    list: vi.fn(async () => [{ sessionId: "stored", title: "Native title" }]),
    create: vi.fn(async () => "fresh"),
    models: vi.fn(async () => ({ models: [], current: {} })),
    read: vi.fn(async () => {}),
    start: vi.fn(async () => ({ stopReason: "end_turn" as const })),
    steer: vi.fn(async () => {}),
    interrupt: vi.fn(async () => {}),
    dispose: vi.fn(async () => {}),
  };
}
beforeEach(() => vi.resetAllMocks());

it("uses Pi by default and preserves explicit native selections", () => {
  vi.stubEnv("SWARMX_AGENT", undefined);
  expect(selectedAgent()).toBe("pi");
  expect(selectedAgent("codex")).toBe("codex");
  vi.stubEnv("SWARMX_AGENT", "claude");
  expect(selectedAgent()).toBe("claude");
  vi.unstubAllEnvs();
});

it.each(AGENT_IDS)("loads only the selected %s native integration", async (id) => {
  const leaf = native();
  factory[id].mockResolvedValue(leaf);
  const agent = await loadAgent(id, options);
  expect(factory[id]).toHaveBeenCalledWith(options);
  for (const other of AGENT_IDS.filter((name) => name !== id))
    expect(factory[other]).not.toHaveBeenCalled();
  expect(await agent.list()).toEqual([{ sessionId: `${id}:stored`, title: "Native title" }]);
  await agent.dispose();
  expect(leaf.dispose).toHaveBeenCalledOnce();
});

it("keeps native IDs scoped and forwards observers, options and controls directly", async () => {
  const leaf = native();
  const agent = scopeSessions("hermes", leaf);
  expect(() => agent.read("hermes:unknown", observer)).toThrow("does not belong");
  await agent.list();
  expect(await agent.create()).toBe("hermes:fresh");
  const selection = { model: "native-model" };
  await agent.models("hermes:stored");
  await agent.read("hermes:stored", observer);
  await agent.start("hermes:fresh", "work", observer, selection);
  await agent.steer("hermes:fresh", "redirect");
  await agent.interrupt("hermes:fresh");
  expect(leaf.models).toHaveBeenCalledWith("stored");
  expect(leaf.read).toHaveBeenCalledWith("stored", observer);
  expect(leaf.start).toHaveBeenCalledWith("fresh", "work", observer, selection);
  expect(leaf.steer).toHaveBeenCalledWith("fresh", "redirect");
  expect(leaf.interrupt).toHaveBeenCalledWith("fresh");
  expect(() => agent.start("codex:stored", "wrong", observer)).toThrow("does not belong");
});

it("closes a native integration when its initial listing fails", async () => {
  const leaf = native();
  const failure = new Error("Native unavailable");
  vi.mocked(leaf.list).mockRejectedValue(failure);
  factory.hermes.mockResolvedValue(leaf);
  await expect(loadAgent("hermes", options)).rejects.toBe(failure);
  expect(leaf.dispose).toHaveBeenCalledOnce();
  expect(factory.codex).not.toHaveBeenCalled();
});
