import { mkdir, mkdtemp, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AgentCard } from "@a2a-js/sdk";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import * as nativeAgents from "../src/agent.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";
import { AgentRegistry, currentAgentBinding } from "../src/host/agent-registry.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { ProductServices } from "../src/host/product-services.js";
import { recordedAgent } from "../src/host/recorded-agent.js";
import { SettingsStore } from "../src/host/settings-store.js";
import { startDesktopPlatform } from "../src/platform.js";
import { DEFAULT_POLICY } from "../src/settings.js";

const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

it("starts in the requested canonical directory and restores shared settings", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-platform-"));
  roots.push(root);
  const home = join(root, "home");
  const directory = join(root, "analysis");
  const alias = join(root, "alias");
  await mkdir(directory);
  await symlink(directory, alias);
  const load = vi.spyOn(nativeAgents, "loadAgent").mockResolvedValue({
    name: "codex",
    capabilities: HARNESS_CAPABILITIES.codex,
    models: async () => ({ models: [], current: {} }),
    list: async () => [],
    create: async () => "codex:new",
    read: async () => {},
    start: async () => {},
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  });
  try {
    const platform = await startDesktopPlatform({
      productHome: home,
      cwd: alias,
      agentId: "codex",
    });
    try {
      expect(platform.cwd).toBe(await realpath(directory));
      const response = await fetch(`${platform.a2aUrl}/.well-known/agent-card.json`);
      expect(response.status).toBe(200);
      const card = AgentCard.fromJSON(await response.json());
      expect(card.supportedInterfaces[0]?.url).toBe(platform.a2aUrl);
      await platform.operations.updateSettings({ ...DEFAULT_POLICY, cpus: 3 });
      await platform.operations.writeLanguage("en");
    } finally {
      await platform.dispose();
    }
    const reopened = await startDesktopPlatform({ productHome: home, cwd: root, agentId: "codex" });
    try {
      expect(reopened.cwd).toBe(await realpath(root));
      expect((await reopened.operations.settings()).policy.cpus).toBe(3);
      expect((await reopened.operations.bootstrap()).language).toBe("en");
      expect(new SettingsStore(home).path).toBe(join(home, "settings.json"));
    } finally {
      await reopened.dispose();
    }
    const file = join(root, "input.txt");
    await writeFile(file, "data");
    await expect(startDesktopPlatform({ productHome: home, cwd: file })).rejects.toThrow(
      "not a directory",
    );
  } finally {
    load.mockRestore();
  }
});

it.each(["hermes", "openclaw"])(
  "isolates %s global native sessions across execution directories, including empty tasks and restart",
  async (harness) => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-directory-sessions-"));
    roots.push(root);
    const one = new ExecutionJournal(root, "one");
    const two = new ExecutionJournal(root, "two");
    const native = {
      name: harness,
      models: async () => ({ models: [], current: {} }),
      list: async () => [
        { sessionId: `${harness}:first` },
        { sessionId: `${harness}:second` },
        { sessionId: `${harness}:unrelated` },
      ],
      create: vi
        .fn()
        .mockResolvedValueOnce(`${harness}:first`)
        .mockResolvedValueOnce(`${harness}:second`),
      read: vi.fn(async () => {}),
      start: vi.fn(async () => {}),
      steer: vi.fn(async () => {}),
      interrupt: vi.fn(async () => {}),
      dispose: async () => {},
    };
    const first = recordedAgent(one, harness, native);
    const second = recordedAgent(two, harness, native);
    const observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
    try {
      const id = await first.create();
      const other = await second.create();
      two.append(
        { sessionId: id, runId: "review", causedBy: null, attributes: {} },
        { type: EventType.CUSTOM, name: "swarmx.memory.review.failed", value: {} },
      );
      expect(await first.list()).toEqual([{ sessionId: id }]);
      expect(await second.list()).toEqual([{ sessionId: other }]);
      await expect(second.read(id, observer)).rejects.toThrow("this directory");
      await expect(second.start(id, "wrong directory", observer)).rejects.toThrow("this directory");
      expect(native.start).not.toHaveBeenCalled();
    } finally {
      one.close();
      two.close();
    }
    const reopened = new ExecutionJournal(root, "one");
    try {
      expect(await recordedAgent(reopened, harness, native).list()).toEqual([
        { sessionId: `${harness}:first` },
      ]);
    } finally {
      reopened.close();
    }
  },
);

it("shares one host-level native runtime across execution directories", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-shared-agents-"));
  roots.push(root);
  const home = join(root, "home");
  const one = join(root, "one");
  const two = join(root, "two");
  await mkdir(one);
  await mkdir(two);
  const agents = new AgentRegistry();
  const load = vi.spyOn(nativeAgents, "loadAgent").mockResolvedValue({
    name: "codex",
    models: async () => ({ models: [], current: {} }),
    list: async () => [],
    create: async () => "codex:new",
    read: async () => {},
    start: async () => {},
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  });
  const first = await ProductServices.create({
    productHome: home,
    cwd: one,
    agents,
  });
  const second = await ProductServices.create({
    productHome: home,
    cwd: two,
    agents,
  });
  try {
    await first.attachAgents("http://127.0.0.1:8000", undefined, "codex");
    await second.attachAgents("http://127.0.0.1:8000", undefined, "codex");
    expect(load).toHaveBeenCalledOnce();
    await expect(first.rootAgent.create()).resolves.toBe("codex:new");
    await expect(second.rootAgent.create()).resolves.toBe("codex:new");
  } finally {
    await first.dispose();
    await second.dispose();
    await agents.dispose();
    vi.restoreAllMocks();
  }
});

it("passes frozen memory into task creation through the lazy native integration", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-session-memory-"));
  roots.push(root);
  const products = await ProductServices.create({
    productHome: join(root, "home"),
    cwd: root,
  });
  let resolved: { cwd: string; socket: string | undefined } | undefined;
  const create = vi.fn(async () => {
    const binding = currentAgentBinding();
    resolved = { cwd: binding.cwd, socket: binding.mcp.env.SWARMX_MCP_SOCKET };
    return "codex:new";
  });
  const load = vi.spyOn(nativeAgents, "loadAgent").mockResolvedValue({
    name: "codex",
    create,
    models: async () => ({ models: [], current: {} }),
    list: async () => [],
    read: async () => {},
    start: async () => {},
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  });
  vi.spyOn(products.learning, "context").mockResolvedValue("Frozen research memory");
  try {
    await products.attachAgents("http://127.0.0.1:8000", undefined, "codex");
    await products.rootAgent.create();
    expect(create).toHaveBeenCalledWith({ instructions: "Frozen research memory" });
    expect(resolved).toEqual({
      cwd: products.options.cwd,
      socket: products.mcpSocket,
    });
    expect(load).toHaveBeenCalledWith("codex", expect.anything());
  } finally {
    await products.dispose();
    vi.restoreAllMocks();
  }
});
