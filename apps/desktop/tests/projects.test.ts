import { mkdir, mkdtemp, readFile, realpath, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import * as nativeAgents from "../src/agent.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { ProductServices } from "../src/host/product-services.js";
import { recordedAgent } from "../src/host/recorded-agent.js";
import { ProjectStore, resolveWorkspace, SettingsStore } from "../src/host/workspace-settings.js";
import { startDesktopPlatform } from "../src/platform.js";
import { DEFAULT_POLICY } from "../src/settings.js";

const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

it("persists distinct project directories and selection without rebinding existing project data", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-projects-"));
  roots.push(root);
  const home = join(root, "home");
  const one = join(root, "one");
  const two = join(root, "two");
  await mkdir(one);
  await mkdir(two);
  const catalog = new ProjectStore(home);
  const first = catalog.register(await resolveWorkspace(one));
  const settings = new SettingsStore(home, first.id);
  settings.write({ policy: { ...DEFAULT_POLICY, cpus: 3 }, environment: null });
  const second = await catalog.add({ label: "Research two", root: two });
  catalog.select(second.id);
  const reopened = new ProjectStore(home);
  expect(reopened.read()).toEqual({ projects: [first, second], activeId: second.id });
  expect(second.root).toBe(await realpath(two));
  expect(new SettingsStore(home, second.id).read().policy.cpus).toBe(DEFAULT_POLICY.cpus);
  expect(new SettingsStore(home, first.id).read().policy.cpus).toBe(3);
  expect(JSON.parse(await readFile(join(home, "projects.json"), "utf8"))).not.toHaveProperty(
    "root",
  );
  const alias = join(root, "alias");
  await symlink(one, alias);
  expect(await reopened.add({ label: "Duplicate", root: alias })).toEqual(first);
  expect(reopened.read().projects).toHaveLength(2);
  await expect(reopened.add({ label: "Missing", root: join(root, "missing") })).rejects.toThrow();
  expect(reopened.read().projects).toHaveLength(2);
  expect(() => reopened.select("unknown")).toThrow("not found");
  const platform = await startDesktopPlatform({
    productHome: home,
    workspaceRoot: one,
    rendererRoot: root,
  });
  try {
    expect(platform.workspaceRoot).toBe(second.root);
  } finally {
    await platform.dispose();
  }
});

it.each(["hermes", "openclaw"])(
  "isolates %s global native sessions across projects, including empty tasks and restart",
  async (harness) => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-project-sessions-"));
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
      await expect(second.read(id, observer)).rejects.toThrow("this project");
      await expect(second.start(id, "wrong directory", observer)).rejects.toThrow("this project");
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

it("passes project memory into task creation through the lazy native integration", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-project-memory-"));
  roots.push(root);
  const products = await ProductServices.create({
    productHome: join(root, "home"),
    workspace: await resolveWorkspace(root),
  });
  const create = vi.fn(async () => "codex:new");
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
  vi.spyOn(products.learning, "context").mockResolvedValue("Project-specific frozen memory");
  try {
    await products.attachAgents("http://127.0.0.1:8000", "test", undefined, "codex");
    await products.rootAgent.create();
    expect(create).toHaveBeenCalledWith({ instructions: "Project-specific frozen memory" });
    expect(load).toHaveBeenCalledWith(
      "codex",
      expect.objectContaining({
        cwd: products.options.workspace.root,
        mcp: expect.objectContaining({
          url: `http://127.0.0.1:8000/projects/${products.options.workspace.id}/mcp`,
        }),
      }),
    );
  } finally {
    await products.dispose();
    vi.restoreAllMocks();
  }
});
