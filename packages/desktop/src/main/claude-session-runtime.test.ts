import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ClaudeScheduledTaskStore } from "./claude-scheduled-tasks.js";
import {
  ClaudeSessionRuntime,
  ClaudeSessionRuntimeRegistry,
  cronExpressionMatches,
  parseCronExpression,
} from "./claude-session-runtime.js";
import type { WorkspaceShell, WorkspaceShellRunOptions } from "./workspace-shell.js";

afterEach(() => {
  vi.useRealTimers();
});

describe("ClaudeSessionRuntime", () => {
  it("V424/V426 turns bounded monitor stdout into serialized session activations", async () => {
    vi.useFakeTimers();
    let runOptions: WorkspaceShellRunOptions | undefined;
    const stop = vi.fn(async () => monitorSnapshot("stopped"));
    const shell = {
      startBackground: vi.fn(async (_command: string, options: WorkspaceShellRunOptions) => {
        runOptions = options;
        return monitorSnapshot("running");
      }),
      stop,
      close: vi.fn(async () => undefined),
    } as unknown as WorkspaceShell;
    const activate = vi.fn(async () => undefined);
    const runtime = new ClaudeSessionRuntime("/tmp/project", { shell });
    runtime.configure({ activate });

    const result = await runtime.monitor({
      command: "watch-build",
      description: "build <watch>",
      timeoutMs: 15_000,
      persistent: true,
    });
    expect(result).toEqual({ taskId: "7", timeoutMs: 15_000, persistent: true });
    expect(runOptions).toMatchObject({ sessionLifetime: true });

    await runtime.beginForeground();
    runOptions?.onStdout?.("first <line>\nsecond line\n");
    await vi.advanceTimersByTimeAsync(200);
    expect(activate).not.toHaveBeenCalled();
    runtime.endForeground();
    await runtime.flushActivations();

    expect(activate).toHaveBeenCalledOnce();
    expect(activate.mock.calls[0]?.[0]).toMatchObject({ source: "monitor", taskId: "7" });
    expect(activate.mock.calls[0]?.[0].prompt).toContain("build &lt;watch&gt;");
    expect(activate.mock.calls[0]?.[0].prompt).toContain("first &lt;line&gt;");
    expect(activate.mock.calls[0]?.[0].prompt).toContain("untrusted process output");
    runOptions?.onExit?.(monitorSnapshot("completed"));
    await runtime.close();
  });

  it("V427-V428 schedules recurring and one-shot prompts in local time", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 6, 17, 12, 0, 30));
    const activate = vi.fn(async () => undefined);
    const runtime = new ClaudeSessionRuntime("/tmp/project", { shell: fakeShell() });
    runtime.configure({ activate });

    const recurring = await runtime.createCron({
      cron: "* * * * *",
      prompt: "check recurring",
      recurring: true,
      durable: false,
    });
    const oneShot = await runtime.createCron({
      cron: "1 12 17 7 *",
      prompt: "check once",
      recurring: false,
      durable: false,
    });
    expect(await runtime.listCrons()).toEqual({
      jobs: [
        expect.objectContaining({ id: recurring.id, recurring: true }),
        expect.objectContaining({ id: oneShot.id }),
      ],
    });

    await vi.advanceTimersByTimeAsync(30_000);
    await runtime.flushActivations();
    expect(activate).toHaveBeenCalledTimes(2);
    expect(activate.mock.calls.map(([activation]) => activation.prompt)).toEqual(
      expect.arrayContaining([
        expect.stringContaining("check recurring"),
        expect.stringContaining("check once"),
      ]),
    );
    expect((await runtime.listCrons()).jobs.map((job) => job.id)).toEqual([recurring.id]);
    await expect(runtime.deleteCron(recurring.id)).resolves.toEqual({ id: recurring.id });
    await runtime.close();
  });

  it("V428 rejects invalid, impossible, and excess schedules", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 0, 1));
    const runtime = new ClaudeSessionRuntime("/tmp/project", {
      shell: fakeShell(),
      createId: (() => {
        let id = 0;
        return () => `cron_${++id}`;
      })(),
    });

    expect(() => parseCronExpression("* * *")).toThrow(/expected 5 fields/i);
    expect(() => parseCronExpression("60 * * * *")).toThrow(/outside 0\.\.59/i);
    await expect(
      runtime.createCron({
        cron: "0 0 31 2 *",
        prompt: "impossible",
        recurring: false,
        durable: false,
      }),
    ).rejects.toThrow(/next year/i);
    for (let index = 0; index < 50; index++) {
      await runtime.createCron({
        cron: "* * * * *",
        prompt: `job ${index}`,
        recurring: true,
        durable: false,
      });
    }
    await expect(
      runtime.createCron({
        cron: "* * * * *",
        prompt: "job 51",
        recurring: true,
        durable: false,
      }),
    ).rejects.toThrow(/max 50/i);
    await runtime.close();
  });

  it("V432/V437 persists, lists, and deletes a durable job", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 6, 17, 12, 0, 30));
    const root = await mkdtemp(path.join(os.tmpdir(), "swarmx-durable-cron-"));
    const store = scheduledStore(root);
    const runtime = durableRuntime(root, store, "session-1", "deadbeef");
    await runtime.start();

    const created = await runtime.createCron({
      cron: "* * * * *",
      prompt: "durable check",
      recurring: true,
      durable: true,
    });
    expect(created).toMatchObject({ id: "deadbeef", recurring: true, durable: true });
    expect(await runtime.listCrons()).toEqual({
      jobs: [expect.objectContaining({ id: "deadbeef", durable: true, recurring: true })],
    });
    expect(await store.read()).toEqual([
      expect.objectContaining({
        id: "deadbeef",
        prompt: "durable check",
        createdBySessionId: "session-1",
      }),
    ]);
    await runtime.close();

    const reopened = durableRuntime(root, store, "session-1", "unused00");
    await reopened.start();
    expect((await reopened.listCrons()).jobs).toEqual([
      expect.objectContaining({ id: "deadbeef", durable: true }),
    ]);
    await expect(reopened.deleteCron("deadbeef")).resolves.toEqual({ id: "deadbeef" });
    await expect(store.read()).resolves.toEqual([]);
    await reopened.close();
    await store.close();
    await rm(root, { recursive: true, force: true });
  });

  it("V436 removes a missed durable one-shot and asks before executing it", async () => {
    vi.useFakeTimers();
    const now = new Date(2026, 6, 17, 12, 5, 0).getTime();
    vi.setSystemTime(now);
    const root = await mkdtemp(path.join(os.tmpdir(), "swarmx-missed-cron-"));
    const store = scheduledStore(root);
    await store.start();
    await store.add({
      id: "a11ce001",
      cron: "1 12 17 7 *",
      prompt: "deploy production",
      createdAt: new Date(2026, 6, 17, 12, 0, 0).getTime(),
      createdBySessionId: "session-1",
      createdByPid: 101,
      createdByProcStart: "start-101",
    });
    const activate = vi.fn(async () => undefined);
    const runtime = durableRuntime(root, store, "session-1", "unused00");
    runtime.configure({ activate });
    await runtime.start();
    await runtime.flushActivations();

    expect(await store.read()).toEqual([]);
    expect(activate).toHaveBeenCalledOnce();
    const prompt = activate.mock.calls[0]?.[0].prompt ?? "";
    expect(prompt).toContain("Do NOT execute");
    expect(prompt).toContain("ask the user");
    expect(prompt).toContain("deploy production");
    expect(prompt).not.toContain("Execute it now");
    await runtime.close();
    await store.close();
    await rm(root, { recursive: true, force: true });
  });

  it("V436 keeps an old durable one-shot whose first occurrence is still future", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 6, 17, 12, 0, 0));
    const root = await mkdtemp(path.join(os.tmpdir(), "swarmx-future-cron-"));
    const store = scheduledStore(root);
    await store.start();
    await store.add({
      id: "f00d0001",
      cron: "0 9 31 12 *",
      prompt: "year-end task",
      createdAt: new Date(2026, 5, 1, 12, 0, 0).getTime(),
      createdBySessionId: "session-1",
      createdByPid: 101,
      createdByProcStart: "start-101",
    });
    const activate = vi.fn(async () => undefined);
    const runtime = durableRuntime(root, store, "session-1", "unused00");
    runtime.configure({ activate });
    await runtime.start();

    expect(await runtime.listCrons()).toEqual({
      jobs: [expect.objectContaining({ id: "f00d0001", durable: true })],
    });
    expect(activate).not.toHaveBeenCalled();
    await runtime.close();
    await store.close();
    await rm(root, { recursive: true, force: true });
  });

  it("V436 persists recurring lastFiredAt before activating", async () => {
    vi.useFakeTimers();
    const now = new Date(2026, 6, 17, 12, 2, 0).getTime();
    vi.setSystemTime(now);
    const root = await mkdtemp(path.join(os.tmpdir(), "swarmx-recurring-cron-"));
    const store = scheduledStore(root);
    await store.start();
    await store.add({
      id: "beef0001",
      cron: "* * * * *",
      prompt: "check recurring",
      createdAt: new Date(2026, 6, 17, 12, 0, 0).getTime(),
      recurring: true,
      createdBySessionId: "session-1",
      createdByPid: 101,
      createdByProcStart: "start-101",
    });
    const seenLastFired: number[] = [];
    const runtime = durableRuntime(root, store, "session-1", "unused00");
    runtime.configure({
      activate: async () => {
        seenLastFired.push((await store.read())[0]?.lastFiredAt ?? 0);
      },
    });
    await runtime.start();
    await vi.advanceTimersByTimeAsync(0);
    await runtime.flushActivations();

    expect(seenLastFired).toEqual([now]);
    expect((await store.read())[0]?.lastFiredAt).toBe(now);
    await runtime.close();
    await store.close();
    await rm(root, { recursive: true, force: true });
  });

  it("V435 lets a live creator fire once without duplicate lock-owner execution", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 6, 17, 12, 0, 30));
    const root = await mkdtemp(path.join(os.tmpdir(), "swarmx-multisession-cron-"));
    const store = scheduledStore(root);
    const active = new Set(["session-1", "session-2"]);
    const ownerActivation = vi.fn(async () => undefined);
    const creatorActivation = vi.fn(async () => undefined);
    const ownerRuntime = new ClaudeSessionRuntime(root, {
      shell: fakeShell(),
      scheduledTasks: store,
      owner: { sessionId: "session-1", pid: 101, procStart: "start-101" },
      createId: () => "feed0000",
      isSessionActive: (sessionId) => active.has(sessionId),
    });
    const creatorRuntime = new ClaudeSessionRuntime(root, {
      shell: fakeShell(),
      scheduledTasks: store,
      owner: { sessionId: "session-2", pid: 101, procStart: "start-101" },
      createId: () => "feed0001",
      isSessionActive: (sessionId) => active.has(sessionId),
    });
    ownerRuntime.configure({ activate: ownerActivation });
    creatorRuntime.configure({ activate: creatorActivation });
    await ownerRuntime.start();
    await creatorRuntime.start();
    await creatorRuntime.createCron({
      cron: "* * * * *",
      prompt: "creator-only prompt",
      recurring: false,
      durable: true,
    });
    await ownerRuntime.refreshScheduledTasks();
    await creatorRuntime.refreshScheduledTasks();

    await vi.advanceTimersByTimeAsync(30_000);
    await Promise.all([ownerRuntime.flushActivations(), creatorRuntime.flushActivations()]);
    expect(ownerActivation).not.toHaveBeenCalled();
    expect(creatorActivation).toHaveBeenCalledOnce();
    expect(await store.read()).toEqual([]);

    active.clear();
    await creatorRuntime.close();
    await ownerRuntime.close();
    await store.close();
    await rm(root, { recursive: true, force: true });
  });

  it("V435 lets the lock owner adopt a durable task after its creator session closes", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 6, 17, 12, 0, 30));
    const root = await mkdtemp(path.join(os.tmpdir(), "swarmx-orphan-cron-"));
    const store = scheduledStore(root);
    const active = new Set(["session-1", "session-2"]);
    const ownerActivation = vi.fn(async () => undefined);
    const ownerRuntime = new ClaudeSessionRuntime(root, {
      shell: fakeShell(),
      scheduledTasks: store,
      owner: { sessionId: "session-1", pid: 101, procStart: "start-101" },
      createId: () => "feed0002",
      isSessionActive: (sessionId) => active.has(sessionId),
    });
    const creatorRuntime = new ClaudeSessionRuntime(root, {
      shell: fakeShell(),
      scheduledTasks: store,
      owner: { sessionId: "session-2", pid: 101, procStart: "start-101" },
      createId: () => "feed0003",
      isSessionActive: (sessionId) => active.has(sessionId),
    });
    ownerRuntime.configure({ activate: ownerActivation });
    await ownerRuntime.start();
    await creatorRuntime.start();
    await creatorRuntime.createCron({
      cron: "* * * * *",
      prompt: "adopt this prompt",
      recurring: false,
      durable: true,
    });
    active.delete("session-2");
    await creatorRuntime.close();
    await ownerRuntime.refreshScheduledTasks();

    await vi.advanceTimersByTimeAsync(30_000);
    await ownerRuntime.flushActivations();
    expect(ownerActivation).toHaveBeenCalledOnce();
    expect(ownerActivation.mock.calls[0]?.[0]).toMatchObject({ jobId: "feed0003" });
    expect(await store.read()).toEqual([]);

    active.clear();
    await ownerRuntime.close();
    await store.close();
    await rm(root, { recursive: true, force: true });
  });

  it("V428 parses lists, ranges, steps, aliases, Sunday 7, and Vixie day OR semantics", () => {
    const parsed = parseCronExpression("*/15 9-10 1,15 jan,mar 0,7");
    expect([...parsed.minute.values]).toEqual([0, 15, 30, 45]);
    expect([...parsed.hour.values]).toEqual([9, 10]);
    expect([...parsed.dayOfMonth.values]).toEqual([1, 15]);
    expect([...parsed.month.values]).toEqual([1, 3]);
    expect([...parsed.dayOfWeek.values]).toEqual([0]);

    const dayOrWeek = parseCronExpression("0 9 1 jan mon");
    expect(cronExpressionMatches(dayOrWeek, new Date(2026, 0, 1, 9, 0))).toBe(true);
    expect(cronExpressionMatches(dayOrWeek, new Date(2026, 0, 5, 9, 0))).toBe(true);
    expect(cronExpressionMatches(dayOrWeek, new Date(2026, 0, 2, 9, 0))).toBe(false);
    expect(
      cronExpressionMatches(parseCronExpression("0 9 * jan 7"), new Date(2026, 0, 4, 9, 0)),
    ).toBe(true);
  });

  it("V424 replaces a session runtime when its Project root changes", async () => {
    const parent = await mkdtemp(path.join(os.tmpdir(), "swarmx-runtime-registry-"));
    const projectA = path.join(parent, "project-a");
    const projectB = path.join(parent, "project-b");
    const registry = new ClaudeSessionRuntimeRegistry();
    const first = await registry.open("session-1", projectA);
    const same = await registry.open("session-1", projectA);
    expect(registry.isRunning("session-1")).toBe(false);
    await first.beginForeground();
    expect(registry.isRunning("session-1")).toBe(true);
    first.endForeground();
    expect(registry.isRunning("session-1")).toBe(false);
    const replacement = await registry.open("session-1", projectB);

    expect(same).toBe(first);
    expect(replacement).not.toBe(first);
    await expect(first.listCrons()).rejects.toThrow(/closed/i);
    await registry.close();
    await rm(parent, { recursive: true, force: true });
  });
});

function scheduledStore(root: string): ClaudeScheduledTaskStore {
  return new ClaudeScheduledTaskStore(root, {
    watch: false,
    validateCron: (cron) => {
      try {
        parseCronExpression(cron);
        return true;
      } catch {
        return false;
      }
    },
    isProcessAlive: async (pid) => pid === 101,
    processStart: async (pid) => (pid === 101 ? "start-101" : undefined),
  });
}

function durableRuntime(
  root: string,
  store: ClaudeScheduledTaskStore,
  sessionId: string,
  id: string,
): ClaudeSessionRuntime {
  return new ClaudeSessionRuntime(root, {
    shell: fakeShell(),
    scheduledTasks: store,
    owner: { sessionId, pid: 101, procStart: "start-101" },
    createId: () => id,
    isSessionActive: () => true,
  });
}

function monitorSnapshot(status: "running" | "completed" | "stopped") {
  return {
    sessionId: 7,
    status,
    command: "watch-build",
    cwd: "/tmp/project",
    exitCode: status === "running" ? null : 0,
    signal: null,
    stdout: "",
    stderr: "",
    durationMs: 0,
    timedOut: false,
    truncated: false,
  } as const;
}

function fakeShell(): WorkspaceShell {
  return {
    close: vi.fn(async () => undefined),
  } as unknown as WorkspaceShell;
}
