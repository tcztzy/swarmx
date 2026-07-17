import { mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type ClaudeScheduledTaskOwner,
  ClaudeScheduledTaskStore,
} from "./claude-scheduled-tasks.js";
import { parseCronExpression } from "./claude-session-runtime.js";

const roots: string[] = [];

afterEach(async () => {
  vi.useRealTimers();
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("ClaudeScheduledTaskStore", () => {
  it("V432 writes the exact private Claude scheduled_tasks document", async () => {
    const root = await tempRoot();
    const store = taskStore(root, { watch: false });
    await store.start();
    await store.add({
      id: "a1b2c3d4",
      cron: "*/5 * * * *",
      prompt: "check the build",
      createdAt: 1_784_271_000_000,
      recurring: true,
      createdBySessionId: "session-1",
      createdByPid: 101,
      createdByProcStart: "Fri Jul 17 10:00:00 2026",
    });

    const contents = await readFile(store.filePath, "utf8");
    expect(JSON.parse(contents)).toEqual({
      tasks: [
        {
          id: "a1b2c3d4",
          cron: "*/5 * * * *",
          prompt: "check the build",
          createdAt: 1_784_271_000_000,
          recurring: true,
          createdBySessionId: "session-1",
          createdByPid: 101,
          createdByProcStart: "Fri Jul 17 10:00:00 2026",
        },
      ],
    });
    expect(contents.endsWith("\n")).toBe(true);
    expect(contents).not.toContain("durable");
    expect((await stat(store.filePath)).mode & 0o777).toBe(0o600);
    await store.close();
  });

  it("V433 treats corrupt files as empty and filters malformed or invalid tasks", async () => {
    const root = await tempRoot();
    const store = taskStore(root, { watch: false });
    await store.start();
    await writeFile(store.filePath, "not-json", "utf8");
    await expect(store.read()).resolves.toEqual([]);

    await writeFile(
      store.filePath,
      JSON.stringify({
        tasks: [
          { id: "valid001", cron: "0 9 * * *", prompt: "valid", createdAt: 10 },
          { id: "badcron1", cron: "90 * * * *", prompt: "bad", createdAt: 11 },
          { id: "missing1", cron: "0 9 * * *", createdAt: 12 },
          null,
        ],
      }),
      "utf8",
    );
    await expect(store.read()).resolves.toEqual([
      { id: "valid001", cron: "0 9 * * *", prompt: "valid", createdAt: 10 },
    ]);
    await store.close();
  });

  it("V432 rejects a .claude symlink instead of persisting outside the Project", async () => {
    const root = await tempRoot();
    const outside = await tempRoot();
    await mkdir(outside, { recursive: true });
    await symlink(outside, path.join(root, ".claude"));
    const store = taskStore(root, { watch: false });
    await expect(store.start()).rejects.toThrow(/real Project directory/i);
    await store.close();
  });

  it("V433 serializes concurrent mutations without losing tasks", async () => {
    const root = await tempRoot();
    const store = taskStore(root, { watch: false });
    await store.start();
    await Promise.all(
      Array.from({ length: 12 }, (_, index) =>
        store.add({
          id: index.toString(16).padStart(8, "0"),
          cron: "* * * * *",
          prompt: `task ${index}`,
          createdAt: index,
        }),
      ),
    );
    expect(await store.read()).toHaveLength(12);
    await store.close();
  });

  it("V434 preserves a live lock and recovers a stale or PID-reused lock", async () => {
    const root = await tempRoot();
    const starts = new Map<number, string>([
      [101, "owner-start"],
      [202, "contender-start"],
    ]);
    const store = taskStore(root, {
      watch: false,
      isProcessAlive: async (pid) => starts.has(pid),
      processStart: async (pid) => starts.get(pid),
    });
    const owner = lockOwner("owner", 101, "owner-start");
    const contender = lockOwner("contender", 202, "contender-start");
    await store.start();

    await expect(store.acquireLock(owner)).resolves.toBe(true);
    await expect(store.acquireLock(contender)).resolves.toBe(false);
    starts.set(101, "reused-process-start");
    await expect(store.acquireLock(contender)).resolves.toBe(true);
    await store.releaseLock(owner);
    expect(JSON.parse(await readFile(store.lockPath, "utf8"))).toMatchObject({
      sessionId: "contender",
      pid: 202,
    });
    await store.releaseLock(contender);
    await expect(readFile(store.lockPath, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
    await store.close();
  });

  it("V434 notifies subscribers when another writer changes the task file", async () => {
    const root = await tempRoot();
    const store = taskStore(root);
    await store.start();
    const changed = new Promise<void>((resolve) => {
      const unsubscribe = store.subscribe(() => {
        unsubscribe();
        resolve();
      });
    });
    await writeFile(store.filePath, '{"tasks":[]}\n', { mode: 0o600 });
    await expect(Promise.race([changed, rejectAfter(2_000)])).resolves.toBeUndefined();
    await store.close();
  });
});

function taskStore(
  root: string,
  options: ConstructorParameters<typeof ClaudeScheduledTaskStore>[1] = {},
): ClaudeScheduledTaskStore {
  return new ClaudeScheduledTaskStore(root, {
    validateCron: (cron) => {
      try {
        parseCronExpression(cron);
        return true;
      } catch {
        return false;
      }
    },
    ...options,
  });
}

async function tempRoot(): Promise<string> {
  const root = await mkdtemp(path.join(os.tmpdir(), "swarmx-scheduled-tasks-"));
  roots.push(root);
  return root;
}

function lockOwner(sessionId: string, pid: number, procStart: string): ClaudeScheduledTaskOwner {
  return { sessionId, pid, procStart };
}

function rejectAfter(delayMs: number): Promise<never> {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error("Timed out waiting for scheduled task change")), delayMs);
  });
}
