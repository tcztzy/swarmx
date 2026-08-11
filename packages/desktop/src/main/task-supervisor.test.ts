import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  outcomes: [] as Array<unknown | Error>,
  request: vi.fn(async () => {
    const outcome = mocks.outcomes.shift();
    if (outcome instanceof Error) throw outcome;
    return outcome;
  }),
  client: vi.fn(),
  spawn: vi.fn(() => ({ unref: vi.fn() })),
}));

vi.mock("node:child_process", () => ({ spawn: mocks.spawn }));
vi.mock("@swarmx/core", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@swarmx/core")>();
  return {
    ...actual,
    TaskSupervisorClient: class {
      constructor(options: unknown) {
        mocks.client(options);
      }

      request(command: unknown) {
        return mocks.request(command);
      }
    },
  };
});

import { DesktopTaskSupervisor } from "./task-supervisor.js";

const roots: string[] = [];

beforeEach(() => {
  mocks.outcomes.length = 0;
  mocks.request.mockClear();
  mocks.client.mockClear();
  mocks.spawn.mockClear();
  vi.stubEnv("SWARMX_TEST_SECRET", "must-not-reach-supervisor");
});

afterEach(() => {
  vi.unstubAllEnvs();
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("DesktopTaskSupervisor", () => {
  it("reuses an available authenticated supervisor without spawning a process", async () => {
    const rootDir = temporaryRoot();
    mocks.outcomes.push(
      { ok: true, operation: "ping" },
      { ok: true, operation: "list", workItems: [] },
    );
    const supervisor = new DesktopTaskSupervisor({ rootDir });

    await expect(supervisor.request({ operation: "list" })).resolves.toMatchObject({
      operation: "list",
      workItems: [],
    });

    expect(mocks.client).toHaveBeenCalledTimes(1);
    expect(mocks.spawn).not.toHaveBeenCalled();
  });

  it("shares one detached startup and forwards only the supervisor environment allowlist", async () => {
    const rootDir = temporaryRoot();
    const unavailable = Object.assign(new Error("not running"), { code: "ECONNREFUSED" });
    mocks.outcomes.push(
      unavailable,
      { ok: true, operation: "ping" },
      { ok: true, operation: "list", workItems: [] },
      { ok: true, operation: "list", workItems: [] },
    );
    const supervisor = new DesktopTaskSupervisor({
      rootDir,
      executablePath: "/runtime/node",
      entryPath: "/app/task-supervisor-entry.js",
    });

    await expect(
      Promise.all([
        supervisor.request({ operation: "list" }),
        supervisor.request({ operation: "list" }),
      ]),
    ).resolves.toHaveLength(2);

    expect(mocks.client).toHaveBeenCalledTimes(1);
    expect(mocks.spawn).toHaveBeenCalledTimes(1);
    expect(mocks.spawn).toHaveBeenCalledWith(
      "/runtime/node",
      ["/app/task-supervisor-entry.js"],
      expect.objectContaining({
        detached: true,
        stdio: "ignore",
        cwd: rootDir,
        env: expect.objectContaining({
          ELECTRON_RUN_AS_NODE: "1",
          SWARMX_TASK_RUNTIME_ROOT: rootDir,
        }),
      }),
    );
    const environment = mocks.spawn.mock.calls[0]?.[2]?.env;
    expect(environment).not.toHaveProperty("SWARMX_TEST_SECRET");
  });

  it("does not spawn on a non-availability error and retries connection later", async () => {
    const rootDir = temporaryRoot();
    const denied = Object.assign(new Error("denied"), { code: "EACCES" });
    mocks.outcomes.push(
      denied,
      { ok: true, operation: "ping" },
      { ok: true, operation: "list", workItems: [] },
    );
    const supervisor = new DesktopTaskSupervisor({ rootDir });

    await expect(supervisor.request({ operation: "list" })).rejects.toThrow("denied");
    await expect(supervisor.request({ operation: "list" })).resolves.toMatchObject({
      operation: "list",
    });

    expect(mocks.client).toHaveBeenCalledTimes(2);
    expect(mocks.spawn).not.toHaveBeenCalled();
  });
});

function temporaryRoot(): string {
  const root = mkdtempSync(path.join(tmpdir(), "swarmx-desktop-supervisor-"));
  roots.push(root);
  return root;
}
