import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as waitRealtime } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { afterAll, afterEach, expect, it, vi } from "vitest";
import { createExternalAcp } from "../src/agents/external-acp.js";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import type { AgentMemory } from "../src/host/memory.js";
import { ProductServices } from "../src/host/product-services.js";
import { recordedAgent } from "../src/host/recorded-agent.js";
import { policyPermissions } from "../src/permissions.js";
import { DEFAULT_POLICY, type ExecutionPolicy } from "../src/settings.js";

// These cases start real Node/ACP processes and some deliberately restart them.
vi.setConfig({ testTimeout: 15_000, hookTimeout: 15_000 });

const roots: string[] = [];
const pending: Promise<unknown>[] = [];
const startupMs: number[] = [];
const initializeToNewMs: number[] = [];
function pendingOperation<T>(operation: Promise<T>): Promise<T> {
  // Observe early rejection without changing the promise asserted by the test.
  pending.push(
    operation.then(
      () => undefined,
      () => undefined,
    ),
  );
  return operation;
}
const agents: NativeAgent[] = [];
const fixture = fileURLToPath(new URL("./fixtures/external-acp.mjs", import.meta.url));
async function waitWithFrozenTimers(check: () => void | Promise<void>, timeout = 2_000) {
  const deadline = Date.now() + timeout;
  for (;;) {
    try {
      await check();
      return;
    } catch (error) {
      if (Date.now() >= deadline) throw error;
      await waitRealtime(10);
    }
  }
}
const observer = (): Observer => ({
  text: vi.fn(),
  tool: vi.fn(),
  raw: vi.fn(),
  activity: vi.fn(),
  interact: vi.fn(async () => ({ optionId: "no" })),
});
async function setup(
  fixtureEnv: Record<string, string> = {},
  executionPolicy: () => ExecutionPolicy = () => ({ ...DEFAULT_POLICY, harnesses: { acp: null } }),
) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-acp-test-"));
  roots.push(root);
  const path = join(root, "agent.json");
  await writeFile(
    path,
    JSON.stringify({
      id: "fixture",
      command: process.execPath,
      args: [fixture],
      env: { ACP_FIXTURE_HOME: root, ...fixtureEnv },
    }),
  );
  vi.stubEnv("SWARMX_ACP_AGENT", path);
  const open = async () => {
    const agent = await createExternalAcp({
      cwd: root,
      executionPolicy,
      mcp: { command: "secret", args: [], env: { SWARMX_MCP_TOKEN: "do-not-forward" } },
    });
    agents.push(agent);
    return agent;
  };
  const events = async () =>
    (await readFile(join(root, "events.jsonl"), "utf8").catch(() => ""))
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  const wait = async (event: string) =>
    vi.waitFor(async () => expect((await events()).some((row) => row.event === event)).toBe(true), {
      timeout: 5_000,
    });
  const touch = (name: string) => writeFile(join(root, name), "");
  return { root, path, open, events, wait, touch };
}
afterEach(async ({ task }) => {
  const ownedRoots = roots.splice(0);
  try {
    // Hook cleanup runs after all behavioral assertions, including failed waits.
    await Promise.all(
      ownedRoots.flatMap((root) =>
        ["release-resume", "release-config", "release-terminal", "release-exit"].map((name) =>
          writeFile(join(root, name), ""),
        ),
      ),
    );
    await Promise.allSettled(agents.splice(0).map((agent) => agent.dispose()));
    await Promise.all(pending.splice(0));
    for (const root of ownedRoots) {
      const rows = (await readFile(join(root, "events.jsonl"), "utf8").catch(() => ""))
        .trim()
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line));
      if (task.result?.state === "fail")
        console.info("ACP fixture failure phases", {
          test: task.name,
          phases: rows.map((row) => ({ event: row.event, pid: row.pid, at: row.at })),
        });
      for (const started of rows.filter((row) => row.event === "started")) {
        if (typeof started.startupMs === "number") startupMs.push(started.startupMs);
        const initialized = rows.find(
          (row) => row.pid === started.pid && row.event === "initialize",
        );
        const created = rows.find((row) => row.pid === started.pid && row.event === "new");
        if (initialized && created) initializeToNewMs.push(created.at - initialized.at);
      }
    }
  } finally {
    vi.useRealTimers();
    vi.unstubAllEnvs();
    await Promise.all(ownedRoots.map((root) => rm(root, { recursive: true, force: true })));
  }
});
afterAll(() => {
  console.info("ACP fixture phase timings (ms)", { startupMs, initializeToNewMs });
});

it("requires explicit ACP admission and a strict endpoint configuration", async () => {
  expect(policyPermissions({}).harnesses.acp).toBeUndefined();
  expect(policyPermissions({ harnesses: { acp: null } }).harnesses.acp).toBeNull();
  const s = await setup();
  await writeFile(
    s.path,
    JSON.stringify({ id: "fixture", command: process.execPath, trustAll: true }),
  );
  await expect(s.open()).rejects.toThrow();
  expect(await s.events()).toEqual([]);
});

it("rejects a missing ACP grant before launching any child", async () => {
  const s = await setup();
  const agent = await createExternalAcp({
    cwd: s.root,
    executionPolicy: () => DEFAULT_POLICY,
    mcp: { command: "", args: [], env: {} },
  });
  agents.push(agent);
  await expect(agent.list()).rejects.toThrow("explicit Host policy grant");
  expect(await s.events()).toEqual([]);
});

it.each(["list", "resume"] as const)(
  "rejects missing native %s capability before any session or tool dispatch",
  async (missing) => {
    // Handlers still exist in the fixture: only its advertised capability is absent.
    const s = await setup({ ACP_FIXTURE_MISSING_CAPABILITY: missing });
    const a = await s.open();
    const out = observer();
    const attempt = missing === "list" ? a.list() : a.start("fixture:existing", "never", out);
    await expect(attempt).rejects.toThrow(
      "External ACP agent must advertise native list and resume support.",
    );
    const rows = await s.events();
    expect(rows.map((row) => row.event).sort()).toEqual(["closed", "initialize", "started"]);
    const pid = rows.find((row) => row.event === "started").pid;
    expect(() => process.kill(pid, 0)).toThrow(expect.objectContaining({ code: "ESRCH" }));
    expect(out.interact).not.toHaveBeenCalled();
    expect(out.tool).not.toHaveBeenCalled();
    expect(out.text).not.toHaveBeenCalled();
    await a.dispose();
    console.info("ACP rejected capability evidence", { missing, pid, events: rows });
  },
);

it("uses a real stdio child and remote history across process restart without recreating the session", async () => {
  const s = await setup();
  const a = await s.open();
  expect(await a.models()).toEqual({ models: [], current: {} });
  const id = await a.create();
  expect(id).toMatch(/^fixture:/);
  const out = observer();
  expect(await a.start(id, "first", out, { model: "local", effort: "off" })).toEqual({
    stopReason: "end_turn",
  });
  expect(out.text).toHaveBeenCalledWith(expect.any(String), "native:first", "assistant");
  expect(out.tool).toHaveBeenLastCalledWith("tool-1", "local", { text: "first" }, { ok: true });
  await a.dispose();
  let rows = await s.events();
  const first = rows.find((row) => row.event === "started").pid;
  expect(() => process.kill(first, 0)).toThrow();
  const b = await s.open();
  expect((await b.list()).map((x) => x.sessionId)).toEqual([id]);
  const history = observer();
  await b.read(id, history);
  expect(history.text).toHaveBeenCalledWith(expect.any(String), "native:first", "assistant");
  await b.start(id, "second", observer());
  await b.dispose();
  rows = await s.events();
  expect(new Set(rows.filter((x) => x.event === "started").map((x) => x.pid)).size).toBe(2);
  expect(rows.filter((x) => x.event === "new")).toHaveLength(1);
  expect(rows.filter((x) => x.event === "tool")).toHaveLength(2);
  expect(rows.filter((x) => x.event === "closed")).toHaveLength(2);
  expect(rows.find((x) => x.event === "new").params.mcpServers).toEqual([]);
  expect(rows.find((x) => x.event === "initialize").params.clientCapabilities).toMatchObject({
    fs: { readTextFile: false, writeTextFile: false },
    terminal: false,
  });
});

it("fails remote resume and unknown selections without a new session or tool fallback", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  await a.dispose();
  const b = await s.open();
  await s.touch("fail-resume");
  await expect(b.start(id, "never", observer())).rejects.toThrow();
  expect((await s.events()).filter((x) => x.event === "new")).toHaveLength(1);
  await rm(join(s.root, "fail-resume"));
  await expect(b.start(id, "never", observer(), { model: "unknown" })).rejects.toThrow(
    "advertised",
  );
  await expect(b.start(id, "never", observer(), { mode: "yolo" })).rejects.toThrow("advertised");
  expect((await s.events()).filter((x) => x.event === "prompt")).toHaveLength(0);
});

it("Stop during remote preparation prevents dispatch after the preparation is released", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  await a.dispose();
  const b = await s.open();
  // Separate real child startup from the deliberately paused remote resume.
  expect((await b.list()).map((session) => session.sessionId)).toEqual([id]);
  await s.touch("pause-resume");
  const run = pendingOperation(b.start(id, "never", observer()));
  await s.wait("resume-paused");
  await b.interrupt(id);
  await s.touch("release-resume");
  expect(await run).toEqual({ stopReason: "cancelled" });
  expect((await s.events()).filter((x) => x.event === "prompt" || x.event === "tool")).toHaveLength(
    0,
  );
  expect(await b.start(id, "retry", observer())).toEqual({ stopReason: "end_turn" });
});

it("Stop waits for the prompt terminal after sending cancel", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  const run = pendingOperation(a.start(id, "cancel", observer()));
  await s.wait("prompt");
  let stopped = false;
  const stop = pendingOperation(
    a.interrupt(id).then(() => {
      stopped = true;
    }),
  );
  await s.wait("terminal-paused");
  expect(stopped).toBe(false);
  await s.touch("release-terminal");
  await stop;
  expect(await run).toEqual({ stopReason: "cancelled" });
});

it.each(["permission", "permission-observed"])(
  "denies a late %s answer after cancellation and never broadens unknown answers",
  async (prompt) => {
    const s = await setup();
    const a = await s.open();
    const id = await a.create();
    const out = observer();
    let answer: (x: unknown) => void = () => {};
    out.interact = vi.fn(
      () =>
        new Promise((resolve) => {
          answer = resolve;
        }),
    );
    const run = pendingOperation(a.start(id, prompt, out));
    await vi.waitFor(() => expect(out.interact).toHaveBeenCalledOnce());
    const stop = pendingOperation(a.interrupt(id));
    answer({ optionId: "yes" });
    await stop;
    expect(await run).toEqual({ stopReason: "cancelled" });
    expect(
      (await s.events()).find((x) => x.event === "permission-result").response.outcome,
    ).toEqual({
      outcome: "cancelled",
    });
    expect((await s.events()).filter((x) => x.event === "tool")).toHaveLength(0);
    const other = observer();
    other.interact = async () => ({ optionId: "invented" });
    expect(await a.start(id, "permission", other)).toEqual({ stopReason: "refusal" });
  },
);

it("fails a child exit without a terminal and disposal waits for the owned child", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  await expect(a.start(id, "exit", observer())).rejects.toThrow();
  await a.dispose();
  const pid = (await s.events()).find((x) => x.event === "started").pid;
  expect(() => process.kill(pid, 0)).toThrow();
  await expect(a.start(id, "never", observer())).rejects.toThrow("closed");
});

it("keeps Host memory out of external sessions while retaining observed events and explicit instruction errors", async () => {
  const s = await setup();
  const native = await s.open();
  const journal = new ExecutionJournal(s.root, "external");
  const memory = {
    context: vi.fn(async () => "private Host context"),
    snapshot: vi.fn(async () => "private Host context"),
    queueReview: vi.fn(),
  } as unknown as AgentMemory;
  const wrapped = recordedAgent(journal, "acp", native, memory);
  try {
    await expect(wrapped.create({ instructions: "explicit system override" })).rejects.toThrow(
      "system instructions",
    );
    const id = await wrapped.create();
    expect(await wrapped.start(id, "ordinary", observer())).toEqual({ stopReason: "end_turn" });
    expect(memory.context).not.toHaveBeenCalled();
    expect(memory.snapshot).not.toHaveBeenCalled();
    expect(
      journal.read({ session: id }).events.some((row) => row.event.type === "RUN_FINISHED"),
    ).toBe(true);
  } finally {
    await wrapped.dispose();
    journal.close();
  }
});

it("does not dispatch after cancellation during an asynchronous config acknowledgement", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  await s.touch("pause-config");
  const run = pendingOperation(a.start(id, "never", observer(), { model: "other" }));
  await s.wait("config-paused");
  await a.interrupt(id);
  await s.touch("release-config");
  expect(await run).toEqual({ stopReason: "cancelled" });
  expect((await s.events()).filter((x) => x.event === "prompt" || x.event === "tool")).toHaveLength(
    0,
  );
});

it("disposal waits for an active prompt terminal and then the actual child exit", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  const run = pendingOperation(a.start(id, "cancel", observer()));
  await s.wait("prompt");
  // Only the parent's shutdown deadlines are frozen while the fixture holds terminal.
  // Real child execution, cancel, terminal delivery, EOF and process exit remain live.
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  let disposed = false;
  const dispose = pendingOperation(
    a.dispose().then(() => {
      disposed = true;
    }),
  );
  try {
    await waitWithFrozenTimers(async () => {
      const rows = await s.events();
      expect(rows.some((row) => row.event === "terminal-paused")).toBe(true);
      expect(rows.some((row) => row.event === "closed")).toBe(false);
      const pid = rows.find((row) => row.event === "started").pid;
      expect(() => process.kill(pid, 0)).not.toThrow();
    }, 5_000);
    expect(disposed).toBe(false);
    await s.touch("release-terminal");
    expect(await run).toEqual({ stopReason: "cancelled" });
    await dispose;
    const rows = await s.events();
    const pid = rows.find((x) => x.event === "started").pid;
    expect(() => process.kill(pid, 0)).toThrow();
    expect(rows.findIndex((x) => x.event === "terminal")).toBeLessThan(
      rows.findIndex((x) => x.event === "closed"),
    );
    expect(rows.find((x) => x.event === "started").hostToken).toBeNull();
  } finally {
    await s.touch("release-terminal");
    const cleanup = pendingOperation(a.dispose());
    try {
      await vi.runAllTimersAsync();
      await Promise.allSettled([run, dispose, cleanup]);
    } finally {
      vi.useRealTimers();
    }
  }
});

it.skipIf(process.platform === "win32")(
  "emergency disposal force-kills the owned process group without executing a late approval",
  async () => {
    const s = await setup({
      ACP_FIXTURE_HOLD_EXIT: "1",
      ACP_FIXTURE_HOLD_PERMISSION_TERMINAL: "1",
    });
    const a = await s.open();
    const id = await a.create();
    const out = observer();
    const answer = Promise.withResolvers<unknown>();
    out.interact = vi.fn(() => answer.promise);
    let runSettled = false;
    const run = a.start(id, "permission", out).then(
      (result) => {
        runSettled = true;
        return { result, error: undefined };
      },
      (error: unknown) => {
        runSettled = true;
        return { result: undefined, error };
      },
    );
    await vi.waitFor(() => expect(out.interact).toHaveBeenCalledOnce(), { timeout: 5_000 });
    const pid = (await s.events()).find((row) => row.event === "started").pid;
    // Only parent shutdown timers are controlled; SDK stdio, child execution, signals,
    // and process exit remain real. The two 3 s deadlines are deliberately separate.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const kill = vi.spyOn(process, "kill");
    const signals = () => kill.mock.calls.filter(([, signal]) => signal !== 0);
    let disposed = false;
    const dispose = pendingOperation(
      a.dispose().then(() => {
        disposed = true;
      }),
    );
    try {
      await waitWithFrozenTimers(async () => {
        const rows = await s.events();
        expect(rows.some((row) => row.event === "cancel")).toBe(true);
        expect(rows.some((row) => row.event === "terminal-paused")).toBe(true);
        expect(rows.find((row) => row.event === "permission-result").response.outcome).toEqual({
          outcome: "cancelled",
        });
      }, 5_000);
      // Release the UI answer while the real child and connection are still alive.
      answer.resolve({ optionId: "yes" });
      await answer.promise;
      for (const operation of [
        () => a.create(),
        () => a.list(),
        () => a.read("fixture:other", observer()),
        () => a.models(id),
        () => a.start("fixture:other", "never", observer()),
      ])
        await expect(operation()).rejects.toThrow("closed");
      await vi.advanceTimersByTimeAsync(2_999);
      expect(runSettled).toBe(false);
      expect(disposed).toBe(false);
      expect(signals()).toEqual([]);
      expect((await s.events()).some((row) => row.event === "exit-paused")).toBe(false);

      await vi.advanceTimersByTimeAsync(1);
      await waitWithFrozenTimers(async () => {
        expect((await s.events()).some((row) => row.event === "exit-paused")).toBe(true);
        expect(runSettled).toBe(true);
      }, 5_000);
      const outcome = await run;
      expect(outcome.result).toBeUndefined();
      expect(outcome.error).toBeInstanceOf(Error);
      expect(disposed).toBe(false);
      expect(() => process.kill(pid, 0)).not.toThrow();

      await vi.advanceTimersByTimeAsync(499);
      expect(signals()).toEqual([]);
      await vi.advanceTimersByTimeAsync(1);
      await waitWithFrozenTimers(async () => {
        expect((await s.events()).some((row) => row.event === "dispose-waiting")).toBe(true);
      }, 5_000);
      expect(signals()).toEqual([[-pid, "SIGTERM"]]);
      expect(() => process.kill(pid, 0)).not.toThrow();
      await vi.advanceTimersByTimeAsync(2_499);
      expect(signals()).toEqual([[-pid, "SIGTERM"]]);
      expect(disposed).toBe(false);
      expect(() => process.kill(pid, 0)).not.toThrow();

      await vi.advanceTimersByTimeAsync(1);
      await waitWithFrozenTimers(() => expect(disposed).toBe(true), 5_000);
      await dispose;
      expect(signals()).toEqual([
        [-pid, "SIGTERM"],
        [-pid, "SIGKILL"],
      ]);
      expect(() => process.kill(pid, 0)).toThrow(expect.objectContaining({ code: "ESRCH" }));
      // Releasing all native pauses after disposal cannot resurrect work or approval.
      await s.touch("release-terminal");
      await s.touch("release-exit");
      await expect(a.start(id, "never-after-exit", observer())).rejects.toThrow("closed");
      const rows = await s.events();
      expect(rows.filter((row) => row.event === "started")).toHaveLength(1);
      expect(rows.filter((row) => row.event === "initialize")).toHaveLength(1);
      expect(rows.filter((row) => row.event === "new")).toHaveLength(1);
      expect(rows.filter((row) => row.event === "prompt").map((row) => row.text)).toEqual([
        "permission",
      ]);
      expect(rows.filter((row) => row.event === "cancel")).toHaveLength(1);
      expect(rows.filter((row) => row.event === "permission-result")).toHaveLength(1);
      expect(rows.filter((row) => ["tool", "terminal", "closed"].includes(row.event))).toHaveLength(
        0,
      );
      expect(out.interact).toHaveBeenCalledOnce();
      expect(out.tool).not.toHaveBeenCalled();
      expect(out.text).not.toHaveBeenCalled();
      console.info("ACP emergency disposal evidence", {
        pid,
        signals: signals(),
        promptError: (outcome.error as Error).message,
        parentTimerBoundariesMs: [2_999, 3_000, 3_499, 3_500, 5_999, 6_000],
        events: rows,
      });
    } finally {
      answer.resolve({ optionId: "no" });
      await s.touch("release-terminal");
      await s.touch("release-exit");
      const cleanup = pendingOperation(a.dispose());
      try {
        await vi.runAllTimersAsync();
        await Promise.allSettled([run, dispose, cleanup]);
      } finally {
        kill.mockRestore();
        vi.useRealTimers();
      }
    }
  },
);

it("reconnects concurrent readers through one owned child and resumes the exact existing session", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  await expect(a.start(id, "exit", observer())).rejects.toThrow();
  const [sessions, models] = await Promise.all([a.list(), a.models(id)]);
  expect(sessions.map((x) => x.sessionId)).toEqual([id]);
  expect(models.current.model).toBe("local");
  await a.dispose();
  const rows = await s.events();
  expect(rows.filter((x) => x.event === "started")).toHaveLength(2);
  expect(rows.filter((x) => x.event === "new")).toHaveLength(1);
  for (const row of rows.filter((x) => x.event === "started"))
    expect(() => process.kill(row.pid, 0)).toThrow();
});

it.each([true, false])(
  "enforces ACP admission through the real Host registry with history=%s",
  async (historySupported) => {
    const s = await setup(historySupported ? {} : { ACP_FIXTURE_MISSING_CAPABILITY: "load" });
    const options = { cwd: s.root, productHome: join(s.root, "host") };
    const denied = await ProductServices.create(options);
    try {
      await expect(denied.attachAgents("http://localhost", undefined, "acp")).rejects.toThrow(
        "explicit Host policy grant",
      );
      expect(await s.events()).toEqual([]);
    } finally {
      await denied.dispose();
    }
    const products = await ProductServices.create(options);
    try {
      products.updatePolicy({
        ...DEFAULT_POLICY,
        harnesses: { acp: ["local"] },
        tools: [],
        delegation: false,
      });
      await products.attachAgents("http://localhost", undefined, "acp");
      const id = await products.rootAgent.create();
      expect(id).toMatch(/^acp:fixture:/);
      const out = observer();
      await expect(products.rootAgent.start(id, "host", out, { model: "local" })).resolves.toEqual({
        stopReason: "end_turn",
      });
      await expect(
        products.rootAgent.start(id, "never", observer(), { model: "other" }),
      ).rejects.toThrow("permitted model");
      expect(products.rootAgent.capabilities.history).toBe(historySupported);
      const history = observer();
      if (historySupported) {
        await products.rootAgent.read(id, history);
        expect(history.text).toHaveBeenCalledWith(expect.any(String), "native:host", "assistant");
      } else {
        await expect(products.rootAgent.read(id, history)).rejects.toThrow(
          "does not support native history replay",
        );
        expect(history.text).not.toHaveBeenCalled();
      }
      expect(
        products.journal
          .read({ session: id })
          .events.some((row) => row.event.type === "RUN_FINISHED"),
      ).toBe(true);
      expect((await s.events()).filter((x) => x.event === "prompt").map((x) => x.text)).toEqual([
        "host",
      ]);
    } finally {
      await products.dispose();
    }
    for (const row of (await s.events()).filter((x) => x.event === "started"))
      expect(() => process.kill(row.pid, 0)).toThrow();
  },
);

it.each(["close", "revoke"] as const)(
  "does not spawn a replacement if the Host %s occurs while an old child is disposing",
  async (action) => {
    let granted = true;
    const s = await setup({ ACP_FIXTURE_HOLD_EXIT: "1" }, () => ({
      ...DEFAULT_POLICY,
      harnesses: granted ? { acp: null } : {},
    }));
    const a = await s.open();
    const id = await a.create();
    await expect(a.start(id, "disconnect-hold", observer())).rejects.toThrow();
    // Freeze only parent timeout callbacks: the real child and stdio keep running.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const retried = a.list().then(
      (sessions) => ({ sessions, error: undefined }),
      (error: unknown) => ({ sessions: undefined, error }),
    );
    let disposed: Promise<void> | undefined;
    try {
      await waitWithFrozenTimers(() => expect(vi.getTimerCount()).toBeGreaterThanOrEqual(2));
      // Deliver graceful termination, then hold the force-kill clock at this boundary.
      await vi.advanceTimersByTimeAsync(500);
      await waitWithFrozenTimers(async () => {
        const rows = await s.events();
        expect(rows.some((row) => row.event === "dispose-waiting")).toBe(true);
        expect(rows.some((row) => row.event === "exit-paused")).toBe(true);
        expect(rows.some((row) => row.event === "closed")).toBe(false);
        const pid = rows.find((row) => row.event === "started").pid;
        expect(() => process.kill(pid, 0)).not.toThrow();
      });
      if (action === "close") disposed = a.dispose();
      else granted = false;
      await s.touch("release-exit");
      const result = await retried;
      expect(result.error).toBeInstanceOf(Error);
      expect((result.error as Error).message).toContain(
        action === "close" ? "closed" : "explicit Host policy grant",
      );
      await disposed;
      const rows = await s.events();
      expect(rows.filter((row) => row.event === "started")).toHaveLength(1);
      expect(rows.filter((row) => row.event === "initialize")).toHaveLength(1);
      expect(rows.filter((row) => row.event === "tool")).toHaveLength(0);
    } finally {
      await s.touch("release-exit");
      const cleanup = a.dispose();
      await vi.runAllTimersAsync();
      await cleanup;
      vi.useRealTimers();
      await retried;
    }
  },
  10_000,
);

it("retains legacy modes when a model config response only contains configOptions", async () => {
  const s = await setup({ ACP_FIXTURE_LEGACY_MODES: "1" });
  const a = await s.open();
  const id = await a.create();
  expect((await a.models(id)).modes?.map((mode) => mode.id)).toEqual(["safe", "plan"]);
  expect(await a.start(id, "configured", observer(), { model: "other", mode: "plan" })).toEqual({
    stopReason: "end_turn",
  });
  const rows = await s.events();
  const configIndex = rows.findIndex((row) => row.event === "config");
  const modeIndex = rows.findIndex((row) => row.event === "mode");
  expect(configIndex).toBeGreaterThan(-1);
  expect(modeIndex).toBeGreaterThan(configIndex);
  expect(rows[modeIndex].params.modeId).toBe("plan");
  expect(rows.filter((row) => row.event === "prompt")).toHaveLength(1);
  const advertised = await a.models(id);
  expect(advertised.current).toMatchObject({ model: "other", mode: "plan" });
  expect(advertised.modes?.map((mode) => mode.id)).toEqual(["safe", "plan"]);
});

it("filters Host credentials from the actual parent environment before spawning", async () => {
  vi.stubEnv("SWARMX_API_TOKEN", "fake-parent-host-api-token");
  vi.stubEnv("SWARMX_MCP_TOKEN", "fake-parent-host-mcp-token");
  const s = await setup();
  const a = await s.open();
  await a.create();
  await a.dispose();
  expect((await s.events()).find((row) => row.event === "started")).toMatchObject({
    inheritedHostApiToken: false,
    inheritedHostMcpToken: false,
    hostToken: null,
  });
});

it.each(["SWARMX_API_TOKEN", "SWARMX_MCP_TOKEN", "swarmx_api_token"])(
  "rejects explicit Host credential %s in endpoint configuration before launching",
  async (name) => {
    const s = await setup();
    const endpoint = JSON.parse(await readFile(s.path, "utf8"));
    endpoint.env[name] = "fake-explicit-host-token";
    await writeFile(s.path, JSON.stringify(endpoint));
    await expect(s.open()).rejects.toThrow("Host credentials");
    expect(await s.events()).toEqual([]);
  },
);

it("presents the exact remote tool input before requesting approval", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  const out = observer();
  expect(await a.start(id, "permission", out)).toEqual({ stopReason: "refusal" });
  expect(out.interact).toHaveBeenCalledWith(
    expect.objectContaining({
      title: "Run local fixture",
      approval: expect.objectContaining({
        toolId: "tool-1",
        input: { command: "printf '%s\\n' 'ACP exact approval'", cwd: s.root },
        choices: [
          { id: "yes", label: "Allow once", kind: "allow_once", answer: { optionId: "yes" } },
          { id: "no", label: "Reject once", kind: "reject_once", answer: { optionId: "no" } },
        ],
      }),
    }),
    expect.any(AbortSignal),
  );
  expect((await s.events()).filter((row) => row.event === "tool")).toHaveLength(0);
});

it("keeps opaque permission IDs separate from visible option names and kinds", async () => {
  const s = await setup({ ACP_FIXTURE_OPAQUE_IDS: "1" });
  const a = await s.open();
  const id = await a.create();
  const out = observer();
  out.interact = vi.fn(async () => ({ optionId: "opaque-allow-42" }));
  expect(await a.start(id, "permission", out)).toEqual({ stopReason: "end_turn" });
  expect(out.interact).toHaveBeenCalledWith(
    expect.objectContaining({
      schema: expect.objectContaining({
        properties: {
          optionId: {
            type: "string",
            oneOf: [
              { const: "opaque-allow-42", title: "Allow once (allow_once)" },
              { const: "opaque-deny-99", title: "Reject once (reject_once)" },
            ],
          },
        },
      }),
      approval: expect.objectContaining({
        choices: [
          {
            id: "opaque-allow-42",
            label: "Allow once",
            kind: "allow_once",
            answer: { optionId: "opaque-allow-42" },
          },
          {
            id: "opaque-deny-99",
            label: "Reject once",
            kind: "reject_once",
            answer: { optionId: "opaque-deny-99" },
          },
        ],
      }),
    }),
    expect.any(AbortSignal),
  );
  const rows = await s.events();
  expect(rows.find((row) => row.event === "permission-result").response.outcome).toEqual({
    outcome: "selected",
    optionId: "opaque-allow-42",
  });
  expect(rows.filter((row) => row.event === "tool")).toHaveLength(1);
});

it("accepts native resume without history replay and advertises it truthfully", async () => {
  const s = await setup({ ACP_FIXTURE_MISSING_CAPABILITY: "load" });
  const a = await s.open();
  expect(a.capabilities.history).toBe(false);
  const id = await a.create();
  expect(a.capabilities).toMatchObject({ history: false, list: true, resume: true });
  expect(await a.start(id, "first", observer())).toEqual({ stopReason: "end_turn" });
  await a.dispose();
  const b = await s.open();
  expect((await b.list()).map((entry) => entry.sessionId)).toEqual([id]);
  expect(b.capabilities.history).toBe(false);
  const out = observer();
  await expect(b.read(id, out)).rejects.toThrow("does not support native history replay");
  expect(out.text).not.toHaveBeenCalled();
  expect(await b.start(id, "second", observer())).toEqual({ stopReason: "end_turn" });
  const rows = await s.events();
  expect(rows.filter((row) => row.event === "load")).toHaveLength(0);
  expect(rows.filter((row) => row.event === "new")).toHaveLength(1);
  expect(rows.filter((row) => row.event === "resume")).toHaveLength(1);
  expect(rows.filter((row) => row.event === "prompt").map((row) => row.text)).toEqual([
    "first",
    "second",
  ]);
});

it.each(["permission-observed", "permission-override", "permission-null"])(
  "uses current tool updates and explicit request fields for %s",
  async (prompt) => {
    const s = await setup({ ACP_FIXTURE_OPAQUE_IDS: "1" });
    const a = await s.open();
    const id = await a.create();
    const out = observer();
    out.interact = vi.fn(async () => ({ optionId: "opaque-deny-99" }));
    expect(await a.start(id, prompt, out)).toEqual({ stopReason: "refusal" });
    expect(out.interact).toHaveBeenCalledWith(
      expect.objectContaining({
        title: prompt === "permission-override" ? "Run replacement fixture" : "Run local fixture",
        approval: expect.objectContaining({
          toolId: "tool-1",
          input:
            prompt === "permission-null"
              ? null
              : {
                  command:
                    prompt === "permission-override"
                      ? "printf replacement"
                      : "printf '%s\\n' 'ACP exact approval'",
                  cwd: s.root,
                },
        }),
      }),
      expect.any(AbortSignal),
    );
    const rows = await s.events();
    expect(rows.find((row) => row.event === "permission-result").response.outcome).toEqual({
      outcome: "selected",
      optionId: "opaque-deny-99",
    });
    expect(rows.filter((row) => row.event === "tool")).toHaveLength(0);
  },
);

it("does not borrow approval input from another active session with the same tool ID", async () => {
  const s = await setup();
  const a = await s.open();
  const first = await a.create();
  const second = await a.create();
  const out = observer();
  const answer = Promise.withResolvers<unknown>();
  out.interact = vi.fn(() => answer.promise);
  const run = pendingOperation(a.start(first, "permission-observed", out));
  try {
    await vi.waitFor(() => expect(out.interact).toHaveBeenCalledOnce());
    const other = observer();
    expect(await a.start(second, "permission-unobserved", other)).toEqual({
      stopReason: "refusal",
    });
    expect(other.interact).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "External agent permission",
        approval: expect.not.objectContaining({ input: expect.anything() }),
      }),
      expect.any(AbortSignal),
    );
    expect((await s.events()).filter((row) => row.event === "tool")).toHaveLength(0);
  } finally {
    answer.resolve({ optionId: "no" });
    await run;
  }
});

it("discards approval input from earlier turns and history replay", async () => {
  const s = await setup();
  const a = await s.open();
  const id = await a.create();
  expect(await a.start(id, "first", observer())).toEqual({ stopReason: "end_turn" });
  for (const replay of [false, true]) {
    if (replay) await a.read(id, observer());
    const out = observer();
    expect(await a.start(id, "permission-unobserved", out)).toEqual({ stopReason: "refusal" });
    expect(out.interact).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "External agent permission",
        approval: expect.not.objectContaining({ input: expect.anything() }),
      }),
      expect.any(AbortSignal),
    );
  }
  expect((await s.events()).filter((row) => row.event === "tool")).toHaveLength(1);
});
