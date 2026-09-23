import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { BrowserWindow } from "electron";
import { beforeEach, expect, it, vi } from "vitest";
import { registerIpc } from "../src/ipc.js";
import type { DesktopPlatform } from "../src/platform.js";

const handlers = vi.hoisted(() => new Map<string, (event: unknown, payload: unknown) => unknown>());
vi.mock("electron", () => ({
  BrowserWindow: { fromWebContents: vi.fn(() => ({})) },
  ipcMain: {
    handle: (name: string, handle: (event: unknown, payload: unknown) => unknown) =>
      handlers.set(name, handle),
  },
}));
beforeEach(() => {
  handlers.clear();
  vi.mocked(BrowserWindow.fromWebContents).mockReturnValue({} as BrowserWindow);
});

const event = { sender: {}, senderFrame: { parent: null } };
const cycle = {
  id: "cycle",
  project: "research",
  budgetUsd: 3,
  concurrency: 1,
  reviewReserveUsd: 0,
  configurations: [{ id: "local", harness: "pi", model: "fixture" }],
};
const feedback = {
  id: "feedback",
  attemptId: "attempt",
  criteriaVersion: "v1",
  verdict: "passed",
  accepted: true,
  fraction: 1,
  report: "User checked the result against the requested analysis.",
  artifacts: [{ id: "sx:analysis/result@1", revision: "1" }],
  intervention: "none",
};

function handler(channel: string) {
  const invoke = handlers.get(`swarmx:work:${channel}`);
  if (!invoke) throw new Error(`Missing work ${channel} IPC handler`);
  return invoke;
}

it.each(["read", "command"])(
  "allows work %s only from the top-level application window",
  async (channel) => {
    const operation = vi.fn(async (input: unknown) => ({ input }));
    registerIpc({
      operations: { workRead: operation, workCommand: operation },
    } as unknown as DesktopPlatform);
    const invoke = handler(channel);
    const input = channel === "read" ? { cycleId: "cycle" } : { action: "start", workId: "item" };
    await expect(invoke(event, input)).resolves.toEqual({ input });
    expect(operation).toHaveBeenCalledExactlyOnceWith(input);
    expect(BrowserWindow.fromWebContents).toHaveBeenCalledWith(event.sender);
    expect(() => invoke({ ...event, senderFrame: { parent: {} } }, input)).toThrow(
      "application window",
    );
    vi.mocked(BrowserWindow.fromWebContents).mockReturnValue(null);
    expect(() => invoke(event, input)).toThrow("application window");
    expect(operation).toHaveBeenCalledTimes(1);
  },
);

it("forwards goal, budget, execution and acceptance commands without dropping their values", async () => {
  const workRead = vi.fn(async () => ({ cycles: [] }));
  const workCommand = vi.fn(async (input: unknown) => ({ input }));
  registerIpc({ operations: { workRead, workCommand } } as unknown as DesktopPlatform);
  await expect(handler("read")(event, {})).resolves.toEqual({ cycles: [] });
  expect(workRead).toHaveBeenCalledExactlyOnceWith({});
  const commands = [
    { action: "createCycle", request: cycle },
    {
      action: "createItem",
      request: {
        id: "item",
        cycleId: "cycle",
        goal: "Compare the changed data with the initial analysis.",
        criteria: "Report the observed difference and its uncertainty.",
        criteriaVersion: "v1",
        taskClass: "analysis",
        priority: 2,
        value: 4,
        risk: "high",
        deadline: "2026-10-01T00:00:00.000Z",
        dependencies: ["initial-analysis"],
        policy: "fixed",
        fixedConfiguration: "local",
        mode: "manual",
        runtime: { budgetUsd: 1, timeoutMs: 60_000 },
      },
    },
    { action: "setBudget", request: { cycleId: "cycle", expectedBudgetUsd: 3, budgetUsd: 4 } },
    {
      action: "revise",
      request: {
        id: "item",
        expectedCriteriaVersion: "v1",
        criteriaVersion: "v2",
        goal: "Repeat the analysis with the corrected exclusions.",
        criteria: "Use the revised exclusions and explain the changed estimate.",
      },
    },
    { action: "start", workId: "item" },
    { action: "startNext", cycleId: "cycle" },
    { action: "stop", workId: "item" },
    { action: "respond", workId: "item", interactionId: "approval", answer: { allow: true } },
    { action: "respond", workId: "item", interactionId: "approval", cancel: true },
    { action: "accept", request: feedback },
    {
      action: "reconcileCharge",
      request: {
        id: "invoice",
        reservationId: "attempt",
        costUsd: 0.8,
        source: "invoice",
        reference: "receipt-1",
      },
    },
    {
      action: "reconcileOutcome",
      request: {
        reservationId: "attempt",
        outcome: "cancelled",
        reference: "User checked native execution status.",
      },
    },
  ];
  for (const input of commands) {
    await expect(handler("command")(event, input)).resolves.toEqual({ input });
    expect(workCommand).toHaveBeenLastCalledWith(input);
  }
  expect(workCommand).toHaveBeenCalledTimes(commands.length);
});

it("rejects malformed budgets, authority claims and directory overrides before operations run", async () => {
  const workRead = vi.fn(async () => ({}));
  const workCommand = vi.fn(async () => ({}));
  registerIpc({ operations: { workRead, workCommand } } as unknown as DesktopPlatform);
  for (const input of [
    null,
    { cycleId: "" },
    { cycleId: "cycle", directory: "/another-workspace" },
  ])
    await expect(Promise.resolve().then(() => handler("read")(event, input))).rejects.toThrow();
  const invalid = [
    ...[-1, NaN, Infinity, "3"].map((budgetUsd) => ({
      action: "createCycle",
      request: { ...cycle, budgetUsd },
    })),
    ...[0, -1, NaN, Infinity].map((budgetUsd) => ({
      action: "start",
      workId: "item",
      options: { runtime: { budgetUsd } },
    })),
    { action: "createCycle", request: { ...cycle, directory: "/another-workspace" } },
    { action: "start", workId: "item", directory: "/another-workspace" },
    { action: "setBudget", request: { cycleId: "cycle", expectedBudgetUsd: 3, budgetUsd: NaN } },
    ...[
      { source: "validator" },
      { source: "user" },
      { layer: "scientific-evidence" },
      { evaluator: "trusted-validator" },
      { evaluatorVersion: "v1" },
      { validator: "trusted-validator" },
      { directory: "/another-workspace" },
    ].map((claim) => ({ action: "accept", request: { ...feedback, ...claim } })),
    { action: "accept", request: { ...feedback, fraction: NaN } },
    { action: "accept", request: { ...feedback, fraction: 2 } },
    { action: "accept", request: { ...feedback, artifacts: [{ id: "artifact" }] } },
    { action: "validator", request: feedback },
  ];
  for (const input of invalid)
    await expect(Promise.resolve().then(() => handler("command")(event, input))).rejects.toThrow();
  expect(workRead).not.toHaveBeenCalled();
  expect(workCommand).not.toHaveBeenCalled();
});

it("preserves work operation failures for the renderer", async () => {
  const workCommand = vi.fn(async () => {
    throw new Error("The work criteria changed. Refresh before accepting this result.");
  });
  registerIpc({ operations: { workCommand } } as unknown as DesktopPlatform);
  await expect(handler("command")(event, { action: "accept", request: feedback })).rejects.toThrow(
    "The work criteria changed",
  );
  expect(workCommand).toHaveBeenCalledTimes(1);
});

it("exposes only named work methods and retains existing preload channels and subscriptions", async () => {
  const invoke = vi.fn(async () => ({ ok: true }));
  const expose = vi.fn();
  const on = vi.fn();
  const removeListener = vi.fn();
  runInNewContext(readFileSync(new URL("../preload.cjs", import.meta.url), "utf8"), {
    require: () => ({
      contextBridge: { exposeInMainWorld: expose },
      ipcRenderer: { invoke, on, removeListener },
    }),
  });
  expect(expose).toHaveBeenCalledTimes(1);
  expect(expose.mock.calls[0]?.[0]).toBe("swarmx");
  const exposed = expose.mock.calls[0]?.[1];
  expect(Object.keys(exposed).sort()).toEqual(
    [
      "bootstrap",
      "tool",
      "cancelTool",
      "settings",
      "language",
      "environment",
      "sessions",
      "models",
      "logs",
      "runs",
      "work",
      "science",
      "agui",
    ].sort(),
  );
  expect(Object.keys(exposed.work).sort()).toEqual(["command", "read"]);
  expect(exposed.invoke).toBeUndefined();
  await expect(exposed.work.read({ cycleId: "cycle" })).resolves.toEqual({ ok: true });
  expect(invoke).toHaveBeenLastCalledWith("swarmx:work:read", { cycleId: "cycle" });
  const command = { action: "createCycle", request: cycle };
  await expect(exposed.work.command(command)).resolves.toEqual({ ok: true });
  expect(invoke).toHaveBeenLastCalledWith("swarmx:work:command", command);
  for (const [operation, channel] of [
    [exposed.bootstrap, "bootstrap"],
    [exposed.tool, "tool"],
    [exposed.cancelTool, "tool:cancel"],
    [exposed.settings.read, "settings:read"],
    [exposed.settings.update, "settings:update"],
    [exposed.language.write, "language:write"],
    [exposed.environment.read, "environment:read"],
    [exposed.environment.act, "environment:act"],
    [exposed.sessions.list, "sessions:list"],
    [exposed.sessions.create, "sessions:create"],
    [exposed.sessions.history, "sessions:history"],
    [exposed.models.read, "models:read"],
    [exposed.logs.read, "logs:read"],
    [exposed.logs.evidence, "logs:evidence"],
    [exposed.runs.control, "runs:control"],
    [exposed.science.workspace, "science:workspace"],
    [exposed.science.researchObject, "science:research-object"],
    [exposed.science.notebookExecutions, "science:notebook-executions"],
    [exposed.science.artifactPreview, "science:artifact-preview"],
    [exposed.science.artifactContent, "science:artifact-content"],
    [exposed.science.import, "science:import"],
    [exposed.agui.start, "agui:start"],
    [exposed.agui.cancel, "agui:cancel"],
  ]) {
    await operation({ fixture: channel });
    expect(invoke).toHaveBeenLastCalledWith(`swarmx:${channel}`, { fixture: channel });
  }
  const listener = vi.fn();
  const unsubscribe = exposed.agui.subscribe(listener);
  const nativeHandler = on.mock.calls[0]?.[1];
  expect(on).toHaveBeenCalledWith("swarmx:agui:event", nativeHandler);
  const message = { threadId: "thread", done: true };
  nativeHandler({}, message);
  expect(listener).toHaveBeenCalledExactlyOnceWith(message);
  unsubscribe();
  expect(removeListener).toHaveBeenCalledExactlyOnceWith("swarmx:agui:event", nativeHandler);
});
