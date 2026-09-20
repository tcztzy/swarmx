import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { BrowserWindow } from "electron";
import { beforeEach, expect, it, vi } from "vitest";
import { LogsEvidencePayloadSchema } from "../src/bridge-contract.js";
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

it("only forwards bounded exact execution references from the trusted application window", async () => {
  const evidence = vi.fn(async ({ sources }: { sources: string[] }) => ({ sources }));
  registerIpc({ operations: { evidence } } as unknown as DesktopPlatform);
  const invoke = handlers.get("swarmx:logs:evidence");
  if (!invoke) throw new Error("Missing evidence IPC handler");
  const event = { sender: {}, senderFrame: { parent: null } };
  const sources = [`urn:swarmx:execution:${randomUUID()}`];
  await expect(invoke(event, { sources })).resolves.toEqual({ sources });
  expect(evidence).toHaveBeenCalledExactlyOnceWith({ sources });
  for (const input of [
    { sources: [] },
    { sources: Array(65).fill(sources[0]) },
    { sources: ["sx:a/figure@1"] },
    { sources: ["urn:swarmx:execution:not-an-event"] },
    { sources, directory: "another-workspace" },
  ])
    expect(LogsEvidencePayloadSchema.safeParse(input).success).toBe(false);
  expect(() => invoke({ ...event, senderFrame: { parent: {} } }, { sources })).toThrow(
    "application window",
  );
  vi.mocked(BrowserWindow.fromWebContents).mockReturnValue(null);
  expect(() => invoke(event, { sources })).toThrow("application window");
  expect(evidence).toHaveBeenCalledTimes(1);
});

it("preserves resolver failures instead of substituting available history", async () => {
  const evidence = vi.fn(async () => {
    throw new Error("Execution source is unavailable in this directory.");
  });
  registerIpc({ operations: { evidence } } as unknown as DesktopPlatform);
  const invoke = handlers.get("swarmx:logs:evidence");
  if (!invoke) throw new Error("Missing evidence IPC handler");
  await expect(
    invoke(
      { sender: {}, senderFrame: { parent: null } },
      {
        sources: [`urn:swarmx:execution:${randomUUID()}`],
      },
    ),
  ).rejects.toThrow("unavailable in this directory");
});

it("exposes logs.evidence through the preload without a general invoke escape hatch", async () => {
  const invoke = vi.fn(async () => ({ records: [] }));
  const expose = vi.fn();
  runInNewContext(readFileSync(new URL("../preload.cjs", import.meta.url), "utf8"), {
    require: () => ({ contextBridge: { exposeInMainWorld: expose }, ipcRenderer: { invoke } }),
  });
  const exposed = expose.mock.calls[0]?.[1];
  const sources = [`urn:swarmx:execution:${randomUUID()}`];
  await exposed.logs.evidence({ sources });
  expect(invoke).toHaveBeenCalledExactlyOnceWith("swarmx:logs:evidence", { sources });
  expect(exposed.invoke).toBeUndefined();
});
