import { describe, expect, it, vi } from "vitest";
import { WorkspaceInspectionInvokeContracts } from "../shared/ipc-contracts/workspace-inspection.js";
import { createDesktopIpcRegistrar, createSemanticAuditReceipt } from "./ipc-router.js";
import { registerWorkspaceInspectionIpc } from "./workspace-inspection-ipc.js";

describe("Workspace inspection IPC router", () => {
  it("registers every contract, resolves cwd, and projects host-only fields", async () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        WorkspaceInspectionInvokeContracts[
          channel as keyof typeof WorkspaceInspectionInvokeContracts
        ].audit,
    });
    const review = vi.fn(async () => ({
      root: "/workspace/real",
      branch: null,
      isRepository: false,
      files: [
        {
          path: "src/App.tsx",
          previousPath: "src/OldApp.tsx",
          status: "R ",
          patch: "@@ -1 +1 @@",
          binary: false,
          additions: 1,
          deletions: 1,
          truncated: false,
          error: "partial diff",
          hostOnly: "not transported",
        },
        {
          path: "README.md",
          status: " M",
          patch: "",
          binary: false,
          additions: 0,
          deletions: 0,
          truncated: false,
        },
      ],
      truncated: false,
      error: "partial review",
      hostOnly: "not transported",
    }));
    const listDirectory = vi.fn(async (path = "") => ({
      path,
      entries: [
        {
          name: "App.tsx",
          path: "src/App.tsx",
          kind: "file" as const,
          hostOnly: "not transported",
        },
      ],
      truncated: false,
    }));
    const readFile = vi.fn(async (path: string) => ({
      path,
      content: "export function App() {}",
      size: 24,
      truncated: false,
      sha256: "a".repeat(64),
    }));
    const tools = { root: "/workspace/real", review, listDirectory, readFile };
    const normalizeWorkingDirectory = vi.fn(async (cwd?: string) =>
      cwd ? "/workspace/real" : undefined,
    );
    const toolsFor = vi.fn(() => tools);

    registerWorkspaceInspectionIpc(registrar, {
      workspaceRoot: "/workspace/default",
      normalizeWorkingDirectory,
      toolsFor,
    });
    const invoke = (channel: string, ...args: unknown[]) =>
      handlers.get(channel)?.({}, createSemanticAuditReceipt(), ...args);

    expect([...handlers.keys()]).toEqual(Object.keys(WorkspaceInspectionInvokeContracts));
    expect(invoke("workspace:root")).toBe("/workspace/default");
    await expect(invoke("workspace:review", { cwd: "/workspace/link" })).resolves.toEqual({
      root: "/workspace/real",
      branch: null,
      isRepository: false,
      files: [
        {
          path: "src/App.tsx",
          previousPath: "src/OldApp.tsx",
          status: "R ",
          patch: "@@ -1 +1 @@",
          binary: false,
          additions: 1,
          deletions: 1,
          truncated: false,
          error: "partial diff",
        },
        {
          path: "README.md",
          status: " M",
          patch: "",
          binary: false,
          additions: 0,
          deletions: 0,
          truncated: false,
        },
      ],
      truncated: false,
      error: "partial review",
    });
    await expect(
      invoke("workspace:listDirectory", { path: "src", cwd: "/workspace/link" }),
    ).resolves.toEqual({
      root: "/workspace/real",
      path: "src",
      entries: [{ name: "App.tsx", path: "src/App.tsx", kind: "file" }],
      truncated: false,
    });
    await expect(
      invoke("workspace:readFile", { path: "src/App.tsx", cwd: "/workspace/link" }),
    ).resolves.toEqual({
      root: "/workspace/real",
      path: "src/App.tsx",
      content: "export function App() {}",
      size: 24,
      binary: false,
      truncated: false,
    });
    expect(normalizeWorkingDirectory).toHaveBeenCalledTimes(3);
    expect(toolsFor).toHaveBeenCalledWith("/workspace/real");
    expect(listDirectory).toHaveBeenCalledWith("src");
    expect(readFile).toHaveBeenCalledWith("src/App.tsx");
  });

  it("rejects invalid input before workspace effects and invalid host results before return", async () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: () => "intent_outcome",
    });
    const normalizeWorkingDirectory = vi.fn(async () => undefined);
    const review = vi.fn(async () => ({
      root: "/workspace",
      branch: undefined,
      isRepository: false,
      files: [],
      truncated: false,
    }));
    registerWorkspaceInspectionIpc(registrar, {
      workspaceRoot: "/workspace",
      normalizeWorkingDirectory,
      toolsFor: () => ({
        root: "/workspace",
        review,
        listDirectory: vi.fn(),
        readFile: vi.fn(),
      }),
    });

    const read = handlers.get("workspace:readFile");
    expect(() =>
      read?.({}, createSemanticAuditReceipt(), {
        path: "README.md",
        rawCredential: "secret",
      }),
    ).toThrow(/arguments failed validation/i);
    expect(normalizeWorkingDirectory).not.toHaveBeenCalled();

    const inspect = handlers.get("workspace:review");
    await expect(inspect?.({}, createSemanticAuditReceipt(), {})).rejects.toThrow(
      /result failed validation/i,
    );
    expect(review).toHaveBeenCalledOnce();
  });
});
