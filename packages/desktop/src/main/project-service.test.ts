import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";

const adapters = vi.hoisted(() => ({
  archiveSessions: vi.fn(),
  dismiss: vi.fn(),
  list: vi.fn(),
  register: vi.fn(),
  registerDefault: vi.fn(),
  rename: vi.fn(),
  sessions: vi.fn(),
  setPinned: vi.fn(),
  showItemInFolder: vi.fn(),
  showOpenDialog: vi.fn(),
  showSaveDialog: vi.fn(),
}));

vi.mock("electron", () => ({
  dialog: {
    showOpenDialog: adapters.showOpenDialog,
    showSaveDialog: adapters.showSaveDialog,
  },
  shell: { showItemInFolder: adapters.showItemInFolder },
}));
vi.mock("@swarmx/core", () => ({
  archiveProjectSessions: adapters.archiveSessions,
  listSessionSummaries: adapters.sessions,
}));
vi.mock("@swarmx/core/project", () => ({
  dismissProject: adapters.dismiss,
  listProjects: adapters.list,
  registerDefaultProject: adapters.registerDefault,
  registerProject: adapters.register,
  renameProject: adapters.rename,
  setProjectPinned: adapters.setPinned,
}));

import { ProjectService } from "./project-service.js";

const project = {
  id: "project-1",
  name: "Project One",
  cwd: "/workspace/project-one",
  pinned: false,
  createdAt: "2026-08-13T00:00:00.000Z",
  updatedAt: "2026-08-13T00:00:00.000Z",
};

beforeEach(() => {
  for (const adapter of Object.values(adapters)) adapter.mockReset();
  adapters.archiveSessions.mockReturnValue(0);
  adapters.dismiss.mockReturnValue(true);
  adapters.list.mockReturnValue([project]);
  adapters.register.mockReturnValue(project);
  adapters.registerDefault.mockReturnValue(project);
  adapters.rename.mockReturnValue(project);
  adapters.sessions.mockReturnValue([]);
  adapters.setPinned.mockReturnValue(project);
});

describe("Project service", () => {
  it("registers the default Project before listing and uses soft removal", () => {
    const { service } = createService();

    expect(service.list()).toEqual([project]);
    expect(adapters.registerDefault).toHaveBeenCalledWith("/workspace/default");
    expect(adapters.list).toHaveBeenCalledAfter(adapters.registerDefault);
    expect(service.remove(project.id)).toBe(true);
    expect(adapters.dismiss).toHaveBeenCalledWith(project.id);
    adapters.dismiss.mockReturnValueOnce(false);
    expect(service.remove("missing")).toBe(false);
  });

  it("registers an existing folder only after explicit selection", async () => {
    const { service } = createService();
    adapters.showOpenDialog
      .mockResolvedValueOnce({ canceled: true, filePaths: ["/ignored"] })
      .mockResolvedValueOnce({ canceled: false, filePaths: [] })
      .mockResolvedValueOnce({ canceled: false, filePaths: ["/workspace/existing"] });

    await expect(service.addExisting()).resolves.toBeNull();
    await expect(service.addExisting()).resolves.toBeNull();
    await expect(service.addExisting()).resolves.toEqual(project);
    expect(adapters.register).toHaveBeenCalledTimes(1);
    expect(adapters.register).toHaveBeenCalledWith("/workspace/existing");
    expect(adapters.showOpenDialog).toHaveBeenCalledWith({
      title: "Use an existing project folder",
      buttonLabel: "Use folder",
      defaultPath: "/workspace/default",
      properties: ["openDirectory", "createDirectory"],
    });
  });

  it("creates a confirmed scratch folder before registering it", async () => {
    const parent = await mkdtemp(path.join(tmpdir(), "swarmx-project-service-"));
    const scratch = path.join(parent, "scratch");
    const { service } = createService(parent);
    adapters.showSaveDialog
      .mockResolvedValueOnce({ canceled: true, filePath: "/ignored" })
      .mockResolvedValueOnce({ canceled: false })
      .mockResolvedValueOnce({ canceled: false, filePath: scratch });
    adapters.register.mockImplementationOnce((cwd: string) => {
      expect(existsSync(cwd)).toBe(true);
      return { ...project, cwd };
    });

    try {
      await expect(service.createScratch()).resolves.toBeNull();
      await expect(service.createScratch()).resolves.toBeNull();
      await expect(service.createScratch()).resolves.toMatchObject({ cwd: scratch });
      expect(adapters.register).toHaveBeenCalledTimes(1);
      expect(adapters.showSaveDialog).toHaveBeenCalledWith({
        title: "Create a new project",
        buttonLabel: "Create project",
        defaultPath: path.join(path.dirname(parent), "untitled-project"),
        nameFieldLabel: "Project name",
        properties: ["createDirectory"],
      });
    } finally {
      await rm(parent, { recursive: true, force: true });
    }
  });

  it("does not register a scratch Project when directory creation fails", async () => {
    const parent = await mkdtemp(path.join(tmpdir(), "swarmx-project-service-failure-"));
    const blocked = path.join(parent, "missing-parent", "scratch");
    const { service } = createService(parent);
    adapters.showSaveDialog.mockResolvedValueOnce({ canceled: false, filePath: blocked });

    try {
      await expect(service.createScratch()).rejects.toThrow();
      expect(adapters.register).not.toHaveBeenCalled();
    } finally {
      await rm(parent, { recursive: true, force: true });
    }
  });

  it("forwards valid mutations and rejects unknown Projects", () => {
    const { service } = createService();

    expect(service.setPinned(project.id, true)).toEqual(project);
    expect(adapters.setPinned).toHaveBeenCalledWith(project.id, true);
    expect(service.rename(project.id, "Renamed")).toEqual(project);
    expect(adapters.rename).toHaveBeenCalledWith(project.id, "Renamed");

    adapters.setPinned.mockReturnValueOnce(null);
    adapters.rename.mockReturnValueOnce(null);
    expect(() => service.setPinned("missing", false)).toThrow("Unknown project: missing");
    expect(() => service.rename("missing", "Name")).toThrow("Unknown project: missing");
  });

  it("reveals only registered Projects", () => {
    const { service } = createService();

    expect(service.reveal(project.id)).toBe(true);
    expect(adapters.showItemInFolder).toHaveBeenCalledWith(project.cwd);
    adapters.list.mockReturnValueOnce([]);
    expect(service.reveal("missing")).toBe(false);
    expect(adapters.showItemInFolder).toHaveBeenCalledTimes(1);
  });

  it("blocks running Project Sessions and clears only archived parents", () => {
    const { service, isSessionRunning, clearSideChats } = createService();
    adapters.sessions.mockReturnValue([
      { id: "session-1", projectId: project.id, cwd: project.cwd },
      { id: "session-2", cwd: project.cwd },
      { id: "empty-cwd", cwd: "" },
      { id: "other", projectId: "other-project", cwd: "/workspace/other" },
    ]);
    isSessionRunning.mockImplementation((id) => id === "session-1");

    expect(() => service.archiveTasks(project.id)).toThrow(/Stop all running tasks/);
    expect(adapters.archiveSessions).not.toHaveBeenCalled();
    expect(clearSideChats).not.toHaveBeenCalled();

    isSessionRunning.mockReturnValue(false);
    adapters.archiveSessions.mockReturnValue(2);
    expect(service.archiveTasks(project.id)).toBe(2);
    expect(adapters.archiveSessions).toHaveBeenCalledWith({
      projectId: project.id,
      cwd: project.cwd,
    });
    expect(clearSideChats.mock.calls).toEqual([["session-1"], ["session-2"]]);
  });

  it("rejects archival for an unknown Project before reading Sessions", () => {
    const { service } = createService();
    adapters.list.mockReturnValue([]);

    expect(() => service.archiveTasks("missing")).toThrow("Unknown project: missing");
    expect(adapters.sessions).not.toHaveBeenCalled();
  });

  it("does not clear Side Chats when archival fails", () => {
    const { service, clearSideChats } = createService();
    adapters.sessions.mockReturnValue([{ id: "session-1", projectId: project.id }]);
    adapters.archiveSessions.mockImplementationOnce(() => {
      throw new Error("archive failed");
    });

    expect(() => service.archiveTasks(project.id)).toThrow("archive failed");
    expect(clearSideChats).not.toHaveBeenCalled();
  });
});

function createService(workspaceRoot = "/workspace/default") {
  const isSessionRunning = vi.fn(() => false);
  const clearSideChats = vi.fn();
  return {
    service: new ProjectService({ workspaceRoot, isSessionRunning, clearSideChats }),
    isSessionRunning,
    clearSideChats,
  };
}
