import { mkdir } from "node:fs/promises";
import path from "node:path";
import { archiveProjectSessions, listSessionSummaries } from "@swarmx/core";
import {
  dismissProject,
  listProjects,
  type ProjectData,
  registerDefaultProject,
  registerProject,
  renameProject,
  setProjectPinned,
} from "@swarmx/core/project";
import { dialog, shell } from "electron";

export interface ProjectServiceLike {
  list(): ProjectData[];
  addExisting(): Promise<ProjectData | null>;
  createScratch(): Promise<ProjectData | null>;
  setPinned(id: string, pinned: boolean): ProjectData;
  rename(id: string, name: string): ProjectData;
  reveal(id: string): boolean;
  archiveTasks(id: string): number;
  remove(id: string): boolean;
}

export interface ProjectServiceOptions {
  workspaceRoot: string;
  isSessionRunning(id: string): boolean;
  clearSideChats(parentSessionId: string): void;
}

export class ProjectService implements ProjectServiceLike {
  constructor(private readonly options: ProjectServiceOptions) {}

  list(): ProjectData[] {
    registerDefaultProject(this.options.workspaceRoot);
    return listProjects();
  }

  async addExisting(): Promise<ProjectData | null> {
    const result = await dialog.showOpenDialog({
      title: "Use an existing project folder",
      buttonLabel: "Use folder",
      defaultPath: this.options.workspaceRoot,
      properties: ["openDirectory", "createDirectory"],
    });
    const cwd = result.filePaths[0];
    return result.canceled || !cwd ? null : registerProject(cwd);
  }

  async createScratch(): Promise<ProjectData | null> {
    const result = await dialog.showSaveDialog({
      title: "Create a new project",
      buttonLabel: "Create project",
      defaultPath: path.join(path.dirname(this.options.workspaceRoot), "untitled-project"),
      nameFieldLabel: "Project name",
      properties: ["createDirectory"],
    });
    if (result.canceled || !result.filePath) return null;
    await mkdir(result.filePath);
    return registerProject(result.filePath);
  }

  setPinned(id: string, pinned: boolean): ProjectData {
    return requiredProject(setProjectPinned(id, pinned), id);
  }

  rename(id: string, name: string): ProjectData {
    return requiredProject(renameProject(id, name), id);
  }

  reveal(id: string): boolean {
    const project = listProjects().find((candidate) => candidate.id === id);
    if (!project) return false;
    shell.showItemInFolder(project.cwd);
    return true;
  }

  archiveTasks(id: string): number {
    const project = requiredProject(
      listProjects().find((candidate) => candidate.id === id) ?? null,
      id,
    );
    const sessions = listSessionSummaries().filter((session) => belongsToProject(session, project));
    if (sessions.some((session) => this.options.isSessionRunning(session.id))) {
      throw new Error("Stop all running tasks in this project before archiving them.");
    }
    const archived = archiveProjectSessions({ projectId: project.id, cwd: project.cwd });
    for (const session of sessions) this.options.clearSideChats(session.id);
    return archived;
  }

  remove(id: string): boolean {
    return dismissProject(id);
  }
}

function requiredProject(project: ProjectData | null, id: string): ProjectData {
  if (!project) throw new Error(`Unknown project: ${id}`);
  return project;
}

function belongsToProject(
  session: { projectId?: string; cwd?: string },
  project: ProjectData,
): boolean {
  return (
    session.projectId === project.id ||
    Boolean(session.cwd && path.resolve(session.cwd) === path.resolve(project.cwd))
  );
}
