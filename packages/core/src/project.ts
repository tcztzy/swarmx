import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";

export const ProjectDataSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  cwd: z.string().min(1),
  pinned: z.boolean().default(false),
  createdAt: z.string(),
  updatedAt: z.string(),
  removedAt: z.string().optional(),
});

export type ProjectData = z.infer<typeof ProjectDataSchema>;

const PROJECTS_DIR = path.join(homedir(), ".swarmx");
const PROJECTS_FILE = path.join(PROJECTS_DIR, "projects.json");

export function registerProject(cwd: string, name?: string): ProjectData {
  const projectRoot = normalizeProjectRoot(cwd);
  const projects = readProjects();
  const existing = projects.find((project) => project.cwd === projectRoot);
  if (existing && !existing.removedAt) return existing;

  const now = new Date().toISOString();
  const project = ProjectDataSchema.parse(
    existing
      ? {
          ...existing,
          name: name?.trim() || existing.name,
          updatedAt: now,
          removedAt: undefined,
        }
      : {
          id: uuidv4(),
          name: name?.trim() || path.basename(projectRoot) || projectRoot,
          cwd: projectRoot,
          createdAt: now,
          updatedAt: now,
        },
  );
  writeProjects(existing ? replaceProject(projects, project) : [...projects, project]);
  return project;
}

export function registerDefaultProject(cwd: string, name?: string): ProjectData | null {
  const projectRoot = normalizeProjectRoot(cwd);
  const existing = readProjects().find((project) => project.cwd === projectRoot);
  if (existing?.removedAt) return null;
  return existing ?? registerProject(projectRoot, name);
}

export function listProjects(): ProjectData[] {
  return sortProjects(readProjects().filter((project) => !project.removedAt));
}

export function renameProject(id: string, name: string): ProjectData | null {
  const nextName = name.trim();
  if (!nextName) throw new Error("Project name must not be empty");
  return updateProject(id, (project, now) => ({ ...project, name: nextName, updatedAt: now }));
}

export function setProjectPinned(id: string, pinned: boolean): ProjectData | null {
  return updateProject(id, (project, now) => ({ ...project, pinned, updatedAt: now }));
}

export function dismissProject(id: string): boolean {
  const project = updateProject(id, (current, now) => ({
    ...current,
    pinned: false,
    updatedAt: now,
    removedAt: now,
  }));
  return project !== null;
}

export function removeProject(id: string): boolean {
  const projects = readProjects();
  const next = projects.filter((project) => project.id !== id);
  if (next.length === projects.length) return false;
  writeProjects(next);
  return true;
}

function readProjects(): ProjectData[] {
  ensureProjectsDir();
  if (!fs.existsSync(PROJECTS_FILE)) return [];

  try {
    const parsed = z
      .array(ProjectDataSchema)
      .safeParse(JSON.parse(fs.readFileSync(PROJECTS_FILE, "utf8")));
    if (!parsed.success) return [];
    return parsed.data;
  } catch {
    return [];
  }
}

function updateProject(
  id: string,
  update: (project: ProjectData, now: string) => ProjectData,
): ProjectData | null {
  const projects = readProjects();
  const current = projects.find((project) => project.id === id && !project.removedAt);
  if (!current) return null;
  const next = ProjectDataSchema.parse(update(current, new Date().toISOString()));
  writeProjects(replaceProject(projects, next));
  return next;
}

function replaceProject(projects: ProjectData[], next: ProjectData): ProjectData[] {
  return projects.map((project) => (project.id === next.id ? next : project));
}

function sortProjects(projects: ProjectData[]): ProjectData[] {
  return [...projects].sort((left, right) => {
    if (left.pinned !== right.pinned) return left.pinned ? -1 : 1;
    return left.name.localeCompare(right.name);
  });
}

function normalizeProjectRoot(cwd: string): string {
  const resolved = path.resolve(cwd.trim());
  const stat = fs.statSync(resolved);
  if (!stat.isDirectory()) throw new Error(`Project path must be a directory: ${resolved}`);
  return fs.realpathSync(resolved);
}

function ensureProjectsDir(): void {
  fs.mkdirSync(PROJECTS_DIR, { recursive: true, mode: 0o700 });
}

function writeProjects(projects: ProjectData[]): void {
  ensureProjectsDir();
  const temporaryFile = `${PROJECTS_FILE}.${process.pid}.tmp`;
  fs.writeFileSync(temporaryFile, `${JSON.stringify(projects, null, 2)}\n`, { mode: 0o600 });
  fs.renameSync(temporaryFile, PROJECTS_FILE);
}
