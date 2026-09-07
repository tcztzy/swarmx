import { createHash, randomUUID } from "node:crypto";
import {
  closeSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { z } from "zod";
import { MemorySettingsSchema } from "../memory.js";
import {
  AddProjectSchema,
  DEFAULT_POLICY,
  LanguageSchema,
  ProjectCatalogSchema,
  ProjectSchema,
  type WorkspaceSettings,
  WorkspaceSettingsSchema,
} from "../settings.js";

export async function resolveWorkspace(path: string) {
  const root = await realpath(resolve(path));
  if (!(await stat(root)).isDirectory()) throw new Error("Workspace is not a directory.");
  return {
    root,
    label: basename(root),
    id: createHash("sha256").update(root).digest("hex").slice(0, 12),
  };
}

export class ProjectStore {
  private readonly path: string;
  constructor(productHome: string) {
    this.path = join(productHome, "projects.json");
  }

  read(): z.infer<typeof ProjectCatalogSchema> {
    try {
      return ProjectCatalogSchema.parse(JSON.parse(readFileSync(this.path, "utf8")));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      return { projects: [], activeId: null };
    }
  }

  get(id: string) {
    const project = this.read().projects.find((project) => project.id === id);
    if (!project) throw new Error("Project not found.");
    return project;
  }

  register(raw: z.infer<typeof ProjectSchema>) {
    const project = ProjectSchema.parse(raw);
    const catalog = this.read();
    const existing = catalog.projects.find(({ root }) => root === project.root);
    if (existing) return existing;
    const next = ProjectCatalogSchema.parse({
      projects: [...catalog.projects, project],
      activeId: catalog.activeId ?? project.id,
    });
    writePrivateJson(this.path, next);
    return project;
  }

  async add(raw: unknown) {
    const input = AddProjectSchema.parse(raw);
    return this.register({ ...(await resolveWorkspace(input.root)), label: input.label });
  }

  select(id: string) {
    const project = this.get(id);
    writePrivateJson(this.path, { ...this.read(), activeId: id });
    return project;
  }
}

export class SettingsStore {
  private value: WorkspaceSettings;
  readonly path: string;
  private readonly languagePath: string;

  constructor(productHome: string, workspaceId: string) {
    this.path = join(productHome, "workspaces", workspaceId, "settings.json");
    this.languagePath = join(productHome, "language.json");
    try {
      const raw: unknown = JSON.parse(readFileSync(this.path, "utf8"));
      const legacy = z.object({ policy: z.object({ approval: z.string() }) }).safeParse(raw);
      if (legacy.success)
        throw new Error(
          `Legacy native permission policy in ${this.path}. Remove policy.approval and set policy.tools explicitly; filesystem now controls only the research container. See docs/permissions.md.`,
        );
      this.value = WorkspaceSettingsSchema.parse(raw);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      this.value = { policy: { ...DEFAULT_POLICY }, environment: null };
    }
  }

  read(): WorkspaceSettings {
    return structuredClone(this.value);
  }

  readMemory() {
    try {
      return MemorySettingsSchema.parse(
        JSON.parse(readFileSync(join(dirname(this.path), "memory.json"), "utf8")),
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      return MemorySettingsSchema.parse({});
    }
  }

  writeMemory(raw: unknown) {
    const value = MemorySettingsSchema.parse(raw);
    writePrivateJson(join(dirname(this.path), "memory.json"), value);
    return value;
  }

  write(value: WorkspaceSettings): WorkspaceSettings {
    const parsed = WorkspaceSettingsSchema.parse(value);
    writePrivateJson(this.path, parsed);
    this.value = parsed;
    return this.read();
  }

  readLanguage(): "zh" | "en" | null {
    try {
      return LanguageSchema.parse(JSON.parse(readFileSync(this.languagePath, "utf8")));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      return null;
    }
  }

  writeLanguage(language: "zh" | "en"): void {
    writePrivateJson(this.languagePath, LanguageSchema.parse(language));
  }
}

function writePrivateJson(path: string, value: unknown) {
  const directory = dirname(path);
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  const temporary = `${path}.${randomUUID()}`;
  const descriptor = openSync(temporary, "wx", 0o600);
  try {
    writeFileSync(descriptor, `${JSON.stringify(value, null, 2)}\n`);
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
  renameSync(temporary, path);
}
