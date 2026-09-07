import { lstat, mkdir, readFile, realpath } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import lockfile from "proper-lockfile";
import writeFileAtomic from "write-file-atomic";
import { z } from "zod";
import { MemoryError } from "./errors.js";
import { conceptRevision } from "./markdown.js";

export const coreMemoryTargetSchema = z.enum(["user", "workspace"]);
export const coreMemoryUpdateSchema = z.strictObject({
  target: coreMemoryTargetSchema,
  content: z.string().max(8_800),
  expectedRevision: z.string().regex(/^sha256:[a-f0-9]{64}$/u),
});
export const CORE_MEMORY_LIMITS = { user: 1375, workspace: 2200 } as const;

export class CoreMemory {
  private readonly root: string;
  constructor(
    root: string,
    private readonly workspaceId: string,
  ) {
    this.root = resolve(root);
    z.string()
      .regex(/^[a-zA-Z0-9_-]{1,128}$/u)
      .parse(workspaceId);
  }

  private path(target: z.infer<typeof coreMemoryTargetSchema>) {
    return target === "user"
      ? join(this.root, "USER.md")
      : join(this.root, "workspaces", this.workspaceId, "MEMORY.md");
  }

  async initialize() {
    await mkdir(join(this.root, "workspaces", this.workspaceId), { recursive: true, mode: 0o700 });
  }

  async read(target: z.infer<typeof coreMemoryTargetSchema>) {
    coreMemoryTargetSchema.parse(target);
    const path = this.path(target);
    let content = "";
    try {
      const info = await lstat(path);
      const canonical = await realpath(this.root);
      if (
        !info.isFile() ||
        info.isSymbolicLink() ||
        (await realpath(path)) !== path.replace(this.root, canonical)
      )
        throw new MemoryError("Core memory must be a regular, unredirected file.", "UNSAFE_PATH");
      if (info.size > CORE_MEMORY_LIMITS[target] * 4)
        throw new MemoryError("Core memory exceeds its character limit.", "INVALID_CONCEPT");
      content = new TextDecoder("utf-8", { fatal: true }).decode(await readFile(path));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    if (Array.from(content).length > CORE_MEMORY_LIMITS[target])
      throw new MemoryError("Core memory exceeds its character limit.", "INVALID_CONCEPT");
    return {
      target,
      content,
      revision: conceptRevision(content),
      limit: CORE_MEMORY_LIMITS[target],
    };
  }

  async update(raw: z.infer<typeof coreMemoryUpdateSchema>, signal?: AbortSignal) {
    const input = coreMemoryUpdateSchema.parse(raw);
    if (Array.from(input.content).length > CORE_MEMORY_LIMITS[input.target])
      throw new MemoryError(
        "Core memory is full. Consolidate entries before saving.",
        "INVALID_REQUEST",
      );
    await this.initialize();
    const canonicalRoot = await realpath(this.root);
    if (
      (await realpath(dirname(this.path(input.target)))) !==
      join(canonicalRoot, relative(this.root, dirname(this.path(input.target))))
    )
      throw new MemoryError("Core memory directory is redirected.", "UNSAFE_PATH");
    const release = await lockfile.lock(this.path(input.target), { realpath: false });
    try {
      signal?.throwIfAborted();
      const current = await this.read(input.target);
      if (current.revision !== input.expectedRevision)
        throw new MemoryError(
          "Core memory changed; read it again before editing.",
          "REVISION_CONFLICT",
        );
      signal?.throwIfAborted();
      await writeFileAtomic(this.path(input.target), input.content, { mode: 0o600, fsync: true });
      return this.read(input.target);
    } finally {
      await release();
    }
  }
}
