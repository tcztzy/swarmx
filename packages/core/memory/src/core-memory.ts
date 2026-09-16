import { lstat, mkdir, readFile, realpath } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import lockfile from "proper-lockfile";
import writeFileAtomic from "write-file-atomic";
import { z } from "zod";
import { MemoryError } from "./errors.js";
import { conceptRevision } from "./markdown.js";

export const coreMemoryUpdateSchema = z.strictObject({
  content: z.string().max(8_800),
  expectedRevision: z.string().regex(/^sha256:[a-f0-9]{64}$/u),
});
export const CORE_MEMORY_LIMIT = 1_375;

export class CoreMemory {
  private readonly root: string;
  private readonly path: string;

  constructor(root: string) {
    this.root = resolve(root);
    this.path = join(this.root, "USER.md");
  }

  async initialize() {
    await mkdir(this.root, { recursive: true, mode: 0o700 });
  }

  async read() {
    let content = "";
    try {
      const info = await lstat(this.path);
      const canonical = await realpath(this.root);
      if (
        !info.isFile() ||
        info.isSymbolicLink() ||
        (await realpath(this.path)) !== this.path.replace(this.root, canonical)
      )
        throw new MemoryError("Core memory must be a regular, unredirected file.", "UNSAFE_PATH");
      if (info.size > CORE_MEMORY_LIMIT * 4)
        throw new MemoryError("Core memory exceeds its character limit.", "INVALID_CONCEPT");
      content = new TextDecoder("utf-8", { fatal: true }).decode(await readFile(this.path));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    if (Array.from(content).length > CORE_MEMORY_LIMIT)
      throw new MemoryError("Core memory exceeds its character limit.", "INVALID_CONCEPT");
    return { content, revision: conceptRevision(content), limit: CORE_MEMORY_LIMIT };
  }

  async update(raw: z.infer<typeof coreMemoryUpdateSchema>, signal?: AbortSignal) {
    const input = coreMemoryUpdateSchema.parse(raw);
    if (Array.from(input.content).length > CORE_MEMORY_LIMIT)
      throw new MemoryError(
        "Core memory is full. Consolidate entries before saving.",
        "INVALID_REQUEST",
      );
    await this.initialize();
    const canonicalRoot = await realpath(this.root);
    if (
      (await realpath(dirname(this.path))) !==
      join(canonicalRoot, relative(this.root, dirname(this.path)))
    )
      throw new MemoryError("Core memory directory is redirected.", "UNSAFE_PATH");
    // proper-lockfile appends ".lock" to the lockfilePath, keeping the lock outside the vault.
    const release = await lockfile.lock(this.path, {
      realpath: false,
      lockfilePath: `${this.root}.lock`,
    });
    try {
      signal?.throwIfAborted();
      const current = await this.read();
      if (current.revision !== input.expectedRevision)
        throw new MemoryError(
          "Core memory changed; read it again before editing.",
          "REVISION_CONFLICT",
        );
      signal?.throwIfAborted();
      await writeFileAtomic(this.path, input.content, { mode: 0o600, fsync: true });
      return this.read();
    } finally {
      await release();
    }
  }
}
