import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { chmod, lstat, open, realpath, rename, unlink } from "node:fs/promises";
import { basename, dirname, isAbsolute, join } from "node:path";
import { evaluationSchema } from "@swarmx/memory";
import { z } from "zod";

const revision = (content: string) =>
  `sha256:${createHash("sha256").update(content).digest("hex")}`;
const RevisionSchema = z.string().regex(/^sha256:[a-f0-9]{64}$/u);
const ContentSchema = z
  .string()
  .min(1)
  .refine((value) => Buffer.byteLength(value) <= 131_072);
const ResourceSchema = z.strictObject({
  id: z.string().min(1).max(128),
  kind: z.enum(["agent", "skill"]),
  path: z.string().min(1).max(2048).regex(/\.md$/iu),
  validate: z.array(z.string().min(1).max(4096)).min(1).max(32),
});
const ConfigSchema = z.strictObject({ resources: z.array(ResourceSchema).max(10) });
export const ResourceUpdateSchema = z.strictObject({
  action: z.literal("update_resource"),
  request: z.strictObject({
    id: ResourceSchema.shape.id,
    expectedRevision: RevisionSchema,
    content: ContentSchema,
    evaluation: evaluationSchema.optional(),
  }),
});
export const ResourceSnapshotSchema = ResourceSchema.extend({
  configurationRevision: RevisionSchema,
  expectedRevision: RevisionSchema,
  content: ContentSchema,
});
export type ResourceSnapshot = z.infer<typeof ResourceSnapshotSchema>;

export class LearningResources {
  private readonly applying = new Set<string>();
  constructor(private readonly cwd: string) {}

  private async read(path: string, signal: AbortSignal) {
    signal.throwIfAborted();
    if (
      isAbsolute(path) ||
      /\\/u.test(path) ||
      path.split("/").some((part) => !part || part === "." || part === "..")
    )
      throw new Error("Learning resource path must stay inside the project.");
    let target = await realpath(this.cwd);
    for (const part of path.split("/")) {
      target = join(target, part);
      if ((await lstat(target)).isSymbolicLink())
        throw new Error("Learning resources cannot use symbolic links.");
    }
    const file = await open(target, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const status = await file.stat();
      if (!status.isFile() || status.size > 131_072)
        throw new Error("Learning resource must be a regular file of at most 128 KiB.");
      const content = new TextDecoder("utf-8", { fatal: true }).decode(
        await file.readFile({ signal }),
      );
      return { target, content, mode: status.mode & 0o777 };
    } finally {
      await file.close();
    }
  }

  private async configuration(signal: AbortSignal) {
    let content: string;
    try {
      ({ content } = await this.read(".swarmx/learning.json", signal));
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") return undefined;
      throw error;
    }
    const config = ConfigSchema.parse(JSON.parse(content));
    if (
      new Set(config.resources.map(({ id }) => id)).size !== config.resources.length ||
      new Set(config.resources.map(({ path }) => path)).size !== config.resources.length
    )
      throw new Error("Learning resource IDs and paths must be unique.");
    return { ...config, revision: revision(content) };
  }

  async snapshot(signal: AbortSignal): Promise<ResourceSnapshot[]> {
    const config = await this.configuration(signal);
    if (!config) return [];
    const snapshots: ResourceSnapshot[] = [];
    for (const resource of config.resources) {
      const { content } = await this.read(resource.path, signal);
      ContentSchema.parse(content);
      snapshots.push({
        ...resource,
        configurationRevision: config.revision,
        expectedRevision: revision(content),
        content,
      });
      if (JSON.stringify(snapshots).length > 32_000)
        throw new Error("Learning resource snapshots exceed 32,000 characters.");
    }
    return snapshots;
  }

  async apply(
    request: z.infer<typeof ResourceUpdateSchema>["request"],
    snapshot: ResourceSnapshot,
    signal: AbortSignal,
  ) {
    const input = ResourceUpdateSchema.shape.request.parse(request);
    if (
      input.id !== snapshot.id ||
      input.expectedRevision !== snapshot.expectedRevision ||
      revision(snapshot.content) !== snapshot.expectedRevision
    )
      throw new Error("Resource update does not match the supplied snapshot revision.");
    if (this.applying.has(input.id)) throw new Error("Resource update is already running.");
    this.applying.add(input.id);
    try {
      const check = async () => {
        signal.throwIfAborted();
        const config = await this.configuration(signal);
        const resource = config?.resources.find(({ id }) => id === snapshot.id);
        if (
          config?.revision !== snapshot.configurationRevision ||
          !resource ||
          resource.path !== snapshot.path ||
          resource.kind !== snapshot.kind ||
          JSON.stringify(resource.validate) !== JSON.stringify(snapshot.validate)
        )
          throw new Error("Learning resource registration changed.");
        const current = await this.read(resource.path, signal);
        if (
          revision(current.content) !== input.expectedRevision &&
          current.content !== input.content
        )
          throw new Error("Learning resource revision changed.");
        return current;
      };
      const original = await check();
      const candidate = join(
        dirname(original.target),
        `.${basename(original.target)}.${randomUUID()}.candidate.md`,
      );
      const file = await open(candidate, "wx", 0o600);
      try {
        try {
          await file.writeFile(input.content);
          await file.sync();
        } finally {
          await file.close();
        }
        const validationSignal = AbortSignal.any([signal, AbortSignal.timeout(30_000)]);
        validationSignal.throwIfAborted();
        const [command, ...args] = snapshot.validate;
        if (!command) throw new Error("Missing learning resource validator.");
        await new Promise<void>((resolve, reject) => {
          const child = spawn(command, [...args, candidate], {
            cwd: this.cwd,
            stdio: "ignore",
            signal: validationSignal,
            killSignal: "SIGKILL",
            shell: false,
          });
          let failure: Error | undefined;
          child.on("error", (error) => {
            failure = error;
          });
          child.on("close", (code) => {
            if (failure) reject(failure);
            else if (code !== 0)
              reject(new Error(`Learning resource validation failed (exit ${code}).`));
            else resolve();
          });
        });
        const validated = await this.read(
          join(dirname(snapshot.path), basename(candidate)),
          signal,
        );
        if (validated.content !== input.content)
          throw new Error("Learning resource validator changed the candidate.");
        await chmod(candidate, original.mode);
        const current = await check();
        const changed = current.content !== input.content;
        if (changed) {
          signal.throwIfAborted();
          await rename(candidate, current.target);
        }
        return {
          id: input.id,
          path: snapshot.path,
          revision: revision(input.content),
          changed,
          validation: "passed" as const,
        };
      } finally {
        await unlink(candidate).catch((error: unknown) => {
          if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error;
        });
      }
    } finally {
      this.applying.delete(input.id);
    }
  }
}
